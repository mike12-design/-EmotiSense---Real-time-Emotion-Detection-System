"""
完整情绪识别模型评估脚本

对比三种模型：
1. HSEmotion 单体
2. 决策融合 - 规则基（Rule-based）
3. 决策融合 - 元学习器（Meta-Learner）

用法：
python evaluate_all_models.py \
  --dataset-path /Users/asahiyang/Downloads/FacialEmotion/dataset \
  --samples-per-class 100
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteEmotionEvaluator:
    """完整情绪识别评估器"""

    EMOTION_LABELS = {
        'Angry': 'angry',
        'Happy': 'happy',
        'Neutral': 'neutral',
        'Sad': 'sad',
        'Surprise': 'surprise'
    }

    LABEL_TO_IDX = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3,
        'surprise': 4
    }
    IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

    def __init__(self, dataset_path: str, samples_per_class: int = 100, fusion_k: float = 0.5):
        self.dataset_path = Path(dataset_path)
        self.samples_per_class = samples_per_class
        self.fusion_k = fusion_k
        self.test_data = []
        self.ground_truth = []

        # 三个模型
        self.hsemotion = None
        self.rule_fusion_detector = None  # 规则融合
        self.meta_fusion_detector = None  # 元学习器融合
        self.device = None

        self._init_models()

    def _init_models(self):
        """初始化所有模型"""
        try:
            import torch
            from core.advanced_detectors import HSEmotionDetector
            from core.decision_fusion_detector import create_improved_decision_fusion_detector
            from core.config import Config

            # 初始化设备
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Using device: {self.device}")

            config = Config()

            # 1. HSEmotion 单体
            logger.info("🔧 初始化 HSEmotion 单体模型...")
            self.hsemotion = HSEmotionDetector(config)
            logger.info("✅ HSEmotion 模型加载成功")

            # 2. 规则融合检测器
            logger.info("🔧 初始化规则融合检测器...")
            self.rule_fusion_detector = create_improved_decision_fusion_detector(
                config, use_meta_learner=False
            )
            logger.info("✅ 规则融合检测器加载成功")

            # 3. 元学习器融合检测器
            logger.info("🔧 初始化元学习器融合检测器...")
            self.meta_fusion_detector = create_improved_decision_fusion_detector(
                config, use_meta_learner=True
            )
            logger.info("✅ 元学习器融合检测器加载成功")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败：{e}")
            raise

    def load_dataset(self) -> Tuple[List, List]:
        """加载并打乱数据集"""
        logger.info(f"📁 从 {self.dataset_path} 加载数据集...")

        import random

        test_images = []
        gt_labels = []

        for folder_name, emotion_label in self.EMOTION_LABELS.items():
            folder_path = self.dataset_path / folder_name

            if not folder_path.exists():
                logger.warning(f"⚠️ 文件夹不存在：{folder_path}")
                continue

            image_files = list(folder_path.glob('*.jpg')) + \
                         list(folder_path.glob('*.png')) + \
                         list(folder_path.glob('*.jpeg'))

            sampled_files = random.sample(
                image_files,
                min(self.samples_per_class, len(image_files))
            )

            logger.info(f"  {folder_name}: 加载 {len(sampled_files)} 张图像")

            for img_path in sampled_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    img = cv2.resize(img, (224, 224))
                    test_images.append(img)
                    gt_labels.append(emotion_label)

                except Exception as e:
                    logger.warning(f"⚠️ 加载图像失败 {img_path}: {e}")
                    continue

        # 打乱数据集
        combined = list(zip(test_images, gt_labels))
        random.shuffle(combined)
        self.test_data, self.ground_truth = zip(*combined)
        self.test_data = list(self.test_data)
        self.ground_truth = list(self.ground_truth)

        logger.info(f"✅ 数据集加载完成：共 {len(self.test_data)} 张图像")

        return self.test_data, self.ground_truth

    def evaluate_hsemotion(self) -> Dict:
        """评估 HSEmotion 单体"""
        logger.info("🔬 开始评估 HSEmotion 单体模型...")

        predictions = []
        confidences = []

        for idx, img in enumerate(tqdm(self.test_data, desc="HSEmotion")):
            try:
                emotions = self.hsemotion.get_all_emotions(img)

                if not emotions:
                    predictions.append('neutral')
                    confidences.append(0.0)
                    continue

                top_emotion = max(emotions.items(), key=lambda x: x[1])
                predictions.append(top_emotion[0])
                confidences.append(float(top_emotion[1]))

            except Exception as e:
                logger.warning(f"⚠️ 预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')
                confidences.append(0.0)

        accuracy = accuracy_score(self.ground_truth, predictions)

        gt_normalized = [str(l).lower() for l in self.ground_truth]
        pred_normalized = [str(p).lower() for p in predictions]

        results = {
            'model': 'HSEmotion',
            'accuracy': float(accuracy),
            'predictions': predictions,
            'classification_report': classification_report(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys()),
                zero_division=0
            ),
            'confusion_matrix': confusion_matrix(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys())
            )
        }

        logger.info(f"✅ HSEmotion 准确率：{accuracy*100:.2f}%")

        return results

    def evaluate_rule_fusion(self) -> Dict:
        """评估规则融合"""
        logger.info("🔬 开始评估规则融合模型...")

        predictions = []
        confidences = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Rule Fusion")):
            try:
                # 🔴 关键：重置 EMA 状态
                self.rule_fusion_detector._emotion_ema = {}

                emotion, confidence = self.rule_fusion_detector.analyze_emotion(img)
                predictions.append(emotion)
                confidences.append(float(confidence))

            except Exception as e:
                logger.warning(f"⚠️ 预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')
                confidences.append(0.0)

        accuracy = accuracy_score(self.ground_truth, predictions)

        gt_normalized = [str(l).lower() for l in self.ground_truth]
        pred_normalized = [str(p).lower() for p in predictions]

        results = {
            'model': 'Rule-based Fusion',
            'accuracy': float(accuracy),
            'predictions': predictions,
            'classification_report': classification_report(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys()),
                zero_division=0
            ),
            'confusion_matrix': confusion_matrix(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys())
            )
        }

        logger.info(f"✅ 规则融合准确率：{accuracy*100:.2f}%")

        return results

    def evaluate_meta_learner(self) -> Dict:
        """评估元学习器融合"""
        logger.info("🔬 开始评估元学习器融合模型...")

        predictions = []
        confidences = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Meta Learner")):
            try:
                # 🔴 关键：重置 EMA 状态
                self.meta_fusion_detector._emotion_ema = {}

                emotion, confidence = self.meta_fusion_detector.analyze_emotion(img)
                predictions.append(emotion)
                confidences.append(float(confidence))

            except Exception as e:
                logger.warning(f"⚠️ 预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')
                confidences.append(0.0)

        accuracy = accuracy_score(self.ground_truth, predictions)

        gt_normalized = [str(l).lower() for l in self.ground_truth]
        pred_normalized = [str(p).lower() for p in predictions]

        results = {
            'model': 'Meta-Learner Fusion',
            'accuracy': float(accuracy),
            'predictions': predictions,
            'classification_report': classification_report(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys()),
                zero_division=0
            ),
            'confusion_matrix': confusion_matrix(
                gt_normalized, pred_normalized,
                labels=list(self.LABEL_TO_IDX.keys())
            )
        }

        logger.info(f"✅ 元学习器准确率：{accuracy*100:.2f}%")

        return results

    def generate_report(self, hsemotion_results: Dict, rule_results: Dict, meta_results: Dict):
        """生成对比报告"""
        print("\n" + "="*80)
        print("📊 完整情绪识别模型对比报告")
        print("="*80)

        # 基本统计
        print(f"\n📈 基本统计：")
        print(f"  总样本数：{len(self.test_data)}")
        print(f"  类别数：{len(set(self.ground_truth))}")
        print(f"  类别分布：")
        for emotion, count in sorted([(e, self.ground_truth.count(e)) for e in set(self.ground_truth)]):
            print(f"    - {emotion}: {count}")

        # 准确率对比
        print(f"\n🎯 准确率对比：")
        hsemotion_acc = hsemotion_results.get('accuracy', 0) * 100
        rule_acc = rule_results.get('accuracy', 0) * 100
        meta_acc = meta_results.get('accuracy', 0) * 100

        print(f"  HSEmotion 单体：     {hsemotion_acc:.2f}%")
        print(f"  规则融合：          {rule_acc:.2f}%")
        print(f"  元学习器融合：       {meta_acc:.2f}%")

        # 改进对比
        rule_improvement = rule_acc - hsemotion_acc
        meta_improvement = meta_acc - hsemotion_acc
        best_improvement = max(meta_improvement, rule_improvement)

        print(f"\n  规则融合改进：      {rule_improvement:+.2f}% {'✅' if rule_improvement >= 0 else '❌'}")
        print(f"  元学习器改进：      {meta_improvement:+.2f}% {'✅' if meta_improvement >= 0 else '❌'}")

        # 最佳模型
        best_model = max(
            [('HSEmotion', hsemotion_acc), ('Rule-based', rule_acc), ('Meta-Learner', meta_acc)],
            key=lambda x: x[1]
        )
        print(f"\n🏆 最佳模型：{best_model[0]} ({best_model[1]:.2f}%)")

        # 分类报告
        print(f"\n📋 HSEmotion 分类报告：")
        print(hsemotion_results.get('classification_report', 'N/A'))

        print(f"\n📋 规则融合分类报告：")
        print(rule_results.get('classification_report', 'N/A'))

        print(f"\n📋 元学习器分类报告：")
        print(meta_results.get('classification_report', 'N/A'))

        # 混淆矩阵可视化
        self._plot_confusion_matrices(
            hsemotion_results.get('confusion_matrix'),
            rule_results.get('confusion_matrix'),
            meta_results.get('confusion_matrix')
        )

        # 保存 JSON 报告
        report_path = self.dataset_path.parent / 'evaluation_results_complete.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': str(self.dataset_path),
                'samples_per_class': self.samples_per_class,
                'fusion_k': self.fusion_k,
                'total_samples': len(self.test_data),
                'results': {
                    'hsemotion': {
                        'accuracy': hsemotion_acc,
                        'report': hsemotion_results.get('classification_report', '')
                    },
                    'rule_based': {
                        'accuracy': rule_acc,
                        'report': rule_results.get('classification_report', ''),
                        'improvement': rule_improvement
                    },
                    'meta_learner': {
                        'accuracy': meta_acc,
                        'report': meta_results.get('classification_report', ''),
                        'improvement': meta_improvement
                    }
                },
                'best_model': best_model[0],
                'best_accuracy': best_model[1]
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ JSON 报告已保存到 {report_path}")

    def _plot_confusion_matrices(self, cm_hse, cm_rule, cm_meta):
        """绘制混淆矩阵"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            labels = list(self.LABEL_TO_IDX.keys())

            if cm_hse is not None:
                sns.heatmap(cm_hse, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels, ax=axes[0])
                axes[0].set_title('HSEmotion 单体')
                axes[0].set_ylabel('True Label')
                axes[0].set_xlabel('Predicted Label')

            if cm_rule is not None:
                sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Greens',
                           xticklabels=labels, yticklabels=labels, ax=axes[1])
                axes[1].set_title('规则融合')
                axes[1].set_ylabel('True Label')
                axes[1].set_xlabel('Predicted Label')

            if cm_meta is not None:
                sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Reds',
                           xticklabels=labels, yticklabels=labels, ax=axes[2])
                axes[2].set_title('元学习器融合')
                axes[2].set_ylabel('True Label')
                axes[2].set_xlabel('Predicted Label')

            plt.tight_layout()
            plot_path = self.dataset_path.parent / 'confusion_matrices_all_models.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='完整情绪识别模型评估 - HSEmotion vs 规则融合 vs 元学习器')
    parser.add_argument('--dataset-path', type=str,
                       default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                       help='数据集路径')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='每个类别的样本数（默认 100）')
    parser.add_argument('--fusion-k', type=float, default=0.5,
                       help='规则融合中眼睛权重 k（默认 0.5）')

    args = parser.parse_args()

    # 创建评估器
    evaluator = CompleteEmotionEvaluator(
        args.dataset_path,
        args.samples_per_class,
        fusion_k=args.fusion_k
    )

    # 加载数据集
    evaluator.load_dataset()

    # 评估三个模型
    hsemotion_results = evaluator.evaluate_hsemotion()
    rule_results = evaluator.evaluate_rule_fusion()
    meta_results = evaluator.evaluate_meta_learner()

    # 生成对比报告
    evaluator.generate_report(hsemotion_results, rule_results, meta_results)

    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == '__main__':
    main()
