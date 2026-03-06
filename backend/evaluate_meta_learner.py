"""
元学习器 vs 规则融合 对比评估脚本

对比两种融合策略：
1. 规则融合 (Rule-based): 手动加权 + 阈值判断
2. 元学习器 (Meta-Learner): 使用训练好的 Logistic Regression / SVM 模型

用法：
python evaluate_meta_learner.py \
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
import seaborn as sns
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaLearnerEvaluator:
    """元学习器评估器"""

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

    def __init__(self, dataset_path: str, samples_per_class: int = 100):
        self.dataset_path = Path(dataset_path)
        self.samples_per_class = samples_per_class
        self.test_data = []
        self.ground_truth = []

        self.rule_fusion_detector = None
        self.meta_learner_detector = None

        self._init_models()

    def _init_models(self):
        """初始化两种融合检测器"""
        from core.config import Config
        from core.decision_fusion_detector import create_improved_decision_fusion_detector

        config = Config()

        # 1. 规则融合检测器
        logger.info("🔧 初始化规则融合检测器...")
        self.rule_fusion_detector = create_improved_decision_fusion_detector(
            config, use_meta_learner=False
        )
        logger.info("✅ 规则融合检测器就绪")

        # 2. 元学习器检测器
        logger.info("🤖 初始化元学习器检测器...")
        self.meta_learner_detector = create_improved_decision_fusion_detector(
            config, use_meta_learner=True
        )
        logger.info("✅ 元学习器检测器就绪")

    def load_dataset(self):
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

        logger.info(f"✅ 数据集加载并打乱完成：共 {len(self.test_data)} 张图像")

    def evaluate_rule_based(self) -> Dict[str, any]:
        """评估规则融合"""
        if self.rule_fusion_detector is None:
            logger.error("❌ 规则融合检测器未加载")
            return {}

        logger.info("🔬 开始评估规则融合模型...")

        predictions = []
        confidences = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Rule-based")):
            try:
                # 重置 EMA 状态（防止数据泄露）
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
            'confidences': confidences,
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

    def evaluate_meta_learner(self) -> Dict[str, any]:
        """评估元学习器"""
        if self.meta_learner_detector is None:
            logger.error("❌ 元学习器检测器未加载")
            return {}

        logger.info("🔬 开始评估元学习器模型...")

        predictions = []
        confidences = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Meta-Learner")):
            try:
                # 重置 EMA 状态
                self.meta_learner_detector._emotion_ema = {}

                emotion, confidence = self.meta_learner_detector.analyze_emotion(img)
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
            'confidences': confidences,
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

    def generate_report(self, rule_results: Dict, meta_results: Dict):
        """生成对比报告"""
        print("\n" + "="*80)
        print("📊 融合策略对比报告 - 规则融合 vs 元学习器")
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
        rule_acc = rule_results.get('accuracy', 0) * 100
        meta_acc = meta_results.get('accuracy', 0) * 100
        improvement = meta_acc - rule_acc

        print(f"  规则融合：     {rule_acc:.2f}%")
        print(f"  元学习器：     {meta_acc:.2f}%")
        print(f"  改进：        {improvement:+.2f}% {'✅' if improvement >= 0 else '❌'}")

        # 分类报告
        print(f"\n📋 规则融合分类报告：")
        print(rule_results.get('classification_report', 'N/A'))

        print(f"\n📋 元学习器分类报告：")
        print(meta_results.get('classification_report', 'N/A'))

        # 混淆矩阵可视化
        self._plot_confusion_matrices(
            rule_results.get('confusion_matrix'),
            meta_results.get('confusion_matrix')
        )

        # 保存 JSON 报告
        report_path = self.dataset_path.parent / 'evaluation_results_rule_vs_meta_learner.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': str(self.dataset_path),
                'samples_per_class': self.samples_per_class,
                'total_samples': len(self.test_data),
                'rule_based': {
                    'accuracy': rule_acc,
                    'report': rule_results.get('classification_report', '')
                },
                'meta_learner': {
                    'accuracy': meta_acc,
                    'report': meta_results.get('classification_report', ''),
                    'improvement': improvement
                }
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ JSON 报告已保存到 {report_path}")

    def _plot_confusion_matrices(self, cm_rule, cm_meta):
        """绘制混淆矩阵"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            labels = list(self.LABEL_TO_IDX.keys())

            if cm_rule is not None:
                sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels, ax=axes[0])
                axes[0].set_title('规则融合混淆矩阵')
                axes[0].set_ylabel('True Label')
                axes[0].set_xlabel('Predicted Label')

            if cm_meta is not None:
                sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Greens',
                           xticklabels=labels, yticklabels=labels, ax=axes[1])
                axes[1].set_title('元学习器混淆矩阵')
                axes[1].set_ylabel('True Label')
                axes[1].set_xlabel('Predicted Label')

            plt.tight_layout()
            plot_path = self.dataset_path.parent / 'confusion_matrices_rule_vs_meta.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='元学习器 vs 规则融合 对比评估')
    parser.add_argument('--dataset-path', type=str,
                       default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                       help='数据集路径')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='每个类别的样本数（默认 100）')

    args = parser.parse_args()

    # 创建评估器
    evaluator = MetaLearnerEvaluator(args.dataset_path, args.samples_per_class)

    # 加载数据集
    evaluator.load_dataset()

    # 评估规则融合
    rule_results = evaluator.evaluate_rule_based()

    # 评估元学习器
    meta_results = evaluator.evaluate_meta_learner()

    # 生成对比报告
    evaluator.generate_report(rule_results, meta_results)


if __name__ == '__main__':
    main()
