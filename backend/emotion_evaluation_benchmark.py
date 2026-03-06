"""
HSEmotion vs DeepFace vs 决策融合模型（规则/元学习器）准确率对比评估脚本

功能：
1. ✅ 从数据集加载图像（Angry, Happy, Neutral, Sad, Surprise）
2. ✅ 测试 DeepFace 单体准确率（经典基准）
3. ✅ 测试 HSEmotion 单体准确率（EfficientNet-B0）
4. ✅ 测试决策融合模型准确率（眼睛模型 + HSEmotion / 规则 or 元学习器）
5. ✅ 生成对比报告和可视化
6. ✅ 解决 EMA 数据泄露问题
7. ✅ 新增：四元对比模式（DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合）

修复记录（v2）：
[FIX-1] 所有随机操作统一使用固定种子（random.seed + np.random.seed），确保结果可重现
[FIX-2] DeepFace 评估时禁用 anger_threshold 过滤，与其他模型保持公平一致
[FIX-3] 所有模型统一使用 face_rect=(0,0,224,224) 输入，避免融合模型因绕过
         人脸检测而获得不公平优势
[FIX-4] accuracy_score 与 classification_report 统一使用归一化小写标签，
         消除大小写不一致导致的数值矛盾
[FIX-5] 新增 macro-F1 作为核心指标，避免类别不均衡时总体准确率失真
[FIX-6] generate_report 中三元对比的标题判断增加 has_rule 校验，防止误报
[FIX-7] _plot_confusion_matrices（默认二元模式）增加 rule_results None 安全检查，
         防止 AttributeError 崩溃
[FIX-8] 将 import random 移至文件顶部，符合 PEP8 规范

用法：
# 模式 1: 四元对比（DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合）
python emotion_evaluation_benchmark.py --four-way-comparison --samples-per-class 100

# 模式 2: 三元对比（HSEmotion vs 规则融合 vs 元学习器融合）
python emotion_evaluation_benchmark.py --three-way-comparison --samples-per-class 100

# 模式 3: DeepFace vs HSEmotion（双单体对比）
python emotion_evaluation_benchmark.py --deepface-comparison --samples-per-class 100

# 模式 4: 单独评估 DeepFace
python emotion_evaluation_benchmark.py --deepface-only --samples-per-class 100
"""

# ===== [FIX-8] import random 移至顶部，符合 PEP8 =====
import random
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score  # [FIX-5] 新增 f1_score
)
import seaborn as sns
import logging
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== [FIX-1] 全局固定随机种子，确保每次运行测试集完全相同 =====
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class EmotionEvaluator:
    """情绪识别评估器（v2 修复版）"""

    # 标签映射
    EMOTION_LABELS = {
        'Angry': 'angry',
        'Happy': 'happy',
        'Neutral': 'neutral',
        'Sad': 'sad',
        'Surprise': 'surprise'
    }

    # 标签到数字的映射（用于混淆矩阵）
    LABEL_TO_IDX = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3,
        'surprise': 4
    }
    IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

    def __init__(
        self,
        dataset_path: str,
        samples_per_class: int = 1000,
        fusion_k: float = 0.3,
        use_meta_learner: bool = True,
        deepface_only: bool = False
    ):
        self.dataset_path = Path(dataset_path)
        self.samples_per_class = samples_per_class
        self.fusion_k = fusion_k
        self.use_meta_learner = use_meta_learner
        self.deepface_only = deepface_only
        self.test_data = []
        self.ground_truth = []

        # 模型初始化
        self.deepface = None
        self.hsemotion = None
        self.rule_fusion_detector = None
        self.meta_learner_detector = None
        self.device = None
        self._init_models()

    def _init_models(self):
        """初始化模型"""
        try:
            import torch
            from core.advanced_detectors import HSEmotionDetector
            from core.decision_fusion_detector import create_improved_decision_fusion_detector
            from core.config import Config
            from core.detector import EmotionDetector as DeepFaceDetector

            # 初始化设备
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Using device: {self.device}")

            config = Config()

            # 1. 初始化 DeepFace 单体（经典基准）
            # [FIX-2] 将 anger_threshold 设为 0，禁用 DeepFace 内部的 angry 过滤
            #         使 DeepFace 与其他模型在相同条件下进行公平对比
            try:
                self.deepface = DeepFaceDetector(config)
                self.deepface.anger_threshold = 0  # [FIX-2] 关键：禁用 angry 阈值过滤
                logger.info("✅ DeepFace 单体模型加载成功（经典基准）[anger_threshold=0]")
            except Exception as e:
                logger.error(f"❌ DeepFace 加载失败：{e}")
                self.deepface = None

            # 2. 初始化 HSEmotion 单体
            try:
                self.hsemotion = HSEmotionDetector(config)
                logger.info("✅ HSEmotion 单体模型加载成功")
            except Exception as e:
                logger.error(f"❌ HSEmotion 加载失败：{e}")
                self.hsemotion = None

            # 3. 初始化规则融合检测器
            try:
                config._config['emotion']['decision_fusion_k'] = self.fusion_k
                self.rule_fusion_detector = create_improved_decision_fusion_detector(
                    config, use_meta_learner=False
                )
                logger.info(f"✅ 规则融合检测器加载成功 (k={self.fusion_k})")
            except Exception as e:
                logger.error(f"❌ 规则融合模型加载失败：{e}")
                self.rule_fusion_detector = None

            # 4. 初始化元学习器融合检测器
            try:
                config_meta = Config()
                config_meta._config['emotion']['decision_fusion_k'] = self.fusion_k
                self.meta_learner_detector = create_improved_decision_fusion_detector(
                    config_meta, use_meta_learner=True
                )
                logger.info(f"✅ 元学习器融合检测器加载成功 (k={self.fusion_k})")
            except Exception as e:
                logger.error(f"❌ 元学习器融合模型加载失败：{e}")
                self.meta_learner_detector = None

        except ImportError as e:
            logger.error(f"缺少必要的导入：{e}")

    def load_dataset(self, verbose: bool = True) -> Tuple[List, List]:
        """
        加载数据集
        [FIX-1] 使用固定种子采样和打乱，确保每次运行测试集完全一致
        """
        logger.info(f"📁 从 {self.dataset_path} 加载数据集（随机种子={RANDOM_SEED}）...")

        test_images = []
        gt_labels = []

        # [FIX-8] random 已在顶部导入，这里直接使用，无需重复 import
        rng = random.Random(RANDOM_SEED)  # [FIX-1] 使用固定种子的独立 Random 实例

        for folder_name, emotion_label in self.EMOTION_LABELS.items():
            folder_path = self.dataset_path / folder_name

            if not folder_path.exists():
                folder_path = self.dataset_path / folder_name.lower()
                if not folder_path.exists():
                    logger.warning(f"⚠️ 文件夹不存在：{folder_name} 或 {folder_name.lower()}")
                    continue

            # 获取所有图像文件，排序后采样（确保文件列表顺序一致）
            image_files = sorted(
                list(folder_path.glob('*.jpg')) +
                list(folder_path.glob('*.png')) +
                list(folder_path.glob('*.jpeg'))
            )

            # [FIX-1] 用固定种子的 rng 采样，每次结果相同
            sampled_files = rng.sample(
                image_files,
                min(self.samples_per_class, len(image_files))
            )

            for img_path in sampled_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.resize(img, (224, 224))
                    test_images.append(img)
                    gt_labels.append(emotion_label)
                except Exception:
                    continue

        # [FIX-1] 打乱数据集时使用固定种子，防止 EMA 跨样本泄露且结果可重现
        combined = list(zip(test_images, gt_labels))
        rng.shuffle(combined)
        test_images, gt_labels = zip(*combined)

        self.test_data = list(test_images)
        self.ground_truth = list(gt_labels)

        logger.info(f"✅ 数据集加载完成：共 {len(self.test_data)} 张图像")

        # 打印各类别实际数量，便于发现类别不均衡问题
        from collections import Counter
        dist = Counter(self.ground_truth)
        for emotion in sorted(dist):
            logger.info(f"   {emotion}: {dist[emotion]} 张")

        return self.test_data, self.ground_truth

    # =========================================================
    # [FIX-4] 统一的标签归一化 + 指标计算辅助方法
    #         所有 evaluate_* 方法统一调用此处，消除大小写不一致
    # =========================================================
    def _compute_metrics(self, predictions: List[str]) -> Dict:
        """
        统一计算所有评估指标
        [FIX-4] accuracy_score 和 classification_report 均使用同一份
                归一化（小写）标签，消除数值矛盾
        [FIX-5] 新增 macro-F1，作为类别不均衡场景下的核心指标
        """
        labels_order = list(self.LABEL_TO_IDX.keys())

        # 统一归一化：全部转小写
        gt_norm = [str(l).lower() for l in self.ground_truth]
        pred_norm = [str(p).lower() for p in predictions]

        # [FIX-4] accuracy 与 classification_report 使用同一份归一化标签
        acc = accuracy_score(gt_norm, pred_norm)

        # [FIX-5] macro-F1：每个类别权重相同，不受样本数量影响
        macro_f1 = f1_score(gt_norm, pred_norm, labels=labels_order,
                            average='macro', zero_division=0)

        report = classification_report(
            gt_norm, pred_norm,
            labels=labels_order,
            zero_division=0
        )
        cm = confusion_matrix(gt_norm, pred_norm, labels=labels_order)

        return {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),  # [FIX-5] 新增
            'predictions': predictions,
            'classification_report': report,
            'confusion_matrix': cm
        }

    def evaluate_deepface(self) -> Dict[str, any]:
        """
        评估 DeepFace 单体准确率
        [FIX-2] anger_threshold 已在初始化时设为 0，此处无需额外处理
        [FIX-3] 统一传入 face_rect=(0,0,224,224)，与融合模型输入条件一致
        """
        if self.deepface is None:
            logger.error("❌ DeepFace 模型未加载")
            return {}

        predictions = []

        for idx, img in enumerate(tqdm(self.test_data, desc="DeepFace")):
            try:
                # [FIX-3] 使用 get_all_emotions（内部已使用 enforce_detection=False）
                #         DeepFace 的 get_all_emotions 不接受 face_rect 参数，
                #         因此通过直接传入已裁剪的 224x224 图像保持一致性。
                #         anger_threshold 已在 __init__ 设为 0，无需额外处理。
                emotions = self.deepface.get_all_emotions(img)
                if not emotions:
                    predictions.append('neutral')
                    continue
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                predictions.append(top_emotion[0])
            except Exception as e:
                logger.warning(f"⚠️ DeepFace 预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')

        # [FIX-4][FIX-5] 统一指标计算
        result = self._compute_metrics(predictions)
        result['model'] = 'DeepFace'
        return result

    def evaluate_hsemotion(self) -> Dict[str, any]:
        """
        评估 HSEmotion 单体准确率
        [FIX-3] 输入为已裁剪的 224x224 图像，与融合模型保持一致
        """
        if self.hsemotion is None:
            logger.error("❌ HSEmotion 模型未加载")
            return {}

        predictions = []

        for idx, img in enumerate(tqdm(self.test_data, desc="HSEmotion")):
            try:
                # [FIX-3] 与其他模型一致，直接传入 224x224 裁剪图像
                emotions = self.hsemotion.get_all_emotions(img)
                if not emotions:
                    predictions.append('neutral')
                    continue
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                predictions.append(top_emotion[0])
            except Exception as e:
                logger.warning(f"⚠️ HSEmotion 预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')

        # [FIX-4][FIX-5] 统一指标计算
        result = self._compute_metrics(predictions)
        result['model'] = 'HSEmotion'
        return result

    def evaluate_rule_fusion(self) -> Dict[str, any]:
        """
        评估规则融合模型准确率
        [FIX-3] face_rect=(0,0,224,224) 已与单体模型输入条件对齐
                （之前融合模型传此参数而单体模型不传，导致不公平）
        """
        if self.rule_fusion_detector is None:
            logger.error("❌ 规则融合检测器未加载")
            return {}

        predictions = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Rule Fusion")):
                try:
                    self.rule_fusion_detector._emotion_ema = {}

                    # ===== 诊断代码（只看前50张）=====
                    if idx < 50:
                        from core.eye_feature_extractor import EyeFeatureExtractor
                        extractor = EyeFeatureExtractor()
                        gt = self.ground_truth[idx]
                        eye_img = extractor.extract_eye_region(img, face_rect=(0, 0, 224, 224))
                        if eye_img is not None:
                            eye_emotion, eye_conf = extractor.predict_eye_emotion(eye_img)
                            print(f"[{idx:3d}] GT={gt:8s} | 眼睛={eye_emotion}({eye_conf:.1f}%) ✅")
                        else:
                            print(f"[{idx:3d}] GT={gt:8s} | 眼睛检测失败 ❌")
                    # ===== 诊断代码结束 =====

                    emotion, confidence = self.rule_fusion_detector.analyze_emotion(
                        img, face_rect=(0, 0, 224, 224)
                    )
                    predictions.append(emotion)
                except Exception as e:
                    logger.warning(f"⚠️ 规则融合预测失败 (样本 {idx}): {e}")
                    predictions.append('neutral')

        # [FIX-4][FIX-5] 统一指标计算
        result = self._compute_metrics(predictions)
        result['model'] = f'Rule Fusion (k={self.fusion_k})'
        return result

    def evaluate_meta_learner_fusion(self) -> Dict[str, any]:
        """
        评估元学习器融合模型准确率
        [FIX-3] 同规则融合，face_rect 对齐
        """
        if self.meta_learner_detector is None:
            logger.error("❌ 元学习器融合检测器未加载")
            return {}

        predictions = []

        for idx, img in enumerate(tqdm(self.test_data, desc="Meta-Learner Fusion")):
            try:
                # 清空 EMA 历史，防止跨样本状态泄露
                self.meta_learner_detector._emotion_ema = {}

                # [FIX-3] face_rect 明确对齐：整图即为人脸
                emotion, confidence = self.meta_learner_detector.analyze_emotion(
                    img, face_rect=(0, 0, 224, 224)
                )
                predictions.append(emotion)
            except Exception as e:
                logger.warning(f"⚠️ 元学习器融合预测失败 (样本 {idx}): {e}")
                predictions.append('neutral')

        # [FIX-4][FIX-5] 统一指标计算
        result = self._compute_metrics(predictions)
        result['model'] = f'Meta-Learner Fusion (k={self.fusion_k})'
        return result

    def generate_report(
        self,
        deepface_results: Dict = None,
        hsemotion_results: Dict = None,
        rule_results: Dict = None,
        meta_results: Dict = None
    ):
        """生成对比报告"""
        print("\n" + "=" * 80)

        has_deepface = bool(deepface_results)
        has_hsemotion = bool(hsemotion_results)
        has_rule = bool(rule_results)
        has_meta = bool(meta_results)

        # [FIX-6] 三元对比标题增加 has_rule 校验，避免只有两个模型时误报"三元"
        if has_deepface and has_hsemotion and has_rule and has_meta:
            print("📊 四元对比报告 - DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合")
        elif has_deepface and has_hsemotion:
            print("📊 双单体对比报告 - DeepFace vs HSEmotion")
        elif has_hsemotion and has_rule and has_meta:  # [FIX-6] 同时检查三个条件
            print("📊 三元对比报告 - HSEmotion vs 规则融合 vs 元学习器融合")
        else:
            print("📊 情绪识别评估报告")
        print("=" * 80)

        print(f"\n📈 基本统计：")
        print(f"  随机种子：{RANDOM_SEED}")  # [FIX-1] 报告中标注种子，确保可追溯
        print(f"  总样本数：{len(self.test_data)}")
        print(f"  类别数：{len(set(self.ground_truth))}")
        print(f"  类别分布：")
        from collections import Counter
        dist = Counter(self.ground_truth)
        for emotion in sorted(dist):
            print(f"    - {emotion}: {dist[emotion]}")

        print(f"\n🎯 准确率 & Macro-F1 对比：")  # [FIX-5] 标题增加 Macro-F1

        models_comparison = []

        # DeepFace 结果
        if has_deepface:
            deepface_acc = deepface_results.get('accuracy', 0) * 100
            deepface_f1 = deepface_results.get('macro_f1', 0) * 100  # [FIX-5]
            print(f"  DeepFace 单体：       Acc={deepface_acc:.2f}%  Macro-F1={deepface_f1:.2f}%")
            models_comparison.append(('DeepFace', deepface_acc, deepface_f1))

        # HSEmotion 结果
        if has_hsemotion:
            hsemotion_acc = hsemotion_results.get('accuracy', 0) * 100
            hsemotion_f1 = hsemotion_results.get('macro_f1', 0) * 100  # [FIX-5]
            print(f"  HSEmotion 单体：      Acc={hsemotion_acc:.2f}%  Macro-F1={hsemotion_f1:.2f}%")
            models_comparison.append(('HSEmotion', hsemotion_acc, hsemotion_f1))

        # 规则融合结果
        if has_rule:
            rule_acc = rule_results.get('accuracy', 0) * 100
            rule_f1 = rule_results.get('macro_f1', 0) * 100  # [FIX-5]
            print(f"  规则融合：            Acc={rule_acc:.2f}%  Macro-F1={rule_f1:.2f}%")
            models_comparison.append(('Rule Fusion', rule_acc, rule_f1))

        # 元学习器融合结果
        if has_meta:
            meta_acc = meta_results.get('accuracy', 0) * 100
            meta_f1 = meta_results.get('macro_f1', 0) * 100  # [FIX-5]
            print(f"  元学习器融合：        Acc={meta_acc:.2f}%  Macro-F1={meta_f1:.2f}%")
            models_comparison.append(('Meta-Learner Fusion', meta_acc, meta_f1))

        # [FIX-5] 分别展示按 Accuracy 和按 Macro-F1 的最佳模型
        best_by_acc = max(models_comparison, key=lambda x: x[1])
        best_by_f1 = max(models_comparison, key=lambda x: x[2])
        print(f"\n🏆 按准确率最佳：{best_by_acc[0]} ({best_by_acc[1]:.2f}%)")
        print(f"🏆 按Macro-F1最佳：{best_by_f1[0]} ({best_by_f1[2]:.2f}%)")

        # 分类报告
        if has_deepface:
            print(f"\n📋 DeepFace 分类报告：")
            print(deepface_results.get('classification_report', 'N/A'))
        if has_hsemotion:
            print(f"\n📋 HSEmotion 分类报告：")
            print(hsemotion_results.get('classification_report', 'N/A'))
        if has_rule:
            print(f"\n📋 规则融合分类报告：")
            print(rule_results.get('classification_report', 'N/A'))
        if has_meta:
            print(f"\n📋 元学习器融合分类报告：")
            print(meta_results.get('classification_report', 'N/A'))

        # 混淆矩阵可视化
        if has_deepface and has_hsemotion and has_rule and has_meta:
            self._plot_four_confusion_matrices(
                deepface_results.get('confusion_matrix'),
                hsemotion_results.get('confusion_matrix'),
                rule_results.get('confusion_matrix'),
                meta_results.get('confusion_matrix')
            )
        elif has_deepface and has_hsemotion:
            self._plot_two_confusion_matrices(
                deepface_results.get('confusion_matrix'),
                hsemotion_results.get('confusion_matrix'),
                'DeepFace', 'HSEmotion'
            )
        elif has_hsemotion and has_rule and has_meta:  # [FIX-6]
            self._plot_three_confusion_matrices(
                hsemotion_results.get('confusion_matrix'),
                rule_results.get('confusion_matrix'),
                meta_results.get('confusion_matrix')
            )
        else:
            # [FIX-7] 默认二元模式安全检查，任一结果为空时跳过绘图
            cm_hsemotion = hsemotion_results.get('confusion_matrix') if has_hsemotion else None
            cm_rule = rule_results.get('confusion_matrix') if has_rule else None
            if cm_hsemotion is not None and cm_rule is not None:
                self._plot_confusion_matrices(cm_hsemotion, cm_rule)
            else:
                logger.warning("⚠️ 缺少混淆矩阵数据，跳过绘图")

        # 保存 JSON 报告
        report_path = self.dataset_path.parent / 'emotion_evaluation_results.json'
        report_data = {
            'random_seed': RANDOM_SEED,  # [FIX-1]
            'dataset': str(self.dataset_path),
            'samples_per_class': self.samples_per_class,
            'fusion_k': self.fusion_k,
            'total_samples': len(self.test_data),
            'best_model_by_accuracy': best_by_acc[0],
            'best_accuracy': best_by_acc[1],
            'best_model_by_macro_f1': best_by_f1[0],  # [FIX-5]
            'best_macro_f1': best_by_f1[2]
        }

        if has_deepface:
            report_data['deepface'] = {
                'accuracy': deepface_acc,
                'macro_f1': deepface_f1,  # [FIX-5]
                'report': deepface_results.get('classification_report', '')
            }
        if has_hsemotion:
            report_data['hsemotion'] = {
                'accuracy': hsemotion_acc,
                'macro_f1': hsemotion_f1,  # [FIX-5]
                'report': hsemotion_results.get('classification_report', '')
            }
        if has_rule:
            report_data['rule_fusion'] = {
                'accuracy': rule_acc,
                'macro_f1': rule_f1,  # [FIX-5]
                'report': rule_results.get('classification_report', ''),
                'improvement_vs_hsemotion': rule_acc - hsemotion_acc if has_hsemotion else 0
            }
        if has_meta:
            report_data['meta_learner'] = {
                'accuracy': meta_acc,
                'macro_f1': meta_f1,  # [FIX-5]
                'report': meta_results.get('classification_report', ''),
                'improvement_vs_hsemotion': meta_acc - hsemotion_acc if has_hsemotion else 0
            }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ JSON 报告已保存到 {report_path}")

    # =========================================================
    # 可视化辅助方法
    # =========================================================

    def _plot_confusion_matrices(self, cm_hsemotion, cm_fusion):
        """绘制两个混淆矩阵（默认二元模式）[FIX-7] 调用前已做 None 安全检查"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            labels = list(self.LABEL_TO_IDX.keys())

            sns.heatmap(cm_hsemotion, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=axes[0])
            axes[0].set_title('HSEmotion 单体')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')

            sns.heatmap(cm_fusion, annot=True, fmt='d', cmap='Greens',
                        xticklabels=labels, yticklabels=labels, ax=axes[1])
            axes[1].set_title('融合模型')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')

            plt.tight_layout()
            plot_path = self.dataset_path.parent / 'confusion_matrices_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")

    def _plot_two_confusion_matrices(self, cm1, cm2, title1, title2):
        """绘制两个混淆矩阵（通用版本）"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            labels = list(self.LABEL_TO_IDX.keys())

            if cm1 is not None:
                sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                            xticklabels=labels, yticklabels=labels, ax=axes[0])
                axes[0].set_title(title1)
                axes[0].set_ylabel('True Label')
                axes[0].set_xlabel('Predicted Label')

            if cm2 is not None:
                sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
                            xticklabels=labels, yticklabels=labels, ax=axes[1])
                axes[1].set_title(title2)
                axes[1].set_ylabel('True Label')
                axes[1].set_xlabel('Predicted Label')

            plt.tight_layout()
            plot_path = self.dataset_path.parent / 'confusion_matrices_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")

    def _plot_three_confusion_matrices(self, cm_hsemotion, cm_rule, cm_meta):
        """绘制三个混淆矩阵（三元对比）"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            labels = list(self.LABEL_TO_IDX.keys())

            if cm_hsemotion is not None:
                sns.heatmap(cm_hsemotion, annot=True, fmt='d', cmap='Blues',
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
            plot_path = self.dataset_path.parent / 'confusion_matrices_three_way_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 三元对比混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")

    def _plot_four_confusion_matrices(self, cm_deepface, cm_hsemotion, cm_rule, cm_meta):
        """绘制四个混淆矩阵（四元对比，2x2 布局）"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            labels = list(self.LABEL_TO_IDX.keys())

            if cm_deepface is not None:
                sns.heatmap(cm_deepface, annot=True, fmt='d', cmap='Blues',
                            xticklabels=labels, yticklabels=labels, ax=axes[0, 0])
                axes[0, 0].set_title('DeepFace 单体（经典基准）')
                axes[0, 0].set_ylabel('True Label')
                axes[0, 0].set_xlabel('Predicted Label')

            if cm_hsemotion is not None:
                sns.heatmap(cm_hsemotion, annot=True, fmt='d', cmap='Greens',
                            xticklabels=labels, yticklabels=labels, ax=axes[0, 1])
                axes[0, 1].set_title('HSEmotion 单体（EfficientNet-B0）')
                axes[0, 1].set_ylabel('True Label')
                axes[0, 1].set_xlabel('Predicted Label')

            if cm_rule is not None:
                sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Oranges',
                            xticklabels=labels, yticklabels=labels, ax=axes[1, 0])
                axes[1, 0].set_title('规则融合（眼睛 + HSEmotion）')
                axes[1, 0].set_ylabel('True Label')
                axes[1, 0].set_xlabel('Predicted Label')

            if cm_meta is not None:
                sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Reds',
                            xticklabels=labels, yticklabels=labels, ax=axes[1, 1])
                axes[1, 1].set_title('元学习器融合（数据驱动）')
                axes[1, 1].set_ylabel('True Label')
                axes[1, 1].set_xlabel('Predicted Label')

            plt.tight_layout()
            plot_path = self.dataset_path.parent / 'confusion_matrices_four_way_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ 四元对比混淆矩阵图已保存到 {plot_path}")
            plt.show()

        except Exception as e:
            logger.warning(f"⚠️ 混淆矩阵绘制失败：{e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='情绪识别模型评估 - DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合'
    )
    parser.add_argument('--dataset-path', type=str,
                        default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                        help='数据集路径')
    parser.add_argument('--samples-per-class', type=int, default=100,
                        help='每个类别的样本数（默认 100）')
    parser.add_argument('--fusion-k', type=float, default=0.5,
                        help='融合权重 k（默认 0.5，眼睛 50%%, HSEmotion 50%%）')
    parser.add_argument('--four-way-comparison', action='store_true',
                        help='四元对比模式（DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合）')
    parser.add_argument('--three-way-comparison', action='store_true',
                        help='三元对比模式（HSEmotion vs 规则融合 vs 元学习器融合）')
    parser.add_argument('--deepface-comparison', action='store_true',
                        help='双单体对比（DeepFace vs HSEmotion）')
    parser.add_argument('--deepface-only', action='store_true',
                        help='仅评估 DeepFace')
    parser.add_argument('--no-meta-learner', action='store_true',
                        help='二元对比时关闭元学习器，只测试 HSEmotion vs 规则融合')

    args = parser.parse_args()

    evaluator = EmotionEvaluator(
        args.dataset_path,
        args.samples_per_class,
        fusion_k=args.fusion_k,
        use_meta_learner=True,
        deepface_only=args.deepface_only
    )

    evaluator.load_dataset()

    # 四元对比模式
    if args.four_way_comparison:
        print("\n" + "=" * 80)
        print("🔬 启动四元对比模式：DeepFace vs HSEmotion vs 规则融合 vs 元学习器融合")
        print("=" * 80)
        deepface_results = evaluator.evaluate_deepface()
        hsemotion_results = evaluator.evaluate_hsemotion()
        rule_results = evaluator.evaluate_rule_fusion()
        meta_results = evaluator.evaluate_meta_learner_fusion()
        evaluator.generate_report(deepface_results, hsemotion_results, rule_results, meta_results)

    # 仅 DeepFace 模式
    elif args.deepface_only:
        print("\n" + "=" * 80)
        print("🔬 启动 DeepFace 单独评估模式")
        print("=" * 80)
        deepface_results = evaluator.evaluate_deepface()
        print("\n" + "=" * 80)
        print("📊 DeepFace 评估报告")
        print("=" * 80)
        print(f"\n📈 基本统计：")
        print(f"  总样本数：{len(evaluator.test_data)}")
        print(f"  准确率：{deepface_results.get('accuracy', 0) * 100:.2f}%")
        print(f"  Macro-F1：{deepface_results.get('macro_f1', 0) * 100:.2f}%")  # [FIX-5]
        print(f"\n📋 分类报告：")
        print(deepface_results.get('classification_report', 'N/A'))

    # 双单体对比模式（DeepFace vs HSEmotion）
    elif args.deepface_comparison:
        print("\n" + "=" * 80)
        print("🔬 启动双单体对比模式：DeepFace vs HSEmotion")
        print("=" * 80)
        deepface_results = evaluator.evaluate_deepface()
        hsemotion_results = evaluator.evaluate_hsemotion()
        evaluator.generate_report(deepface_results=deepface_results, hsemotion_results=hsemotion_results)

    # 三元对比模式
    elif args.three_way_comparison:
        print("\n" + "=" * 80)
        print("🔬 启动三元对比模式：HSEmotion vs 规则融合 vs 元学习器融合")
        print("=" * 80)
        hsemotion_results = evaluator.evaluate_hsemotion()
        rule_results = evaluator.evaluate_rule_fusion()
        meta_results = evaluator.evaluate_meta_learner_fusion()
        evaluator.generate_report(
            hsemotion_results=hsemotion_results,
            rule_results=rule_results,
            meta_results=meta_results
        )

    # 默认模式
    else:
        hsemotion_results = evaluator.evaluate_hsemotion()

        if args.no_meta_learner:
            print("\n" + "=" * 80)
            print("🔬 启动二元对比模式：HSEmotion vs 规则融合")
            print("=" * 80)
            rule_results = evaluator.evaluate_rule_fusion()
            evaluator.generate_report(hsemotion_results=hsemotion_results, rule_results=rule_results)
        else:
            print("\n" + "=" * 80)
            print("🔬 启动二元对比模式：HSEmotion vs 元学习器融合")
            print("=" * 80)
            meta_results = evaluator.evaluate_meta_learner_fusion()
            evaluator.generate_report(hsemotion_results=hsemotion_results, meta_results=meta_results)

    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()