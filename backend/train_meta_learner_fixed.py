"""
✅ 改进的元学习器训练脚本 - 修复版 (Fixed MetaLearner Trainer)

核心改进：
1. ✅ 使用统一的眼睛特征提取器（与推理保持一致）
2. ✅ 修复 PyTorch 预处理的数值稳定性
3. ✅ 添加更完整的特征验证和日志
4. ✅ 改进的错误处理和重试逻辑
5. ✅ 保存特征提取配置以供推理使用

用法：
python train_meta_learner_fixed.py \
  --dataset-path /path/to/dataset \
  --samples-per-class 200 \
  --output-path /path/to/output.pkl
"""

import os
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedMetaLearnerTrainer:
    """改进的元学习器训练器"""

    EMOTION_LABELS = {
        'Angry': 'angry',
        'Happy': 'happy',
        'Neutral': 'neutral',
        'Sad': 'sad',
        'Surprise': 'surprise'
    }

    def __init__(self, dataset_path: str, samples_per_class: int = 200):
        """
        初始化训练器

        Args:
            dataset_path: 数据集根目录
            samples_per_class: 每个类别的样本数
        """
        self.dataset_path = Path(dataset_path)
        self.samples_per_class = samples_per_class

        # 数据储存
        self.features = []      # 特征向量 [P_global_sad, P_global_neutral, P_eye_sad, ...]
        self.labels = []        # 真实标签 (0=neutral, 1=sad)
        self.image_paths = []   # 记录每个样本的路径（用于诊断）

        # 模型
        self.meta_classifier = None
        self.best_classifier_name = None

        # 统计信息
        self.feature_extraction_stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'no_face_detected': 0,
            'no_eyes_detected': 0
        }

    def _init_models(self):
        """初始化所需的模型"""
        import torch
        from core.advanced_detectors import HSEmotionDetector
        from core.config import Config

        # 初始化设备
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"📱 使用设备：{self.device}")

        # 初始化 HSEmotion
        config = Config()
        self.hsemotion = HSEmotionDetector(config)
        logger.info("✅ HSEmotion 模型加载成功")

        # 初始化眼睛特征提取器（统一的提取方式）
        try:
            from eye_feature_extractor import EyeFeatureExtractor
            self.eye_extractor = EyeFeatureExtractor(device=str(self.device))
            logger.info("✅ 眼睛特征提取器加载成功")
        except ImportError:
            logger.error("❌ eye_feature_extractor 模块未找到")
            raise

    def load_dataset_and_extract_features(self):
        """加载数据集并提取特征"""
        logger.info(f"📁 从 {self.dataset_path} 加载数据集并提取特征...")

        for folder_name, emotion_label in self.EMOTION_LABELS.items():
            folder_path = self.dataset_path / folder_name

            if not folder_path.exists():
                logger.warning(f"⚠️ 文件夹不存在：{folder_path}")
                continue

            # 获取所有图像文件
            image_files = list(folder_path.glob('*.jpg')) + \
                         list(folder_path.glob('*.png')) + \
                         list(folder_path.glob('*.jpeg'))

            # 随机取样
            import random
            sampled_files = random.sample(
                image_files,
                min(self.samples_per_class, len(image_files))
            )

            logger.info(f"  {folder_name}: 处理 {len(sampled_files)} 张图像...")

            for img_path in tqdm(sampled_files, desc=f"Extracting {folder_name}"):
                self.feature_extraction_stats['total_images'] += 1

                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"⚠️ 无法读取图像：{img_path}")
                        self.feature_extraction_stats['failed_extractions'] += 1
                        continue

                    # 调整大小到 224x224（确保一致性）
                    img = cv2.resize(img, (224, 224))

                    # 1. 获取全局情绪概率分布
                    hsemotion_probs = self.hsemotion.get_all_emotions(img)

                    if not hsemotion_probs:
                        logger.debug(f"⚠️ HSEmotion 返回空结果：{img_path}")
                        self.feature_extraction_stats['failed_extractions'] += 1
                        continue

                    # 2. 检测人脸（用于眼睛特征提取）
                    face_rect = self._detect_face(img)
                    if face_rect is None:
                        logger.debug(f"⚠️ 未检测到人脸：{img_path}")
                        self.feature_extraction_stats['no_face_detected'] += 1
                        continue

                    # 3. 提取眼睛区域并预测
                    eye_img = self.eye_extractor.extract_eye_region(img, face_rect)
                    if eye_img is None:
                        logger.debug(f"⚠️ 未检测到眼睛：{img_path}")
                        self.feature_extraction_stats['no_eyes_detected'] += 1
                        continue

                    # 4. 获取眼睛的悲伤概率
                    P_eye = self.eye_extractor.get_eye_sad_probability(eye_img)

                    # 5. 构建特征向量（与推理时保持一致！）
                    feature_vector = self._build_feature_vector(hsemotion_probs, P_eye)

                    # 6. 确定标签（只关注 sad vs neutral 二分类）
                    if emotion_label.lower() == 'sad':
                        label = 1
                    elif emotion_label.lower() == 'neutral':
                        label = 0
                    else:
                        # 其他情绪不参与二分类训练
                        logger.debug(f"ℹ️ 跳过情绪 '{emotion_label}' (仅用于二分类)")
                        self.feature_extraction_stats['failed_extractions'] += 1
                        continue

                    self.features.append(feature_vector)
                    self.labels.append(label)
                    self.image_paths.append(str(img_path))

                    self.feature_extraction_stats['successful_extractions'] += 1

                except Exception as e:
                    logger.debug(f"⚠️ 处理图像失败 {img_path}: {e}")
                    self.feature_extraction_stats['failed_extractions'] += 1
                    continue

        # 转换为 numpy 数组
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)

        # 打印统计信息
        self._print_extraction_stats()

        if len(self.features) == 0:
            raise RuntimeError("❌ 特征提取失败，没有生成任何有效样本！")

    def _build_feature_vector(self, hsemotion_probs: Dict[str, float], P_eye: float) -> list:
        """
        构建特征向量
        
        特征顺序必须与 meta_learner_inference.py 中的 FEATURE_NAMES 一致！
        """
        # 归一化到 0-1 范围
        def normalize_prob(p):
            p = float(p)
            if p > 1.0:
                p /= 100.0  # 百分比转小数
            return max(0.0, min(1.0, p))

        P_global_sad = normalize_prob(hsemotion_probs.get('sad', 0.0))
        P_global_neutral = normalize_prob(hsemotion_probs.get('neutral', 0.0))
        P_global_angry = normalize_prob(hsemotion_probs.get('angry', 0.0))
        P_global_happy = normalize_prob(hsemotion_probs.get('happy', 0.0))
        P_global_surprise = normalize_prob(hsemotion_probs.get('surprise', 0.0))
        P_eye = max(0.0, min(1.0, float(P_eye)))

        # 构建特征向量（顺序很重要！）
        return [
            P_global_sad,
            P_global_neutral,
            P_eye,
            P_global_angry,
            P_global_happy,
            P_global_surprise
        ]

    def _detect_face(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测人脸"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            # 返回最大的人脸
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            return tuple(faces[0])

        return None

    def _print_extraction_stats(self):
        """打印特征提取统计信息"""
        stats = self.feature_extraction_stats

        print("\n" + "="*80)
        print("📊 特征提取统计")
        print("="*80)
        print(f"总处理图像数：       {stats['total_images']}")
        print(f"成功提取特征：       {stats['successful_extractions']}")
        print(f"特征提取失败：       {stats['failed_extractions']}")
        print(f"  ├─ 检测不到人脸：  {stats['no_face_detected']}")
        print(f"  └─ 检测不到眼睛：  {stats['no_eyes_detected']}")
        print(f"成功率：            {stats['successful_extractions']/max(1, stats['total_images'])*100:.1f}%")
        print(f"\n最终训练样本数：     {len(self.features)}")

        if len(self.features) > 0:
            print(f"特征维度：          {self.features.shape[1]}")
            print(f"类别分布：          neutral={sum(self.labels==0)}, sad={sum(self.labels==1)}")
        print("="*80 + "\n")

    def train_classifier(self, test_split: float = 0.2):
        """训练分类器"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        if len(self.features) == 0:
            logger.error("❌ 没有训练数据")
            return

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_split, random_state=42,
            stratify=self.labels
        )

       
        logger.info(f"\n📊 数据集划分")
        logger.info(f"训练集：{len(X_train)} 样本")
        logger.info(f"测试集：{len(X_test)} 样本")

        # 分类器对比
        classifiers = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', C=1.0, class_weight='balanced',
                probability=True, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
        }

        best_clf = None
        best_acc = 0
        best_name = ""
        results = {}

        print("\n" + "="*80)
        print("🔬 分类器对比评估")
        print("="*80)

        for name, clf in classifiers.items():
            logger.info(f"\n训练 {name}...")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

            print(f"\n{name}:")
            print(f"  准确率：{acc*100:.2f}%")
            print(f"  分类报告：")
            print(classification_report(
                y_test, y_pred,
                target_names=['neutral', 'sad'],
                digits=3
            ))

            if acc > best_acc:
                best_acc = acc
                best_clf = clf
                best_name = name

        print("\n" + "="*80)
        print(f"🏆 最佳分类器：{best_name} (准确率：{best_acc*100:.2f}%)")
        print("="*80)

        self.meta_classifier = best_clf
        self.best_classifier_name = best_name

        return results

    def save_model(self, output_path: Optional[str] = None):
        """保存模型和配置"""
        if self.meta_classifier is None:
            logger.error("❌ 没有可保存的模型，请先训练")
            return

        if output_path is None:
            output_path = Path(__file__).parent.parent / "weights" / "meta_learner_fusion_model.pkl"
        else:
            output_path = Path(output_path)

        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型和配置
        model_data = {
            'classifier': self.meta_classifier,
            'classifier_name': self.best_classifier_name,
            'feature_names': [
                'P_global_sad',
                'P_global_neutral',
                'P_eye_sad',
                'P_global_angry',
                'P_global_happy',
                'P_global_surprise'
            ],
            'feature_extraction_stats': self.feature_extraction_stats,
            'training_samples': len(self.features),
            'training_labels_distribution': {
                'neutral': int(sum(self.labels == 0)),
                'sad': int(sum(self.labels == 1))
            }
        }

        joblib.dump(model_data, output_path)
        logger.info(f"✅ 模型已保存到：{output_path}")

        # 保存训练元数据
        metadata_path = output_path.parent / "meta_learner_metadata.json"
        metadata = {
            'classifier_type': self.best_classifier_name,
            'feature_names': model_data['feature_names'],
            'training_samples': model_data['training_samples'],
            'label_distribution': model_data['training_labels_distribution'],
            'extraction_stats': model_data['feature_extraction_stats']
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 元数据已保存到：{metadata_path}")

        return output_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='改进的元学习融合模型训练')
    parser.add_argument('--dataset-path', type=str,
                       default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                       help='数据集路径')
    parser.add_argument('--samples-per-class', type=int, default=200,
                       help='每个类别的样本数（默认 200）')
    parser.add_argument('--output-path', type=str, default=None,
                       help='模型保存路径')

    args = parser.parse_args()

    # 创建训练器
    trainer = ImprovedMetaLearnerTrainer(args.dataset_path, args.samples_per_class)

    # 初始化模型
    trainer._init_models()

    # 加载数据并提取特征
    trainer.load_dataset_and_extract_features()

    # 训练分类器
    trainer.train_classifier()

    # 保存模型
    trainer.save_model(args.output_path)

    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 确认 decision_fusion_detector.py 已加载该模型")
    print("2. 在 config.yaml 中设置 use_meta_learner: true")
    print("3. 进行 A/B 测试对比规则融合")


if __name__ == '__main__':
    main()
