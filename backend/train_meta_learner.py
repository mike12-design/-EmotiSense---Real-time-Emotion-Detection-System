"""
元学习器训练脚本 - 训练一个 Meta-learner 来替代手动规则的融合

核心思想：
1. 收集 HSEmotion 和眼睛模型的概率分布
2. 使用 sklearn 训练 Logistic Regression / SVM 分类器
3. 学习最佳的融合权重和非线性决策边界
4. 保存模型供推理时使用

这就是工业界做模型融合（Ensemble）的真正做法 - Stacking!
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


class MetaLearnerTrainer:
    """元学习器训练器"""

    # 标签映射
    EMOTION_LABELS = {
        'Angry': 'angry',
        'Happy': 'happy',
        'Neutral': 'neutral',
        'Sad': 'sad',
        'Surprise': 'surprise'
    }

    # 用于融合的特征标签（sad vs neutral 二分类）
    FUSION_LABELS = {'sad': 1, 'neutral': 0}

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
        self.features = []  # 特征向量 [P_global_sad, P_eye_sad, P_global_neutral, ...]
        self.labels = []    # 真实标签 (0=neutral, 1=sad)

        # 模型
        self.meta_classifier = None

    def _init_models(self):
        """初始化所需的模型"""
        import torch
        from core.advanced_detectors import HSEmotionDetector
        from core.decision_fusion_detector import ImprovedDecisionFusionDetector
        from core.config import Config

        # 初始化设备
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        # 初始化 HSEmotion
        config = Config()
        self.hsemotion = HSEmotionDetector(config)
        logger.info("✅ HSEmotion 模型加载成功")

        # 初始化融合检测器（用于获取眼睛模型预测）
        self.fusion_detector = ImprovedDecisionFusionDetector(config, k=0.3)
        self.fusion_detector._lazy_init()
        logger.info("✅ 融合检测器加载成功")

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
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # 调整大小到 224x224
                    img = cv2.resize(img, (224, 224))

                    # 1. 获取 HSEmotion 概率分布
                    hsemotion_probs = self.hsemotion.get_all_emotions(img)

                    if not hsemotion_probs:
                        continue

                    # 2. 获取眼睛模型概率
                    # 需要先检测人脸
                    face_rect = self._detect_face(img)
                    if face_rect is None:
                        continue

                    P_eye = self._get_eye_sad_prob(img, face_rect)

                    # 3. 构建特征向量
                    # 核心特征：HSEmotion 的 sad 和 neutral 概率 + 眼睛模型的 sad 概率
                    feature_vector = [
                        float(hsemotion_probs.get('sad', 0.0)) / 100.0,      # P_global_sad (归一化到 0-1)
                        float(hsemotion_probs.get('neutral', 0.0)) / 100.0,  # P_global_neutral
                        P_eye,                                                # P_eye_sad
                        float(hsemotion_probs.get('angry', 0.0)) / 100.0,    # P_global_angry
                        float(hsemotion_probs.get('happy', 0.0)) / 100.0,    # P_global_happy
                        float(hsemotion_probs.get('surprise', 0.0)) / 100.0, # P_global_surprise
                    ]

                    # 4. 确定标签（只关注 sad vs neutral 二分类）
                    if emotion_label.lower() == 'sad':
                        label = 1
                    elif emotion_label.lower() == 'neutral':
                        label = 0
                    else:
                        # 其他情绪不参与二分类训练，但保留用于多分类扩展
                        continue

                    self.features.append(feature_vector)
                    self.labels.append(label)

                except Exception as e:
                    logger.warning(f"⚠️ 处理图像失败 {img_path}: {e}")
                    continue

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

        logger.info(f"✅ 特征提取完成：{len(self.features)} 个样本，{self.features.shape[1]} 维特征")
        logger.info(f"   类别分布：neutral={sum(self.labels==0)}, sad={sum(self.labels==1)}")

    def _detect_face(self, img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测人脸"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.fusion_detector.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            return tuple(faces[0])
        return None

    def _get_eye_sad_prob(self, frame_bgr: np.ndarray, face_rect: Tuple[int, int, int, int]) -> float:
        """获取眼睛模型的悲伤概率"""
        try:
            # 提取眼睛区域
            x, y, w, h = face_rect
            eye_top = y + int(h * 0.20)
            eye_bottom = y + int(h * 0.50)
            eye_left = x + int(w * 0.10)
            eye_right = x + int(w * 0.90)

            # 边界保护
            eye_top = max(0, eye_top)
            eye_bottom = min(frame_bgr.shape[0], eye_bottom)
            eye_left = max(0, eye_left)
            eye_right = min(frame_bgr.shape[1], eye_right)

            eye_region = frame_bgr[eye_top:eye_bottom, eye_left:eye_right]

            if eye_region.size == 0:
                return 0.5

            # 预处理
            import torch
            from torchvision import models
            import cv2

            eye_region = cv2.resize(eye_region, (224, 224))
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
            eye_region = eye_region.astype(np.float32) / 255.0

            # 修正：mean/std 形状为 (1, 1, 3) 以匹配 (H, W, C) 格式
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            eye_region = (eye_region - mean) / std

            eye_tensor = torch.from_numpy(eye_region.transpose(2, 0, 1)).float().unsqueeze(0).to(self.fusion_detector.device)

            # 预测
            with torch.no_grad():
                logits = self.fusion_detector.eye_model(eye_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                return float(probs[1])  # sad 概率

        except Exception as e:
            logger.warning(f"眼睛模型预测失败：{e}")
            return 0.5

    def train_classifier(self, test_split: float = 0.2):
        """训练分类器"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        if len(self.features) == 0:
            logger.error("❌ 没有训练数据，请先调用 load_dataset_and_extract_features()")
            return

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_split, random_state=42, stratify=self.labels
        )

        logger.info(f"\n📊 训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")

        # 尝试多种分类器
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
            'SVM (RBF)': SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        }

        best_clf = None
        best_acc = 0
        best_name = ""
        results = {}

        print("\n" + "="*80)
        print("🔬 分类器对比")
        print("="*80)

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

            print(f"\n{name}:")
            print(f"  准确率：{acc*100:.2f}%")
            print(classification_report(y_test, y_pred, target_names=['neutral', 'sad']))

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

    def save_model(self, output_path: str = None):
        """保存模型"""
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
            'feature_names': ['P_global_sad', 'P_global_neutral', 'P_eye_sad', 'P_global_angry', 'P_global_happy', 'P_global_surprise'],
        }

        joblib.dump(model_data, output_path)
        logger.info(f"✅ 模型已保存到：{output_path}")

        return output_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='元学习融合模型训练')
    parser.add_argument('--dataset-path', type=str,
                       default='/Users/asahiyang/Downloads/FacialEmotion/dataset',
                       help='数据集路径')
    parser.add_argument('--samples-per-class', type=int, default=200,
                       help='每个类别的样本数（默认 200）')
    parser.add_argument('--output-path', type=str,
                       default=None,
                       help='模型保存路径')

    args = parser.parse_args()

    # 创建训练器
    trainer = MetaLearnerTrainer(args.dataset_path, args.samples_per_class)

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
    print("1. 修改 decision_fusion_detector.py 加载这个模型")
    print("2. 用 model.predict() 替代手动的加权融合逻辑")


if __name__ == '__main__':
    main()
