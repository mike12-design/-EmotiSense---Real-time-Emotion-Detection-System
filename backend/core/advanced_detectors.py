# backend/core/advanced_detectors.py

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import timm
logger = logging.getLogger(__name__)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

class HSEmotionDetector:
    """
    HSEmotion (EmotiEffLib) - High-Speed Emotion Recognition.
    
    State-of-the-art emotion recognition model with:
    - High accuracy (66%+ on AffectNet)
    - Fast inference (~60ms on mobile)
    - Pre-trained on VGGFace2 and AffectNet
    """
    def is_high_confidence(self, probability: float, threshold: float = 95.0) -> bool:
        """
        判断概率是否超过高置信度阈值
        
        Args:
            probability: 置信度百分比 (0~100)
            threshold: 高置信度阈值，默认 95%
        
        Returns:
            True 如果概率 >= 阈值，否则 False
        """
        return probability >= threshold
    def __init__(self, config: Any):
        """
        Initialize HSEmotion detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self._initialized = False
        
        # Emotion labels for 8-class model
        self.emotion_labels_8 = [
            'angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise'
        ]
        
        # Emotion labels for 7-class model (without contempt)
        self.emotion_labels_7 = [
            'angry', 'disgust', 'fear', 'happy',
            'neutral', 'sad', 'surprise'
        ]
        
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            
            # Use 8-class model by default (can be configured)
            model_name = self.config.get('emotion.hsemotion_model', 'enet_b0_8_best_afew')
            
            logger.info(f"Loading HSEmotion model: {model_name}")
            self.model = HSEmotionRecognizer(model_name=model_name)
            
            # Determine which labels to use
            if '8' in model_name:
                self.emotion_labels = self.emotion_labels_8
            else:
                self.emotion_labels = self.emotion_labels_7
            
            self._initialized = True
            logger.info("HSEmotion model loaded successfully")
            
        except ImportError:
            logger.error("HSEmotion library not installed. Install with: pip install hsemotion")
            raise
        except Exception as e:
            logger.error(f"Failed to load HSEmotion model: {e}")
            raise
    
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion in a face image.
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        self._lazy_init()
        
        try:
            # HSEmotion expects RGB format
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get emotion prediction
            emotion, scores = self.model.predict_emotions(face_rgb, logits=False)
            
            # Get the top emotion
            top_emotion_idx = np.argmax(scores)
            emotion_name = self.emotion_labels[top_emotion_idx]
            confidence = float(scores[top_emotion_idx] * 100)
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"HSEmotion analysis error: {e}")
            return 'unknown', 0.0
    
    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Get all emotion scores.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        self._lazy_init()
        
        try:
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            emotion, scores = self.model.predict_emotions(face_rgb, logits=False)
            
            return {
                label: float(score * 100)
                for label, score in zip(self.emotion_labels, scores)
            }
            
        except Exception as e:
            logger.error(f"HSEmotion analysis error: {e}")
            return {}


class FERDetector:
    """
    FER (Facial Expression Recognition) library detector.
    
    Uses deep learning CNN for emotion recognition.
    Supports real-time video analysis.
    """
    
    def __init__(self, config: Any):
        """
        Initialize FER detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.detector = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from fer import FER as FERModel
            
            logger.info("Loading FER model...")
            # mtcnn=True uses MTCNN for face detection (more accurate but slower)
            use_mtcnn = self.config.get('emotion.fer_use_mtcnn', False)
            self.detector = FERModel(mtcnn=use_mtcnn)
            
            self._initialized = True
            logger.info("FER model loaded successfully")
            
        except ImportError:
            logger.error("FER library not installed. Install with: pip install fer")
            raise
        except Exception as e:
            logger.error(f"Failed to load FER model: {e}")
            raise
    
    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion in a face image.
        
        Args:
            face_img: Face image (BGR format from OpenCV)
            
        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        self._lazy_init()
        
        try:
            # FER expects RGB format
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            result = self.detector.detect_emotions(face_rgb)
            
            if not result:
                return 'unknown', 0.0
            
            # Get the first face's emotions
            emotions = result[0]['emotions']
            
            # Find top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name = top_emotion[0]
            confidence = float(top_emotion[1] * 100)
            
            return emotion_name, confidence
            
        except Exception as e:
            logger.error(f"FER analysis error: {e}")
            return 'unknown', 0.0
    
    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Get all emotion scores.
        
        Args:analyze_emotion
            face_img: Face image (BGR format)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        self._lazy_init()
        
        try:
            import cv2
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            result = self.detector.detect_emotions(face_rgb)
            
            if not result:
                return {}
            
            emotions = result[0]['emotions']
            
            return {
                emotion: float(score * 100)
                for emotion, score in emotions.items()
            }
            
        except Exception as e:
            logger.error(f"FER analysis error: {e}")
            return {}


class EnsembleEmotionDetector:
    """
    Ensemble detector that combines multiple models for better accuracy.
    """

    def __init__(self, config: Any):
        """
        Initialize ensemble detector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.detectors = []

        # Initialize available detectors
        enabled_models = config.get('emotion.ensemble_models', ['hsemotion'])

        for model_name in enabled_models:
            try:
                if model_name == 'hsemotion':
                    self.detectors.append(HSEmotionDetector(config))
                    logger.info("Added HSEmotion to ensemble")
                elif model_name == 'fer':
                    self.detectors.append(FERDetector(config))
                    logger.info("Added FER to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {model_name} to ensemble: {e}")

        if not self.detectors:
            raise RuntimeError("No detectors available for ensemble")

    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion using ensemble of models.

        Args:
            face_img: Face image (BGR format)

        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        # Collect predictions from all models
        all_emotions = {}

        for detector in self.detectors:
            try:
                emotions = detector.get_all_emotions(face_img)
                for emotion, score in emotions.items():
                    if emotion not in all_emotions:
                        all_emotions[emotion] = []
                    all_emotions[emotion].append(score)
            except Exception as e:
                logger.warning(f"Detector failed: {e}")

        if not all_emotions:
            return 'unknown', 0.0

        # Average scores across models
        averaged_emotions = {
            emotion: np.mean(scores)
            for emotion, scores in all_emotions.items()
        }

        # Get top emotion
        top_emotion = max(averaged_emotions.items(), key=lambda x: x[1])

        return top_emotion[0], float(top_emotion[1])


class EyeFusionDetector:
    """
    眼睛区域融合检测器 - 将专用眼睛模型与全局情绪特征融合

    特征级融合架构:
    1. 眼睛专用模型 (PyTorch): 擅长识别中性 vs 悲伤的细微肌肉差异
    2. 全局情绪检测器 (DeepFace/HSEmotion): 提供完整的情绪概率分布
    3. 随机森林融合分类器: 学习如何加权两种特征进行最终决策

    特征向量 (9 维):
    [eye_neutral, eye_sad, angry, disgust, fear, happy, sad, surprise, neutral]
    """

    EMOTION_LABELS = ['neutral', 'sad', 'hidden_sad']

    def __init__(self, config: Any):
        """
        Initialize Eye Fusion Detector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.eye_model = None
        self.rf_model = None
        self._initialized = False
        self._detector = None  # 全局情绪检测器

        # 模型路径
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        self.eye_model_path = base_dir / "app" / "models" / "paper_style_eye_region_fold1_best.pth"
        self.rf_model_path = base_dir.parent / "weights" / "rf_eye_fusion_model.pkl"

    def _lazy_init(self):
        """Lazy initialization"""
        if self._initialized:
            return

        import torch

        # 1. 加载 PyTorch 眼睛模型
        try:
            if not self.eye_model_path.exists():
                logger.error(f"眼睛模型不存在：{self.eye_model_path}")
                raise FileNotFoundError(f"眼睛模型不存在：{self.eye_model_path}")

            self.eye_model = torch.load(self.eye_model_path, map_location='cpu', weights_only=False)
            self.eye_model.eval()
            logger.info(f"✅ 眼睛模型已加载：{self.eye_model_path}")
        except Exception as e:
            logger.error(f"加载眼睛模型失败：{e}")
            raise

        # 2. 加载 RF 融合分类器
        try:
            if self.rf_model_path.exists():
                import joblib
                self.rf_model = joblib.load(self.rf_model_path)
                logger.info(f"✅ RF 融合模型已加载：{self.rf_model_path}")
            else:
                logger.warning(f"RF 模型不存在：{self.rf_model_path}, 将使用规则融合")
        except Exception as e:
            logger.warning(f"加载 RF 模型失败：{e}")

        # 3. 初始化全局情绪检测器
        try:
            from .detector import create_emotion_detector
            self._detector = create_emotion_detector(self.config)
            logger.info("✅ 全局情绪检测器已初始化")
        except Exception as e:
            logger.error(f"初始化全局检测器失败：{e}")
            raise

        self._initialized = True

    def _extract_eye_features(self, face_img: np.ndarray, already_face_roi: bool = True) -> Dict[str, float]:
        """
        使用 PyTorch 模型提取眼睛情绪特征

        Args:
            face_img: 人脸图像 (BGR)
            already_face_roi: 是否已经是人脸 ROI（来自外部裁剪），默认 True

        Returns:
            {'neutral': float, 'sad': float}
        """
        import torch
        from torchvision import transforms
        import cv2

        if already_face_roi:
            # 直接按比例裁剪，不重复检测人脸
            # 训练时的眼部区域定义：上半脸的 20%-50% 位置
            h, w = face_img.shape[:2]
            eye_top = int(h * 0.20)
            eye_bottom = int(h * 0.50)
            eye_left = int(w * 0.10)
            eye_right = int(w * 0.90)
            eye_region = face_img[eye_top:eye_bottom, eye_left:eye_right]
        else:
            # 旧逻辑：先检测人脸再裁剪（用于处理完整图像输入）
            h, w = face_img.shape[:2]
            eye_region = face_img[0:h//2, :]

        #  resize 到模型输入尺寸
        eye_region = cv2.resize(eye_region, (48, 48))

        # 预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        input_tensor = transform(eye_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.eye_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

            return {
                'neutral': float(probs[0].item()),
                'sad': float(probs[1].item())
            }

    def analyze_emotion(self, frame: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
        """
        分析情绪 (融合眼睛特征和全局特征)

        Args:
            frame: 完整帧图像 (BGR 格式)
            face_rect: 人脸框坐标 (x, y, w, h)，可选

        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        self._lazy_init()

        # 1. 提取人脸 ROI (用于眼睛特征提取)
        if face_rect:
            x, y, w, h = face_rect
            face_img = frame[y:y+h, x:x+w]
        else:
            face_img = frame

        # 2. 眼睛模型预测（使用裁剪后的人脸 ROI）
        eye_probs = self._extract_eye_features(face_img, already_face_roi=True)

        # 3. 全局情绪预测 ⚠️ 关键修改：传入完整帧，让全局检测器自己处理上下文
        global_probs = self._detector.get_all_emotions(frame)

        # 3. 融合决策
        if self.rf_model is not None:
            # 使用 RF 分类器
            import numpy as np
            feature_vector = np.array([
                eye_probs['neutral'],
                eye_probs['sad'],
                global_probs.get('angry', 0.0),
                global_probs.get('disgust', 0.0),
                global_probs.get('fear', 0.0),
                global_probs.get('happy', 0.0),
                global_probs.get('sad', 0.0),
                global_probs.get('surprise', 0.0),
                global_probs.get('neutral', 0.0)
            ]).reshape(1, -1)

            prediction = self.rf_model.predict(feature_vector)[0]
            proba = self.rf_model.predict_proba(feature_vector)[0]
            confidence = float(np.max(proba) * 100)

            # 映射标签
            label_map = {0: 'neutral', 1: 'sad', 2: 'sad'}
            return label_map.get(prediction, 'neutral'), confidence
        else:
            # 规则融合：眼睛模型权重更高
            eye_weight = 0.6
            global_weight = 0.4

            sad_score = (eye_probs['sad'] * eye_weight +
                        global_probs.get('sad', 0.0) * global_weight)

            # 如果两者都认为悲伤，增强置信度
            if eye_probs['sad'] > 0.5 and global_probs.get('sad', 0.0) > 0.3:
                sad_score = min(sad_score * 1.2, 1.0)

            # 阈值从 0.45 提升到 0.60，给 neutral 留更多空间
            if sad_score > 0.60:
                return 'sad', sad_score * 100
            else:
                return 'neutral', (1 - sad_score) * 100

