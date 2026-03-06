"""
眼睛区域融合检测器 - 改进版 (Eye Region Fusion Detector - FIXED)

修复问题：
1. 添加 sad 阈值过滤（类似 anger 处理）
2. 调整融合权重，降低眼睛模型权重
3. 提高融合 sad 阈值
4. 添加中性表情保护机制
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib

logger = logging.getLogger(__name__)


class EyeFusionDetectorFixed:
    """
    眼睛区域融合检测器 - 改进版
    
    修复要点：
    - 降低眼睛模型权重从 0.6 → 0.4
    - 提高全局权重从 0.4 → 0.6
    - 添加 sad 置信度阈值（类似 anger）
    - 提高融合 sad 阈值从 0.45 → 0.55
    - 增加中性表情保护（当全局模型同意时）
    """

    # 情绪标签映射
    GLOBAL_EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    FUSION_LABELS = ['neutral', 'sad', 'hidden_sad']

    def __init__(self, config: Any, eye_model_path: Optional[str] = None,
                 rf_model_path: Optional[str] = None):
        """
        初始化融合检测器（改进版）
        
        新增参数：
        - sad_confidence_threshold: sad 置信度过滤阈值（0-100），默认30
          当眼睛模型的 sad 概率对应置信度 < 此值时，将其清零
        """
        self.config = config
        self.eye_model = None
        self.rf_model = None
        self._initialized = False

        # 设置默认模型路径
        base_dir = Path(__file__).parent.parent
        self.default_eye_model = base_dir / "models" / "paper_style_eye_region_fold1_best.pth"
        self.default_rf_model = base_dir.parent / "weights" / "rf_eye_fusion_model.pkl"

        self.eye_model_path = eye_model_path or str(self.default_eye_model)
        self.rf_model_path = rf_model_path or str(self.default_rf_model)

        # 眼睛检测器
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # ========== 新增配置参数 ==========
        # sad 置信度阈值：当眼睛模型输出的 sad 置信度 < 30% 时，认为不是 sad
        self.sad_confidence_threshold = config.get('emotion.sad_threshold', 30)
        
        # 融合权重调整：降低眼睛权重，提升全局权重
        self.eye_weight = 0.4      # 原来 0.6，现在 0.4
        self.global_weight = 0.6   # 原来 0.4，现在 0.6
        
        # 融合 sad 判断阈值：更严格的标准
        self.sad_fusion_threshold = 0.55  # 原来 0.45，现在 0.55

    def _lazy_init(self):
        """延迟加载模型"""
        if self._initialized:
            return

        logger.info(f"加载眼睛情绪模型：{self.eye_model_path}")

        # 1. 加载 PyTorch 眼睛模型
        try:
            import torch
            if not Path(self.eye_model_path).exists():
                logger.error(f"眼睛模型文件不存在：{self.eye_model_path}")
                raise FileNotFoundError(f"眼睛模型文件不存在：{self.eye_model_path}")

            self.eye_model = torch.load(self.eye_model_path, map_location='cpu')
            self.eye_model.eval()
            logger.info("✅ PyTorch 眼睛模型加载成功")
        except Exception as e:
            logger.error(f"加载 PyTorch 眼睛模型失败：{e}")
            raise

        # 2. 加载随机森林融合分类器
        try:
            if not Path(self.rf_model_path).exists():
                logger.warning(f"RF 融合模型不存在：{self.rf_model_path}，将使用加权融合")
                self.rf_model = None
            else:
                self.rf_model = joblib.load(self.rf_model_path)
                logger.info("✅ 随机森林融合模型加载成功")
        except Exception as e:
            logger.warning(f"加载 RF 模型失败：{e}，将使用加权融合")
            self.rf_model = None

        self._initialized = True

    def _detect_eyes(self, frame: np.ndarray, face_rect: Optional[Tuple] = None) -> Optional[np.ndarray]:
        """从帧中检测并裁剪眼睛区域"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_rect:
            x, y, w, h = face_rect
            face_roi = gray[y:y+h, x:x+w]
            upper_face = face_roi[0:h//2, :]

            eyes = self.eye_cascade.detectMultiScale(
                upper_face, scaleFactor=1.1, minNeighbors=3,
                minSize=(20, 20), maxSize=(w//2, h//3)
            )

            if len(eyes) > 0:
                ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                eye_x, eye_y = x + ex, y + ey
                eye_roi = frame[eye_y:eye_y+eh, eye_x:eye_x+ew]
                return cv2.resize(eye_roi, (48, 48))
        else:
            eyes = self.eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3,
                minSize=(30, 30), maxSize=(100, 100)
            )

            if len(eyes) > 0:
                ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                eye_roi = frame[ey:ey+eh, ex:ex+ew]
                return cv2.resize(eye_roi, (48, 48))

        return None

    def _predict_eye_emotion(self, eye_img: np.ndarray) -> Dict[str, float]:
        """
        使用 PyTorch 模型预测眼睛情绪
        返回 {'neutral': float, 'sad': float} 概率字典
        """
        try:
            import torch
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(eye_rgb).unsqueeze(0)

            with torch.no_grad():
                outputs = self.eye_model(input_tensor)

                if outputs.shape[1] == 2:
                    probs = torch.softmax(outputs, dim=1)[0]
                    return {
                        'neutral': float(probs[0].item()),
                        'sad': float(probs[1].item())
                    }
                else:
                    probs = torch.softmax(outputs[:, :2], dim=1)[0]
                    return {
                        'neutral': float(probs[0].item()),
                        'sad': float(probs[1].item())
                    }

        except Exception as e:
            logger.error(f"眼睛模型预测失败：{e}")
            return {'neutral': 0.5, 'sad': 0.5}

    def _get_global_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """获取全局情绪概率"""
        from .detector import create_emotion_detector

        try:
            detector = create_emotion_detector(self.config)
            all_emotions = detector.get_all_emotions(face_img)

            result = {}
            for emotion in self.GLOBAL_EMOTION_ORDER:
                result[emotion] = all_emotions.get(emotion, 0.0)

            return result
        except Exception as e:
            logger.error(f"全局情绪分析失败：{e}")
            return {emotion: 1/7 for emotion in self.GLOBAL_EMOTION_ORDER}
   




    def _fuse_and_classify(self, eye_probs, global_probs):
        """
        智能融合版本 - 眼睛模型只在 neutral vs sad 区间工作
        
        核心逻辑：
        1️⃣ 如果全局模型明确是 happy / angry / surprise → 直接信全局
        2️⃣ 只有在 neutral / sad 纠缠区间才融合
        3️⃣ 其他情况默认使用全局最高类别
        """

        # =========================================================
        # 第一阶段：如果全局已经明确是其他情绪，直接使用全局
        # =========================================================
        
        dominance_threshold = 0.15  # 明确情绪阈值
        
        if global_probs['happy'] > dominance_threshold:
            return 'happy', global_probs['happy'] * 100
        
        if global_probs['angry'] > dominance_threshold:
            return 'angry', global_probs['angry'] * 100
        
        if global_probs['surprise'] > dominance_threshold:
            return 'surprise', global_probs['surprise'] * 100
        
        if global_probs['fear'] > dominance_threshold:
            return 'fear', global_probs['fear'] * 100
        
        if global_probs['disgust'] > dominance_threshold:
            return 'disgust', global_probs['disgust'] * 100

        # =========================================================
        # 第二阶段：进入 neutral vs sad 专属区间
        # =========================================================
        
        # sad 置信度过滤
        if eye_probs['sad'] * 100 < 30:
            eye_probs = {'neutral': 1.0, 'sad': 0.0}

        # 加权融合（只针对 sad）
        sad_score = (
            eye_probs['sad'] * 0.4 +
            global_probs['sad'] * 0.6
        )

        # 中性保护机制
        if global_probs['neutral'] > 0.35:
            sad_score *= 0.7

        if sad_score > 0.55:
            return 'sad', sad_score * 100
        else:
            return 'neutral', (1 - sad_score) * 100

        # =========================================================
        # 理论不会走到这里，但保险处理
        # =========================================================
        top_emotion = max(global_probs.items(), key=lambda x: x[1])
        return top_emotion[0], top_emotion[1] * 100



def create_eye_fusion_detector_fixed(config: Any) -> EyeFusionDetectorFixed:
    return EyeFusionDetectorFixed(config)