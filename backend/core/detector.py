"""
Face and emotion detection module for EmotiSense.
Handles face detection, eye detection, and emotion analysis.
"""
# backend/core/detector.py

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from deepface import DeepFace
import logging
from pathlib import Path
from ultralytics import YOLO

from .config import Config

logger = logging.getLogger(__name__)


def create_emotion_detector(config: Config, use_meta_learner: Optional[bool] = None):
    """
    Factory function to create the appropriate emotion detector.

    Args:
        config: Configuration object
        use_meta_learner: Whether to use meta-learner fusion (overrides config if set)

    Returns:
        Emotion detector instance
    """
    detector_type = config.get('emotion.detector_type', 'deepface')

    # 如果 detector_type 是 'meta_learner'，自动使用 decision_fusion + use_meta_learner=True
    if detector_type == 'meta_learner':
        from .decision_fusion_detector import create_improved_decision_fusion_detector
        logger.info("Using Meta-Learner Fusion detector (Stacking 分类器)")
        return create_improved_decision_fusion_detector(config, use_meta_learner=True)

    # 如果是 decision_fusion，检查配置中的 use_meta_learner 选项
    if detector_type == 'decision_fusion':
        from .decision_fusion_detector import create_improved_decision_fusion_detector
        # 优先使用传入的参数，否则从 config 读取
        if use_meta_learner is None:
            use_meta_learner = config.get('emotion.use_meta_learner', False)
        if use_meta_learner:
            logger.info("Using Decision Fusion detector with Meta-Learner (元学习器融合)")
        else:
            logger.info("Using Decision Fusion detector (Rule-based 规则融合)")
        return create_improved_decision_fusion_detector(config, use_meta_learner=use_meta_learner)

    if detector_type == 'hsemotion':
        from .advanced_detectors import HSEmotionDetector
        logger.info("Using HSEmotion detector")
        return HSEmotionDetector(config)
    elif detector_type == 'fer':
        from .advanced_detectors import FERDetector
        logger.info("Using FER detector")
        return FERDetector(config)
    elif detector_type == 'eye_fusion':
        from .advanced_detectors import EyeFusionDetector
        logger.info("Using Eye Fusion detector (眼睛模型 + 全局特征)")
        return EyeFusionDetector(config)
    elif detector_type == 'ensemble':
        from .advanced_detectors import EnsembleEmotionDetector
        logger.info("Using Ensemble detector")
        return EnsembleEmotionDetector(config)
    else:  # 'deepface' or default
        logger.info("Using DeepFace detector")
        return EmotionDetector(config)


class FaceDetector:
    """
    Handles face detection using YOLOv8 (State-of-the-art)
    and eye detection using OpenCV Haar Cascades (Classic).
    """

    def __init__(self, config: Config):
        """
        Initialize face detector.
        """
        self.config = config

        # 1. 初始化 YOLOv8 人脸模型
        # 自动寻找模型路径，建议放在 backend/assets/models/ 下
        model_path = Path(__file__).parent.parent / "models" / "yolov8m-face-lindevs.pt"

        if not model_path.exists():
            logger.warning(f"⚠️ 未找到本地模型：{model_path}，将尝试自动下载或使用标准 yolov8n.pt (效果可能稍差)")
            self.model_type = "yolo_standard"
            self.face_model = YOLO("yolov8n.pt") # 回退方案
        else:
            logger.info(f"🚀 加载 YOLOv8 人脸专用模型：{model_path}")
            self.model_type = "yolo_face"
            self.face_model = YOLO(str(model_path))

        # 2. 保留 Haar Cascade 仅用于眼部检测 (在人脸 ROI 内检测)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        # 平滑参数
        self.last_face_rect: Optional[Tuple[int, int, int, int]] = None
        self.smoothing_factor = config.get('face_detection.smoothing_factor', 0.3)
        self.conf_threshold = 0.5  # 置信度阈值

    def smooth_face_rect(self, face_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """对人脸框进行指数平滑，减少视频抖动"""
        if self.last_face_rect is None:
            self.last_face_rect = face_rect
            return face_rect

        x, y, w, h = face_rect
        last_x, last_y, last_w, last_h = self.last_face_rect

        # 指数移动平均 (EMA)
        sf = self.smoothing_factor
        smooth_x = int(last_x * (1 - sf) + x * sf)
        smooth_y = int(last_y * (1 - sf) + y * sf)
        smooth_w = int(last_w * (1 - sf) + w * sf)
        smooth_h = int(last_h * (1 - sf) + h * sf)

        smoothed = (smooth_x, smooth_y, smooth_w, smooth_h)
        self.last_face_rect = smoothed

        return smoothed

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用 YOLOv8 检测人脸
        返回格式：[(x, y, w, h), ...]
        """
        # YOLO 期望输入是 BGR (OpenCV 默认) 或 RGB
        # verbose=False 关闭控制台打印
        results = self.face_model(frame, verbose=False, conf=self.conf_threshold)

        detected_faces = []

        # 解析 YOLO 结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 如果使用的是通用 yolov8n.pt，类别 0 是 person
                if self.model_type == "yolo_standard" and int(box.cls) != 0:
                    continue

                # 获取坐标 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # 转换为 (x, y, w, h)
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # 边界保护
                h_img, w_img = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                detected_faces.append((x, y, w, h))

        # 如果有脸，对第一张脸（主脸）进行平滑处理
        if len(detected_faces) > 0:
            # 这里假设最大的脸是主用户
            main_face = max(detected_faces, key=lambda f: f[2] * f[3])
            smoothed_face = self.smooth_face_rect(main_face)
            return [smoothed_face] # 目前系统逻辑主要处理单人

        return []

    def detect_faces_with_roi(self, frame: np.ndarray) -> List[Dict]:
        """
        高级检测：YOLO 检测人脸 + Haar 检测眼部
        返回全局坐标，用于可解释人工智能 (XAI) 分析

        注意：返回结果按人脸面积降序排序，保证：
        - 列表第一个是最大（最近）的人脸
        - 多人场景下取主用户时结果稳定
        """
        # 1. 先用 YOLO 获取人脸框
        faces = self.detect_faces(frame)

        results = []

        # 2. 在每个人脸框内检测眼睛
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            # 提取人脸 ROI (灰度图，用于 Haar)
            # 优化：只取上半脸找眼睛，减少误判（把嘴巴当眼睛）
            roi_h = int(h * 0.6)
            face_roi_gray = gray_frame[y:y+roi_h, x:x+w]

            eyes = []
            if face_roi_gray.size > 0:
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20)
                )

            # 转换眼睛坐标到全局系
            global_eyes = []
            for (ex, ey, ew, eh) in eyes:
                global_eyes.append((x + ex, y + ey, ew, eh))

            results.append({
                "rect": (x, y, w, h),
                "has_eyes": len(global_eyes) >= 1,
                "eye_coords": global_eyes,
                "area": w * h  # 用于排序
            })

        # 按面积降序排序：最大（最近）的人脸排在前面
        results.sort(key=lambda x: x["area"], reverse=True)

        # 移除临时的 area 字段
        for r in results:
            r.pop("area", None)

        return results

    def reset_tracking(self):
        self.last_face_rect = None


class EmotionDetector:
    """Handles emotion detection using DeepFace."""

    def __init__(self, config: Config):
        """
        Initialize emotion detector.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.emotion_model = None
        self.anger_threshold = config.get('emotion.anger_threshold', 50)

    def analyze_emotion(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        Analyze emotion from face image.

        Args:
            face_img: Face image (BGR format)

        Returns:
            Tuple of (emotion_name, confidence_percentage)
        """
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip' if self.emotion_model is not None else 'opencv'
            )

            # Get emotion results
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']

            # Initialize model flag after first successful analysis
            if self.emotion_model is None:
                self.emotion_model = emotions

            # Apply anger threshold filter
            if emotions.get('angry', 0) < self.anger_threshold:
                emotions['angry'] = 0

            # Get top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])

            return top_emotion

        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return ('Unknown', 0.0)

    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """
        Get all emotion scores.

        Args:
            face_img: Face image (BGR format)

        Returns:
            Dictionary mapping emotion names to confidence scores (0-100)
        """
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']

            # Apply anger threshold filter
            if emotions.get('angry', 0) < self.anger_threshold:
                emotions['angry'] = 0

            # Normalize to percentage (0-100)
            total = sum(emotions.values())
            if total > 0:
                return {k: (v / total) * 100 for k, v in emotions.items()}
            else:
                return {k: 100 / len(emotions) for k in emotions.keys()}

        except Exception as e:
            logger.error(f"Get all emotions error: {e}")
            return {}

    def is_high_confidence(self, confidence: float) -> bool:
        """
        Check if emotion confidence is high enough to log.

        Args:
            confidence: Confidence percentage

        Returns:
            True if confidence exceeds threshold
        """
        threshold = self.config.get('emotion.high_confidence_threshold', 95)
        return confidence > threshold


def find_identity(face_img: np.ndarray, face_db: list) -> str:
    """
    Identify person based on face embedding.
    """
    try:
        # 1. 提取当前人脸特征
        result = DeepFace.represent(
            img_path=face_img,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=False
        )

        if not result:
            return "Stranger"

        current_embedding = np.array(result[0]["embedding"])

        # 2. 初始化比对变量
        best_name = "Stranger"
        best_distance = 1.0

        # 【建议】将阈值调优为 0.6，适配普通摄像头和多变光线
        THRESHOLD = 0.6

        # 3. 遍历数据库进行比对
        for item in face_db:
            db_embedding = np.array(item["embedding"])

            # 计算余弦距离
            dot_product = np.dot(current_embedding, db_embedding)
            norm_curr = np.linalg.norm(current_embedding)
            norm_db = np.linalg.norm(db_embedding)

            # 避免除以 0
            if norm_curr == 0 or norm_db == 0:
                continue

            distance = 1 - (dot_product / (norm_curr * norm_db))

            # 只记录调试信息，不在这里做最终决定
            # print(f"DEBUG: 正在对比 {item['username']}, 距离：{distance:.4f}")

            if distance < best_distance:
                best_distance = distance
                best_name = item["username"]

        # 4. 【关键：在所有对比结束后再做判断】
        if best_distance < THRESHOLD:
            # 如果最好的匹配结果小于阈值，说明找到了
            print(f"✅ 识别成功：{best_name} (距离：{best_distance:.4f})")
            return best_name
        else:
            # 否则即便有"最接近的人"，也不认为是同一个人
            # print(f"❌ 识别失败：最接近 {best_name} 但距离 ({best_distance:.4f}) 超过阈值")
            return "Stranger"

    except Exception as e:
        logger.warning(f"find_identity error: {e}")
        return "Stranger"
