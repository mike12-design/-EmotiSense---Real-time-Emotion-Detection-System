"""
改进的决策级融合检测器 (Improved Decision-Level Fusion Detector)

核心改进：
1. ✅ 修复 HSEmotion 失败时的 fallback 机制
2. ✅ 不再排除 'happy' 等情绪，改为全局融合
3. ✅ 眼睛模型专注于 sad/neutral，其他情绪直通 DeepFace
4. ✅ 添加置信度门控，防止弱判断导致的误判
5. ✅ 完整的错误处理和诊断信息
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from .advanced_detectors import HSEmotionDetector
from deepface import DeepFace
from torchvision import models

from .config import Config


# 在文件顶部添加
from .eye_feature_extractor import EyeFeatureExtractor
from .meta_learner_inference import MetaLearnerPrediction, predict_with_meta_learner_complete

# 在 __init__ 中添加

    # ... 现有代码 ...
    
    # 新增：初始化眼睛特征提取器
    




logger = logging.getLogger(__name__)


class ImprovedDecisionFusionDetector:
    """
    改进的决策级融合检测器
    
    融合策略：
    - sad/neutral: 使用眼睛模型 (70%) + HSEmotion (30%) 的加权融合
    - 其他情绪: 直接使用 HSEmotion（眼睛模型对其他情绪无特化）
    - HSEmotion 失败: 回退到纯眼睛模型（仅限 sad/neutral）或 DeepFace
    """

    # 眼睛模型配置
    DEFAULT_EYE_MODEL_PATH = Path(__file__).parent.parent / "models" / "paper_style_eye_region_fold12_best.pth"
    TARGET_SIZE = (224, 224)
    EYE_REGION_SIZE = (100, 30)

    # 情绪标签映射
    EYE_LABELS = ['neutral', 'sad']
    # HSEmotion 可能的情绪标签（8类或7类）
    GLOBAL_EMOTION_LABELS_8 = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    GLOBAL_EMOTION_LABELS_7 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # 融合仅限于这些情绪（眼睛模型优化的范围）
    FUSION_TARGET_EMOTIONS = {'sad', 'neutral'}

    # 元学习器模型路径
    DEFAULT_META_LEARNER_PATH = Path(__file__).parent.parent / "weights" / "meta_learner_fusion_model.pkl"




    def __init__(self, config: Config, eye_model_path: Optional[str] = None, k: float = 0.3,
             ema_alpha: float = 0.3, use_meta_learner: bool = False):
        self.eye_feature_extractor = None

 
        """
        初始化改进的决策融合检测器

        Args:
            config: 配置对象
            eye_model_path: PyTorch 眼睛模型路径
            k: 眼睛模型权重 (0.0-1.0)，默认 0.7
            ema_alpha: EMA 平滑系数
            use_meta_learner: 是否使用元学习器（替代手动规则融合）
        """
        self.config = config
        self.k = max(0.0, min(1.0, k))
        self.eye_model = None
        self.device = None
        self._initialized = False
        self.use_meta_learner = use_meta_learner

        # 元学习器
        self.meta_learner = None
        self.meta_learner_path = self.DEFAULT_META_LEARNER_PATH

        # EMA 平滑状态
        self._emotion_ema: Dict[str, float] = {}
        self._ema_alpha: float = float(ema_alpha)

        # HSEmotion 状态追踪（用于诊断）
        self._hsemotion_error_count = 0
        self._hsemotion_available = True

        # 备用检测器（当 HSEmotion 失败时）
        self._deepface_detector = None

        if eye_model_path:
            self.eye_model_path = Path(eye_model_path)
        else:
            self.eye_model_path = self.DEFAULT_EYE_MODEL_PATH

        logger.info(f"[ImprovedDecisionFusion] 初始化中...")
        logger.info(f"  - 眼睛模型权重 k={self.k} (眼睛{self.k*100:.0f}%, HSEmotion{(1-self.k)*100:.0f}%)")
        logger.info(f"  - EMA 平滑系数 alpha={self._ema_alpha}")
        logger.info(f"  - 融合目标情绪：{self.FUSION_TARGET_EMOTIONS}")
        logger.info(f"  - 使用元学习器：{self.use_meta_learner}")

        # 人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _lazy_init(self):
        """延迟加载所有模型"""
        if self._initialized:
            return

        logger.info("[系统初始化] 正在加载改进的决策级融合检测器...")

        # 1. 设置计算设备
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("[Device] 使用 MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("[Device] 使用 CUDA")
        else:
            self.device = torch.device("cpu")
            logger.info("[Device] 使用 CPU")

        try:
            # ✅ 修正：显式传递 eye_model_path，防止 EyeFeatureExtractor 内部默认路径出错
            self.eye_feature_extractor = EyeFeatureExtractor(
                eye_model_path=str(self.eye_model_path), 
                device=str(self.device)
            )
            logger.info("✅ 眼睛特征提取器初始化完成")
        except Exception as e:
            logger.error(f"⚠️ 眼睛特征提取器加载失败：{e}")
            self.eye_feature_extractor = None

        # 2. 加载眼睛专家模型
        self._load_eye_expert_model()

        # 3. 尝试加载 HSEmotion
        self._load_hsemotion_model()

        # 4. 初始化 DeepFace 作为备用
        self._deepface_detector = DeepFaceBackup()

        # 5. 加载元学习器（如果启用）
        if self.use_meta_learner:
            self._load_meta_learner()

        self._initialized = True
        logger.info("✅ 改进决策级融合检测器初始化完成")

    def _load_eye_expert_model(self):
        """加载眼睛专家模型"""
        try:
            if not self.eye_model_path.exists():
                raise FileNotFoundError(f"眼睛模型文件不存在：{self.eye_model_path}")

            logger.info(f"加载眼睛专家模型：{self.eye_model_path.name}...")

            # 构建 ResNet18 架构
            self.eye_model = models.resnet18(weights=None)
            num_ftrs = self.eye_model.fc.in_features
            self.eye_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, 2)  # 2 分类：neutral, sad
            )

            checkpoint = torch.load(self.eye_model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.eye_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.eye_model.load_state_dict(checkpoint)

            self.eye_model.to(self.device)
            self.eye_model.eval()

            logger.info("✅ 眼睛专家模型加载成功")

        except Exception as e:
            logger.error(f"❌ 加载眼睛模型失败：{e}")
            raise

    def _load_hsemotion_model(self):
        """尝试加载 HSEmotion 模型，失败则标记不可用"""
        try:
            logger.info("尝试加载 HSEmotion 模型...")
            self.global_detector = HSEmotionDetector(self.config)
            
            # 测试模型是否真的可用
            test_img = self._create_test_face_image()
            test_result = self.global_detector.get_all_emotions(test_img)
            
            if test_result:
                logger.info(f"✅ HSEmotion 模型加载成功，支持标签：{list(test_result.keys())}")
                self._hsemotion_available = True
            else:
                logger.warning("⚠️ HSEmotion 返回空结果，标记为不可用")
                self._hsemotion_available = False
                
        except Exception as e:
            logger.error(f"❌ HSEmotion 加载失败：{e}")
            logger.warning("⚠️ 将使用 DeepFace 作为备用全局检测器")
            self._hsemotion_available = False

    def _load_meta_learner(self):
        """加载元学习器模型"""
        try:
            import joblib

            if not self.meta_learner_path.exists():
                logger.warning(f"⚠️ 元学习器模型文件不存在：{self.meta_learner_path}，将使用规则融合")
                self.use_meta_learner = False
                return

            logger.info(f"加载元学习器模型：{self.meta_learner_path.name}...")
            model_data = joblib.load(self.meta_learner_path)

            self.meta_learner = model_data.get('classifier')
            self.meta_learner_name = model_data.get('classifier_name', 'Unknown')

            logger.info(f"✅ 元学习器加载成功：{self.meta_learner_name}")

        except Exception as e:
            logger.error(f"❌ 加载元学习器失败：{e}")
            logger.warning("⚠️ 将使用规则融合")
            self.use_meta_learner = False
    def _predict_with_meta_learner(self, frame_bgr: np.ndarray,
                                    face_rect: Optional[Tuple[int, int, int, int]] = None,
                                    all_emotions: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
            """
            使用元学习器进行完整预测
            ✅ 修复：必须添加 all_emotions 参数，否则调用时会报错参数过多
            """
            try:
                # 如果外层没传，才自己去算
                if all_emotions is None:
                    all_emotions = self._get_global_emotions(frame_bgr)
                
                # 如果算出来还是空（模型失败），做兜底处理
                if not all_emotions:
                    return 'neutral', 50.0

                # 检测人脸框
                if face_rect is None:
                    face_rect = self._detect_face_once(frame_bgr)

                # 3️⃣ 获取眼睛情绪概率
                P_eye = self._get_eye_sad_prob(frame_bgr, face_rect) if face_rect else 0.5

                # 4️⃣ 构建特征向量 [P_global_sad, P_global_neutral, P_eye, ...]
                features = MetaLearnerPrediction.extract_features(all_emotions, P_eye)

                # 5️⃣ 用二分类 RF 预测 (结果必然是 neutral 或 sad)
                emotion, confidence = MetaLearnerPrediction.predict(self.meta_learner, features)

                return emotion, confidence

            except Exception as e:
                logger.error(f"元学习器推理异常: {e}")
                # 出错回退到全局最高的那项
                if all_emotions:
                    top_em = max(all_emotions.items(), key=lambda x: x[1])[0]
                    return top_em, float(all_emotions[top_em])
                return 'neutral', 50.0

 

    def _create_test_face_image(self) -> np.ndarray:
        """
        创建一个简单的测试人脸图像（比全黑图像更有意义）
        这样可以真正测试模型是否可用
        """
        # 创建一个 224x224 的中等灰度图像（代表人脸皮肤色调）
        img = np.full((224, 224, 3), 150, dtype=np.uint8)
        return img

    def _detect_face_once(self, full_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测单张人脸"""
        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            return tuple(faces[0])
        return None

    def _extract_eye_region(self, full_img: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """从全脸提取眼部区域"""
        if face_rect is not None:
            x, y, w, h = face_rect
            eye_top = y + int(h * 0.20)
            eye_bottom = y + int(h * 0.50)
            eye_left = x + int(w * 0.10)
            eye_right = x + int(w * 0.90)
        else:
            # 无人脸框时的降级方案
            img_h, img_w = full_img.shape[:2]
            eye_top = int(img_h * 0.15)
            eye_bottom = int(img_h * 0.35)
            eye_left = int(img_w * 0.30)
            eye_right = int(img_w * 0.70)

        # 边界保护
        eye_top = max(0, eye_top)
        eye_bottom = min(full_img.shape[0], eye_bottom)
        eye_left = max(0, eye_left)
        eye_right = min(full_img.shape[1], eye_right)

        eye_region = full_img[eye_top:eye_bottom, eye_left:eye_right]

        if eye_region.size == 0:
            eye_region = np.zeros((self.EYE_REGION_SIZE[1], self.EYE_REGION_SIZE[0], 3), dtype=np.uint8)
        else:
            eye_region = cv2.resize(eye_region, self.EYE_REGION_SIZE)

        return eye_region
    def extract_eye_region(self, frame_bgr: np.ndarray, 
                          face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        从帧中提取眼睛区域 - 包含 Debug 存图功能和硬裁剪兜底
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        eye_found = False
        
        # 🎯 尝试方法 1：Haar 级联检测
        if face_rect is not None:
            x, y, w, h = face_rect
            upper_face_roi = gray[y:y+int(h*0.6), x:x+w]
            
            if upper_face_roi.size > 0:
                eyes = self.eye_cascade.detectMultiScale(
                    upper_face_roi, scaleFactor=1.1, minNeighbors=4, 
                    minSize=(20, 20), maxSize=(w//2, int(h*0.3))
                )
                if len(eyes) > 0:
                    ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                    eye_x, eye_y = x + ex, y + ey
                    eye_roi = frame_bgr[eye_y:eye_y+eh, eye_x:eye_x+ew]
                    if eye_roi.size > 0:
                        eye_found = True
                        return cv2.resize(eye_roi, self.EYE_REGION_SIZE)
        
        # 🎯 尝试方法 2：无人脸框时的全局检测
        if not eye_found and face_rect is None:
            eyes = self.eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, 
                minSize=(30, 30), maxSize=(150, 100)
            )
            if len(eyes) > 0:
                ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                eye_roi = frame_bgr[ey:ey+eh, ex:ex+ew]
                if eye_roi.size > 0:
                    eye_found = True
                    return cv2.resize(eye_roi, self.EYE_REGION_SIZE)

        # =========================================================
        # 🚨 触发警告：如果走到这里，说明 Haar 检测失败了！
        # =========================================================
        logger.warning("⚠️ 未检测到眼睛，触发存图并启用硬裁剪兜底")
        
        # 1. Debug 存图逻辑：保存失败的图片
        try:
            import time
            from pathlib import Path
            import cv2
            
            # 在项目根目录创建一个 debug 文件夹
            debug_dir = Path(__file__).parent.parent.parent / "debug_failed_eyes"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            save_path = debug_dir / f"failed_eye_{timestamp}.jpg"
            
            # 画个人脸红框，方便你看看检测器是在哪个区域内找眼睛失败的
            debug_img = frame_bgr.copy()
            if face_rect is not None:
                bx, by, bw, bh = face_rect
                cv2.rectangle(debug_img, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                # 画出它搜索眼睛的“上半脸”区域 (绿色框)
                cv2.rectangle(debug_img, (bx, by), (bx+bw, by+int(bh*0.6)), (0, 255, 0), 2)
                
            cv2.imwrite(str(save_path), debug_img)
            # logger.info(f"📸 失败图片已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存 Debug 图片失败: {e}")

        # 2. 终极兜底方案：几何比例硬裁剪 (Blind Crop)
        # 不要返回 None，强行切出眼睛所在的大致位置送给模型！
        if face_rect is not None:
            x, y, w, h = face_rect
        else:
            x, y, w, h = 0, 0, frame_bgr.shape[1], frame_bgr.shape[0]

        # 眼睛大概在面部 20% 到 50% 的高度区间，水平 10% 到 90% 的区间
        eye_top = max(0, y + int(h * 0.20))
        eye_bottom = min(frame_bgr.shape[0], y + int(h * 0.50))
        eye_left = max(0, x + int(w * 0.10))
        eye_right = min(frame_bgr.shape[1], x + int(w * 0.90))

        eye_roi = frame_bgr[eye_top:eye_bottom, eye_left:eye_right]
        
        if eye_roi.size > 0:
            return cv2.resize(eye_roi, self.EYE_REGION_SIZE)
            
        return None
        """将眼部图片转换为模型输入"""
        eye_region = cv2.resize(eye_img, self.TARGET_SIZE)
        eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        eye_region = eye_region.astype(np.float32) / 255.0
        eye_region = eye_region.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        eye_region = (eye_region - mean) / std
        return torch.from_numpy(eye_region).float().unsqueeze(0).to(self.device)
    def _get_eye_sad_prob(self, frame_bgr: np.ndarray,
                        face_rect: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        获取眼睛的悲伤概率 (0.0 ~ 1.0)
        
        使用统一的特征提取器确保训练和推理一致
        """
        try:
            if self.eye_feature_extractor is None:
                logger.warning("⚠️ 眼睛特征提取器未初始化")
                return 0.5

            # 1. 提取眼睛区域
            eye_img = self.eye_feature_extractor.extract_eye_region(frame_bgr, face_rect)

            if eye_img is None:
                logger.debug("⚠️ 未检测到眼睛")
                return -1.0


            # 2. 获取悲伤概率
            P_sad = self.eye_feature_extractor.get_eye_sad_probability(eye_img)

            return float(max(0.0, min(1.0, P_sad)))

        except Exception as e:
            logger.warning(f"⚠️ 眼睛概率获取失败：{e}")
            return 0.5

    def _get_global_emotions(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        """
        获取全局情绪概率（优先 HSEmotion，失败则用 DeepFace）
        
        Returns:
            {'emotion_name': confidence_percentage, ...}
        """
        if not self._hsemotion_available:
            return self._deepface_detector.get_all_emotions(frame_bgr)

        try:
            emotions = self.global_detector.get_all_emotions(frame_bgr)
            
            if not emotions:
                # HSEmotion 返回空，则递推备用方案
                logger.warning("HSEmotion 返回空结果，切换到 DeepFace")
                self._hsemotion_error_count += 1
                if self._hsemotion_error_count >= 5:
                    logger.error("HSEmotion 连续失败 5 次，标记为不可用")
                    self._hsemotion_available = False
                
                return self._deepface_detector.get_all_emotions(frame_bgr)
            
            self._hsemotion_error_count = 0  # 重置错误计数
            return emotions

        except Exception as e:
            logger.warning(f"HSEmotion 执行失败：{e}，切换到 DeepFace")
            self._hsemotion_error_count += 1
            if self._hsemotion_error_count >= 5:
                logger.error("HSEmotion 连续失败 5 次，标记为不可用")
                self._hsemotion_available = False
            
            return self._deepface_detector.get_all_emotions(frame_bgr)

    def _smooth_emotions(self, all_emotions: dict) -> dict:
        """对全局情绪做 EMA 平滑"""
        if not isinstance(all_emotions, dict) or len(all_emotions) == 0:
            return dict(self._emotion_ema)

        for emotion, prob in all_emotions.items():
            try:
                prob_val = float(prob)
            except Exception:
                prob_val = 0.0

            if emotion not in self._emotion_ema:
                self._emotion_ema[emotion] = prob_val
            else:
                self._emotion_ema[emotion] = (
                    self._ema_alpha * prob_val +
                    (1.0 - self._ema_alpha) * self._emotion_ema[emotion]
                )

        return dict(self._emotion_ema)

#     def analyze_emotion(self, frame_bgr: np.ndarray, k: Optional[float] = None,
#                         face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
#         """
#         执行改进的决策级融合分析
        
#         核心改进：
#         1. 获取完整的全局情绪分布
#         2. 根据情绪类型决定是否需要融合
#         3. sad/neutral 由眼睛模型增强
#         4. 其他情绪直接返回全局检测器的结果
#         """
#         self._lazy_init()

#         k = k if k is not None else self.k
#         k = max(0.0, min(1.0, k))

#         # Step 1：获取全局情绪分布
#         all_emotions = {}
#         try:
#             all_emotions = self._get_global_emotions(frame_bgr)
#             all_emotions = self._smooth_emotions(all_emotions)

#             if len(all_emotions) > 0:
#                 top_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
#                 top_confidence = float(all_emotions[top_emotion])
#             else:
#                 top_emotion = 'neutral'
#                 top_confidence = 50.0

#         except Exception as e:
#             logger.warning(f"全局情绪分析失败：{e}")
#             top_emotion = 'neutral'
#             top_confidence = 50.0
#             all_emotions = {}
#         # 调试输出
#         try:
#             debug_all = {em: round(float(v), 1) for em, v in all_emotions.items()} if all_emotions else {}
#             print(f"[DEBUG] 全局检测：{top_emotion} {top_confidence:.1f}%  详细：{debug_all}")
#         except Exception:
#             pass

#         # Step 2：路由决策
#         # ✅ 改进：对所有情绪都支持，只在 sad/neutral 时启动融合
#         # 🔴 如果使用元学习器，直接用元学习器预测
        
#         if top_confidence >= 85.0:
#             return top_emotion, top_confidence

#         # 2. 提取潜在的 sad 和 neutral 概率
#         p_sad = float(all_emotions.get('sad', 0.0))
#         p_neutral = float(all_emotions.get('neutral', 0.0))

#         # 3. 动态触发条件：
#         # 条件 A: Top 1 本身就是 sad 或 neutral
#         # 条件 B: sad 或 neutral 虽然不是 Top 1，但概率 > 20%（说明模型在犹豫）
#         is_target_top = (top_emotion in self.FUSION_TARGET_EMOTIONS)
#         is_target_competitive = (p_sad > 20.0) or (p_neutral > 20.0)

#         # 如果和 sad/neutral 完全无关，直接返回
#         if not (is_target_top or is_target_competitive):
#             return top_emotion, top_confidence
#         if self.use_meta_learner:
#             meta_emotion, meta_conf = self._predict_with_meta_learner(frame_bgr, face_rect, all_emotions)
            
#             # 【关键防御机制】：如果元学习器给出的置信度也很低 (< 60%)，
#             # 且最初的全局情绪不在 target 里，说明元学习器被迫在 sad/neutral 里选了一个弱选项，此时听全局的
#             if meta_conf < 60.0 and not is_target_top:
#                 return top_emotion, top_confidence
                
#             return meta_emotion, meta_conf

# # ✅ 关键修正 2：Sad/Neutral 且低置信度，才进入元学习器或规则融合
#         if self.use_meta_learner:
#             return self._predict_with_meta_learner(frame_bgr, face_rect, all_emotions)

# # Step 4：加时赛（低置信度的 sad/neutral，引入眼睛专家）
#         print(f"[→] 情绪 '{top_emotion}' 处于模棱两可状态 ({top_confidence:.1f}%)，引入眼睛专家决胜...")


      


#         # =========================================================
#         # Step 4：加时赛（模棱两可的图片，引入眼睛专家）
#         # =========================================================
#         print(f"[→] 情绪 '{top_emotion}' 处于模棱两可状态 ({top_confidence:.1f}%)，引入眼睛专家决胜...")

#         if face_rect is None:
#             face_rect = self._detect_face_once(frame_bgr)

#         # 获取眼睛模型的悲伤概率
#         P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)
#         print(f"P_eye: {P_eye}")  # 应该在 0-1 之间
        
#         # 提取全局悲伤概率并归一化
#         P_global_sad = float(all_emotions.get('sad', 0.0))
#         if P_global_sad > 1.0: 
#             P_global_sad /= 100.0
#         P_global_sad = max(0.0, min(1.0, P_global_sad))

#         # 在低置信度区间，我们可以放心大胆地进行加权融合
#         # 此时给眼睛模型更高的权重 (k=0.45) 来打破僵局
#         k = 0.45 
#         var_final = (P_eye * k) + (P_global_sad * (1.0 - k))
        
#         # 只要有一方拉了一把，就判定为 sad
#         SAD_THRESHOLD = 0.45
#         final_emotion = "sad" if var_final > SAD_THRESHOLD else "neutral"

#         # 重新计算置信度并钳制，防止 API 报错
#         decision_strength = abs(var_final - 0.5) / 0.5
#         confidence = 50.0 + (decision_strength * 50.0)
#         confidence = float(max(0.0, min(100.0, confidence)))

#         return final_emotion, confidence
    

         
    # def analyze_emotion(self, frame_bgr: np.ndarray, k: Optional[float] = None,
    #                     face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
    #     """
    #     执行改进的决策级融合分析
    #     """
    #     self._lazy_init()

    #     k = k if k is not None else self.k
    #     k = max(0.0, min(1.0, k))

    #     # ==========================================
    #     # Step 1：获取全局情绪分布
    #     # ==========================================
    #     all_emotions = {}
    #     try:
    #         all_emotions = self._get_global_emotions(frame_bgr)
    #         all_emotions = self._smooth_emotions(all_emotions)

    #         if len(all_emotions) > 0:
    #             top_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
    #             top_confidence = float(all_emotions[top_emotion])
    #         else:
    #             top_emotion, top_confidence = 'neutral', 50.0

    #     except Exception as e:
    #         logger.warning(f"全局情绪分析失败：{e}")
    #         top_emotion, top_confidence = 'neutral', 50.0
    #         all_emotions = {}

    #     # 调试输出
    #     try:
    #         debug_all = {em: round(float(v), 1) for em, v in all_emotions.items()} if all_emotions else {}
    #         # print(f"[DEBUG] 全局检测：{top_emotion} {top_confidence:.1f}%  详细：{debug_all}")
    #     except Exception:
    #         pass

    #     # ==========================================
    #     # Step 2：绝对高置信度直通 (防止元学习器帮倒忙)
    #     # ==========================================
    #     if top_emotion not in self.FUSION_TARGET_EMOTIONS:
    #         return top_emotion, top_confidence
    #     if top_confidence >= 80.0:
    #         return top_emotion, top_confidence

    #     # ==========================================
    #     # Step 3：严苛的门控路由机制 (Gating)
    #     # ==========================================
    #     p_sad = float(all_emotions.get('sad', 0.0))
    #     p_neutral = float(all_emotions.get('neutral', 0.0))
        
    #     is_target_top = (top_emotion in self.FUSION_TARGET_EMOTIONS) # Top1 是 sad 或 neutral

    #     if not is_target_top:
    #         # 如果全局判断是 surprise/happy/angry...
    #         # 只有当全局非常不自信 (<55%)，且 sad/neutral 概率追得很紧 (>25%) 时，才允许求助专家
    #         if top_confidence > 55.0 or (p_sad < 25.0 and p_neutral < 25.0):
    #             return top_emotion, top_confidence

    #     # ==========================================
    #     # Step 4：元学习器决策 (Meta-Learner)
    #     # ==========================================
    #     if self.use_meta_learner:
    #         meta_emotion, meta_conf = self._predict_with_meta_learner(frame_bgr, face_rect, all_emotions)
            
    #         # 【弃权保护】：如果元学习器返回 abstain，听全局的
    #         if meta_emotion == 'abstain':
    #             return top_emotion, top_confidence
                
    #         # 【弱判决保护】：如果元学习器自己都不确定 (<55%)，不许推翻全局
    #         if meta_conf < 55.0:
    #             return top_emotion, top_confidence
                
    #         # 【跨界翻盘保护】：如果全局认为是 surprise，元学习器非要说是 neutral
    #         # 必须要求元学习器具有压倒性的自信 (>75%)，否则驳回
    #         if not is_target_top and meta_conf < 75.0:
    #             return top_emotion, top_confidence
                
    #         return meta_emotion, meta_conf

    #     # ==========================================
    #     # Step 5：规则融合加时赛 (Rule Fusion)
    #     # 如果未启用元学习器，走这条路
    #     # ==========================================
    #     if face_rect is None:
    #         face_rect = self._detect_face_once(frame_bgr)

    #     # 获取眼睛模型的悲伤概率
    #     P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)
        
    #     P_global_sad = float(all_emotions.get('sad', 0.0))
    #     if P_global_sad > 1.0: 
    #         P_global_sad /= 100.0
    #     P_global_sad = max(0.0, min(1.0, P_global_sad))

    #     # 动态调整权重
    #     k_dynamic = 0.45 
    #     var_final = (P_eye * k_dynamic) + (P_global_sad * (1.0 - k_dynamic))
        
    #     SAD_THRESHOLD = 0.45
    #     final_emotion = "sad" if var_final > SAD_THRESHOLD else "neutral"

    #     decision_strength = abs(var_final - 0.5) / 0.5
    #     confidence = 50.0 + (decision_strength * 50.0)
    #     confidence = float(max(0.0, min(100.0, confidence)))

    #     # 【跨界翻盘保护】(同上)
    #     if not is_target_top and confidence < 75.0:
    #         return top_emotion, top_confidence

    #     return final_emotion, confidence

    # def analyze_emotion(self, frame_bgr: np.ndarray, k: Optional[float] = None,
    #                     face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
    #     """
    #     执行改进的决策级融合分析
    #     """
    #     self._lazy_init()

    #     k = k if k is not None else self.k
    #     k = max(0.0, min(1.0, k))

    #     # ==========================================
    #     # Step 1：获取全局情绪分布
    #     # ==========================================
    #     all_emotions = {}
    #     try:
    #         all_emotions = self._get_global_emotions(frame_bgr)
    #         all_emotions = self._smooth_emotions(all_emotions)

    #         if len(all_emotions) > 0:
    #             top_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
    #             top_confidence = float(all_emotions[top_emotion])
    #         else:
    #             top_emotion, top_confidence = 'neutral', 50.0
    #     except Exception as e:
    #         logger.warning(f"全局情绪分析失败：{e}")
    #         top_emotion, top_confidence = 'neutral', 50.0

    #     # ==========================================
    #     # Step 2：铁血门控 (绝对隔离区)
    #     # ==========================================
    #     # 只要全局判定第一名不是 sad 也不是 neutral，直接相信全局！
    #     # 彻底保护 surprise, happy, angry 等情绪不被元学习器误杀
    #     if top_emotion not in self.FUSION_TARGET_EMOTIONS:
    #         return top_emotion, top_confidence

    #     # ==========================================
    #     # Step 3：高置信度直通
    #     # ==========================================
    #     # 如果是 sad/neutral，但 HSEmotion 已经极度自信，不需要专家插手
    #     if top_confidence >= 80.0:
    #         return top_emotion, top_confidence

    #     # ==========================================
    #     # Step 4：元学习器决策 (Meta-Learner)
    #     # (此时必定满足: top_emotion 是 sad/neutral 且 置信度 < 80)
    #     # ==========================================
    #     if self.use_meta_learner:
    #         meta_emotion, meta_conf = self._predict_with_meta_learner(frame_bgr, face_rect, all_emotions)
            
    #         # 【弱判决保护】：如果元学习器自己都不确信 (<55%)，或者发出了 abstain 弃权信号，听 HSEmotion 原判
    #         if meta_conf < 55.0 or meta_emotion == 'abstain':
    #             return top_emotion, top_confidence
                
    #         return meta_emotion, meta_conf

    #     # ==========================================
    #     # Step 5：规则融合加时赛 (Rule Fusion)
    #     # 如果未启用元学习器，走这条路
    #     # ==========================================
    #     if face_rect is None:
    #         face_rect = self._detect_face_once(frame_bgr)

    #     # 获取眼睛模型的悲伤概率
    #     P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)
        
    #     P_global_sad = float(all_emotions.get('sad', 0.0))
    #     if P_global_sad > 1.0: 
    #         P_global_sad /= 100.0
    #     P_global_sad = max(0.0, min(1.0, P_global_sad))

    #     # 动态调整权重
    #     k_dynamic = 0.45 
    #     var_final = (P_eye * k_dynamic) + (P_global_sad * (1.0 - k_dynamic))
        
    #     SAD_THRESHOLD = 0.45
    #     final_emotion = "sad" if var_final > SAD_THRESHOLD else "neutral"

    #     decision_strength = abs(var_final - 0.5) / 0.5
    #     confidence = 50.0 + (decision_strength * 50.0)
    #     confidence = float(max(0.0, min(100.0, confidence)))

    #     return final_emotion, confidence
    def analyze_emotion(self, frame_bgr: np.ndarray, k: Optional[float] = None,
                        face_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
        self._lazy_init()

        k = k if k is not None else self.k
        k = max(0.0, min(1.0, k))

        # ==========================================
        # Step 1：获取全局情绪分布
        # ==========================================
        try:
            all_emotions = self._get_global_emotions(frame_bgr)
            all_emotions = self._smooth_emotions(all_emotions)
            top_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
            top_confidence = float(all_emotions[top_emotion])
        except Exception:
            top_emotion, top_confidence, all_emotions = 'neutral', 50.0, {}

        # ==========================================
        # Step 2：高置信度直通与门控拦截
        # ==========================================
        if top_confidence >= 80.0:
            return top_emotion, top_confidence

        p_sad = float(all_emotions.get('sad', 0.0))
        p_neutral = float(all_emotions.get('neutral', 0.0))
        is_target_top = (top_emotion in self.FUSION_TARGET_EMOTIONS)

        if not is_target_top:
            if top_confidence > 55.0 or (p_sad < 25.0 and p_neutral < 25.0):
                return top_emotion, top_confidence

        # ==========================================
        # Step 3：获取眼睛概率 & 🚀 遮挡免疫机制 🚀
        # ==========================================
        if face_rect is None:
            face_rect = self._detect_face_once(frame_bgr)

        P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)
        
        # 🚨 戴了墨镜 / 闭眼 / 没找到眼睛 -> 局部专家瞎了，100% 听全局！
        if P_eye < 0:
            return top_emotion, top_confidence

        # ==========================================
        # Step 4：元学习器决策 (Meta-Learner)
        # ==========================================
        if self.use_meta_learner:
            # 直接提取特征，省去嵌套调用
            features = MetaLearnerPrediction.extract_features(all_emotions, P_eye)
            meta_emotion, meta_conf = MetaLearnerPrediction.predict(self.meta_learner, features)
            
            if meta_emotion == 'abstain': 
                return top_emotion, top_confidence
            if meta_conf < 55.0: 
                return top_emotion, top_confidence
            if not is_target_top and meta_conf < 75.0: 
                return top_emotion, top_confidence
                
            return meta_emotion, meta_conf

        # ==========================================
        # Step 5：规则融合加时赛 (Rule Fusion)
        # ==========================================
        P_global_sad = float(all_emotions.get('sad', 0.0))
        if P_global_sad > 1.0: P_global_sad /= 100.0
        
        k_dynamic = 0.45 
        var_final = (P_eye * k_dynamic) + (P_global_sad * (1.0 - k_dynamic))
        
        SAD_THRESHOLD = 0.45
        final_emotion = "sad" if var_final > SAD_THRESHOLD else "neutral"

        decision_strength = abs(var_final - 0.5) / 0.5
        confidence = 50.0 + (decision_strength * 50.0)
        confidence = float(max(0.0, min(100.0, confidence)))

        if not is_target_top and confidence < 75.0:
            return top_emotion, top_confidence

        return final_emotion, confidence
      

    def get_fusion_details(self, frame_bgr: np.ndarray, k: Optional[float] = None,
                           face_rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """获取融合分析的详细信息"""
        self._lazy_init()

        k = k if k is not None else self.k
        k = max(0.0, min(1.0, k))

        if face_rect is None:
            face_rect = self._detect_face_once(frame_bgr)

        all_emotions = self._get_global_emotions(frame_bgr)

        # Bug 4 修复：自动归一化
        P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)
        P_global_sad = float(all_emotions.get('sad', 0.0))
        if P_global_sad > 1.0:
            P_global_sad /= 100.0
        P_global_sad = max(0.0, min(1.0, P_global_sad))

        var_final = P_global_sad 

        # 2. 眼睛模型发挥作用的时刻：
        if P_eye > 0.60:
            # 【一票赞成】如果眼睛极其悲伤（比如强颜欢笑但眼神悲伤），直接提权！
            var_final = max(var_final, P_eye) 
        elif P_eye < 0.25:
            # 【轻微否决】如果眼睛完全没悲伤，稍微压低一下全局分数，防误报
            var_final = var_final * 0.85
        else:
            # 【模棱两可】普通的加权融合 (取 k=0.4)
            var_final = (P_eye * 0.4) + (P_global_sad * 0.6)

        # 3. 放宽悲伤门槛，保护召回率 (从 0.45 降到 0.40)
        SAD_THRESHOLD = 0.40
        final_emotion = "sad" if var_final > SAD_THRESHOLD else "neutral"

        # 置信度计算
        decision_strength = abs(var_final - 0.5) / 0.5
        confidence = 50.0 + (decision_strength * 50.0)
        confidence = float(max(0.0, min(100.0, confidence)))

        return {
            "final_emotion": final_emotion,
            "confidence": float(confidence),
            "details": {
                "P_eye": float(P_eye),
                "P_global_sad": float(P_global_sad),
                "fusion_result": float(var_final),
                "weight_k": float(k),
                "all_emotions": all_emotions,
                "hsemotion_available": self._hsemotion_available,
                "error_count": self._hsemotion_error_count
            },
            "model_info": {
                "eye_model_path": str(self.eye_model_path),
                "device": str(self.device),
                "fusion_target_emotions": list(self.FUSION_TARGET_EMOTIONS)
            }
        }


class DeepFaceBackup:
    """DeepFace 备用检测器（当 HSEmotion 失败时）"""

    def __init__(self):
        self.initialized = False

    def _ensure_initialized(self):
        """延迟初始化"""
        if self.initialized:
            return
        self.initialized = True

    def get_all_emotions(self, face_img: np.ndarray) -> Dict[str, float]:
        """使用 DeepFace 获取情绪"""
        self._ensure_initialized()
        
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            
            # 标准化为百分比
            total = sum(emotions.values())
            if total > 0:
                return {k: (v/total)*100 for k, v in emotions.items()}
            else:
                return {k: 100/len(emotions) for k in emotions.keys()}

        except Exception as e:
            logger.error(f"DeepFace 备用检测失败：{e}")
            return {}


def create_improved_decision_fusion_detector(config: Config, use_meta_learner: bool = False) -> ImprovedDecisionFusionDetector:
    """工厂函数：创建改进的决策融合检测器

    Args:
        config: 配置对象
        use_meta_learner: 是否使用元学习器（替代手动规则融合）

    Returns:
        ImprovedDecisionFusionDetector 实例
    """
    k = config.get('emotion.decision_fusion_k', 0.3)
    eye_model_path = config.get('emotion.eye_model_path', None)

    logger.info(f"创建改进决策融合检测器 (k={k}, use_meta_learner={use_meta_learner})")
    return ImprovedDecisionFusionDetector(config, eye_model_path=eye_model_path, k=k, use_meta_learner=use_meta_learner)
