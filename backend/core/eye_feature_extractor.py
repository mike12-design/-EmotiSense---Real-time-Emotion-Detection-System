"""
眼睛特征提取统一模块 (Unified Eye Feature Extractor)

核心目标：确保训练和推理时的眼睛特征提取方式**完全一致**，
避免分布偏移（Domain Shift）导致的准确率下降。

用法：
    from eye_feature_extractor import EyeFeatureExtractor
    
    extractor = EyeFeatureExtractor(device='mps')
    eye_img = extractor.extract_eye_region(frame, face_rect)
    eye_sad_prob = extractor.predict_eye_emotion(eye_img)
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EyeFeatureExtractor:
    """
    统一的眼睛区域提取和情绪预测器
    
    确保训练和推理时的特征提取方式完全一致
    """
    
    # 眼睛模型配置
    # 改成
    DEFAULT_EYE_MODEL_PATH = Path(__file__).parent.parent / "models" / "eye_model_finetuned.pth"   
    EYE_REGION_SIZE = (224, 224)
    EYE_LABELS = ['neutral', 'sad']
    
    def __init__(self, eye_model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化眼睛特征提取器
        
        Args:
            eye_model_path: PyTorch 眼睛模型路径
            device: 计算设备 ('cpu', 'cuda', 'mps')
        """
        self.eye_model = None
        self.device = device or self._detect_device()
        
        if eye_model_path:
            self.eye_model_path = Path(eye_model_path)
        else:
            self.eye_model_path = self.DEFAULT_EYE_MODEL_PATH
        
        # Haar Cascade 眼睛检测器
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        self._load_model()
    
    def _detect_device(self) -> str:
        """自动检测最优计算设备"""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self):
        """加载 PyTorch 眼睛模型"""
        try:
            if not self.eye_model_path.exists():
                raise FileNotFoundError(f"眼睛模型不存在：{self.eye_model_path}")
            
            import torch.nn as nn
            from torchvision import models
            
            # 构建 ResNet18
            self.eye_model = models.resnet18(weights=None)
            num_ftrs = self.eye_model.fc.in_features
            self.eye_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, 2)  # 2分类：neutral, sad
            )
            
            checkpoint = torch.load(self.eye_model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.eye_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.eye_model.load_state_dict(checkpoint)
            
            self.eye_model.to(self.device)
            self.eye_model.eval()
            
            logger.info(f"✅ 眼睛模型加载成功，设备：{self.device}")
        
        except Exception as e:
            logger.error(f"❌ 加载眼睛模型失败：{e}")
            raise
    
    # def extract_eye_region(self, frame_bgr: np.ndarray, 
    #                       face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    #     """
    #     从帧中提取眼睛区域 - 核心方法
        
    #     策略：
    #     1. 如果提供了人脸框，在人脸上半部分检测眼睛
    #     2. 如果没有人脸框，从整帧检测眼睛
    #     3. 返回最大的眼睛区域
        
    #     Args:
    #         frame_bgr: 输入帧 (BGR 格式)
    #         face_rect: 人脸框 (x, y, w, h)，可选
        
    #     Returns:
    #         224x224 的眼睛区域（RGB格式），或 None
    #     """
    #     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
    #     # 🎯 方法 1：有人脸框时，在人脸上半部分检测
    #     if face_rect is not None:
    #         x, y, w, h = face_rect
            
    #         # 【重要】只在人脸上半部分查找眼睛
    #         # 避免把嘴巴误认为眼睛
    #         upper_face_roi = gray[y:y+int(h*0.6), x:x+w]
            
    #         if upper_face_roi.size == 0:
    #             logger.warning("⚠️ 人脸 ROI 为空")
    #             return None
            
    #         eyes = self.eye_cascade.detectMultiScale(
    #             upper_face_roi,
    #             scaleFactor=1.1,
    #             minNeighbors=4,
    #             minSize=(20, 20),
    #             maxSize=(w//2, int(h*0.3))
    #         )
            
    #         if len(eyes) > 0:
    #             # 取最大的眼睛区域
    #             ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                
    #             # 转换回全局坐标
    #             eye_x = x + ex
    #             eye_y = y + ey
                
    #             # 提取眼睛ROI
    #             eye_roi = frame_bgr[eye_y:eye_y+eh, eye_x:eye_x+ew]
                
    #             if eye_roi.size == 0:
    #                 return None
                
    #             # 调整大小到 224x224
    #             eye_img = cv2.resize(eye_roi, self.EYE_REGION_SIZE)
                
    #             logger.debug(f"✅ 从人脸框内检测到眼睛 ({eye_roi.shape})")
    #             return eye_img
        
    #     # 🎯 方法 2：无人脸框时，从整帧检测
    #     else:
    #         eyes = self.eye_cascade.detectMultiScale(
    #             gray,
    #             scaleFactor=1.1,
    #             minNeighbors=4,
    #             minSize=(30, 30),
    #             maxSize=(150, 100)
    #         )
            
    #         if len(eyes) > 0:
    #             ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
    #             eye_roi = frame_bgr[ey:ey+eh, ex:ex+ew]
                
    #             if eye_roi.size == 0:
    #                 return None
                
    #             eye_img = cv2.resize(eye_roi, self.EYE_REGION_SIZE)
                
    #             logger.debug(f"✅ 从整帧检测到眼睛")
    #             return eye_img
        
    #     logger.warning("⚠️ 未检测到眼睛")
    #     return None
    



    def extract_eye_region(self, frame_bgr: np.ndarray, 
                          face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        从帧中提取眼睛区域 - 包含调试存图与硬裁剪兜底
        """
        import os
        import time
        from pathlib import Path
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 🎯 尝试方法 1：有人脸框时，在人脸上半部分检测
        if face_rect is not None:
            x, y, w, h = face_rect
            upper_face_roi = gray[y:y+int(h*0.6), x:x+w]
            
            if upper_face_roi.size > 0:
                eyes = self.eye_cascade.detectMultiScale(
                    upper_face_roi,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20),
                    maxSize=(w//2, int(h*0.3))
                )
                if len(eyes) > 0:
                    ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                    eye_x, eye_y = x + ex, y + ey
                    eye_roi = frame_bgr[eye_y:eye_y+eh, eye_x:eye_x+ew]
                    if eye_roi.size > 0:
                        return cv2.resize(eye_roi, self.EYE_REGION_SIZE)
        
        # 🎯 尝试方法 2：无人脸框时，从整帧检测
        else:
            eyes = self.eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                maxSize=(150, 100)
            )
            if len(eyes) > 0:
                ex, ey, ew, eh = max(eyes, key=lambda r: r[2]*r[3])
                eye_roi = frame_bgr[ey:ey+eh, ex:ex+ew]
                if eye_roi.size > 0:
                    return cv2.resize(eye_roi, self.EYE_REGION_SIZE)

        # ==========================================
        # 🚨 失败了！触发 Debug 存图机制
        # ==========================================
     
        logger.debug("⚠️ 眼睛被遮挡（墨镜/低头），局部专家失效，拒绝瞎猜")
        
        try:
            # 你之前的存图逻辑可以保留在这里，方便你以后继续看
            import os, time
            from pathlib import Path
            debug_dir = Path(__file__).parent.parent.parent / "debug_no_eyes"
            os.makedirs(debug_dir, exist_ok=True)
            debug_img = frame_bgr.copy()
            if face_rect is not None:
                dx, dy, dw, dh = face_rect
                cv2.rectangle(debug_img, (dx, dy), (dx+dw, dy+dh), (0, 0, 255), 2)
            save_path = debug_dir / f"fail_eye_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(save_path), debug_img)
        except Exception as e:
            pass

        # 核心修改：绝对不要硬裁剪！直接告诉上层“我看不见”
        return None
        
      



    def predict_eye_emotion(self, eye_img: np.ndarray) -> Tuple[str, float]:
        """
        预测眼睛情绪
        
        Args:
            eye_img: 224x224 的眼睛区域 (BGR 格式)
        
        Returns:
            (emotion_name, confidence_percentage)
        """
        if self.eye_model is None:
            logger.error("❌ 眼睛模型未加载")
            return 'neutral', 50.0
        
        try:
            # 1. 预处理
            eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_float = eye_rgb.astype(np.float32) / 255.0
            
            # 2. 标准化（ImageNet 标准）
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            eye_normalized = (eye_float - mean) / std
            
            # 3. 转为张量 (C, H, W)
            eye_tensor = torch.from_numpy(
                eye_normalized.transpose(2, 0, 1)
            ).float().unsqueeze(0).to(self.device)
            
            # 4. 推理
            with torch.no_grad():
                logits = self.eye_model(eye_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
                neutral_prob = float(probs[0].item())
                sad_prob = float(probs[1].item())
            
            # 5. 返回最高概率的情绪
            if sad_prob > neutral_prob:
                return 'sad', sad_prob * 100
            else:
                return 'neutral', neutral_prob * 100
        
        except Exception as e:
            logger.error(f"❌ 眼睛情绪预测失败：{e}")
            return 'neutral', 50.0
    
    def get_eye_sad_probability(self, eye_img: np.ndarray) -> float:
        """
        获取眼睛的悲伤概率 (0.0 ~ 1.0)
        
        这个方法用于特征提取，用于元学习器的输入
        """
        emotion, confidence = self.predict_eye_emotion(eye_img)
        
        if emotion == 'sad':
            return confidence / 100.0
        else:
            return (100.0 - confidence) / 100.0


def create_eye_feature_extractor(device: Optional[str] = None) -> EyeFeatureExtractor:
    """工厂函数：创建眼睛特征提取器"""
    return EyeFeatureExtractor(device=device)
