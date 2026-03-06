"""
元学习器推理模块
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict
import torch

logger = logging.getLogger(__name__)


class MetaLearnerPrediction:
    """元学习器推理模块"""
    
    # 特征标签（与训练时保持一致）
    FEATURE_NAMES = [
        'P_global_sad',      # 全局模型的 sad 概率
        'P_global_neutral',  # 全局模型的 neutral 概率
        'P_eye_sad',         # 眼睛模型的 sad 概率
        'P_global_angry',    # 全局模型的 angry 概率
        'P_global_happy',    # 全局模型的 happy 概率
        'P_global_surprise'  # 全局模型的 surprise 概率
    ]
    
    @staticmethod
    def extract_features(all_emotions: Dict[str, float], 
                        P_eye: float) -> np.ndarray:
        """
        构建特征向量
        
        Args:
            all_emotions: 全局模型的所有情绪概率 (字典)
            P_eye: 眼睛模型的 sad 概率 (0.0 ~ 1.0)
        
        Returns:
            6 维特征向量 (numpy array)
        """
        try:
            # 1. 提取概率值，并归一化到 0-1 范围
            P_global_sad = float(all_emotions.get('sad', 0.0))
            if P_global_sad > 1.0:
                P_global_sad /= 100.0  # 如果是百分比，转为小数
            P_global_sad = max(0.0, min(1.0, P_global_sad))
            
            P_global_neutral = float(all_emotions.get('neutral', 0.0))
            if P_global_neutral > 1.0:
                P_global_neutral /= 100.0
            P_global_neutral = max(0.0, min(1.0, P_global_neutral))
            
            P_global_angry = float(all_emotions.get('angry', 0.0))
            if P_global_angry > 1.0:
                P_global_angry /= 100.0
            P_global_angry = max(0.0, min(1.0, P_global_angry))
            
            P_global_happy = float(all_emotions.get('happy', 0.0))
            if P_global_happy > 1.0:
                P_global_happy /= 100.0
            P_global_happy = max(0.0, min(1.0, P_global_happy))
            
            P_global_surprise = float(all_emotions.get('surprise', 0.0))
            if P_global_surprise > 1.0:
                P_global_surprise /= 100.0
            P_global_surprise = max(0.0, min(1.0, P_global_surprise))
            
            # 2. 眼睛概率已经是 0.0 ~ 1.0
            P_eye = max(0.0, min(1.0, float(P_eye)))
            
            # 3. 构建特征向量
            feature_vector = np.array([
                P_global_sad,
                P_global_neutral,
                P_eye,
                P_global_angry,
                P_global_happy,
                P_global_surprise
            ], dtype=np.float32).reshape(1, -1)  # 重要：reshape 为 (1, 6) 以适配 sklearn
            
            logger.debug(f"✅ 特征提取成功：{feature_vector}")
            return feature_vector
        
        except Exception as e:
            logger.error(f"❌ 特征提取失败：{e}")
            # 返回中性的特征向量（避免推理崩溃）
            return np.array([[0.0, 0.5, 0.5, 0.0, 0.2, 0.0]], dtype=np.float32)
    
    @staticmethod
    def predict(meta_learner, features: np.ndarray) -> Tuple[str, float]:
        """
        使用元学习器预测
        
        Args:
            meta_learner: 训练好的 sklearn 分类器
            features: (1, 6) 的特征向量
        
        Returns:
            (emotion_name, confidence_percentage)
        """
        try:
            # 1. 预测类别
            prediction = meta_learner.predict(features)[0]  # 0=neutral, 1=sad
            
            # 2. 获取预测概率（如果可用）
            if hasattr(meta_learner, 'predict_proba'):
                probabilities = meta_learner.predict_proba(features)[0]
                
                if prediction == 1:
                    emotion = 'sad'
                    confidence = probabilities[1] * 100
                else:
                    emotion = 'neutral'
                    confidence = probabilities[0] * 100
            else:
                # 如果分类器不支持 predict_proba（比如某些 SVM）
                emotion = 'sad' if prediction == 1 else 'neutral'
                
                # 通过决策函数估算置信度
                if hasattr(meta_learner, 'decision_function'):
                    score = meta_learner.decision_function(features)[0]
                    confidence = 50.0 + (np.tanh(score) * 50.0)  # 将分数映射到 0-100
                else:
                    confidence = 75.0  # 保守估计
            
            # 3. 钳制置信度
            confidence = float(max(0.0, min(100.0, confidence)))
            
            logger.debug(f"✅ 元学习器预测：{emotion} (置信度 {confidence:.1f}%)")
            return emotion, confidence
        
        except Exception as e:
            logger.error(f"❌ 元学习器预测失败：{e}")
            return 'neutral', 50.0


# ============================================================
# 以下代码应该替换 decision_fusion_detector.py 中的
# _predict_with_meta_learner 方法（第 216-260 行）
# ============================================================

def predict_with_meta_learner_complete(
    detector_instance,  # ImprovedDecisionFusionDetector 的 self
    frame_bgr: np.ndarray,
    face_rect: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[str, float]:
    """
    使用元学习器进行完整预测
    
    这是 ImprovedDecisionFusionDetector._predict_with_meta_learner 的完整实现
    
    Args:
        detector_instance: ImprovedDecisionFusionDetector 实例
        frame_bgr: 输入帧 (BGR 格式)
        face_rect: 人脸框 (x, y, w, h)，可选
    
    Returns:
        (emotion_name, confidence_percentage)
    """
    if detector_instance.meta_learner is None:
        logger.error("❌ 元学习器未加载，回退到规则融合")
        return detector_instance._rule_based_fusion(frame_bgr, face_rect)
    
    try:
        # 1️⃣ 获取全局情绪分布
        all_emotions = detector_instance._get_global_emotions(frame_bgr)
        
        if not all_emotions:
            logger.warning("⚠️ 全局情绪获取失败，使用默认值")
            return 'neutral', 50.0
        
        # 2️⃣ 检测人脸（如果需要）
        if face_rect is None:
            face_rect = detector_instance._detect_face_once(frame_bgr)
        
        # 3️⃣ 获取眼睛模型概率
        P_eye = detector_instance._get_eye_sad_prob(frame_bgr, face_rect) if face_rect else 0.5
        
        # 4️⃣ 构建特征向量
        features = MetaLearnerPrediction.extract_features(all_emotions, P_eye)
        
        # 5️⃣ 用元学习器预测
        emotion, confidence = MetaLearnerPrediction.predict(
            detector_instance.meta_learner,
            features
        )
        
        # 6️⃣ 调试输出
        logger.debug(f"[元学习器] 预测：{emotion} ({confidence:.1f}%) | "
                    f"P_global_sad={all_emotions.get('sad', 0):.1f}% "
                    f"P_eye={P_eye:.3f}")
        
        return emotion, confidence
    
    except Exception as e:
        logger.error(f"❌ 元学习器推理异常：{e}，回退到规则融合")
        return detector_instance._rule_based_fusion(frame_bgr, face_rect)


# ============================================================
# 快速测试脚本
# ============================================================

def test_feature_extraction():
    """测试特征提取"""
    test_emotions = {
        'angry': 5.0,
        'contempt': 2.0,
        'disgust': 3.0,
        'fear': 5.0,
        'happy': 10.0,
        'neutral': 55.0,
        'sad': 20.0,
        'surprise': 0.0
    }
    test_eye_prob = 0.45
    
    features = MetaLearnerPrediction.extract_features(test_emotions, test_eye_prob)
    
    print(f"✅ 特征向量形状：{features.shape}")
    print(f"✅ 特征向量值：{features}")
    print(f"✅ 特征名称：{MetaLearnerPrediction.FEATURE_NAMES}")
    
    assert features.shape == (1, 6), "特征向量形状错误！"
    assert np.all((features >= 0) & (features <= 1)), "特征值超出范围！"
    
    print("✅ 特征提取测试通过！")


if __name__ == '__main__':
    test_feature_extraction()
