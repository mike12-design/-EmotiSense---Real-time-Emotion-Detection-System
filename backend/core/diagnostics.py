"""
EmotiSense 情绪检测诊断脚本

用途：
1. 诊断 HSEmotion 模型加载问题
2. 对比多个情绪检测器的输出
3. 测试决策融合模型的效果
4. 验证笑容识别是否正常
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("EmotiSenseDiagnostics")


class EmotiDiagnostics:
    """情绪检测诊断工具"""
    
    def __init__(self, config=None):
        self.config = config
        self.test_results = []
    
    # ============================================================================
    # 诊断 1：HSEmotion 库状态检查
    # ============================================================================
    
    def diagnose_hsemotion_installation(self) -> Dict[str, bool]:
        """检查 HSEmotion 库是否正确安装"""
        logger.info("\n" + "="*70)
        logger.info("诊断 1：HSEmotion 库安装状态")
        logger.info("="*70)
        
        results = {
            "hsemotion_installed": False,
            "pytorch_installed": False,
            "torch_version_compatible": False,
            "hsemotion_can_import": False,
            "hsemotion_model_loadable": False
        }
        
        # 1. 检查 hsemotion 库
        try:
            import hsemotion
            results["hsemotion_installed"] = True
            logger.info("✅ hsemotion 库已安装")
            print(f"   版本：{hsemotion.__version__ if hasattr(hsemotion, '__version__') else '未知'}")
        except ImportError as e:
            logger.error(f"❌ hsemotion 库未安装：{e}")
            logger.info("修复：pip install --upgrade hsemotion")
            return results
        
        # 2. 检查 PyTorch
        try:
            import torch
            results["pytorch_installed"] = True
            logger.info(f"✅ PyTorch 已安装，版本：{torch.__version__}")
            print(f"   CUDA 可用：{torch.cuda.is_available()}")
            print(f"   设备：{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        except ImportError:
            logger.error("❌ PyTorch 未安装")
            logger.info("修复：pip install torch torchvision")
            return results
        
        # 3. 检查版本兼容性
        try:
            import torch
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version >= (1, 9):
                results["torch_version_compatible"] = True
                logger.info(f"✅ PyTorch 版本 {torch.__version__} 兼容")
            else:
                logger.warning(f"⚠️ PyTorch 版本 {torch.__version__} 可能太旧，建议升级到 1.9+")
        except Exception as e:
            logger.warning(f"⚠️ 无法检查 PyTorch 版本兼容性：{e}")
        
        # 4. 检查 HSEmotion 导入
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            results["hsemotion_can_import"] = True
            logger.info("✅ HSEmotionRecognizer 可以成功导入")
        except ImportError as e:
            logger.error(f"❌ HSEmotionRecognizer 导入失败：{e}")
            logger.info("可能的原因：")
            logger.info("  1. hsemotion 版本太旧")
            logger.info("  2. 依赖库不完整")
            logger.info("修复：pip install --upgrade hsemotion --force-reinstall")
            return results
        
        # 5. 尝试加载模型
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            logger.info("尝试加载 HSEmotion 模型...")
            
            # 尝试加载 8 类模型
            try:
                model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
                logger.info("✅ 8 类模型 (enet_b0_8_best_afew) 加载成功")
                results["hsemotion_model_loadable"] = True
            except Exception as e1:
                logger.warning(f"8 类模型加载失败：{e1}")
                
                # 尝试加载 7 类模型
                try:
                    model = HSEmotionRecognizer(model_name='enet_b0_7_best_afew')
                    logger.info("✅ 7 类模型 (enet_b0_7_best_afew) 加载成功")
                    results["hsemotion_model_loadable"] = True
                except Exception as e2:
                    logger.error(f"❌ 两个模型都加载失败")
                    logger.error(f"   8 类：{e1}")
                    logger.error(f"   7 类：{e2}")
                    logger.info("修复：")
                    logger.info("  1. pip cache purge")
                    logger.info("  2. pip install --upgrade hsemotion --no-cache-dir")
                    return results
        
        except Exception as e:
            logger.error(f"❌ 模型加载异常：{e}")
        
        return results
    
    # ============================================================================
    # 诊断 2：多个检测器的对比测试
    # ============================================================================
    
    def test_emotion_detectors_with_sample_image(self, img_path: str = None) -> Dict[str, Dict]:
        """
        使用示例图像测试多个情绪检测器
        
        Args:
            img_path: 测试图像路径（如果为 None，则生成合成测试图像）
        """
        logger.info("\n" + "="*70)
        logger.info("诊断 2：多检测器对比测试")
        logger.info("="*70)
        
        # 加载或生成测试图像
        if img_path is None:
            logger.info("生成合成测试图像...")
            test_img = self._generate_synthetic_face_image()
        else:
            test_img = cv2.imread(img_path)
            if test_img is None:
                logger.error(f"❌ 无法加载图像：{img_path}")
                return {}
        
        results = {}
        
        # 1. 测试 HSEmotion
        logger.info("\n[1/4] 测试 HSEmotion...")
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
            emotion, scores = model.predict_emotions(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), logits=False)
            
            logger.info(f"✅ HSEmotion 成功")
            logger.info(f"   检测到的主情绪：{emotion}")
            logger.info(f"   置信度分布：{scores}")
            results["hsemotion"] = {
                "status": "success",
                "emotion": emotion,
                "scores": scores
            }
        except Exception as e:
            logger.error(f"❌ HSEmotion 失败：{e}")
            results["hsemotion"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 2. 测试 DeepFace
        logger.info("\n[2/4] 测试 DeepFace...")
        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                test_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            
            logger.info(f"✅ DeepFace 成功")
            logger.info(f"   检测到的主情绪：{top_emotion[0]}")
            logger.info(f"   置信度分布：{emotions}")
            results["deepface"] = {
                "status": "success",
                "emotion": top_emotion[0],
                "emotions": emotions
            }
        except Exception as e:
            logger.error(f"❌ DeepFace 失败：{e}")
            results["deepface"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 3. 测试 FER
        logger.info("\n[3/4] 测试 FER...")
        try:
            from fer import FER
            detector = FER(mtcnn=False)
            result = detector.detect_emotions(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
            
            if result:
                emotions = result[0]['emotions']
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                logger.info(f"✅ FER 成功")
                logger.info(f"   检测到的主情绪：{top_emotion[0]}")
                logger.info(f"   置信度分布：{emotions}")
                results["fer"] = {
                    "status": "success",
                    "emotion": top_emotion[0],
                    "emotions": emotions
                }
            else:
                logger.warning("⚠️ FER 未检测到人脸")
                results["fer"] = {
                    "status": "no_face_detected"
                }
        except Exception as e:
            logger.error(f"❌ FER 失败：{e}")
            results["fer"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # 4. 测试决策融合（如果配置可用）
        logger.info("\n[4/4] 测试决策融合...")
        try:
            if self.config:
                from core.decision_fusion_detector import create_improved_decision_fusion_detector
                detector = create_improved_decision_fusion_detector(self.config)
                emotion, confidence = detector.analyze_emotion(test_img)
                details = detector.get_fusion_details(test_img)
                
                logger.info(f"✅ 决策融合成功")
                logger.info(f"   最终情绪：{emotion} ({confidence:.1f}%)")
                logger.info(f"   详细：{details}")
                results["decision_fusion"] = {
                    "status": "success",
                    "emotion": emotion,
                    "confidence": confidence,
                    "details": details
                }
            else:
                logger.warning("⚠️ 未提供配置，跳过决策融合测试")
        except Exception as e:
            logger.error(f"❌ 决策融合失败：{e}")
            results["decision_fusion"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results
    
    def _generate_synthetic_face_image(self, emotion_type: str = "neutral") -> np.ndarray:
        """生成合成人脸图像用于测试"""
        # 创建一个 224x224 的合成图像
        img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # 肤色背景
        
        # 绘制简单的人脸特征
        cv2.circle(img, (112, 112), 60, (180, 150, 130), -1)  # 脸部圆形
        cv2.circle(img, (95, 95), 8, (50, 50, 50), -1)  # 左眼
        cv2.circle(img, (130, 95), 8, (50, 50, 50), -1)  # 右眼
        
        # 根据情绪类型绘制嘴
        if emotion_type == "happy":
            # 笑脸：上扬的弧线
            cv2.ellipse(img, (112, 120), (20, 15), 0, 0, 180, (50, 50, 50), 2)
        elif emotion_type == "sad":
            # 悲伤脸：下沉的弧线
            cv2.ellipse(img, (112, 130), (20, 15), 0, 180, 360, (50, 50, 50), 2)
        else:
            # 中性脸：直线
            cv2.line(img, (95, 130), (130, 130), (50, 50, 50), 2)
        
        return img
    
    # ============================================================================
    # 诊断 3：验证决策融合对笑容的识别
    # ============================================================================
    
    def test_happy_emotion_recognition(self) -> Dict:
        """
        专门测试笑容（happy emotion）的识别
        这是关键的测试用例
        """
        logger.info("\n" + "="*70)
        logger.info("诊断 3：笑容识别测试 (Happy Emotion)")
        logger.info("="*70)
        
        # 使用合成的笑脸图像
        happy_img = self._generate_synthetic_face_image(emotion_type="happy")
        
        results = {}
        
        # 保存图像用于手动验证
        output_path = "/tmp/test_happy_face.jpg"
        cv2.imwrite(output_path, happy_img)
        logger.info(f"✅ 测试图像已保存到：{output_path}")
        
        # 测试各个检测器对笑脸的识别
        results["hsemotion"] = self._test_detector_on_emotion(happy_img, "HSEmotion", "hsemotion")
        results["deepface"] = self._test_detector_on_emotion(happy_img, "DeepFace", "deepface")
        results["fer"] = self._test_detector_on_emotion(happy_img, "FER", "fer")
        results["decision_fusion"] = self._test_detector_on_emotion(happy_img, "Decision Fusion", "decision_fusion")
        
        # 总结
        logger.info("\n📊 笑容识别总结：")
        happy_detected_count = sum(
            1 for r in results.values() 
            if isinstance(r, dict) and r.get("emotion") in ["happy", "happiness"]
        )
        logger.info(f"在 {len(results)} 个检测器中，{happy_detected_count} 个识别到了笑容")
        
        return results
    
    def _test_detector_on_emotion(self, img: np.ndarray, detector_name: str, detector_type: str) -> Dict:
        """测试单个检测器"""
        try:
            if detector_type == "hsemotion":
                from hsemotion.facial_emotions import HSEmotionRecognizer
                model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
                emotion, scores = model.predict_emotions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), logits=False)
                logger.info(f"  {detector_name}: {emotion}")
                return {"emotion": emotion, "status": "success"}
            
            elif detector_type == "deepface":
                from deepface import DeepFace
                result = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                emotions = result[0]['emotion']
                top = max(emotions.items(), key=lambda x: x[1])
                logger.info(f"  {detector_name}: {top[0]}")
                return {"emotion": top[0], "status": "success"}
            
            elif detector_type == "fer":
                from fer import FER
                detector = FER(mtcnn=False)
                result = detector.detect_emotions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if result:
                    emotions = result[0]['emotions']
                    top = max(emotions.items(), key=lambda x: x[1])
                    logger.info(f"  {detector_name}: {top[0]}")
                    return {"emotion": top[0], "status": "success"}
                else:
                    logger.warning(f"  {detector_name}: 未检测到人脸")
                    return {"status": "no_face"}
            
            else:
                return {"status": "unknown"}
        
        except Exception as e:
            logger.error(f"  {detector_name} 失败：{e}")
            return {"status": "failed", "error": str(e)}
    
    # ============================================================================
    # 运行完整诊断
    # ============================================================================
    
    def run_full_diagnostics(self, test_img_path: str = None):
        """运行完整的诊断套件"""
        logger.info("\n" + "="*80)
        logger.info("EmotiSense 完整诊断开始")
        logger.info("="*80)
        
        # 诊断 1
        install_results = self.diagnose_hsemotion_installation()
        
        # 诊断 2
        detector_results = self.test_emotion_detectors_with_sample_image(test_img_path)
        
        # 诊断 3
        happy_results = self.test_happy_emotion_recognition()
        
        # 生成诊断报告
        self._generate_diagnostic_report(install_results, detector_results, happy_results)
    
    def _generate_diagnostic_report(self, install_results, detector_results, happy_results):
        """生成诊断报告"""
        logger.info("\n" + "="*80)
        logger.info("📋 诊断报告")
        logger.info("="*80)
        
        logger.info("\n1️⃣ 库安装状态：")
        for key, value in install_results.items():
            status = "✅" if value else "❌"
            logger.info(f"   {status} {key}")
        
        logger.info("\n2️⃣ 检测器功能：")
        for detector, result in detector_results.items():
            status = "✅" if result.get("status") == "success" else "❌"
            logger.info(f"   {status} {detector}")
        
        logger.info("\n3️⃣ 笑容识别：")
        for detector, result in happy_results.items():
            emotion = result.get("emotion", "未知")
            is_happy = emotion in ["happy", "happiness"]
            status = "✅" if is_happy else "❌"
            logger.info(f"   {status} {detector}: {emotion}")
        
        logger.info("\n" + "="*80)
        logger.info("诊断完成")
        logger.info("="*80)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建诊断工具
    diagnostics = EmotiDiagnostics()
    
    # 运行完整诊断
    diagnostics.run_full_diagnostics()
    
    # 或者单独运行各部分诊断
    # diagnostics.diagnose_hsemotion_installation()
    # diagnostics.test_emotion_detectors_with_sample_image()
    # diagnostics.test_happy_emotion_recognition()
