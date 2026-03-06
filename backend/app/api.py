# ============================================================================
# EmotiSense API - 后端核心路由文件
# ============================================================================
# 功能：处理所有 HTTP 请求，包括视频流、情绪检测、用户管理、数据分析等
# 端点数量：~50 个 API 路由
# 主要依赖：FastAPI, OpenCV, DeepFace, SQLAlchemy, PyTorch
#
# 核心流程：
#   视频帧 → 人脸检测 → 情绪分析 → 稳定器平滑 → 动力学计算 → 数据库写入 → 干预决策
# ============================================================================

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse

import logging
import random
import time
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import inspect
import httpx
import cv2
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case

from deepface import DeepFace

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import os
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from core.models import (
    MusicLibrary,
    SessionLocal,
    User,
    EmotionLog,
    Diary,
    ComfortScript,
    SystemEvent
)

from core.config import get_config
from core.detector import FaceDetector, create_emotion_detector, find_identity
from core.stabilizer import EmotionStabilizer
from core.audio_manager import AudioManager
from core.emotion_dynamics import EmotionDynamicsEngine
from core.advanced_analyzer import AdvancedEmotionAnalyzer, DiarySentimentAnalyzer


# ============================================================================
# 第一部分：全局配置与状态 (Global Config & State)
# ============================================================================

router = APIRouter()
logger = logging.getLogger("EmotiSenseAPI")

# 系统组件全局变量 (在 init_components 中初始化)
face_detector = None        # 人脸检测器 (YOLOv8 + Haar Cascade)
emotion_detector = None     # 情绪检测器 (DecisionFusion / DeepFace)
data_manager = None         # 数据管理器
audio_manager = None        # 音频管理器 (音乐播放 + TTS)
stabilizer = None           # 情绪稳定器 (窗口投票平滑)
video_cap = None            # 视频捕获对象
dynamics_engine = EmotionDynamicsEngine()  # 情绪动力学引擎 (价态追踪)

# 人脸特征缓存相关
face_db_cache = []                  # 缓存格式：List[Dict] -> [{"username": str, "embedding": np.array}]
face_db_last_update = 0             # 上次刷新时间戳
FACE_DB_REFRESH_INTERVAL = 10       # 缓存刷新间隔 (秒) - 避免每帧都查数据库

# 系统运行状态 (用于音频干预逻辑)
state = {
    "last_emotion": "neutral",      # 当前情绪状态
    "mood_score": 0.5,              # 心情指数 (0-1, 0=极负，1=极正)
    "stress": 0.0                   # 压力水平
}

# 高级情绪分析器实例
advanced_analyzer = AdvancedEmotionAnalyzer()   # 吸引子模型、RMSSD 分析
diary_analyzer = DiarySentimentAnalyzer()       # 日记情感分析

# ===== 情绪日志节流控制 (Emotion Logging Gate) =====
# 功能：防止数据库写入过频，只有情绪持续 EMOTION_CHANGE_MIN_DURATION 秒以上才写入
# 节流间隔：WRITE_INTERVAL = 5 秒

last_logged_emotion = None      # 上次写入数据库的情绪
last_db_log_time = 0            # 上次写入时间戳

WRITE_INTERVAL = 5              # 秒 - 写入节流间隔
EMOTION_CHANGE_MIN_DURATION = 3 # 秒 - 情绪持续时间阈值 (超过此值才考虑写入)

emotion_since_time = time.time()  # 当前情绪开始的时间


# ============================================================================
# 第二部分：辅助函数 - 一言 API (Hitokoto Quote API)
# ============================================================================
# 功能：从 hitokoto.cn 获取治愈系句子，用于情绪干预
# 超时设置：2 秒，防止第三方 API 拖慢系统
# ============================================================================

async def get_hitokoto_by_emotion(emotion: str):
    """根据情绪获取特定分类的一言句子

    Args:
        emotion: 情绪类型 (happy/sad/angry/neutral/fear/surprise)

    Returns:
        格式化的一言句子："正文 —— 作者「来源」"，失败返回 None
    """
    # 情绪到一言分类的映射字典
    mapping = {
        "happy":   "d&c=l",  # 文学+抖机灵
        "sad":     "i&c=j",  # 诗词+网易云
        "angry":   "k",      # 哲学
        "neutral": "a&c=b",  # 动画+漫画
        "fear":    "h",      # 影视
        "surprise":"e"       # 原创
    }
    cat = mapping.get(emotion.lower(), "d") # 默认文学
    
    url = f"https://v1.hitokoto.cn/?c={cat}&max_length=35"
    
    try:
        async with httpx.AsyncClient() as client:
            # 设置 2 秒超时，防止第三方 API 拖慢系统
            response = await client.get(url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                # 拼接格式：正文 —— 作者 「来源」
                author = data['from_who'] if data['from_who'] else "佚名"
                return f"{data['hitokoto']} —— {author} 「{data['from']}」"
    except Exception as e:
        print(f"Hitokoto API 失败: {e}")
    return None


async def get_hitokoto_quote():
    """从一言官方 API 获取一条随机治愈系句子

    Returns:
        格式化的句子："正文 ——《出处》"，失败返回 None
    """
    # 参数说明：c=d(文学) c=i(诗词) c=k(哲学)，这些比较适合治愈系背景
    url = "https://v1.hitokoto.cn/?c=d&c=i&c=k&min_length=10&max_length=30"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                # 返回格式：句子 + 出处
                return f"{data['hitokoto']} ——《{data['from']}》"
    except Exception as e:
        logger.warning(f"Hitokoto API 请求失败: {e}")
    return None


def write_stable_log(db, emotion, score, user_id=None, is_stranger=True):
    global last_logged_emotion, last_db_log_time, emotion_since_time

    now_ts = time.time()
    # 情绪变化检测：如果情绪变了，重置持续时间计时器
    if emotion != last_logged_emotion:
        emotion_since_time = now_ts
        last_logged_emotion = emotion
        return

    # 只有当一种情绪持续了 EMOTION_CHANGE_MIN_DURATION 秒以上，才考虑写入
    if now_ts - emotion_since_time < EMOTION_CHANGE_MIN_DURATION:
        return

    # 写入频率控制 (节流)
    if now_ts - last_db_log_time < WRITE_INTERVAL:
        return

    try:
        log = EmotionLog(
            timestamp=datetime.now(),
            user_id=user_id,
            is_stranger=is_stranger,
            emotion=emotion,
            score=score  # 这里的 score 已经是 dynamics_engine 计算出的 0-1 心情指数
        )
        db.add(log)
        db.commit()
        last_db_log_time = now_ts
        logger.info(f"💾 数据库已记录稳定情绪: {emotion} (指数: {score:.2f})")
    except Exception as e:
        db.rollback()
        logger.error(f"写入日志失败: {e}")


# ==========================================
# 1. 视频流处理核心类 (Video Processing Core)
# ==========================================

class VideoCapture:
    """
    摄像头控制类：负责打开摄像头、读取原始帧。
    """

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

    def read_raw(self) -> Optional[np.ndarray]:
        """返回原始 BGR numpy frame（未镜像翻转）。"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def get_frame(self) -> Optional[bytes]:
        """返回编码好的 jpeg bytes（用于简单的直接流回退场景）。"""
        raw = self.read_raw()
        if raw is None:
            return None
        # 翻转图像以产生镜面效果
        frame = cv2.flip(raw, 1)
        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes() if ret else None

    def release(self):
        """释放摄像头资源"""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

class FrameProcessor:
    """
    帧处理器：包含图像转灰度、人脸检测、情绪分析、身份识别、数据库写入等静态逻辑。
    """

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def process_frame(
        frame: np.ndarray,
        face_detector,
        emotion_detector,
     
       
        session_factory,  
        face_db_cache: List[Dict],
        update_face_db_fn=None,
        draw_labels=True
    ) -> Tuple[Optional[bytes], List[Dict]]:
        
        logger = logging.getLogger("FrameProcessor")
        face_details: List[Dict] = []

        try:
            color_frame = frame  # BGR 彩色帧，给 YOLOv8 / detect_faces 使用
            gray = FrameProcessor.to_grayscale(frame)  # 仅用于 Haar Cascade 等需要灰度的检测器
            faces_data = []
            
            # --- 1. 人脸检测 (兼容新旧检测器) ---
            if face_detector:
                try:
                    # 优先调用新的高级 ROI 检测方法
                    if hasattr(face_detector, 'detect_faces_with_roi'):
                        faces_data = face_detector.detect_faces_with_roi(color_frame)
                    else:
                        # 兼容老版本检测器 (防止 KeyError)
                        raw_faces = face_detector.detect_faces(color_frame)
                       
                        faces_data = [{
                            'rect': f, 
                            'has_eyes': False, 
                            'eye_coords': []
                        } for f in raw_faces]
                except Exception as e:
                    logger.warning("face_detector failed: %s", e)
         

            # 刷新人脸库缓存
            if update_face_db_fn:
                try:
                    face_db_cache = update_face_db_fn()
                except Exception:
                    pass

            db = session_factory()

            # --- 2. 遍历每个人脸进行处理 ---
            for face_info in faces_data:
                # 解包数据
                x, y, w, h = face_info["rect"]
                has_eyes = face_info.get("has_eyes", False)
                global_eyes = face_info.get("eye_coords", [])
                
                # 边界保护
                x, y = max(0, int(x)), max(0, int(y))
                w, h = int(w), int(h)

                # 截取人脸图像
                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                # --- 3. 情绪分析 ---
                # 自适应调用：通过 inspect 检查方法签名，动态决定传什么参数
                try:
                    sig = inspect.signature(emotion_detector.analyze_emotion)

                    if 'face_rect' in sig.parameters:
                        # 决策融合等高级检测器：需要全帧 + ROI 坐标
                        base_emotion, base_confidence = emotion_detector.analyze_emotion(frame, face_rect=(x, y, w, h))
                    else:
                        # DeepFace/HSEmotion 等传统检测器：只需要裁剪后的人脸图像
                        base_emotion, base_confidence = emotion_detector.analyze_emotion(face_img)

                except Exception as e:
                    logger.error(f"Emotion detection failed: {e}")
                    base_emotion, base_confidence = "neutral", 0.0

                # --- 4. 🧠 吸收论文思想：基于 ROI 的动态置信度调整 ---
                # 定义高度依赖眼睛区域的情绪
                eye_dependent_emotions = ["surprise", "fear", "sad", "sadness"]
                # 定义高度依赖嘴巴区域的情绪
                mouth_dependent_emotions = ["happy", "angry", "disgust"]
                
                final_confidence = base_confidence

                if base_emotion in eye_dependent_emotions:
                    if has_eyes:
                        # 眼睛清晰可见，提升置信度 (模拟注意力机制)
                        final_confidence = min(100.0, base_confidence * 1.15) 
                    else:
                        # 未检测到眼睛但识别出惊讶/恐惧，可能是误判，降低权重
                        final_confidence = base_confidence * 0.8 

                # --- 5. 身份识别 ---
                name = "Stranger"
                if face_db_cache:
                    try:
                        name = find_identity(face_img, face_db_cache)
                    except Exception as e:
                        pass # 忽略识别错误

                # --- 6. 数据库关联 ---
                user_id = None
                is_stranger = True
                if name != "Stranger":
                    user = db.query(User).filter(User.username == name).first()
                    if user:
                        user_id = user.id
                        is_stranger = False

                face_details.append({
                    "bbox": (x, y, w, h),
                    "emotion": base_emotion,
                    "confidence": float(final_confidence), # 使用调整后的置信度
                    "name": name,
                    "user_id": user_id,
                    "is_stranger": is_stranger
                })

                # --- 7. 👁️ 可解释人工智能 (XAI) 视觉展示 ---
                if draw_labels:
                    # 基础框颜色：陌生人灰色，熟人绿色
                    color = (0, 255, 0) if not is_stranger else (200, 200, 200)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 准备热力图覆盖层
                    overlay = frame.copy()
                    xai_text = ""
                    highlight_drawn = False
                    
                    # 情况 A：惊讶/恐惧/悲伤 -> 高亮眼睛区域 (蓝色)
                    if base_emotion in eye_dependent_emotions and has_eyes:
                        for (ex, ey, ew, eh) in global_eyes:
                            cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), -1) # BGR: 蓝色偏亮
                        xai_text = "XAI: Eye Region Activation"
                        highlight_drawn = True
                        
                    # 情况 B：开心/愤怒 -> 高亮嘴部区域 (红色)
                    elif base_emotion in mouth_dependent_emotions:
                        # 估算嘴巴位置：人脸下 1/3 处
                        mouth_y = y + int(h * 0.68)
                        mouth_h = int(h * 0.25)
                        # 边界检查防止画出屏幕
                        if mouth_y + mouth_h < frame.shape[0]:
                            cv2.rectangle(overlay, (x, mouth_y), (x + w, mouth_y + mouth_h), (0, 0, 255), -1) # BGR: 红色
                            xai_text = "XAI: Mouth Region Activation"
                            highlight_drawn = True
                    
                    # 如果绘制了高亮区域，混合图层产生半透明热力图效果
                    if highlight_drawn:
                        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                        # 在框下方绘制 XAI 解释文本 (黄色字体)
                        cv2.putText(frame, xai_text, (x, y + h + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # 顶部标签：姓名 | 情绪 | 置信度
                    label_text = f"{name} | {base_emotion} {final_confidence:.0f}%"
                    cv2.putText(frame, label_text, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            try:
                db.close()
            except Exception:
                pass

        except Exception as e:
            logger.exception("Frame processing error: %s", e)

        # 8. 编码为 JPEG 返回给前端
        try:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                return jpeg.tobytes(), face_details
            else:
                return None, face_details
        except Exception:
            return None, face_details


# ============================================================================
# 第四部分：辅助函数 (Helper Functions)
# ============================================================================
# get_db       - 数据库会话依赖注入
# load_face_db - 人脸特征缓存加载 (10 秒刷新，避免频繁查询数据库)
# gen_from_video - 视频流生成器 (含情绪检测、稳定器、动力学引擎)
# gen          - 备用视频流生成器
# init_components - 系统初始化 (AI 模型、摄像头、话术库)
# ============================================================================

def get_db():
    """FastAPI 依赖注入：获取数据库会话 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_face_db():
    """
    加载并缓存所有用户的面部特征向量。
    带有时间间隔限制 (FACE_DB_REFRESH_INTERVAL=10 秒)，避免每帧都查询数据库。

    Returns:
        List[Dict]: [{"username": str, "embedding": np.array}]
    """
    global face_db_cache, face_db_last_update
    now = time.time()
    if now - face_db_last_update < FACE_DB_REFRESH_INTERVAL:
        return face_db_cache

    db = SessionLocal()
    users = db.query(User).filter(User.face_encoding != None).all()
    db.close()

    face_db_cache = [
        {
            "username": u.username,
            "embedding": np.array(u.face_encoding)
        }
        for u in users
    ]
    face_db_last_update = now
    return face_db_cache


def gen_from_video():
    """
    视频流生成器 (主流程) - 用于 /video_feed 端点

    核心流程:
    1. 读取摄像头原始帧
    2. FrameProcessor.process_frame() 处理 (人脸检测 + 情绪分析 + 身份识别 + XAI 可视化)
    3. 情绪稳定器平滑处理
    4. 情绪动力学引擎计算 (mood_score, stress_level)
    5. 写入稳定情绪日志 (节流控制)
    6. 编码为 JPEG 返回给前端

    Yields:
        MJPEG 格式的视频帧 (multipart/x-mixed-replace)
    """
    global video_cap, face_detector, emotion_detector, stabilizer, dynamics_engine, face_db_cache

    while True:
        if video_cap is None:
            time.sleep(0.1)
            continue

        raw = video_cap.read_raw()
        if raw is None:
            continue

        try:
            # 1. AI 识别处理
            jpeg_bytes, face_details = FrameProcessor.process_frame(
                raw, face_detector, emotion_detector, 
                SessionLocal, face_db_cache,
                update_face_db_fn=load_face_db, draw_labels=True
            )

            # 2. 情绪动力学处理 (Dynamics)
            if face_details and stabilizer:
                primary_face = face_details[0]
                current_emo = primary_face["emotion"]

                # 👉 送入稳定器
                stabilizer.add_prediction(current_emo)

                # 👉 获取稳定结果
                stable_emo = stabilizer.get_stable_emotion()

                # dynamics 更新
                dyn = dynamics_engine.update(
                    emotion=stable_emo,
                    confidence=primary_face["confidence"] / 100.0
                )

                # 更新状态
                state["last_emotion"] = dyn.primary_emotion
                state["mood_score"] = dyn.mood_score
                state["stress"] = dyn.stress_level

                # 写日志（节流）
                db = SessionLocal()
                try:
                    # 归一化：[-1, 1] → [0, 1]
                    score_normalized = (dyn.mood_score + 1.0) / 2.0
                    write_stable_log(
                        db,
                        dyn.primary_emotion,
                        score_normalized,
                        primary_face["user_id"],
                        primary_face["is_stranger"]
                    )
                finally:
                    db.close()

            # 4. 画面输出
            if jpeg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n\r\n')

        except Exception as e:
            logger.error(f"视频流生成错误: {e}")
            time.sleep(0.01)


def gen(camera):
    """简单的视频流生成器 (备用)"""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)


# ============================================================================
# 第五部分：系统初始化 (System Initialization)
# ============================================================================
# init_components: 应用启动时调用，初始化 AI 模型、摄像头、话术库
# 初始化组件：FaceDetector, emotion_detector, AudioManager, EmotionStabilizer, VideoCapture
# 话术库初始化：预设 4 条安慰话术 (sad/angry/happy)
# ============================================================================

async def init_components():
    """
    应用启动时调用，初始化 AI 模型和硬件资源。

    初始化内容:
    1. 注册退出处理 (atexit) - 关闭摄像头
    2. 初始化话术库 (如果为空)
    3. 加载配置 (config.yaml)
    4. 初始化 AI 核心组件：
       - AudioManager: 音乐播放 + TTS
       - EmotionStabilizer: 窗口投票情绪平滑 (window_size=15)
       - FaceDetector: YOLOv8 人脸检测
       - emotion_detector: 情绪检测器 (DecisionFusion/DeepFace)
       - VideoCapture: 摄像头控制
    """
    import atexit

    atexit.register(lambda: video_cap.release() if video_cap else None)

    global face_detector, emotion_detector, data_manager, audio_manager, stabilizer, video_cap
    db = SessionLocal()
    try:
        if db.query(ComfortScript).count() == 0:
            print("正在初始化话术库...")
            scripts = [
                # 【这里写入你想让它说的话】
                ComfortScript(emotion_tag="sad", content="别难过，我会陪着你。"),
                ComfortScript(emotion_tag="sad", content="不管发生什么，我都在这里。"),
                ComfortScript(emotion_tag="angry", content="深呼吸，不要生气啦。"),
                ComfortScript(emotion_tag="happy", content="你笑起来真好看！"),
            ]
            db.add_all(scripts)
            db.commit()
    except Exception as e:
        print(f"话术初始化错误: {e}")
    finally:
        db.close()

    logger.info("正在加载系统组件...")
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = get_config(str(config_path))

    # 初始化 AI 核心
    audio_manager = AudioManager()
    stabilizer = EmotionStabilizer(window_size=15)

    # 初始化摄像头
    try:
        # 注意：此处有重复初始化 detector 的代码，保留原样以符合“不修改代码”的要求
        face_detector = FaceDetector(config)
        emotion_detector = create_emotion_detector(config)
        video_cap = VideoCapture()
        print("✅ 后端所有组件初始化成功，摄像头已就绪")

    except Exception as e:
        logger.warning(f"摄像头启动失败: {e}")

    logger.info("✅ 所有组件初始化完成")


# ============================================================================
# 第六部分：API 路由定义 (API Route Definitions)
# ============================================================================
#
# 端点分类:
# | 前缀                      | 功能                    | 认证  |
# |--------------------------|------------------------|-------|
# | /video_feed              | MJPEG 实时视频流         | 无    |
# | /api/status              | 系统状态轮询            | 无    |
# | /api/my/*                | 用户端 (历史/日记/资源)  | 需要  |
# | /api/admin/*             | 管理后台 (用户/日志/分析) | 需要  |
# | /api/debug/*             | 调试与测试              | 无    |
#
# ============================================================================

# ---------------------------------------------------------------------------
# 视频流 API
# ---------------------------------------------------------------------------

@router.get("/video_feed")
async def video_feed():
    """
    前端通过 <img> 标签调用此接口获取实时视频流

    返回：
        MJPEG 流 (multipart/x-mixed-replace)
        包含：人脸框、情绪标签、身份识别、XAI 热力图可视化
    """
    global video_cap
    if video_cap is None or video_cap.cap is None or not video_cap.cap.isOpened():
        return JSONResponse(status_code=503, content={"message": "Camera not available"})
    return StreamingResponse(gen_from_video(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# 状态 API - 前端获取当前情绪
# ---------------------------------------------------------------------------

@router.get("/api/my/history/stats")
async def get_my_history_stats(username: str, range_type: str = 'day', db: Session = Depends(get_db)):
    # 1. 查找用户
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"labels": [], "scores": []}

    now = datetime.now()
    
    # 2. 基础查询构建
    logs_query = db.query(EmotionLog).filter(EmotionLog.user_id == user.id)

    # 3. 根据 range_type 定义 logs 和 label_func
    # -------------------------------------------------------
    if range_type == 'day':
        # 查询今日数据
        logs = logs_query.filter(func.date(EmotionLog.timestamp) == func.date(now)).all()
        # 横坐标显示小时 (09:00, 10:00)
        label_func = lambda t: t.strftime('%H:00')

    elif range_type == 'week':
        # 查询本周数据
        start_of_week = now - timedelta(days=now.weekday())
        logs = logs_query.filter(EmotionLog.timestamp >= start_of_week).all()
        # 横坐标显示日期 (2024-02-19)
        label_func = lambda t: t.strftime('%Y-%m-%d')

    else: # month
        # 查询本月数据 (SQLite/MySQL 语法略有不同，这里兼容通用逻辑)
        # 如果是 SQLite: func.strftime('%Y-%m', ...)
        # 这里使用 Python 过滤兜底，确保兼容性
        all_logs = logs_query.all()
        current_month_str = now.strftime('%Y-%m')
        logs = [l for l in all_logs if l.timestamp.strftime('%Y-%m') == current_month_str]
        
        # 横坐标显示日期
        label_func = lambda t: t.strftime('%Y-%m-%d')
    # -------------------------------------------------------

    # 4. 定义情绪目标分数 (100=极乐, 50=平静, 0=愤怒)
    TARGET_SCORES = {
        "happy": 100,
        "surprise": 85,
        "neutral": 50,   # 基准线
        "fear": 40,
        "sad": 20,
        "angry": 0,
        "disgust": 0
    }
    
    BASE_SCORE = 50.0 # 平静基准分
    bucket = defaultdict(list)

    # 5. 遍历日志，计算插值分数
    for log in logs:
        emotion_key = log.emotion.lower()
        
        # 获取置信度 (假设数据库存的是 0.0 - 1.0 的小数)
        # 如果数据库存的是整数(0-100)，请除以 100.0
        confidence = log.score 
        
        # 获取该情绪的目标极值
        target = TARGET_SCORES.get(emotion_key, 50)
        
        # ⭐ 核心算法：线性插值 (Lerp)
        # 公式：当前分 = 平静分 + (目标分 - 平静分) * 置信度
        # 例子1 (Happy, 0.8): 50 + (100 - 50) * 0.8 = 90 分
        # 例子2 (Sad, 0.5):   50 + (20 - 50) * 0.5 = 35 分
        real_score = BASE_SCORE + (target - BASE_SCORE) * confidence
        
        # 限制范围在 0-100 防止溢出
        real_score = max(0, min(100, real_score))
        
        # 分组聚合
        label = label_func(log.timestamp)
        bucket[label].append(real_score)

    # 6. 计算平均值并返回
    labels = sorted(bucket.keys())
    scores = []
    for l in labels:
        avg_score = sum(bucket[l]) / len(bucket[l])
        scores.append(int(avg_score)) # 转为整数给前端

    return {"labels": labels, "scores": scores}


@router.get("/api/status")
async def get_status(db: Session = Depends(get_db)):
    global state

    emotion = state.get("last_emotion", "neutral")
    mood = state.get("mood_score", 0.5)
    stress = state.get("stress", 0.2)

    should_intervene = stress > 0.6

    text_content = ""

    if should_intervene:
        script_obj = db.query(ComfortScript) \
            .filter(ComfortScript.emotion_tag == emotion) \
            .order_by(func.random()) \
            .first()

        if script_obj:
            text_content = script_obj.content
        else:
            default_map = {
                "sad": "别难过，我在这里陪你。",
                "angry": "慢慢呼吸，放松一下。",
                "fear": "你是安全的。",
                "happy": "继续保持好心情！"
            }
            text_content = default_map.get(emotion, "")

    return {
        "current_emotion": emotion,
        "mood_score": mood,
        "stress_level": stress,
        "should_intervene": should_intervene,
        "resource": {
            "text": text_content,
            "audio_url": f"assets/music/{emotion}.mp3"
        } if should_intervene else None
    }


# ---------------------------------------------------------------------------
# 图片分析 API (非流式)
# ---------------------------------------------------------------------------
# POST /analyze - 单帧图片情绪分析
# 流程：人脸检测 → 情绪分析 → 稳定器平滑 → 动力学计算 → 数据库写入
# ---------------------------------------------------------------------------

@router.post("/analyze")
async def analyze_frame(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    单帧图片情绪分析接口 (非流式)

    处理流程:
    1. 解码上传的图片为 BGR numpy 数组
    2. 人脸检测 (YOLOv8)
    3. 情绪分析 (DecisionFusion/DeepFace)
    4. 身份识别 (人脸特征匹配)
    5. 稳定器平滑处理
    6. 动力学引擎计算 (mood_score, stress_level)
    7. 写入数据库 (节流控制)

    Args:
        file: 上传的图片文件

    Returns:
        {
            "face_count": int,
            "dominant_emotion": str,
            "mood": float (0-1),
            "faces": List[face_details]
        }
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(frame)

    current_frame_emotions = []
    face_details = []
    face_db = load_face_db()

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        if face_img.size == 0:
            continue

        # 自适应调用：通过 inspect 检查方法签名，动态决定传什么参数
        try:
            sig = inspect.signature(emotion_detector.analyze_emotion)
            if 'face_rect' in sig.parameters:
                # 决策融合等高级检测器：需要全帧 + ROI 坐标
                emotion, confidence = emotion_detector.analyze_emotion(frame, face_rect=(x, y, w, h))
            else:
                # DeepFace/HSEmotion 等传统检测器：只需要裁剪后的人脸图像
                emotion, confidence = emotion_detector.analyze_emotion(face_img)
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            emotion, confidence = "neutral", 0.0

        current_frame_emotions.append(emotion)

        name = find_identity(face_img, face_db)

        face_details.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": float(confidence),
            "name": name
        })

    # ===== 稳定器 =====
    dominant_now = "neutral"
    if current_frame_emotions:
        dominant_now = Counter(current_frame_emotions).most_common(1)[0][0]

    if stabilizer:
        stabilizer.add_prediction(dominant_now)
        stable_emotion = stabilizer.get_stable_emotion()
    else:
        stable_emotion = dominant_now

    # ===== dynamics =====
    confidence = face_details[0]["confidence"] if face_details else 0.3
    dyn = dynamics_engine.update(stable_emotion, confidence)

    score = max(0.0, min(1.0, dyn.mood_score))

    user_id = None
    is_stranger = True

    if face_details:
        name = face_details[0]["name"]
        if name != "Stranger":
            user = db.query(User).filter(User.username == name).first()
            if user:
                user_id = user.id
                is_stranger = False

    write_stable_log(
        db,
        stable_emotion,
        dyn.mood_score,
        user_id,
        is_stranger
    )

    state["last_emotion"] = dyn.primary_emotion
    state["mood_score"] = dyn.mood_score
    state["stress"] = dyn.stress_level

    return {
        "face_count": len(face_details),
        "dominant_emotion": stable_emotion,
        "mood": score,
        "faces": face_details
    }


# --- 管理后台接口 (Admin) ---




# backend/app/api.py

@router.get("/api/admin/logs")
async def get_admin_logs(
    page: int = 1, 
    page_size: int = 10, 
    username: str = None, # 新增过滤参数
    db: Session = Depends(get_db)
):
    """获取全系统的情绪日志（仅限已注册用户，支持按人筛选）"""
    offset = (page - 1) * page_size
    
    # 使用 join 而不是 outerjoin，会自动过滤掉 user_id 为空或无法匹配 User 表的记录
    query = db.query(
        EmotionLog.id,
        EmotionLog.timestamp,
        EmotionLog.emotion,
        EmotionLog.score,
        User.username
    ).join(User, EmotionLog.user_id == User.id) 

    # 如果传了用户名，则进行过滤
    if username:
        query = query.filter(User.username == username)

    total = query.count()
    
    results = query.order_by(desc(EmotionLog.timestamp))\
                   .offset(offset).limit(page_size).all()

    logs = []
    for item in results:
        logs.append({
            "id": item.id,
            "timestamp": item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": item.emotion,
            "score": item.score,
            "username": item.username # 这里肯定有值，因为是内连接
        })

    return {"total": total, "data": logs}


@router.get("/api/admin/analytics/stats")
async def get_analytics_stats(time_range: str = "7d", user_id: int = None, db: Session = Depends(get_db)):
    """获取仪表盘统计数据 (增强版)

    Args:
        time_range: 时间范围 - 24h, 7d, 30d
        user_id: 用户 ID，可选，如果提供则只统计该用户的数据
    """
    # 根据时间范围计算时间边界
    now = datetime.now()
    if time_range == "24h":
        time_delta = timedelta(hours=24)
        hour_format = "%H:00"
    elif time_range == "30d":
        time_delta = timedelta(days=30)
        hour_format = "%m-%d"
    else:  # 7d
        time_delta = timedelta(days=7)
        hour_format = "%m-%d"

    start_time = now - time_delta

    # 构建基础查询过滤器
    base_filter = EmotionLog.timestamp >= start_time
    if user_id:
        base_filter = base_filter & (EmotionLog.user_id == user_id)

    # ============ 1. 概览统计 ============
    total_logs = db.query(EmotionLog).filter(base_filter).count()

    # 积极情绪率 (happy + surprise)
    positive_count = db.query(EmotionLog).filter(
        EmotionLog.emotion.in_(['happy', 'surprise']),
        base_filter
    ).count()
    positive_rate = (positive_count / total_logs * 100) if total_logs > 0 else 0

    if user_id:
        # 按用户统计时，显示该用户的活跃天数和今天记录数
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_count = db.query(EmotionLog).filter(
            EmotionLog.user_id == user_id,
            EmotionLog.timestamp >= today_start
        ).count()
        # 计算活跃天数（有记录的不同日期数量）
        active_days = db.query(
            func.date(EmotionLog.timestamp).label('log_date')
        ).filter(
            EmotionLog.user_id == user_id,
            EmotionLog.timestamp >= start_time
        ).distinct().count()
        overview = {
            "total_users": 1,
            "total_logs": total_logs,
            "avg_positive_rate": round(positive_rate, 1),
            "active_users_today": today_count,
            "active_days": active_days
        }
    else:
        total_users = db.query(User).count()
        active_users = db.query(EmotionLog.user_id).filter(
            EmotionLog.user_id != None,
            EmotionLog.timestamp >= start_time
        ).distinct().count()
        overview = {
            "total_users": total_users,
            "total_logs": total_logs,
            "avg_positive_rate": round(positive_rate, 1),
            "active_users_today": active_users,
            "active_days": 0
        }

    # ============ 2. 饼图：情绪分布 ============
    pie_query = db.query(EmotionLog.emotion, func.count(EmotionLog.id)) \
        .filter(base_filter) \
        .group_by(EmotionLog.emotion).all()
    pie_data = [{"name": item[0], "value": item[1]} for item in pie_query]

    # ============ 3. 折线图：多情绪趋势 ============
    # 按时间分组统计各情绪数量
    emotions_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    trend_labels = []
    trend_series = {emo: [] for emo in emotions_list}

    # 生成时间序列
    if time_range == "24h":
        time_points = 12  # 每 2 小时
        step = timedelta(hours=2)
    elif time_range == "30d":
        time_points = 15  # 每 2 天
        step = timedelta(days=2)
    else:
        time_points = 7  # 每天
        step = timedelta(days=1)

    for i in range(time_points):
        point_time = start_time + (step * i)
        next_time = point_time + step
        trend_labels.append(point_time.strftime(hour_format))

        for emo in emotions_list:
            count = db.query(EmotionLog).filter(
                EmotionLog.emotion == emo,
                EmotionLog.timestamp >= point_time,
                EmotionLog.timestamp < next_time,
                base_filter
            ).count()
            trend_series[emo].append(count)

    trend_data = {
        "labels": trend_labels,
        "series": trend_series
    }

    # ============ 4. 柱状图：情绪对比 ============
    bar_query = db.query(EmotionLog.emotion, func.count(EmotionLog.id)) \
        .filter(base_filter) \
        .group_by(EmotionLog.emotion).all()
    emotion_breakdown = {
        "emotions": [item[0] for item in bar_query],
        "counts": [item[1] for item in bar_query]
    }

    # ============ 5. 热力图：小时 × 情绪 ============
    # 统计每个小时段各情绪的出现次数
    heatmap_data = []
    for hour in range(0, 24, 2):
        for emo_idx, emo in enumerate(emotions_list):
            count = db.query(EmotionLog).filter(
                EmotionLog.emotion == emo,
                base_filter,
                func.strftime('%H', EmotionLog.timestamp) == f"{hour:02d}"
            ).count()
            if count > 0:
                heatmap_data.append([emo, f"{hour:02d}:00", count])

    # ============ 6. 雷达图：用户情绪特征 ============
    if user_id:
        # 按用户查询时，只显示该用户的情绪特征
        user_emotions = db.query(EmotionLog.emotion, func.count(EmotionLog.id)) \
            .filter(EmotionLog.user_id == user_id, base_filter) \
            .group_by(EmotionLog.emotion).all()
        emo_dict = {e[0]: e[1] for e in user_emotions}
        data = [emo_dict.get(e, 0) for e in emotions_list[:5]]
        radar_series = [{"name": "该用户", "data": data}]
        radar_data = {
            "indicators": [{"name": emo, "max": max([data[i]] + [10])} for i, emo in enumerate(emotions_list[:5])],
            "series": radar_series
        }
    else:
        # 全局查询时，显示前 2 个活跃用户的情绪分布
        top_users = db.query(
            EmotionLog.user_id, User.username,
            func.count(EmotionLog.id).label('log_count')
        ).join(User, EmotionLog.user_id == User.id) \
         .filter(EmotionLog.user_id != None, base_filter) \
         .group_by(EmotionLog.user_id, User.username) \
         .order_by(desc('log_count')) \
         .limit(2).all()

        radar_series = []
        for user in top_users:
            user_emotions = db.query(EmotionLog.emotion, func.count(EmotionLog.id)) \
                .filter(EmotionLog.user_id == user.user_id, base_filter) \
                .group_by(EmotionLog.emotion).all()
            emo_dict = {e[0]: e[1] for e in user_emotions}
            data = [emo_dict.get(e, 0) for e in emotions_list[:5]]
            radar_series.append({"name": user.username, "data": data})

        radar_data = {
            "indicators": [{"name": emo, "max": max([s["data"][i] for s in radar_series] + [10])} for i, emo in enumerate(emotions_list[:5])],
            "series": radar_series
        }

    # ============ 7. Top 用户分析 ============
    if user_id:
        # 按用户查询时，不显示 Top 用户对比（返回空数据）
        top_users_data = {"users": []}
    else:
        top_users_query = db.query(
            EmotionLog.user_id, User.username,
            func.count(EmotionLog.id).label('total'),
            func.sum(case((EmotionLog.emotion == 'happy', 1), else_=0)).label('happy_count')
        ).join(User, EmotionLog.user_id == User.id) \
         .filter(EmotionLog.user_id != None, base_filter) \
         .group_by(EmotionLog.user_id, User.username) \
         .order_by(desc('total')) \
         .limit(5).all()

        top_users_data = {
            "users": [
                {
                    "username": u.username,
                    "log_count": u.total,
                    "happy_rate": round((u.happy_count / u.total * 100) if u.total > 0 else 0, 1)
                }
                for u in top_users_query
            ]
        }

    return {
        "overview": overview,
        "pie_data": pie_data,
        "trend_data": trend_data,
        "emotion_breakdown": emotion_breakdown,
        "heatmap_data": {"data": heatmap_data},
        "user_comparison": radar_data,
        "top_users": top_users_data
    }


@router.get("/api/admin/analytics")
async def get_analytics(db: Session = Depends(get_db)):
    """简单的分析统计接口 (备用)"""
    counts = db.query(EmotionLog.emotion, func.count(EmotionLog.id)) \
        .group_by(EmotionLog.emotion).all()
    return {
        "pie_data": [{"name": c[0], "value": c[1]} for c in counts],
        "trend_data": {
            "labels": ["08:00", "12:00", "16:00", "20:00"],
            "values": [10, 25, 15, 30]
        }
    }


# ==========================================
# 高级情绪分析 API (Advanced Analytics)
# ==========================================

@router.get("/api/admin/analytics/advanced/{user_id}")
async def get_advanced_analytics(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    高级情绪分析 - 吸引子模型、RMSSD 波动、干预决策

    返回：
    - attractor: 情绪吸引子 (基线均值)
    - attractor_std: 标准差
    - rmssd: 情绪波动指数
    - current_valence: 当前情绪值
    - deviation: 偏离度 (σ 单位)
    - smoothed_valence: 卡尔曼滤波后的序列
    - intervention_needed: 是否需要干预
    - intervention_type: 干预类型 (none/music/tts/tts_urgency)
    - risk_level: 风险等级 (low/medium/high)
    """
    # 获取历史数据
    cutoff = datetime.now() - timedelta(days=days)
    logs = db.query(EmotionLog).filter(
        EmotionLog.user_id == user_id,
        EmotionLog.timestamp >= cutoff
    ).order_by(EmotionLog.timestamp).all()

    if not logs or len(logs) == 0:
        return {
            "error": "无足够数据",
            "attractor": 0.0,
            "rmssd": 0.0,
            "intervention_needed": False
        }

    # 转换为分析器需要的格式
    logs_data = [
        {
            "timestamp": log.timestamp.timestamp(),
            "emotion": log.emotion,
            "score": log.score
        }
        for log in logs
    ]

    # 执行分析
    result = advanced_analyzer.analyze(logs_data, days=days)

    # 计算情绪趋势方向
    _, valence_series = advanced_analyzer.convert_to_valence_series(logs_data)
    trend_direction = advanced_analyzer.get_trend_direction(valence_series)

    # 计算情绪惯性
    inertia = advanced_analyzer.calculate_emotion_inertia(valence_series)

    # 生成干预建议
    suggestions = []
    if result.intervention_needed:
        if result.intervention_type == "tts_urgency":
            suggestions = [
                {"type": "tts", "priority": "high", "description": "检测到您情绪持续低落，建议进行深呼吸放松"},
                {"type": "music", "priority": "high", "description": "播放舒缓的自然音乐"},
                {"type": "hitokoto", "priority": "medium", "description": "诗词/治愈类句子"}
            ]
        elif result.intervention_type == "music":
            suggestions = [
                {"type": "music", "priority": "medium", "description": "播放积极向上的音乐"},
                {"type": "hitokoto", "priority": "low", "description": "文学/抖机灵类句子"}
            ]
        elif result.current_valence < -0.3:
            suggestions = [
                {"type": "music", "priority": "medium", "description": "播放安慰性音乐"},
                {"type": "script", "priority": "low", "description": "播放预定义的安慰话术"}
            ]
    else:
        suggestions = [
            {"type": "info", "priority": "low", "description": "当前情绪状态良好，继续保持"}
        ]

    return {
        "user_id": user_id,
        "analysis_period_days": days,
        "data_points": len(logs),
        "attractor": result.attractor,
        "attractor_std": result.attractor_std,
        "rmssd": result.rmssd,
        "current_valence": result.current_valence,
        "deviation": result.deviation,
        "smoothed_valence": result.smoothed_valence,
        "intervention": {
            "needed": result.intervention_needed,
            "type": result.intervention_type,
            "risk_level": result.risk_level
        },
        "trend": {
            "direction": trend_direction,
            "inertia": round(inertia, 3)
        },
        "suggestions": suggestions,
        "valence_history": [
            {"timestamp": log.timestamp.isoformat(), "value": float(log.score / 100 * advanced_analyzer.valence_map(log.emotion))}
            for log in logs[-20:]  # 最近 20 条用于前端展示
        ]
    }


@router.post("/api/admin/analytics/diary/validate/{user_id}")
async def validate_diary_emotion(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    日记情感闭环校验

    对比视觉识别情绪与日记主观情感的一致性，
    用于验证视觉模型的准确性并提供自校正建议。
    """
    # 获取用户
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 获取日记数据
    cutoff = datetime.now() - timedelta(days=days)
    diaries = db.query(Diary).filter(
        Diary.user_id == user_id,
        Diary.timestamp >= cutoff
    ).all()

    # 获取同期视觉识别数据
    vision_logs = db.query(EmotionLog).filter(
        EmotionLog.user_id == user_id,
        EmotionLog.timestamp >= cutoff
    ).order_by(EmotionLog.timestamp).all()

    if not diaries or not vision_logs:
        return {
            "consistency": "insufficient_data",
            "message": "日记或视觉数据不足",
            "diary_count": len(diaries) if diaries else 0,
            "vision_count": len(vision_logs) if vision_logs else 0
        }

    # 转换为分析器格式
    diary_entries = [
        {"timestamp": d.timestamp.timestamp(), "content": d.content}
        for d in diaries
    ]
    vision_entries = [
        {"timestamp": v.timestamp.timestamp(), "emotion": v.emotion, "score": v.score}
        for v in vision_logs
    ]

    # 执行校验
    result = diary_analyzer.validate_visual_emotion(diary_entries, vision_entries)

    # 优化 3：检测极端不对称性（如视觉极度悲伤 V < -0.5，但日记强颜欢笑 V > 0.2）
    visual_score = result.get("visual_avg", 0)
    diary_score = result.get("diary_avg", 0)

    is_lying = (visual_score < -0.5) and (diary_score > 0.2)
    severe_conflict = abs(visual_score - diary_score) > 0.7

    # 向前端下发指令，触发第三方量表仲裁（如弹窗 PHQ-9 简易问卷）
    trigger_questionnaire = is_lying or severe_conflict

    return {
        "user_id": user_id,
        "analysis_period_days": days,
        "diary_entries": len(diaries),
        "vision_logs": len(vision_logs),
        "trigger_questionnaire": trigger_questionnaire,  # 新增字段：指示前端弹窗
        **result
    }


@router.get("/api/admin/analytics/intervention/suggest/{user_id}")
async def suggest_intervention(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    获取干预建议

    基于当前情绪状态，推荐合适的干预措施：
    - 音乐类型
    - 安慰话术
    - 一言句子类别
    """
    # 获取最近的分析结果
    cutoff = datetime.now() - timedelta(days=1)
    logs = db.query(EmotionLog).filter(
        EmotionLog.user_id == user_id,
        EmotionLog.timestamp >= cutoff
    ).order_by(EmotionLog.timestamp).all()

    if not logs:
        return {"suggestion": "暂无足够数据生成建议"}

    # 转换为分析器格式
    logs_data = [
        {"timestamp": log.timestamp.timestamp(), "emotion": log.emotion, "score": log.score}
        for log in logs
    ]

    result = advanced_analyzer.analyze(logs_data, days=1)

    # 根据情绪状态生成建议
    suggestions = []

    if result.intervention_type == "tts_urgency":
        suggestions = [
            {"type": "tts", "priority": "high", "content": "检测到您情绪持续低落，建议进行深呼吸放松"},
            {"type": "music", "emotion_tag": "calming", "description": "播放舒缓的自然音乐"},
            {"type": "hitokoto", "category": "i&c=j", "description": "诗词/治愈类句子"}
        ]
    elif result.intervention_type == "music":
        suggestions = [
            {"type": "music", "emotion_tag": "uplifting", "description": "播放积极向上的音乐"},
            {"type": "hitokoto", "category": "d&c=l", "description": "文学/抖机灵类句子"}
        ]
    elif result.current_valence < -0.3:
        suggestions = [
            {"type": "music", "emotion_tag": "comforting", "description": "播放安慰性音乐"},
            {"type": "script", "emotion_tag": "sad", "description": "播放预定义的安慰话术"}
        ]
    else:
        suggestions = [
            {"type": "info", "content": "当前情绪状态良好，继续保持"}
        ]

    return {
        "current_state": {
            "valence": result.current_valence,
            "risk_level": result.risk_level
        },
        "suggestions": suggestions
    }


# ==========================================
# 高危干预预警台 API
# ==========================================

@router.get("/api/admin/analytics/alerts")
async def get_alert_feed(limit: int = 20, db: Session = Depends(get_db)):
    """
    获取实时警报滚动数据

    返回最近的干预事件和高风险用户警报
    """
    # 获取最近的系统事件（干预记录）
    events = db.query(SystemEvent).order_by(
        desc(SystemEvent.timestamp)
    ).limit(limit).all()

    alerts = []
    for event in events:
        # 解析事件数据
        event_data = {
            "timestamp": event.timestamp.strftime("%H:%M:%S"),
            "username": "系统",
            "condition": event.event_type,
            "risk_level": "low",
            "intervention": None
        }

        # 如果是干预事件，尝试获取关联用户
        if "intervention" in event.event_type.lower():
            event_data["risk_level"] = "medium"
            event_data["intervention"] = "音乐干预" if "music" in event.event_type.lower() else "语音安抚"

        alerts.append(event_data)

    # 检查当前高风险用户
    now = datetime.now()
    cutoff = now - timedelta(hours=1)

    # 获取所有用户
    users = db.query(User).all()
    for user in users:
        # 获取用户最近的情绪记录
        recent_logs = db.query(EmotionLog).filter(
            EmotionLog.user_id == user.id,
            EmotionLog.timestamp >= cutoff
        ).order_by(desc(EmotionLog.timestamp)).limit(10).all()

        if len(recent_logs) < 3:
            continue

        # 计算最近的平均效价
        valence_map = {
            "happy": 1.0, "surprise": 0.3, "neutral": 0.0,
            "sad": -0.6, "fear": -0.8, "angry": -1.0, "disgust": -0.9
        }

        valences = [valence_map.get(log.emotion, 0) * (log.score / 100) for log in recent_logs]
        avg_valence = sum(valences) / len(valences)

        # 计算 RMSSD（优化 1：引入 Sessioning 机制，只计算 15 分钟内的连续波动）
        diffs = []
        for i in range(len(recent_logs) - 1):
            # 计算相邻两条记录的时间差（秒）
            time_gap = (recent_logs[i+1].timestamp - recent_logs[i].timestamp).total_seconds()

            # 阈值设为 900 秒（15 分钟）。超过 15 分钟视为不同场景，不计算波动差值
            if time_gap < 900:
                diffs.append(valences[i+1] - valences[i])

        if len(diffs) > 0:
            rmssd = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5
            rmssd = min(rmssd, 1.0)  # 归一化到 0-1
        else:
            rmssd = 0.0

        # 高风险判定
        risk_level = "low"
        condition = ""

        if avg_valence < -0.5 and rmssd < 0.1:
            risk_level = "high"
            condition = "触发【极端情绪僵化】(RMSSD<0.1, V<-0.5)"
        elif avg_valence < -0.3 and rmssd < 0.15:
            risk_level = "medium"
            condition = "情绪持续低落，波动率偏低"
        elif rmssd > 0.5:
            risk_level = "medium"
            condition = "情绪波动剧烈"

        if risk_level != "low":
            alerts.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "username": user.username,
                "condition": condition,
                "risk_level": risk_level,
                "intervention": "系统已自动推送轻音乐" if risk_level == "high" else None
            })

    # 按风险等级排序
    risk_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda x: (risk_order.get(x["risk_level"], 3), x["timestamp"]))

    return {"alerts": alerts[:limit]}


@router.get("/api/admin/analytics/quadrant")
async def get_quadrant_data(db: Session = Depends(get_db)):
    """
    获取四象限散点图数据

    X 轴：效价 (Valence) -1 ~ +1
    Y 轴：波动率 (RMSSD) 0 ~ 1
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=24)  # 统计最近 24 小时

    users_data = []

    for user in db.query(User).all():
        if user.role == "admin":
            continue  # 跳过管理员

        # 获取用户情绪记录
        logs = db.query(EmotionLog).filter(
            EmotionLog.user_id == user.id,
            EmotionLog.timestamp >= cutoff
        ).order_by(EmotionLog.timestamp).all()

        if len(logs) < 5:
            continue

        # 计算效价
        valence_map = {
            "happy": 1.0, "surprise": 0.3, "neutral": 0.0,
            "sad": -0.6, "fear": -0.8, "angry": -1.0, "disgust": -0.9
        }

        valences = [valence_map.get(log.emotion, 0) * (log.score / 100) for log in logs]

        # 优化 2：指数移动平均 (EMA)，赋予近期情绪更高权重，远期情绪指数衰减
        if valences:
            ema_valence = valences[0]
            alpha = 0.3  # 平滑系数 (0~1)，0.3 表示当次情绪占 30%，历史积累占 70%
            for v in valences[1:]:
                ema_valence = alpha * v + (1 - alpha) * ema_valence
            avg_valence = ema_valence
        else:
            avg_valence = 0.0

        # 计算 RMSSD
        if len(valences) > 1:
            diffs = [valences[i+1] - valences[i] for i in range(len(valences)-1)]
            rmssd = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5
            # 归一化到 0-1
            rmssd = min(rmssd, 1.0)
        else:
            rmssd = 0

        # 风险等级判定
        if avg_valence < -0.3 and rmssd < 0.15:
            risk_level = "high"
        elif avg_valence < -0.2 or rmssd > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        users_data.append({
            "user_id": user.id,
            "username": user.username,
            "valence": round(avg_valence, 3),
            "rmssd": round(rmssd, 3),
            "risk_level": risk_level
        })

    return {"users": users_data}


@router.get("/api/admin/analytics/interventions/{user_id}")
async def get_intervention_events(
    user_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    获取干预事件时间轴数据
    """
    cutoff = datetime.now() - timedelta(days=days)

    # 获取系统事件中的干预记录
    events = db.query(SystemEvent).filter(
        SystemEvent.timestamp >= cutoff
    ).order_by(SystemEvent.timestamp).all()

    intervention_events = []
    for event in events:
        event_type = event.event_type.lower()
        if "music" in event_type:
            intervention_events.append({
                "timestamp": event.timestamp.isoformat(),
                "type": "music",
                "effect": 0.7  # 模拟效果值
            })
        elif "tts" in event_type or "voice" in event_type:
            intervention_events.append({
                "timestamp": event.timestamp.isoformat(),
                "type": "tts",
                "effect": 0.8
            })

    return {"events": intervention_events}


@router.get("/api/admin/analytics/system-health")
async def get_system_health(db: Session = Depends(get_db)):
    """
    获取 AI 系统健康度数据

    - 置信度分布
    - 情绪类别占比
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=24)

    # 获取最近 24 小时的情绪记录
    logs = db.query(EmotionLog).filter(
        EmotionLog.timestamp >= cutoff
    ).all()

    # 置信度分布统计
    confidence_buckets = [0, 0, 0, 0, 0]  # 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    emotion_counts = {}

    for log in logs:
        # 置信度分布
        score = log.score / 100  # 0-1
        if score < 0.2:
            confidence_buckets[0] += 1
        elif score < 0.4:
            confidence_buckets[1] += 1
        elif score < 0.6:
            confidence_buckets[2] += 1
        elif score < 0.8:
            confidence_buckets[3] += 1
        else:
            confidence_buckets[4] += 1

        # 情绪统计
        emotion_counts[log.emotion] = emotion_counts.get(log.emotion, 0) + 1

    # 计算模型准确率（基于稳定检测的比例）
    total_users = db.query(User).filter(User.role != "admin").count()
    active_users = db.query(EmotionLog.user_id).filter(
        EmotionLog.timestamp >= cutoff
    ).distinct().count()

    model_accuracy = active_users / max(total_users, 1) if total_users > 0 else 0

    return {
        "confidenceDistribution": confidence_buckets,
        "emotionPieData": [{"name": k, "value": v} for k, v in emotion_counts.items()],
        "modelAccuracy": round(model_accuracy, 2),
        "totalRecords": len(logs)
    }


@router.post("/api/admin/capture_face/{user_id}")
async def capture_face(user_id: int, db: Session = Depends(get_db)):
    """
    管理员手动录入人脸：
    1. 抓取摄像头当前帧
    2. 使用 DeepFace 提取 Embedding
    3. 存入 User 表
    """
    global video_cap

    if video_cap is None or not video_cap.cap.isOpened():
        raise HTTPException(status_code=503, detail="摄像头未就绪")

    success, frame = video_cap.cap.read()
    if not success or frame is None:
        raise HTTPException(status_code=500, detail="无法读取摄像头画面")

    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True
        )

        if not results:
            return {"success": False, "message": "未检测到清晰人脸，请正对摄像头"}

        embedding = results[0].get("embedding")
        if embedding is None:
            return {"success": False, "message": "人脸特征提取失败"}

        embedding = [float(x) for x in embedding]

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"success": False, "message": "用户不存在"}

        user.face_encoding = embedding
        db.commit()

        return {"success": True, "message": f"用户 [{user.username}] 人脸特征录入成功"}

    except Exception as e:
        logger.error(f"❌ 人脸特征录入失败: {e}", exc_info=True)
        return {"success": False, "message": "录入失败，请正对摄像头并保持光线充足"}


# --- 用户端专属接口 (User) ---

# backend/app/api.py

@router.get("/api/my/stats")
async def get_my_stats(username: str, db: Session = Depends(get_db)):
    """获取特定用户的个人统计信息"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    logs = db.query(EmotionLog).filter(EmotionLog.user_id == user.id)\
            .order_by(desc(EmotionLog.timestamp)).limit(100).all()
            
    emotion_counts = {}
    for log in logs:
        e = log.emotion
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
    pie_data = [{"name": k, "value": v} for k, v in emotion_counts.items()]
    
    return {
        "user_id": user.id,
        "pie_data": pie_data,
        "total_records": len(logs),
        # ✅ 新增：返回是否有面部特征数据
        "has_face": user.face_encoding is not None 
    }

@router.get("/api/my/history")
async def get_my_history(username: str, page: int = 1, db: Session = Depends(get_db)):
    """获取特定用户的历史记录"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"total": 0, "data": []}

    page_size = 10
    total = db.query(EmotionLog).filter(EmotionLog.user_id == user.id).count()
    logs = db.query(EmotionLog).filter(EmotionLog.user_id == user.id) \
        .order_by(desc(EmotionLog.timestamp)) \
        .offset((page - 1) * page_size).limit(page_size).all()

    return {"total": total, "data": logs}


# --- 调试与测试接口 (Debug) ---

@router.get("/api/debug/seed_data")
async def seed_data(db: Session = Depends(get_db)):
    emotions = ["happy", "sad", "neutral", "angry", "fear"]

    for _ in range(50):
        log = EmotionLog(
            timestamp=datetime.now() - timedelta(minutes=random.randint(1, 1440)),
            user_id=None,
            is_stranger=True,
            emotion=random.choice(emotions),
            score=random.random()
        )
        db.add(log)

    db.commit()
    return {"message": "生成完成"}


@router.get("/api/debug/create_test_user")
async def create_test_user(db: Session = Depends(get_db)):
    """创建默认管理员账户 (admin/123456)"""
    exists = db.query(User).filter(User.username == "admin").first()
    if not exists:
        test_user = User(
            username="admin",
            password_hash="123456",
            role="admin"
        )
        db.add(test_user)
        db.commit()
        return {"message": "测试用户 admin 已创建，密码 123456"}
    return {"message": "用户已存在"}


# --- 日记相关接口 ---

@router.get("/api/my/diaries")
async def get_my_diaries(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []

    diaries = db.query(Diary).filter(Diary.user_id == user.id) \
        .order_by(desc(Diary.timestamp)).all()
    return diaries


@router.post("/api/my/diaries")
async def create_diary(data: dict, db: Session = Depends(get_db)):
    # data 格式: { "username": "admin", "content": "...", "emotion": "...", "timestamp": "2023-..." }
    user = db.query(User).filter(User.username == data["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 如果前端传了时间，就用传的时间（补打卡），否则用当前时间
    custom_time = data.get("timestamp")
    if custom_time:
        # 解析前端传来的 ISO 时间字符串
        try:
            # 简单处理，如果前端传的是 ISO 字符串
            log_time = datetime.fromisoformat(custom_time.replace('Z', ''))
        except:
            log_time = datetime.now()
    else:
        log_time = datetime.now()

    new_entry = Diary(
        user_id=user.id,
        content=data["content"],
        emotion=data.get("emotion", "Neutral"),
        title=data.get("title", log_time.strftime("%Y-%m-%d")),
        timestamp=log_time  # 使用计算后的时间
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return {"success": True, "id": new_entry.id}


@router.put("/api/my/diaries/{diary_id}")
async def update_diary(diary_id: int, data: dict, db: Session = Depends(get_db)):
    diary = db.query(Diary).filter(Diary.id == diary_id).first()
    if not diary:
        raise HTTPException(status_code=404, detail="Diary not found")

    # 简单的权限检查 (实际项目应校验当前登录用户)
    # 更新内容
    if "content" in data:
        diary.content = data["content"]
    if "emotion" in data:
        diary.emotion = data["emotion"]
    if "timestamp" in data and data["timestamp"]:
        try:
            diary.timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', ''))
        except:
            pass

    db.commit()
    return {"success": True, "message": "已更新"}


@router.delete("/api/my/diaries/{diary_id}")
async def delete_diary(diary_id: int, db: Session = Depends(get_db)):
    diary = db.query(Diary).filter(Diary.id == diary_id).first()
    if not diary:
        raise HTTPException(status_code=404, detail="Diary not found")

    db.delete(diary)
    db.commit()
    return {"success": True, "message": "已删除"}


@router.post("/api/register")
async def register(data: dict, db: Session = Depends(get_db)):
    # data: { username, password }
    username = data.get("username")
    password = data.get("password")

    # 检查用户是否存在
    exists = db.query(User).filter(User.username == username).first()
    if exists:
        return {"success": False, "message": "用户名已存在"}

    # 创建新用户 (实际项目应使用 passlib 加密，这里为了演示直接存)
    new_user = User(
        username=username,
        password_hash=password,  # 建议生产环境哈希加密
        role="user"  # 默认注册为普通用户
    )
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "注册成功，请登录"}


# --- 2. 真实登录接口 ---
@router.post("/api/login")
async def login(data: dict, db: Session = Depends(get_db)):
    username = data.get("username")
    password = data.get("password")

    user = db.query(User).filter(User.username == username).first()

    if user and user.password_hash == password:
        return {
            "success": True,
            "role": user.role,
            "username": user.username
        }

    return {"success": False, "message": "用户名或密码错误"}


@router.get("/api/my/daily_mood")
async def get_daily_mood(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []

    # 查询最近 30 天每天最高频的情绪（SQL 逻辑较复杂，这里简化示意）
    # 实际上你可以返回一个字典：{ "2024-02-06": "happy", "2024-02-07": "sad" }
    return {"6": "happy", "7": "neutral"}


# backend/app/api.py
# backend/app/api.py

@router.get("/api/my/calendar_moods")
async def get_calendar_moods(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {}

    # --- 1. 获取人脸识别数据 ---
    face_stats = db.query(
        func.date(EmotionLog.timestamp).label('date_str'),  # 得到 "2026-02-18"
        EmotionLog.emotion,
        func.count(EmotionLog.id).label('count')
    ).filter(EmotionLog.user_id == user.id) \
        .group_by('date_str', EmotionLog.emotion).all()

    daily_face_mood = {}
    temp_counts = {}

    for date_str, emotion, count in face_stats:
        if not date_str:
            continue
        # 直接使用 date_str ("2026-02-18") 作为 Key
        if date_str not in temp_counts:
            temp_counts[date_str] = []
        temp_counts[date_str].append((emotion, count))

    for date_str, emotion_list in temp_counts.items():
        best_emotion = sorted(emotion_list, key=lambda x: x[1], reverse=True)[0][0]
        daily_face_mood[date_str] = best_emotion.lower()

    # --- 2. 获取日记数据 (优先级高) ---
    diaries = db.query(Diary.timestamp, Diary.emotion) \
        .filter(Diary.user_id == user.id).all()

    daily_diary_mood = {}
    for ts, emotion in diaries:
        if not ts:
            continue
        # 格式化为 "2026-02-18"
        date_str = ts.strftime("%Y-%m-%d")
        daily_diary_mood[date_str] = emotion.lower()

    # --- 3. 合并 ---
    final_moods = daily_face_mood.copy()
    for date_str, diary_emotion in daily_diary_mood.items():
        final_moods[date_str] = diary_emotion

    # 返回的数据结构现在是: {"2026-02-18": "happy", "2026-03-18": "sad"}
    return final_moods


@router.get("/api/admin/scripts")
async def admin_get_scripts(target_user: str = 'global', db: Session = Depends(get_db)):
    if target_user == 'global':
        scripts = db.query(ComfortScript).filter(ComfortScript.user_id == None).all()
    else:
        user = db.query(User).filter(User.username == target_user).first()
        if not user:
            return[]
        scripts = db.query(ComfortScript).filter(ComfortScript.user_id == user.id).all()
    return scripts
class AdminScriptCreate(BaseModel):
    content: str
    emotion_tag: str
    target_user: str  # 'global' 或者 'username'


@router.post("/api/admin/scripts")
async def admin_add_script(script: AdminScriptCreate, db: Session = Depends(get_db)):
    if script.target_user == 'global':
        user_id = None
    else:
        user = db.query(User).filter(User.username == script.target_user).first()
        if not user:
            raise HTTPException(status_code=404, detail="目标用户不存在")
        user_id = user.id
        
    new_script = ComfortScript(
        content=script.content,
        emotion_tag=script.emotion_tag,
        user_id=user_id
    )
    db.add(new_script)
    db.commit()
    return {"success": True}

@router.delete("/api/admin/scripts/{script_id}")
async def delete_script(script_id: int, db: Session = Depends(get_db)):
    """删除话术"""
    script = db.query(ComfortScript).get(script_id)
    if script:
        db.delete(script)
        db.commit()
    return {"success": True}

@router.post("/api/user/upload_background")
async def upload_background(
    file: UploadFile = File(...), 
    username: str = Form(...),
    db: Session = Depends(get_db)
):
    # 路径必须指向 main.py 挂载的那个 assets 文件夹
    base_assets = Path(__file__).parent.parent / "assets"
    save_path = base_assets / f"bg_{username}.jpg"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"message": "success", "url": f"/assets/bg_{username}.jpg"}
# --- 修改状态接口：整合话术 ---
@router.get("/api/admin/scripts/daily")
async def get_daily_quote(db: Session = Depends(get_db)):
    """获取每日欢迎语：优先一言 API，失败则回退本地库"""
    # 1. 尝试从一言获取
    external_quote = await get_hitokoto_quote()
    if external_quote:
        return {"content": external_quote, "source": "hitokoto"}

    # 2. 如果一言失败，从本地数据库随机挑一条 happy 话术
    local_script = db.query(ComfortScript)\
                     .filter(ComfortScript.emotion_tag == "happy")\
                     .order_by(func.random()).first()
    
    if local_script:
        return {"content": local_script.content, "source": "local"}
    
    return {"content": "愿你今天拥有好心情！", "source": "default"}

@router.get("/api/my/personalized_quote")
async def get_personalized_quote(username: str, db: Session = Depends(get_db)):
    """
    核心接口：根据用户历史情绪定向推送
    """
    # 1. 查找用户
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"content": "欢迎回来！", "emotion_detected": "unknown"}

    # 2. 查找最近 10 分钟内最频繁的情绪 (如果没有，就看最后一条)
    # 这里我们简化逻辑：直接获取最后一条有效的情绪记录
    last_log = db.query(EmotionLog).filter(EmotionLog.user_id == user.id)\
                 .order_by(desc(EmotionLog.timestamp)).first()
    
    # 默认情绪为平静
    current_emotion = last_log.emotion.lower() if last_log else "neutral"

    # 3. 尝试获取对应的一言
    quote = await get_hitokoto_by_emotion(current_emotion)
    
    # 4. 如果一言接口没响应，回退到本地话术库
    if not quote:
        local_script = db.query(ComfortScript)\
                         .filter(ComfortScript.emotion_tag == current_emotion)\
                         .order_by(func.random()).first()
        quote = local_script.content if local_script else "愿你被世界温柔以待。"

    return {
        "content": quote,
        "emotion_tag": current_emotion,
        "source": "hitokoto" if "「" in quote else "local"
    }

class ScriptCreate(BaseModel):
    content: str
    emotion_tag: str
    username: str  # 新增：告诉后端是谁添加的

@router.post("/api/user/upload_music")
async def upload_user_music(
    file: UploadFile = File(...),
    emotion: str = Form(...),
    username: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 构造唯一文件名
    os.makedirs("assets/music", exist_ok=True)
    timestamp = int(time.time())
    file_name = f"music_{username}_{emotion}_{timestamp}.mp3"
    file_path = os.path.join("assets/music", file_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 直接新增记录，支持一个情绪多首音乐
    new_music = MusicLibrary(
        title=file.filename,
        filepath=file_path,
        emotion_tag=emotion,
        user_id=user.id
    )
    db.add(new_music)
    db.commit()
    return {"success": True, "message": "上传成功"}

# ==========================================
# 💬 2. 用户专属：获取自己的话术列表
# ==========================================
@router.get("/api/user/scripts")
async def get_user_scripts(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return[]
    
    # 只查询属于这个用户的话术
    scripts = db.query(ComfortScript).filter(ComfortScript.user_id == user.id).all()
    return scripts

# ==========================================
# 💬 3. 用户专属：添加自己的话术
# ==========================================
@router.post("/api/user/scripts")
async def add_user_script(script: ScriptCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == script.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
        
    new_script = ComfortScript(
        content=script.content,
        emotion_tag=script.emotion_tag,
        user_id=user.id  # 绑定专属用户
    )
    db.add(new_script)
    db.commit()
    return {"success": True, "message": "添加成功"}

# ==========================================
# 💬 4. 用户专属：删除自己的话术
# ==========================================
@router.delete("/api/user/scripts/{script_id}")
async def delete_user_script(script_id: int, username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    
    script = db.query(ComfortScript).filter(
        ComfortScript.id == script_id,
        ComfortScript.user_id == user.id  # 安全校验：只能删自己的
    ).first()
    
    if not script:
        raise HTTPException(status_code=404, detail="找不到该话术或无权删除")
        
    db.delete(script)
    db.commit()
    return {"success": True, "message": "删除成功"}
# backend/app/api.py

@router.get("/api/admin/users")
async def admin_get_users(db: Session = Depends(get_db)):
    # 1. 改为查询完整的 User 对象，以便获取 face_encoding 字段
    users = db.query(User).all()
    
    # 2. 构造返回列表，增加 has_face 字段
    return {
        "users": [
            {
                "id": u.id, 
                "username": u.username, 
                "role": u.role,
                # 💡 核心修改：判断 face_encoding 是否为空，返回布尔值
                "has_face": u.face_encoding is not None 
            } 
            for u in users
        ]
    }

# ==========================================
# 🎵 管理员：获取音乐配置状态 (区分全局和专属)
# ==========================================

# ==========================================
# 🎵 管理员：获取音乐配置状态 (区分全局和专属)
# ==========================================
@router.get("/api/admin/music")
async def admin_get_music(target_user: str = 'global', db: Session = Depends(get_db)):
    music_records = []
    if target_user == 'global':
        music_records = db.query(MusicLibrary).filter(MusicLibrary.user_id == None).all()
    else:
        user = db.query(User).filter(User.username == target_user).first()
        if user:
            music_records = db.query(MusicLibrary).filter(MusicLibrary.user_id == user.id).all()
    
    return [
        {
            "id": m.id,
            "title": m.title,
            "emotion_tag": m.emotion_tag,
            "filepath": m.filepath,
            "is_active": m.is_active
        } for m in music_records
    ]
# ==========================================
# 🎵 管理员：删除音乐记录
# ==========================================
@router.delete("/api/admin/music/{music_id}")
async def admin_delete_music(music_id: int, db: Session = Depends(get_db)):
    music = db.query(MusicLibrary).filter(MusicLibrary.id == music_id).first()
    if not music:
        raise HTTPException(status_code=404, detail="Music not found")

    # 尝试删除物理文件
    try:
        # 这里的路径处理要小心，确保存储的是相对路径
        base_path = Path(__file__).parent.parent 
        full_path = base_path / music.filepath
        if full_path.exists():
            os.remove(full_path)
    except Exception as e:
        print(f"Delete file error: {e}")

    db.delete(music)
    db.commit()
    return {"message": "deleted"}

# ==========================================
# 🎵 管理员：上传音乐到音乐库 (区分全局和专属)
# ==========================================
@router.post("/api/admin/upload_music")
async def admin_upload_music(
    file: UploadFile = File(...),
    emotion: str = Form(...),
    target_user: str = Form('global'),
    db: Session = Depends(get_db)
):
    # 定义存储目录 (对应 main.py 中的 assets 挂载点)
    # 因为 main.py 挂载的是 backend/assets，所以我们要存到这个目录下
    base_assets = Path(__file__).parent.parent / "assets"
    music_dir = base_assets / "music"
    music_dir.mkdir(parents=True, exist_ok=True)

    # 为了支持多个音乐，文件名加入时间戳防止覆盖
    timestamp = int(time.time())
    ext = os.path.splitext(file.filename)[1]
    safe_name = f"{target_user}_{emotion}_{timestamp}{ext}"
    save_path = music_dir / safe_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 存入数据库
    new_music = MusicLibrary(
        title=file.filename, # 保留原名展示
        filepath=f"assets/music/{safe_name}", # 存储相对路径供前端访问
        emotion_tag=emotion,
        is_active=True
    )

    if target_user != 'global':
        user = db.query(User).filter(User.username == target_user).first()
        if user:
            new_music.user_id = user.id

    db.add(new_music)
    db.commit()
    return {"message": "success", "filename": safe_name}

# ==========================================
# 🧠 AI 综合情绪诊断与智能建议 API
# ==========================================
@router.get("/api/admin/analytics/comprehensive/{user_id}")
async def get_comprehensive_report(
    user_id: int, 
    days: int = 7, 
    db: Session = Depends(get_db)
):
    """
    生成用户的文字版综合分析报告与具体干预建议
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
        
    cutoff = datetime.now() - timedelta(days=days)
    logs = db.query(EmotionLog).filter(
        EmotionLog.user_id == user_id,
        EmotionLog.timestamp >= cutoff
    ).order_by(EmotionLog.timestamp).all()

    if not logs or len(logs) < 3:
        return {
            "has_data": False,
            "summary": "该用户近期数据不足，无法生成有效诊断报告。",
        }

    # 1. 基础数据计算
    valence_map = { 
        "happy": 1.0, "surprise": 0.3, "neutral": 0.0, 
        "sad": -0.6, "fear": -0.8, "angry": -1.0, "disgust": -0.9 
    }
    valences =[valence_map.get(log.emotion, 0) * (log.score / 100) for log in logs]
    
    avg_valence = sum(valences) / len(valences)
    
    # 计算 RMSSD (波动率)
    diffs = [valences[i+1] - valences[i] for i in range(len(valences)-1)]
    rmssd = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5

    # 2. 诊断逻辑树
    risk_level = "low"
    conclusion = "情绪状态平稳"
    summary = "该用户近期情绪保持在健康范围内，没有明显的极端情绪波动或僵化迹象。建议保持日常监控即可。"
    suggestions =[]

    if avg_valence < -0.35 and rmssd < 0.15:
        risk_level = "high"
        conclusion = "重度情绪僵化 / 抑郁风险"
        summary = "该用户近期情绪效价持续处于负面区间（长期悲伤/低落），且波动率(RMSSD)极低。这表现出典型的『情绪僵化（Emotional Inflexibility）』特征，极有可能已陷入抑郁或绝望状态。"
        suggestions =[
            {"type": "medical", "icon": "Document", "title": "下发量表", "desc": "建议立即推送 PHQ-9 抑郁筛查量表进行临床评估。"},
            {"type": "music", "icon": "Headset", "title": "保守音乐干预", "desc": "推送轻柔舒缓的治愈系白噪音，切忌推送欢快音乐产生反差压力。"},
            {"type": "tts", "icon": "Mic", "title": "语音安抚", "desc": "触发看板娘共情话术：'无论发生什么，我都在这里陪着你。'"}
        ]
    elif avg_valence < -0.2 or rmssd > 0.4:
        risk_level = "medium"
        conclusion = "情绪波动剧烈 / 焦虑状态"
        summary = "该用户近期情绪起伏剧烈，可能遭遇了某些应激事件，处于焦虑、烦躁或不稳定的状态。需要预防情绪崩溃。"
        suggestions =[
            {"type": "music", "icon": "Headset", "title": "环境音干预", "desc": "推送自然白噪音（如雨声、海浪声）帮助平复心率。"},
            {"type": "action", "icon": "VideoPlay", "title": "放松引导", "desc": "在用户端推送『4-7-8 深呼吸放松训练』。"}
        ]
    elif avg_valence > 0.4:
        risk_level = "low"
        conclusion = "积极乐观 / 状态极佳"
        summary = "该用户近期情绪非常积极，处于高唤醒的愉悦状态。系统视觉捕获到高频次的微笑特征。"
        suggestions =[
            {"type": "hitokoto", "icon": "ChatLineRound", "title": "正向激励", "desc": "推送充满活力的每日一言，巩固好心情。"}
        ]
    else:
        # 平稳期建议
        suggestions =[
            {"type": "monitor", "icon": "View", "title": "持续观察", "desc": "无需特殊人工干预，维持 AI 后台静默守护。"}
        ]

    return {
        "has_data": True,
        "metrics": {
            "avg_valence": round(avg_valence, 3),
            "rmssd": round(rmssd, 3)
        },
        "risk_level": risk_level,
        "conclusion": conclusion,
        "summary": summary,
        "suggestions": suggestions
    }
# ==========================================
# 🖼️ 删除用户背景图 (恢复默认)
# ==========================================
@router.delete("/api/user/upload_background")
async def delete_background(username: str, db: Session = Depends(get_db)):
    # 路径指向存储背景图的地方
    base_assets = Path(__file__).parent.parent / "assets"
    save_path = base_assets / f"bg_{username}.jpg"

    if save_path.exists():
        try:
            os.remove(save_path)
            return {"success": True, "message": "背景已重置为默认"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除文件失败: {e}")
    
    return {"success": True, "message": "已经是默认背景"}


@router.get("/api/user/music")
async def get_user_music(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []
    
    # 只查询属于该用户的音乐
    musics = db.query(MusicLibrary).filter(MusicLibrary.user_id == user.id).all()
    return [
        {
            "id": m.id,
            "title": m.title,
            "emotion_tag": m.emotion_tag,
            "filepath": m.filepath
        } for m in musics
    ]
@router.delete("/api/user/music/{music_id}")
async def delete_user_music(music_id: int, username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    music = db.query(MusicLibrary).filter(
        MusicLibrary.id == music_id,
        MusicLibrary.user_id == user.id # 安全校验：只能删自己的
    ).first()

    if not music:
        raise HTTPException(status_code=404, detail="找不到该音乐资源或无权删除")

    # 删除物理文件
    if os.path.exists(music.filepath):
        try:
            os.remove(music.filepath)
        except:
            pass

    db.delete(music)
    db.commit()
    return {"success": True, "message": "删除成功"}