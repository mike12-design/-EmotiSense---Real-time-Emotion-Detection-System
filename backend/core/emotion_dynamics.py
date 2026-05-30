# backend/core/emotion_dynamics.py
import time
from collections import deque

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust", "neutral"}

class EmotionDynamicsEngine:
    """情绪动力学引擎 — 基于稳定器输出的滑动窗口投票触发干预"""

    def __init__(self):
        self.primary_emotion = "neutral"
        self.mood_score = 0.0
        self.stress_level = 0.0
        self.trigger_intervention = False

        # 滑动窗口：记录稳定器每次输出的情绪
        self.window = deque(maxlen=20)   # 约 100 秒
        self.last_intervention = 0

        # 可调参数
        self.WINDOW_SIZE = 20
        self.NEGATIVE_RATIO = 0.7
        self.COOLDOWN = 120

    def valence_map(self, emotion):
        mapping = {
            "happy": 1.0, "surprise": 0.3, "neutral": 0.0,
            "sad": -0.6, "fear": -0.8, "angry": -1.0, "disgust": -0.9,
        }
        return mapping.get(emotion.lower(), 0.0)

    def update(self, emotion, confidence):
        now = time.time()
        self.primary_emotion = emotion

        # 更新效价和压力（保留给前端显示用）
        v = self.valence_map(emotion) * confidence
        self.mood_score = self.mood_score * 0.8 + v * 0.2
        self.stress_level = 1.0 if self.mood_score < 0 else max(0, self.stress_level * 0.9)

        # 滑动窗口投票
        self.window.append(emotion.lower())

        # 判断触发
        self.trigger_intervention = False
        if len(self.window) >= self.WINDOW_SIZE and now - self.last_intervention > self.COOLDOWN:
            negative_count = sum(1 for e in self.window if e in NEGATIVE_EMOTIONS)
            ratio = negative_count / len(self.window)
            if ratio >= self.NEGATIVE_RATIO:
                self.last_intervention = now
                self.window.clear()
                self.trigger_intervention = True

        return self

    def get_state(self):
        return {
            "valence": round(self.mood_score, 3),
            "distress": round(self.stress_level, 3),
            "primary_emotion": self.primary_emotion,
        }
