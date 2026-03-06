# backend/core/emotion_dynamics.py
import time
from collections import deque

class EmotionDynamicsEngine:
    """
    Emotional Dynamics Engine
    模拟情绪随时间演化，并增加兼容 API 调用所需的属性
    """

    def __init__(self):
        # EMA 情绪值（-1 ~ 1）
        self.valence_ema = 0.0
        # 压力累积
        self.distress = 0.0
        # 历史窗口
        self.history = deque(maxlen=30)
        # 上次干预时间
        self.last_intervention = 0

        # --- 新增：用于兼容 api.py 调用的属性 ---
        self.primary_emotion = "neutral"
        self.mood_score = 0.0
        self.stress_level = 0.0
        self.trigger_intervention = False # 是否触发了干预

        # 参数
        self.ALPHA = 0.2
        self.DECAY = 0.95
        self.TRIGGER_THRESHOLD = 1.2
        self.COOLDOWN = 60

    def valence_map(self, emotion):
        mapping = {
            "happy": 1.0,
            "surprise": 0.3,
            "neutral": 0.0,
            "sad": -0.6,
            "fear": -0.8,
            "angry": -1.0,
            "disgust": -0.9,
            "contempt": -0.7
        }
        return mapping.get(emotion.lower(), 0.0)

    def update(self, emotion, confidence):
        """
        更新情绪动力学状态
        """
        now = time.time()
        self.primary_emotion = emotion # 记录当前情绪

        # 计算当前的效价权重
        v = self.valence_map(emotion) * confidence

        # EMA 更新
        self.valence_ema = self.ALPHA * v + (1 - self.ALPHA) * self.valence_ema

        # 压力累积逻辑
        if self.valence_ema < 0:
            self.distress = self.distress * self.DECAY + abs(self.valence_ema)
        else:
            self.distress *= self.DECAY

        self.history.append(self.valence_ema)

        # 更新映射属性供外部访问
        self.mood_score = self.valence_ema
        self.stress_level = self.distress

        # 干预判断逻辑
        self.trigger_intervention = False
        if now - self.last_intervention > self.COOLDOWN:
            if self.distress > self.TRIGGER_THRESHOLD:
                self.last_intervention = now
                self.distress *= 0.5
                self.trigger_intervention = True

        # 💡 关键修改：返回 self 确保 api.py 中的 dyn.primary_emotion 不报错
        return self

    def get_state(self):
        return {
            "valence": round(self.valence_ema, 3),
            "distress": round(self.distress, 3),
            "primary_emotion": self.primary_emotion
        }