# backend/core/stabilizer.py
from collections import deque, Counter

class EmotionStabilizer:
    def __init__(self, window_size=5):
        """
        window_size: 窗口大小，越大越稳定，但反应越慢
        建议设置为 5-10 帧
        """
        self.history = deque(maxlen=window_size)
        self.last_stable_emotion = "neutral"

    def add_prediction(self, emotion):
        """添加最新的预测结果"""
        self.history.append(emotion)

    def get_stable_emotion(self):
        """获取当前最稳定的情绪"""
        if not self.history:
            return "neutral"
        
        # 统计窗口内出现次数最多的情绪
        counts = Counter(self.history)
        most_common_emotion, count = counts.most_common(1)[0]
        
        # 只有当该情绪在窗口中占比超过 60% 时，才认为情绪发生了切换
        # 否则保持上一次的稳定情绪（防抖）
        threshold = self.history.maxlen * 0.6
        
        if count >= threshold:
            self.last_stable_emotion = most_common_emotion
            
        return self.last_stable_emotion