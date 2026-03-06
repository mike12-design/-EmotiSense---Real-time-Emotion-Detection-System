# backend/core/advanced_analyzer.py
"""
高级情绪分析模块

实现：
1. 情绪数值化映射 (Valence × Score)
2. 个性化吸引子模型 (Attractor)
3. RMSSD 情绪波动量化
4. 卡尔曼滤波平滑
5. 干预决策引擎
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmotionDataPoint:
    """单个情绪数据点"""
    timestamp: float
    emotion: str
    score: float  # 原始置信度 (0-100)
    valence: float  # 效价值 (-1 ~ 1)
    v_final: float  # 最终情绪值 (valence × normalized_score)


@dataclass
class AnalyticsResult:
    """分析结果"""
    attractor: float  # 情绪吸引子 (基线)
    attractor_std: float  # 标准差
    rmssd: float  # 情绪波动指数
    current_valence: float  # 当前情绪值
    deviation: float  # 当前偏离度 (σ 单位)
    smoothed_valence: List[float]  # 卡尔曼滤波后的序列
    intervention_needed: bool  # 是否需要干预
    intervention_type: str  # 干预类型
    risk_level: str  # 风险等级：low/medium/high


class AdvancedEmotionAnalyzer:
    """
    高级情绪分析器

    核心功能：
    1. 情绪数值化映射
    2. 个性化吸引子计算
    3. RMSSD 波动量化
    4. 卡尔曼滤波平滑
    5. 智能干预决策
    """

    # 情绪效价映射表 (基于心理学研究)
    VALENCE_MAP = {
        "happy": 1.0,
        "surprise": 0.3,
        "neutral": 0.0,
        "sad": -0.6,
        "fear": -0.8,
        "angry": -1.0,
        "disgust": -0.9,
        "contempt": -0.7
    }

    def __init__(
        self,
        kalman_use: bool = True,
        rmssd_window: int = 10,
        deviation_threshold: float = 2.0,
        duration_threshold: int = 5
    ):
        """
        初始化分析器

        Args:
            kalman_use: 是否使用卡尔曼滤波
            rmssd_window: RMSSD 计算窗口大小
            deviation_threshold: 偏离度阈值 (σ 倍数)
            duration_threshold: 持续时间阈值 (记录条数)
        """
        self.kalman_use = kalman_use
        self.rmssd_window = rmssd_window
        self.deviation_threshold = deviation_threshold
        self.duration_threshold = duration_threshold

        # 卡尔曼滤波参数
        self.kalman_Q = 0.01  # 过程噪声协方差
        self.kalman_R = 0.1   # 测量噪声协方差

    def valence_map(self, emotion: str) -> float:
        """获取情绪的效价值"""
        return self.VALENCE_MAP.get(emotion.lower(), 0.0)

    def convert_to_valence_series(
        self,
        logs: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将情绪日志转换为效价时间序列

        Args:
            logs: 情绪日志列表，每条包含 {timestamp, emotion, score}

        Returns:
            (timestamps, v_final_values) 元组
        """
        timestamps = []
        v_final_values = []

        for log in logs:
            emotion = log.get('emotion', 'neutral')
            score = log.get('score', 50.0)
            timestamp = log.get('timestamp', 0)

            # 1. 获取基础效价
            valence = self.valence_map(emotion)

            # 2. 计算最终情绪值：V_final = Valence × (score / 100)
            normalized_score = score / 100.0
            v_final = valence * normalized_score

            timestamps.append(timestamp)
            v_final_values.append(v_final)

        return np.array(timestamps), np.array(v_final_values)

    def calculate_attractor(
        self,
        valence_series: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算情绪吸引子 (基线)

        Args:
            valence_series: 效价时间序列

        Returns:
            (mean, std) 元组
        """
        if len(valence_series) == 0:
            return 0.0, 1.0

        mean = np.mean(valence_series)
        std = np.std(valence_series) if len(valence_series) > 1 else 1.0

        # 防止 std 为 0 导致除零错误
        if std < 0.001:
            std = 1.0

        return mean, std

    def calculate_rmssd(
        self,
        valence_series: np.ndarray,
        window: Optional[int] = None
    ) -> float:
        """
        计算 RMSSD (Root Mean Square of Successive Differences)
        情绪波动指数 - 类似 HRV 心率变异性分析

        Args:
            valence_series: 效价时间序列
            window: 计算窗口大小 (默认使用实例变量)

        Returns:
            RMSSD 值
        """
        if len(valence_series) < 2:
            return 0.0

        # 使用最近的窗口数据
        if window is None:
            window = self.rmssd_window

        if len(valence_series) > window:
            series = valence_series[-window:]
        else:
            series = valence_series

        # 计算相邻差分
        delta_v = np.diff(series)

        # RMSSD = sqrt(mean(delta^2))
        rmssd = np.sqrt(np.mean(delta_v ** 2))

        return rmssd

    def kalman_filter(
        self,
        measurements: np.ndarray
    ) -> np.ndarray:
        """
        一维卡尔曼滤波
        用于提取"慢动态心境"曲线，消除光照/微表情引起的波动

        Args:
            measurements: 原始测量值序列

        Returns:
            平滑后的序列
        """
        if len(measurements) == 0:
            return measurements

        n = len(measurements)

        # 初始化
        x_est = measurements[0]  # 状态估计
        P_est = 1.0  # 估计协方差

        smoothed = [x_est]

        for i in range(1, n):
            # 预测步骤
            x_pred = x_est  # 状态转移矩阵为 1
            P_pred = P_est + self.kalman_Q  # 预测协方差

            # 更新步骤
            K = P_pred / (P_pred + self.kalman_R)  # 卡尔曼增益
            x_est = x_pred + K * (measurements[i] - x_pred)  # 状态更新
            P_est = (1 - K) * P_pred  # 协方差更新

            smoothed.append(x_est)

        return np.array(smoothed)

    def detect_intervention_need(
        self,
        valence_series: np.ndarray,
        attractor: float,
        attractor_std: float,
        rmssd: float
    ) -> Tuple[bool, str, str]:
        """
        检测是否需要干预

        判定逻辑：
        1. 偏离度检查：当前值偏离吸引子 > 2σ
        2. 持续性检查：连续 N 条记录持续偏离 > 1.5σ

        Args:
            valence_series: 效价时间序列
            attractor: 吸引子均值
            attractor_std: 吸引子标准差
            rmssd: 情绪波动指数

        Returns:
            (needed, type, risk_level) 元组
        """
        if len(valence_series) < self.duration_threshold:
            return False, "none", "low"

        # 获取最近 N 条记录
        recent = valence_series[-self.duration_threshold:]
        current_valence = recent[-1]

        # 计算偏离度
        deviation = np.abs(current_valence - attractor)
        deviation_sigma = deviation / attractor_std if attractor_std > 0.001 else 0

        # 检查持续性偏离
        sustained_deviation = np.all(
            np.abs(recent - attractor) > 1.5 * attractor_std
        )

        # 双条件判定
        if deviation_sigma > self.deviation_threshold and sustained_deviation:
            # 判断干预类型
            if rmssd < 0.1 and attractor < -0.3:
                # 低波动 + 低效价 = 陷入悲伤 (最需要关注)
                return True, "tts_urgency", "high"
            elif current_valence < -0.5:
                # 深度负面情绪
                return True, "music", "medium"
            elif deviation_sigma > 3.0:
                # 极端偏离
                return True, "tts", "high"
            else:
                return True, "music", "medium"

        return False, "none", "low"

    def analyze(
        self,
        logs: List[Dict],
        days: int = 7
    ) -> AnalyticsResult:
        """
        完整分析流程

        Args:
            logs: 情绪日志列表
            days: 分析天数 (用于过滤)

        Returns:
            AnalyticsResult 数据类
        """
        if not logs or len(logs) == 0:
            return AnalyticsResult(
                attractor=0.0,
                attractor_std=1.0,
                rmssd=0.0,
                current_valence=0.0,
                deviation=0.0,
                smoothed_valence=[],
                intervention_needed=False,
                intervention_type="none",
                risk_level="low"
            )

        # 1. 转换为效价序列
        timestamps, valence_series = self.convert_to_valence_series(logs)

        # 2. 计算吸引子
        attractor, attractor_std = self.calculate_attractor(valence_series)

        # 3. 计算 RMSSD
        rmssd = self.calculate_rmssd(valence_series)

        # 4. 卡尔曼滤波平滑
        if self.kalman_use and len(valence_series) > 3:
            smoothed = self.kalman_filter(valence_series)
        else:
            smoothed = valence_series.tolist()

        # 5. 当前状态
        current_valence = valence_series[-1] if len(valence_series) > 0 else 0.0
        deviation = np.abs(current_valence - attractor) / attractor_std if attractor_std > 0.001 else 0.0

        # 6. 干预决策
        intervention_needed, intervention_type, risk_level = self.detect_intervention_need(
            valence_series, attractor, attractor_std, rmssd
        )

        return AnalyticsResult(
            attractor=round(attractor, 3),
            attractor_std=round(attractor_std, 3),
            rmssd=round(rmssd, 3),
            current_valence=round(current_valence, 3),
            deviation=round(deviation, 2),
            smoothed_valence=[round(v, 3) for v in smoothed],
            intervention_needed=intervention_needed,
            intervention_type=intervention_type,
            risk_level=risk_level
        )

    def get_trend_direction(self, valence_series: np.ndarray, window: int = 5) -> str:
        """
        判断情绪趋势方向

        Args:
            valence_series: 效价时间序列
            window: 用于趋势计算的窗口大小

        Returns:
            'rising' | 'falling' | 'stable'
        """
        if len(valence_series) < window:
            return "stable"

        recent = valence_series[-window:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > 0.05:
            return "rising"
        elif slope < -0.05:
            return "falling"
        else:
            return "stable"

    def calculate_emotion_inertia(
        self,
        valence_series: np.ndarray
    ) -> float:
        """
        计算情绪惯性 - 衡量用户"陷入"某种情绪无法自拔的程度

        计算方法：自相关系数 at lag=1

        Args:
            valence_series: 效价时间序列

        Returns:
            惯性系数 (0~1)，越高表示情绪越难改变
        """
        if len(valence_series) < 3:
            return 0.5

        # 计算 lag-1 自相关
        n = len(valence_series)
        mean = np.mean(valence_series)
        var = np.var(valence_series)

        if var < 0.001:
            return 1.0  # 方差太小，认为情绪完全稳定 (惯性极大)

        # 自相关公式
        autocorr = np.sum(
            (valence_series[:-1] - mean) * (valence_series[1:] - mean)
        ) / ((n - 1) * var)

        # 归一化到 0~1
        inertia = (autocorr + 1) / 2

        return max(0, min(1, inertia))


# ============ 日记情感分析模块 ============

class DiarySentimentAnalyzer:
    """
    日记情感分析器

    用于：
    1. 分析日记内容的情感倾向
    2. 与视觉识别情绪进行闭环校验
    """

    def __init__(self):
        """初始化分析器"""
        # 简单的情感词库 (可扩展为使用 transformers 等模型)
        self.positive_words = {
            '开心', '快乐', '高兴', '幸福', '满足', '温暖', '感动', '兴奋',
            '美好', '喜欢', '爱', '感激', '感谢', '欣慰', '愉悦', '舒畅',
            'good', 'happy', 'great', 'wonderful', 'love', 'thank', 'excited',
            'joy', 'delighted', 'pleased', 'satisfied'
        }

        self.negative_words = {
            '难过', '悲伤', '伤心', '痛苦', '沮丧', '失望', '绝望', '愤怒',
            '生气', '烦躁', '焦虑', '害怕', '恐惧', '孤独', '无助', '疲惫',
            'sad', 'angry', 'depressed', 'anxious', 'scared', 'lonely', 'tired',
            'upset', 'frustrated', 'disappointed', 'hurt', 'pain'
        }

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        分析文本情感

        Args:
            text: 日记内容

        Returns:
            (sentiment_label, confidence) 元组
            sentiment_label: 'positive' | 'negative' | 'neutral'
        """
        text_lower = text.lower()

        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)

        total = pos_count + neg_count

        if total == 0:
            return "neutral", 0.5

        pos_ratio = pos_count / total

        if pos_ratio > 0.6:
            return "positive", pos_ratio
        elif pos_ratio < 0.4:
            return "negative", 1 - pos_ratio
        else:
            return "neutral", 0.5 + abs(pos_ratio - 0.5)

    def diary_to_valence(self, sentiment: str, confidence: float) -> float:
        """
        将情感分析结果转换为效价值

        Args:
            sentiment: 情感标签
            confidence: 置信度

        Returns:
            效价值 (-1 ~ 1)
        """
        if sentiment == "positive":
            return confidence
        elif sentiment == "negative":
            return -confidence
        else:
            return 0.0

    def validate_visual_emotion(
        self,
        diary_entries: List[Dict],
        vision_logs: List[Dict],
        tolerance_hours: int = 2
    ) -> Dict:
        """
        校验视觉识别情绪与日记主观情绪的一致性

        Args:
            diary_entries: 日记列表 [{timestamp, content}]
            vision_logs: 视觉识别日志 [{timestamp, emotion, score}]
            tolerance_hours: 时间容差 (小时)

        Returns:
            校验结果字典
        """
        if not diary_entries or not vision_logs:
            return {"correlation": 0.0, "consistency": "insufficient_data"}

        # 分析日记情感
        diary_valence = []
        diary_timestamps = []

        for entry in diary_entries:
            sentiment, confidence = self.analyze_sentiment(entry.get('content', ''))
            v = self.diary_to_valence(sentiment, confidence)
            diary_valence.append(v)
            diary_timestamps.append(entry.get('timestamp', 0))

        # 匹配同一时间段的视觉数据
        vision_valence_matched = []

        tolerance_seconds = tolerance_hours * 3600

        for i, dt in enumerate(diary_timestamps):
            # 找到最接近的视觉记录
            for vl in vision_logs:
                if abs(vl.get('timestamp', 0) - dt) <= tolerance_seconds:
                    v = self.valence_map(vl.get('emotion', 'neutral'))
                    vision_valence_matched.append(v)
                    break

        if len(vision_valence_matched) < 2:
            return {"correlation": 0.0, "consistency": "insufficient_matches"}

        # 计算相关系数
        diary_arr = np.array(diary_valence[:len(vision_valence_matched)])
        vision_arr = np.array(vision_valence_matched)

        if np.std(diary_arr) < 0.001 or np.std(vision_arr) < 0.001:
            correlation = 0.0
        else:
            correlation = np.corrcoef(diary_arr, vision_arr)[0, 1]

        # 处理 NaN
        if np.isnan(correlation):
            correlation = 0.0

        # 一致性判断
        if correlation > 0.7:
            consistency = "high"
            suggestion = "视觉模型可信，可继续保持当前权重"
        elif correlation > 0.3:
            consistency = "medium"
            suggestion = "视觉模型基本可靠，建议定期校准"
        else:
            consistency = "low"
            suggestion = "视觉模型可能存在偏差，建议降低权重或检查环境因素"

        return {
            "correlation": round(correlation, 3),
            "consistency": consistency,
            "suggestion": suggestion,
            "matched_samples": len(vision_valence_matched)
        }
