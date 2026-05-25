"""
生成有情绪变化趋势的测试数据

模拟真实的情绪波动：有起伏、渐变、持续低谷/高峰等模式。
"""

import math
import random
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import SessionLocal
from core.models import EmotionLog, User

# 情绪列表（按效价排序）
EMOTIONS_POSITIVE = ['happy', 'surprise']
EMOTIONS_NEGATIVE = ['sad', 'fear', 'angry', 'disgust']
EMOTIONS_NEUTRAL = ['neutral']
ALL_EMOTIONS = EMOTIONS_POSITIVE + EMOTIONS_NEUTRAL + EMOTIONS_NEGATIVE


def generate_trending_logs(
    user_id: int,
    total_logs: int = 100,
    base_time: datetime = None
):
    """
    生成有趋势的情绪日志

    模拟情绪随时间的自然变化：
    - 使用正弦波 + 噪声模拟情绪波动
    - 包含渐升、渐降、低谷持续、高峰等阶段
    """
    if base_time is None:
        base_time = datetime.now() - timedelta(hours=total_logs // 6)

    logs = []
    random.seed(42)

    # 使用复合正弦函数模拟情绪趋势
    for i in range(total_logs):
        # 主周期（慢波动）+ 次周期（快波动）+ 噪声
        main_cycle = math.sin(2 * math.pi * i / 30)      # 30条一个完整周期
        sub_cycle = 0.5 * math.sin(2 * math.pi * i / 8)   # 8条一个小波动
        noise = random.uniform(-0.3, 0.3)

        valence = main_cycle + sub_cycle + noise
        # 归一化到 [-1, 1]
        valence = max(-1.0, min(1.0, valence))

        # 根据效价值选择情绪
        if valence > 0.4:
            emotion = random.choice(EMOTIONS_POSITIVE)
        elif valence > 0.1:
            emotion = 'surprise' if random.random() > 0.5 else random.choice(EMOTIONS_POSITIVE)
        elif valence > -0.1:
            emotion = 'neutral'
        elif valence > -0.5:
            emotion = 'sad'
        elif valence > -0.8:
            emotion = random.choice(['fear', 'sad'])
        else:
            emotion = random.choice(EMOTIONS_NEGATIVE)

        # 置信度：情绪越极端越高，存 50-100（API 内部会除以 100）
        score = 50 + abs(valence) * 40 + random.uniform(0, 10)
        score = round(min(100, score), 2)

        # 时间间隔：10分钟一条
        timestamp = base_time + timedelta(minutes=i * 10)

        log = EmotionLog(
            timestamp=timestamp,
            user_id=user_id,
            is_stranger=False,
            emotion=emotion,
            score=score
        )
        logs.append(log)

    return logs


def generate_distress_episode(user_id: int, base_time: datetime = None):
    """
    生成一段"持续负面情绪"的片段（模拟需要干预的场景）

    情绪从正常逐渐滑向低谷，持续一段时间
    """
    if base_time is None:
        base_time = datetime.now()

    logs = []
    distress_sequence = [
        # 情绪逐渐恶化
        ('neutral', 65),
        ('neutral', 60),
        ('sad', 55),
        ('sad', 58),
        ('sad', 62),
        ('fear', 65),
        ('sad', 70),
        ('angry', 72),
        ('sad', 75),
        ('fear', 78),
        ('angry', 80),
        ('sad', 82),
        # 低谷持续
        ('sad', 85),
        ('sad', 83),
        ('fear', 80),
        ('sad', 88),
        ('angry', 85),
        # 开始恢复
        ('sad', 75),
        ('neutral', 68),
        ('neutral', 62),
        ('happy', 55),
        ('happy', 60),
    ]

    for i, (emotion, score) in enumerate(distress_sequence):
        timestamp = base_time + timedelta(minutes=i * 10)
        log = EmotionLog(
            timestamp=timestamp,
            user_id=user_id,
            is_stranger=False,
            emotion=emotion,
            score=score
        )
        logs.append(log)

    return logs


def main():
    db = SessionLocal()

    # 获取或创建测试用户
    user = db.query(User).filter(User.id == 1).first()
    if not user:
        user = db.query(User).filter(User.username == 'admin').first()
    if not user:
        print("❌ 未找到用户，请先创建用户")
        db.close()
        return

    user_id = user.id
    username = user.username
    print(f"👤 为用户 '{username}' (id={user_id}) 生成数据...\n")

    # 1. 波动趋势数据
    print("📈 生成情绪波动趋势数据（100条）...")
    trending_logs = generate_trending_logs(user_id, total_logs=100)
    db.add_all(trending_logs)
    print(f"  ✅ 已添加 {len(trending_logs)} 条")

    # 2. 持续负面情绪片段（模拟需要干预）
    print("\n😰 生成持续负面情绪片段（22条）...")
    distress_time = datetime.now() - timedelta(hours=2)
    distress_logs = generate_distress_episode(user_id, base_time=distress_time)
    db.add_all(distress_logs)
    print(f"  ✅ 已添加 {len(distress_logs)} 条")

    # 3. 快乐片段
    print("\n😊 生成快乐情绪片段（15条）...")
    happy_time = datetime.now() - timedelta(hours=1)
    happy_logs = []
    for i in range(15):
        emotion = random.choice(['happy', 'happy', 'happy', 'surprise'])
        score = round(70 + random.uniform(0, 30), 2)
        timestamp = happy_time + timedelta(minutes=i * 10)
        happy_logs.append(EmotionLog(
            timestamp=timestamp,
            user_id=user_id,
            is_stranger=False,
            emotion=emotion,
            score=score
        ))
    db.add_all(happy_logs)
    print(f"  ✅ 已添加 {len(happy_logs)} 条")

    db.commit()
    total = len(trending_logs) + len(distress_logs) + len(happy_logs)
    db.close()

    print(f"\n{'='*50}")
    print(f"✅ 共生成 {total} 条情绪日志！")
    print(f"   - 情绪波动趋势：{len(trending_logs)} 条")
    print(f"   - 持续负面片段：{len(distress_logs)} 条（含低谷恢复）")
    print(f"   - 快乐片段：{len(happy_logs)} 条")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
