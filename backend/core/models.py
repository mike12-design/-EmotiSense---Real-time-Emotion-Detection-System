from sqlalchemy import create_engine, event, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./emotisense.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 15,  # 写锁等待超时（秒）
    },
    pool_size=10,       # 连接池大小
    max_overflow=5,     # 溢出连接数
)

@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """启用 WAL 模式，允许读写并发，大幅降低页面查询延迟"""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 1. 用户表
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user") # admin / user
    face_encoding = Column(JSON, nullable=True) # 存储 DeepFace 的 list
    avatar = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 2. 个体情绪日志表
class EmotionLog(Base):
    __tablename__ = "emotion_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_stranger = Column(Boolean, default=True)
    emotion = Column(String, nullable=False)
    score = Column(Float, nullable=False)

# 3. 系统事件表 (多数决结果)
class SystemEvent(Base):
    __tablename__ = "system_events"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    people_count = Column(Integer)
    vote_result = Column(JSON) # 存储 {'sad':3, 'happy':1}
    final_mood = Column(String)
    action_type = Column(String) # music / tts / none
    resource_id = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String, nullable=True)

# 4. 音乐资源库
class MusicLibrary(Base):
    __tablename__ = "music_library"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    emotion_tag = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

# 5. 安慰话术库
class ComfortScript(Base):
    __tablename__ = "comfort_scripts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) 
    content = Column(Text, nullable=False)
    emotion_tag = Column(String, nullable=False)
# 6. 用户日记表
class Diary(Base):
    __tablename__ = "diaries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True) # 日记标题
    content = Column(Text, nullable=False) # 日记正文
    emotion = Column(String, nullable=True) # 记录时的心情 (如 Happy, Sad)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow) # 记录时间

# 7. 用户隐藏全局资源表（用户删全局资源时只对自己隐藏）
class UserHiddenGlobal(Base):
    __tablename__ = "user_hidden_global"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    item_type = Column(String, nullable=False)  # "script" 或 "music"
    item_id = Column(Integer, nullable=False)   # 全局资源的 id

# 修改 init_db 确保新表被创建
def init_db():
    Base.metadata.create_all(bind=engine)
