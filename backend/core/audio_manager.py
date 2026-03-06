import pygame
import edge_tts
import os
import asyncio
import logging
import random
from pathlib import Path
from sqlalchemy.orm import Session
from .models import MusicLibrary, User  # 🔹 修复：相对导入
logger = logging.getLogger("AudioManager")

class AudioManager:
    def __init__(self):
        # 初始化 Pygame 音频混合器
        try:
            pygame.mixer.init()
            logger.info("音频系统初始化成功")
        except Exception as e:
            logger.error(f"音频系统初始化失败: {e}")

        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.current_emotion = None
        self.current_music_path = None # 记录当前播放的文件路径
        self.is_speaking = False 

    def play_music_for_emotion(self, emotion: str, db: Session, username: str = None):
        """
        根据情绪播放背景音乐 (支持多文件随机抽取)
        
        Args:
            emotion: 检测到的情绪标签
            db: 数据库会话
            username: 当前识别到的用户名（用于检索专属资源）
        """
        # 1. 音量管理：如果正在说话，保持低音量
        if self.is_speaking:
            pygame.mixer.music.set_volume(0.2)
        else:
            pygame.mixer.music.set_volume(1.0)

        # 2. 状态检查：如果情绪没变且音乐正在播放，则不中断
        if emotion == self.current_emotion and pygame.mixer.music.get_busy():
            return

        # 3. 检索音乐资源
        music_record = self._get_random_music(emotion, db, username)
        
        if not music_record:
            logger.warning(f"情绪 {emotion} 未找到任何(专属或全局)音乐资源")
            return

        # 4. 播放音乐
        file_path = Path(music_record.filepath)
        if file_path.exists():
            try:
                # 记录状态
                self.current_emotion = emotion
                self.current_music_path = str(file_path)
                
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play(-1) # -1 表示循环播放这首抽中的歌
                logger.info(f"正在播放音乐: [{emotion}] {music_record.title}")
            except Exception as e:
                logger.error(f"播放失败: {e}")
        else:
            logger.error(f"数据库记录的文件不存在: {file_path}")

    def _get_random_music(self, emotion: str, db: Session, username: str = None):
        """核心逻辑：从数据库随机获取一首音乐 (优先级：专属 > 全局)"""
        
        target_music_list = []

        # 1. 尝试获取专属音乐
        if username and username != "Stranger":
            user = db.query(User).filter(User.username == username).first()
            if user:
                target_music_list = db.query(MusicLibrary).filter(
                    MusicLibrary.user_id == user.id,
                    MusicLibrary.emotion_tag == emotion,
                    MusicLibrary.is_active == True
                ).all()

        # 2. 如果专属音乐库为空，则获取全局音乐 (user_id 为空)
        if not target_music_list:
            target_music_list = db.query(MusicLibrary).filter(
                MusicLibrary.user_id == None,
                MusicLibrary.emotion_tag == emotion,
                MusicLibrary.is_active == True
            ).all()

        # 3. 如果找到了资源，随机选一个返回
        if target_music_list:
            return random.choice(target_music_list)
        
        return None

    async def play_comfort_voice(self, text):
        """生成并播放 TTS 语音 (保持逻辑不变)"""
        if not text:
            return

        self.is_speaking = True
        pygame.mixer.music.set_volume(0.1) 

        output_file = self.assets_dir / "tts_output.mp3"
        
        try:
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(str(output_file))
            
            # 暂停背景音乐
            pygame.mixer.music.pause()
            
            # 播放语音
            sound = pygame.mixer.Sound(str(output_file))
            sound.play()
            
            duration = sound.get_length()
            await asyncio.sleep(duration)
            
            # 恢复背景音乐
            pygame.mixer.music.unpause()
            pygame.mixer.music.set_volume(1.0)
            
        except Exception as e:
            logger.error(f"TTS 错误: {e}")
        finally:
            self.is_speaking = False

    def stop(self):
        pygame.mixer.music.stop()
        self.current_emotion = None