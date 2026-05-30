import pygame
import edge_tts
import os
import asyncio
import logging
import random
import subprocess
import platform
from pathlib import Path
from sqlalchemy.orm import Session
from .models import MusicLibrary, User
logger = logging.getLogger("AudioManager")

class AudioManager:
    def __init__(self):
        try:
            pygame.mixer.init()
            logger.info("音频系统初始化成功")
        except Exception as e:
            logger.error(f"音频系统初始化失败: {e}")

        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.current_emotion = None
        self.current_music_path = None
        self.is_speaking = False
        self._music_process = None  # macOS afplay 子进程

    def _is_playing(self):
        """检查音乐是否正在播放"""
        if self._music_process is not None:
            if self._music_process.poll() is None:
                return True
            self._music_process = None
        if pygame.mixer.get_init():
            return pygame.mixer.music.get_busy()
        return False

    def play_music_for_emotion(self, emotion: str, db: Session, username: str = None):
        if self._is_playing():
            return

        music_record = self._get_random_music(emotion, db, username)
        if not music_record:
            logger.warning(f"情绪 {emotion} 未找到任何(专属或全局)音乐资源")
            return

        file_path = Path(music_record.filepath)
        if not file_path.is_absolute():
            file_path = self.assets_dir.parent / file_path
        if not file_path.exists():
            logger.error(f"数据库记录的文件不存在: {file_path}")
            return

        try:
            self.current_emotion = emotion
            self.current_music_path = str(file_path)

            # macOS: 用 afplay 播放，兼容所有音频格式
            if platform.system() == "Darwin":
                self._music_process = subprocess.Popen(
                    ["afplay", str(file_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play(0)

            logger.info(f"正在播放音乐: [{emotion}] {music_record.title}")
        except Exception as e:
            logger.error(f"播放失败: {e}")

    def _get_random_music(self, emotion: str, db: Session, username: str = None):
        """核心逻辑：从数据库随机获取一首音乐 (优先级：专属 > 全局，排除用户隐藏的全局资源)"""
        from .models import UserHiddenGlobal

        logger.info(f"🔍 查找音乐: emotion={emotion}, username={username}")

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
                logger.info(f"🔍 专属音乐查询: user_id={user.id}, found={len(target_music_list)}")
            else:
                logger.warning(f"🔍 用户 {username} 在 users 表中不存在")
        else:
            logger.info(f"🔍 跳过专属查询: username={repr(username)}")

        # 2. 如果专属音乐库为空，则获取全局音乐（排除该用户隐藏的）
        if not target_music_list:
            global_query = db.query(MusicLibrary).filter(
                MusicLibrary.user_id == None,
                MusicLibrary.emotion_tag == emotion,
                MusicLibrary.is_active == True
            )
            target_music_list = global_query.all()
            logger.info(f"🔍 全局音乐查询: found={len(target_music_list)}")

        # 3. 如果找到了资源，随机选一个返回
        if target_music_list:
            return random.choice(target_music_list)

        return None

    async def play_comfort_voice(self, text):
        if not text:
            return

        self.is_speaking = True

        output_file = self.assets_dir / "tts_output.mp3"
        try:
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(str(output_file))

            # macOS: 用 afplay 播 TTS
            if platform.system() == "Darwin":
                proc = subprocess.Popen(
                    ["afplay", str(output_file)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                proc.wait()
            else:
                pygame.mixer.music.pause()
                sound = pygame.mixer.Sound(str(output_file))
                sound.play()
                await asyncio.sleep(sound.get_length())
                pygame.mixer.music.unpause()
        except Exception as e:
            logger.error(f"TTS 错误: {e}")
        finally:
            self.is_speaking = False

    def stop(self):
        if self._music_process is not None:
            try:
                self._music_process.terminate()
                self._music_process.wait(timeout=1)
            except Exception:
                try:
                    self._music_process.kill()
                except Exception:
                    pass
            self._music_process = None
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.current_emotion = None