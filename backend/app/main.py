# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager # 导入 lifespan 必需的装饰器
import os
import sys
from pathlib import Path
import logging

# 确保能导入 core
sys.path.append(str(Path(__file__).parent.parent))

from core.models import init_db
from app.api import router as api_router
import app.api as api_module 

# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotiSenseAPI")

# --- 1. 定义 Lifespan (生命周期管理器) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理系统的整个生命周期：启动 -> 运行 -> 关闭
    """
    # 【启动阶段】: 相当于以前的 startup_event
    logger.info("🚀 [Startup] 系统启动中...")
    
    # 初始化数据库
    try:
        init_db()
        logger.info("✅ 数据库架构已就绪")
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")

    # 加载 AI 组件
    try:
        await api_module.init_components()
        logger.info("✅ AI 组件加载完成")
    except Exception as e:
        logger.error(f"❌ 组件加载失败: {e}")

    # --------------------------------------------------
    yield  # 此时应用正在运行（在这之前的代码是启动逻辑，之后的代码是关闭逻辑）
    # --------------------------------------------------

    # 【关闭阶段】: 相当于以前的 shutdown_event
    logger.info("👋 [Shutdown] 正在关闭服务并释放资源...")
    if api_module.audio_manager:
        api_module.audio_manager.stop()
    logger.info("✅ 服务已安全停止")


# --- 2. 创建 FastAPI 实例并注入 lifespan ---
app = FastAPI(
    title="EmotiSense API",
    lifespan=lifespan  # 在这里绑定生命周期处理器
)

# 3. CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 静态文件挂载
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_path = os.path.join(BASE_DIR, "assets")
if not os.path.exists(assets_path):
    os.makedirs(assets_path)
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# 5. 路由
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "EmotiSense Backend is Running!"}