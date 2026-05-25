# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

EmotiSense — 实时情绪检测系统。Vue 3 前端 + FastAPI 后端，通过摄像头进行面部情绪识别，融合多模型（DeepFace/HSEmotion/FER/元学习器），经稳定器+动力学引擎平滑后写入 SQLite，并触发音乐/TTS 干预。

## 核心规则

- 使用中文交流
- 优先编辑已有文件，避免新建
- 不做过度抽象，不写无谓注释
- 改动后需验证：前端 `npm run build`，后端 `python -m py_compile app/*.py core/*.py`

## 项目结构概要

```
backend/
├── app/          main.py(FastAPI入口+lifespan), api.py(~50个端点), database.py
├── core/         检测器工厂、融合、稳定器、动力学引擎、高级分析、配置、模型
├── config.yaml   主配置(检测器类型、阈值、权重)
├── models/       YOLO人脸、眼部CNN等预训练模型文件
└── assets/       音乐、背景图、TTS输出
frontend/
└── src/
    ├── router/   角色路由守卫(admin/user), localStorage鉴权
    ├── views/    MonitorMode(实时监控), user/(首页/历史/日记/设置), admin/(用户/资源/分析/日志)
    └── layouts/  AdminLayout, UserLayout
tests/
├── api/          7个API测试文件, pytest+TestClient+依赖覆写+外部模块桩
├── e2e/          Playwright测试(5个spec)
└── conftest.py   测试DB隔离、cv2/deepface等桩
```

## 常用命令

```bash
# 后端
cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 前端
cd frontend && npm run dev

# 测试
pytest tests/api/ -v                    # 全部API测试
cd frontend && npm run test:e2e         # E2E测试
```

## 当前任务

无特定进行中任务。
