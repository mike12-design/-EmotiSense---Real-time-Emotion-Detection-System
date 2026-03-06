# EmotiSense - 实时情绪检测系统

一个基于计算机视觉和深度学习的多模态情绪识别系统，能够实时分析人脸情绪并提供音乐、语音等干预反馈。

## 项目架构

```
EmotiSense/
├── backend/           # FastAPI 后端服务
│   ├── app/          # API 路由和应用逻辑
│   ├── core/         # 核心检测算法
│   ├── models/       # 预训练情绪模型
│   ├── weights/      # 模型权重文件
│   └── config.yaml   # 配置文件
├── frontend/         # Vue 3 前端应用
│   ├── src/
│   │   ├── views/    # 页面组件
│   │   ├── layouts/  # 布局组件
│   │   └── assets/   # 静态资源
│   └── dist/         # 构建输出
└── weights/          # 额外模型权重
```

## 核心功能

### 情绪检测
- **多模型融合**: 支持 DeepFace、HSEmotion、FER 等多种情绪识别模型
- **元学习器融合**: 基于 Stacking 分类器的决策级融合，提升识别准确率
- **眼部微表情分析**: 独立的眼部区域情绪检测，增强悲伤等微表情识别
- **实时视频流处理**: 基于 OpenCV 的实时人脸检测和情绪分析

### 用户系统
- 用户注册/登录与人脸识别
- 个人情绪日志记录
- 日记闭环反馈系统

### 干预系统
- **音乐播放**: 根据情绪状态自动播放对应类型的音乐
- **语音安慰**: 播放预设的安慰话术
- **动力学模型**: 基于情绪变化的干预决策算法

### 数据管理
- SQLite 数据库存储
- 情绪日志、系统事件、音乐库、安慰话术管理
- 数据可视化分析

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | Python, FastAPI, OpenCV, SQLAlchemy |
| 前端 | Vue 3, Element Plus, ECharts, Axios |
| 深度学习 | PyTorch, DeepFace, YOLOv8, HSEmotion |
| 数据库 | SQLite |
| 模型融合 | Scikit-learn (Logistic Regression) |

## 安装指南

### 环境要求
- Python 3.8+
- Node.js 20.19+ 或 22.12+
- CUDA (可选，用于 GPU 加速)

### 后端安装

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### 前端安装

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build
```

### 训练元学习器（可选）

```bash
cd backend
python train_meta_learner.py
```

## 配置说明

编辑 `backend/config.yaml` 配置文件：

```yaml
# 视频采集设置
video:
  camera_index: 0
  frame_width: 640
  frame_height: 360

# 情绪检测设置
emotion:
  detector_type: 'meta_learner'  # deepface / decision_fusion / meta_learner
  use_meta_learner: true
  decision_fusion_k: 0.5  # 眼睛模型权重
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/video/start` | POST | 启动视频流 |
| `/api/video/stop` | POST | 停止视频流 |
| `/api/emotion/analyze` | POST | 上传图片进行情绪分析 |
| `/api/user/register` | POST | 用户注册 |
| `/api/user/login` | POST | 用户登录 |
| `/api/diary` | POST/GET | 日记管理 |
| `/api/music` | GET/POST | 音乐库管理 |
| `/api/analytics` | GET | 情绪数据分析 |

## 情绪类别

系统支持以下 7 种基本情绪识别：
- 快乐 (Happy)
- 悲伤 (Sad)
- 愤怒 (Angry)
- 恐惧 (Fear)
- 惊讶 (Surprise)
- 厌恶 (Disgust)
- 中性 (Neutral)

## 数据集

### OAHEGA Emotion Recognition Dataset

本项目使用的情绪识别数据集包含 6 种不同情绪类别：Happy、Angry、Sad、Neutral、Surprise 和 Ahegao。

**数据集特点：**
- 图像格式：RGB 人脸裁剪图像
- 数据来源：Facebook、Instagram 社交网络爬取，YouTube 视频，以及 IMDB、AffectNet 等公开数据集
- 数据组织：按情绪类别分文件夹存储，附带 `data.csv` 包含图像路径和标签

**引用：**
```
Kovenko, Volodymyr; Shevchuk, Vitalii (2021), "OAHEGA : EMOTION RECOGNITION DATASET", Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
```

## 实验结果

### 元学习器融合模型性能评估

在 **8,539 个样本** 的大规模混合情绪数据集上进行的对比测试结果：

**数据分布：** Angry (1306), Happy (2000), Neutral (2000), Sad (2000), Surprise (1233)

| 模型架构 | 总体准确率 (Accuracy) | 宏平均 F1 (Macro-F1) |
|----------|----------------------|---------------------|
| DeepFace 单体 (经典基准) | 30.10% | 35.10% |
| HSEmotion 单体 (SOTA 基线) | 71.94% | 74.07% |
| **Meta-Learner Fusion (本文提出)** | **72.80%** (+0.86%) | **74.73%** (+0.66%) |

### 核心创新机制

1. **感知遮挡的自适应退避机制** - 动态评估眼部局部特征有效性，高遮挡时自动退避至全局模型
2. **非对称置信度门控** - 防止元分类器在优势类别上产生"灾难性推翻"
3. **动态触发边界扩展** - 潜在概率 >25% 即唤醒元分类器进行二次仲裁
4. **基于软概率的领域适应重塑** - 重新训练 Random Forest 元分类器拟合新基座模型决策边界

### 细粒度情绪分析

| 情绪 | 突破点 |
|------|--------|
| **中性 (Neutral)** | 召回率从 0.60 跃升至 0.66（提升 6%），精准纠正"面无表情"误判为悲伤 |
| **愤怒 (Angry)** | 精确率从 0.69 提升至 0.72，中性情绪精准识别带来溢出红利 |
| **悲伤 (Sad)** | 精确率维持 0.87 极高水准，低误报保证干预决策可靠性 |
| **快乐 (Happy)** | 召回率稳定在 0.94，非破坏性融合保留基线优势 |

## 项目特色

1. **多模态融合**: 结合全局人脸和眼部区域特征，提升识别准确率
2. **可配置架构**: 支持多种检测器切换，适应不同场景需求
3. **闭环反馈**: 日记系统用于校准和优化情绪识别参数
4. **实时干预**: 基于情绪状态自动触发音乐、语音等干预措施
5. **数据可视化**: ECharts 图表展示情绪趋势和分析报告

## 许可证

MIT License

## 致谢

- [DeepFace](https://github.com/serengil/deepface)
- [HSEmotion](https://github.com/HSEmotion/emotion)
- [FER](https://github.com/oarriaga/face_classification)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- **学术论文参考**:
  - Gorbova, J., Colovic, M., Marjanovic, M., Njegus, A., & Anbarjafari, G. (2019). Going deeper in hidden sadness recognition using spontaneous micro-expressions database. *Springer Science+Business Media*.

---

---

# EmotiSense - Real-time Emotion Detection System

A multimodal emotion recognition system based on computer vision and deep learning, capable of real-time facial emotion analysis and providing intervention feedback such as music and voice.

## Project Architecture

```
EmotiSense/
├── backend/           # FastAPI Backend Service
│   ├── app/          # API Routes and Application Logic
│   ├── core/         # Core Detection Algorithms
│   ├── models/       # Pre-trained Emotion Models
│   ├── weights/      # Model Weight Files
│   └── config.yaml   # Configuration File
├── frontend/         # Vue 3 Frontend Application
│   ├── src/
│   │   ├── views/    # Page Components
│   │   ├── layouts/  # Layout Components
│   │   └── assets/   # Static Assets
│   └── dist/         # Build Output
└── weights/          # Additional Model Weights
```

## Core Features

### Emotion Detection
- **Multi-Model Fusion**: Supports DeepFace, HSEmotion, FER and other emotion recognition models
- **Meta-Learner Fusion**: Decision-level fusion based on Stacking classifier for improved accuracy
- **Eye Micro-expression Analysis**: Independent eye region emotion detection for enhanced sadness and micro-expression recognition
- **Real-time Video Processing**: Real-time face detection and emotion analysis based on OpenCV

### User System
- User registration/login with face recognition
- Personal emotion log recording
- Diary closed-loop feedback system

### Intervention System
- **Music Playback**: Automatically plays corresponding music based on emotional state
- **Voice Comfort**: Plays preset comfort scripts
- **Kinetic Model**: Intervention decision algorithm based on emotional changes

### Data Management
- SQLite database storage
- Emotion logs, system events, music library, comfort scripts management
- Data visualization and analytics

## Tech Stack

| Module | Technology |
|------|------|
| Backend | Python, FastAPI, OpenCV, SQLAlchemy |
| Frontend | Vue 3, Element Plus, ECharts, Axios |
| Deep Learning | PyTorch, DeepFace, YOLOv8, HSEmotion |
| Database | SQLite |
| Model Fusion | Scikit-learn (Logistic Regression) |

## Installation Guide

### Requirements
- Python 3.8+
- Node.js 20.19+ or 22.12+
- CUDA (Optional, for GPU acceleration)

### Backend Installation

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start service
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Installation

```bash
cd frontend

# Install dependencies
npm install

# Development mode
npm run dev

# Production build
npm run build
```

### Train Meta-Learner (Optional)

```bash
cd backend
python train_meta_learner.py
```

## Configuration

Edit `backend/config.yaml` configuration file:

```yaml
# Video Capture Settings
video:
  camera_index: 0
  frame_width: 640
  frame_height: 360

# Emotion Detection Settings
emotion:
  detector_type: 'meta_learner'  # deepface / decision_fusion / meta_learner
  use_meta_learner: true
  decision_fusion_k: 0.5  # Eye model weight
```

## API Endpoints

| Endpoint | Method | Description |
|------|------|------|
| `/api/video/start` | POST | Start video stream |
| `/api/video/stop` | POST | Stop video stream |
| `/api/emotion/analyze` | POST | Upload image for emotion analysis |
| `/api/user/register` | POST | User registration |
| `/api/user/login` | POST | User login |
| `/api/diary` | POST/GET | Diary management |
| `/api/music` | GET/POST | Music library management |
| `/api/analytics` | GET | Emotion data analytics |

## Emotion Categories

The system supports the following 7 basic emotion recognition:
- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

## Dataset

### OAHEGA Emotion Recognition Dataset

The emotion recognition dataset used in this project contains 6 distinct emotion categories: Happy, Angry, Sad, Neutral, Surprise, and Ahegao.

**Dataset Features:**
- Image Format: RGB cropped face images
- Data Sources: Scraped from Facebook, Instagram, YouTube videos, and public datasets like IMDB and AffectNet
- Organization: Images stored in folders by emotion category, with `data.csv` containing image paths and labels

**Citation:**
```
Kovenko, Volodymyr; Shevchuk, Vitalii (2021), "OAHEGA : EMOTION RECOGNITION DATASET", Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
```

## Experimental Results

### Meta-Learner Fusion Model Performance Evaluation

Comparative testing results on a large-scale mixed emotion dataset with **8,539 samples**:

**Data Distribution:** Angry (1306), Happy (2000), Neutral (2000), Sad (2000), Surprise (1233)

| Model Architecture | Accuracy | Macro-F1 |
|----------|----------|----------|
| DeepFace Single Model (Classic Baseline) | 30.10% | 35.10% |
| HSEmotion Single Model (SOTA Baseline) | 71.94% | 74.07% |
| **Meta-Learner Fusion (Proposed)** | **72.80%** (+0.86%) | **74.73%** (+0.66%) |

### Core Innovation Mechanisms

1. **Occlusion-Aware Adaptive Fallback** - Dynamically evaluates local eye feature validity, automatically falls back to global model under high occlusion
2. **Asymmetric Confidence Gating** - Prevents "catastrophic overriding" by meta-classifier on dominant categories
3. **Dynamic Trigger Boundary** - Activates meta-classifier for secondary arbitration when potential probability >25%
4. **Domain Adaptation via Soft Probabilities** - Retrains Random Forest meta-classifier to fit new base model decision boundaries

### Fine-Grained Emotion Analysis

| Emotion | Breakthrough |
|---------|--------------|
| **Neutral** | Recall increased from 0.60 to 0.66 (+6%), accurately correcting "expressionless" misclassified as sadness |
| **Angry** | Precision improved from 0.69 to 0.72, spillover benefit from precise neutral recognition |
| **Sad** | Precision maintained at 0.87, low false positive rate ensures reliable intervention decisions |
| **Happy** | Recall stable at 0.94, non-destructive fusion preserves baseline advantages |

## Features Highlights

1. **Multi-Modal Fusion**: Combines global face and eye region features for improved accuracy
2. **Configurable Architecture**: Supports multiple detector switching for different scenarios
3. **Closed-Loop Feedback**: Diary system for calibrating and optimizing emotion recognition parameters
4. **Real-Time Intervention**: Automatically triggers music, voice interventions based on emotional state
5. **Data Visualization**: ECharts charts for emotion trends and analytics reports

## License

MIT License

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface)
- [HSEmotion](https://github.com/HSEmotion/emotion)
- [FER](https://github.com/oarriaga/face_classification)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Academic Paper Reference**:
  - Gorbova, J., Colovic, M., Marjanovic, M., Njegus, A., & Anbarjafari, G. (2019). Going deeper in hidden sadness recognition using spontaneous micro-expressions database. *Springer Science+Business Media*.
