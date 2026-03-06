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

## 人脸识别系统架构

### 整体流程

```
视频输入 → 人脸检测 → 人脸对齐 → 特征提取 → 情绪分类 → 融合决策 → 输出结果
              ↓            ↓           ↓
         YOLOv8      关键点定位    全局/局部特征
                                    ↓
                              元学习器融合
```

### 核心模块

#### 1. 人脸检测与对齐
- **检测器**: YOLOv8-Face 或 Haar Cascade
- **关键点定位**: 5 点定位（双眼、鼻尖、嘴角）
- **对齐方式**: 基于关键点仿射变换标准化

#### 2. 全局特征提取（HSEmotion）
- **骨干网络**: EfficientNet-B0
- **输入尺寸**: 48×48 灰度人脸
- **输出**: 7 类情绪概率分布 (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **预训练**: AffectNet 数据集 + OAHEGA 微调

#### 3. 局部特征提取（眼部模型）
- **模型架构**: 自定义 CNN (4 层卷积 + 2 层全连接)
- **输入尺寸**: 224×224 眼部区域裁剪
- **输出**: 2 类概率 (Sad, Neutral)
- **专长**: 悲伤微表情识别（参考 Gorbova 等，2019）

#### 4. 元学习器融合模块
- **元分类器**: Random Forest (100 棵树) / Logistic Regression
- **输入特征**: [全局概率 7 维，局部概率 2 维，遮挡标志位]
- **输出**: 最终情绪类别
- **训练方式**: 在 OAHEGA 验证集上进行 Stacking 训练

## 融合决策原理

### Meta-Learner Fusion 架构

本系统采用基于 Stacking 策略的决策级融合架构，通过元学习器整合全局人脸模型（HSEmotion）和局部眼部模型 的预测结果，实现比单一模型更准确的情绪识别。

```
┌─────────────────────────────────────────────────────────────┐
│                    输入人脸图像                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │ 全局模型 │              │ 局部模型 │
    │HSEmotion│              │ 眼部 CNN │
    └────┬────┘              └────┬────┘
         │                         │
    ┌────▼────────────────────────▼────┐
    │      特征拼接层                   │
    │ [p_happy, p_sad, p_angry,        │
    │  p_fear, p_surprise, p_disgust,  │
    │  p_neutral, eye_sad, eye_neutral]│
    └─────────────┬────────────────────┘
                  │
         ┌────────▼────────┐
         │  元分类器        │
         │ (LogisticRegression│
         │  / Random Forest) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   最终输出       │
         │  (7 类情绪)      │
         └─────────────────┘
```

### 四大核心创新机制

#### 1. 感知遮挡的自适应退避机制 (Occlusion-Aware Adaptive Fallback)

**问题**: 眼部被头发、眼镜或手遮挡时，局部模型预测可能产生噪声。

**解决方案**:
```python
# 伪代码示例
if eye_occlusion_ratio > 0.5:
    # 高遮挡：完全退避至全局模型
    final_prediction = global_model_prediction
else:
    # 低遮挡：送入元学习器融合
    final_prediction = meta_learner.predict(combined_features)
```

**技术要点**:
- 基于 YOLOv8 关键点检测置信度评估遮挡程度
- 动态调整融合权重，遮挡严重时自动降级为全局模型
- 保证系统鲁棒性，避免局部特征误导整体决策

#### 2. 非对称置信度门控 (Asymmetric Confidence Gating)

**问题**: 元分类器可能过度推翻基线模型在高置信度样本上的正确预测（"灾难性推翻"）。

**解决方案**:
- 当全局模型对某类别置信度 >95% 时，直接输出该类别，跳过元学习器
- 仅在中低置信度区间（<80%）激活元学习器进行二次仲裁
- 高置信度样本采用"直通模式"，保护基线模型的优势判断

**效果**: 在 Happy 等高置信度类别上，召回率稳定在 0.94，融合不破坏基线优势。

#### 3. 动态触发边界扩展 (Dynamic Trigger Boundary)

**问题**: 传统阈值法（如 50%）会漏掉潜在情绪（如 26% 的悲伤信号可能是真悲伤）。

**解决方案**:
- 将触发边界扩展至 25%：任何情绪概率 >25% 即唤醒元学习器
- 元学习器接收完整软概率分布，而非硬阈值判断
- 增强对微弱情绪信号的敏感度

**效果**: Neutral 召回率从 0.60 跃升至 0.66（+6%），精准纠正"面无表情"误判为悲伤。

#### 4. 基于软概率的领域适应重塑 (Domain Adaptation via Soft Probabilities)

**问题**: 规则融合（如 k=0.5 加权平均）无法捕捉复杂的类别间依赖关系。

**解决方案**:
- 使用 Random Forest / Logistic Regression 作为元分类器
- 输入为基线模型的**软概率输出**（非硬标签）
- 在 OAHEGA 验证集上重新训练元分类器，拟合新基座模型的决策边界

**训练流程**:
1. 用 HSEmotion 和眼部 CNN 在验证集上生成软概率
2. 拼接为特征向量，训练元分类器
3. 元分类器学习"何时相信全局"、"何时相信局部"的非线性规则

**效果**: 整体准确率从 71.94% 提升至 72.80%（+0.86%），Macro-F1 从 74.07% 提升至 74.73%（+0.66%）。

### 融合决策示例

**场景 1: 识别"假笑"（表面快乐，真实悲伤）**
- 全局模型：Happy 85%, Sad 10%
- 眼部模型：Sad 75%
- 元学习器输出：Sad 60%, Happy 35% ← 捕捉到眼部悲伤微表情

**场景 2: 识别"面无表情"（真中性，非悲伤）**
- 全局模型：Neutral 70%, Sad 25%
- 眼部模型：Neutral 80%
- 元学习器输出：Neutral 82% ← 局部特征强化中性判断

**场景 3: 遮挡场景（手遮住眼睛）**
- 全局模型：Angry 60%
- 眼部模型：检测失败（遮挡）
- 自适应退避：直接采用全局模型输出 Angry 60%

### 实验验证

在 **8,539 样本** OAHEGA 测试集上的表现：

| 情绪 | 指标 | HSEmotion 单体 | Meta-Learner Fusion | 提升 |
|------|------|----------------|---------------------|------|
| **Neutral** | 召回率 | 0.60 | **0.66** | +6% |
| **Angry** | 精确率 | 0.69 | **0.72** | +3% |
| **Sad** | 精确率 | 0.87 | **0.87** | 持平 |
| **Happy** | 召回率 | 0.94 | **0.94** | 持平 |
| **总体** | Accuracy | 71.94% | **72.80%** | +0.86% |
| **总体** | Macro-F1 | 74.07% | **74.73%** | +0.66% |

**结论**: Meta-Learner Fusion 在保持优势类别（Happy, Sad）的同时，显著提升弱势类别（Neutral, Angry）的识别能力，实现全面优化。

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

## Face Recognition System Architecture

### Overall Pipeline

```
Video Input → Face Detection → Face Alignment → Feature Extraction → Emotion Classification → Fusion Decision → Output
                 ↓                ↓                  ↓
            YOLOv8          Landmarks       Global/Local Features
                                                 ↓
                                         Meta-Learner Fusion
```

### Core Modules

#### 1. Face Detection & Alignment
- **Detector**: YOLOv8-Face or Haar Cascade
- **Landmark Localization**: 5-point localization (left eye, right eye, nose tip, left mouth corner, right mouth corner)
- **Alignment**: Affine transformation based on landmarks for standardization

#### 2. Global Feature Extraction (HSEmotion)
- **Backbone**: EfficientNet-B0
- **Input Size**: 48×48 grayscale face
- **Output**: 7-class emotion probability distribution (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Pretraining**: AffectNet dataset + OAHEGA fine-tuning

#### 3. Local Feature Extraction (Eye Region Model)
- **Model Architecture**: Custom CNN (4 convolutional layers + 2 fully connected layers)
- **Input Size**: 224×224 eye region crop
- **Output**: 2-class probability (Sad, Neutral)
- **Specialty**: Sad micro-expression recognition (inspired by Gorbova et al., 2019)

#### 4. Meta-Learner Fusion Module
- **Meta-Classifier**: Random Forest (100 trees) / Logistic Regression
- **Input Features**: [Global probabilities (7D), Local probabilities (2D), Occlusion flag]
- **Output**: Final emotion category
- **Training**: Stacking training on OAHEGA validation set

## Fusion Decision-Making Principles

### Meta-Learner Fusion Architecture

This system employs a decision-level fusion architecture based on Stacking strategy, integrating predictions from the global face model (HSEmotion) and local eye region model through a meta-learner, achieving more accurate emotion recognition than single models.

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Face Image                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │ Global  │              │  Local  │
    │  Model  │              │  Model  │
    │HSEmotion│              │ Eye CNN │
    └────┬────┘              └────┬────┘
         │                         │
    ┌────▼────────────────────────▼────┐
    │      Feature Concatenation        │
    │ [p_happy, p_sad, p_angry,        │
    │  p_fear, p_surprise, p_disgust,  │
    │  p_neutral, eye_sad, eye_neutral]│
    └─────────────┬────────────────────┘
                  │
         ┌────────▼────────┐
         │  Meta-Classifier │
         │ (LogisticRegression│
         │  / Random Forest) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Final Output   │
         │  (7 Emotions)    │
         └─────────────────┘
```

### Four Core Innovation Mechanisms

#### 1. Occlusion-Aware Adaptive Fallback

**Problem**: When eyes are occluded by hair, glasses, or hands, local model predictions may introduce noise.

**Solution**:
```python
# Pseudocode example
if eye_occlusion_ratio > 0.5:
    # High occlusion: fallback to global model only
    final_prediction = global_model_prediction
else:
    # Low occlusion: send to meta-learner for fusion
    final_prediction = meta_learner.predict(combined_features)
```

**Technical Details**:
- Evaluate occlusion level based on YOLOv8 landmark detection confidence
- Dynamically adjust fusion weights, automatically degrade to global model under severe occlusion
- Ensure system robustness, prevent local features from misleading overall decision

#### 2. Asymmetric Confidence Gating

**Problem**: Meta-classifier may overly override correct predictions from baseline model on high-confidence samples ("catastrophic overriding").

**Solution**:
- When global model confidence for a category >95%, directly output that category, bypass meta-learner
- Activate meta-learner for secondary arbitration only in medium-low confidence range (<80%)
- Use "direct pass mode" for high-confidence samples, protecting baseline model's advantageous judgments

**Effect**: On high-confidence categories like Happy, recall rate remains stable at 0.94, fusion does not destroy baseline advantages.

#### 3. Dynamic Trigger Boundary Extension

**Problem**: Traditional threshold methods (e.g., 50%) miss potential emotions (e.g., 26% sadness signal might be genuine sadness).

**Solution**:
- Extend trigger boundary to 25%: any emotion with probability >25% activates meta-learner
- Meta-learner receives complete soft probability distribution, not hard threshold decisions
- Enhance sensitivity to weak emotional signals

**Effect**: Neutral recall rate jumped from 0.60 to 0.66 (+6%), accurately correcting "expressionless" misclassified as sadness.

#### 4. Domain Adaptation via Soft Probabilities

**Problem**: Rule-based fusion (e.g., k=0.5 weighted average) cannot capture complex inter-class dependencies.

**Solution**:
- Use Random Forest / Logistic Regression as meta-classifier
- Input is **soft probability output** from baseline models (not hard labels)
- Retrain meta-classifier on OAHEGA validation set to fit decision boundaries of new base models

**Training Pipeline**:
1. Generate soft probabilities using HSEmotion and Eye CNN on validation set
2. Concatenate as feature vectors, train meta-classifier
3. Meta-classifier learns non-linear rules for "when to trust global" and "when to trust local"

**Effect**: Overall accuracy improved from 71.94% to 72.80% (+0.86%), Macro-F1 from 74.07% to 74.73% (+0.66%).

### Fusion Decision Examples

**Scenario 1: Recognizing "Fake Smile" (surface happiness, underlying sadness)**
- Global Model: Happy 85%, Sad 10%
- Eye Model: Sad 75%
- Meta-Learner Output: Sad 60%, Happy 35% ← Captures sad micro-expression in eyes

**Scenario 2: Recognizing "Expressionless" (true neutral, not sadness)**
- Global Model: Neutral 70%, Sad 25%
- Eye Model: Neutral 80%
- Meta-Learner Output: Neutral 82% ← Local features reinforce neutral judgment

**Scenario 3: Occlusion Scenario (hand covering eyes)**
- Global Model: Angry 60%
- Eye Model: Detection failed (occluded)
- Adaptive Fallback: Directly adopt global model output Angry 60%

### Experimental Validation

Performance on **8,539 samples** OAHEGA test set:

| Emotion | Metric | HSEmotion Single | Meta-Learner Fusion | Improvement |
|---------|--------|------------------|---------------------|-------------|
| **Neutral** | Recall | 0.60 | **0.66** | +6% |
| **Angry** | Precision | 0.69 | **0.72** | +3% |
| **Sad** | Precision | 0.87 | **0.87** |持平 |
| **Happy** | Recall | 0.94 | **0.94** |持平 |
| **Overall** | Accuracy | 71.94% | **72.80%** | +0.86% |
| **Overall** | Macro-F1 | 74.07% | **74.73%** | +0.66% |

**Conclusion**: Meta-Learner Fusion significantly improves recognition on weak categories (Neutral, Angry) while maintaining advantages on strong categories (Happy, Sad), achieving comprehensive optimization.

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
