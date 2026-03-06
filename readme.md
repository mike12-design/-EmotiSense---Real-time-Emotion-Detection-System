
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
- **Multi-Model Fusion**: Supports DeepFace, HSEmotion, FER and other emotion recognition models.
- **Meta-Learner Fusion**: Decision-level fusion based on Stacking classifier for improved accuracy.
- **Eye Micro-expression Analysis**: Independent eye region emotion detection for enhanced sadness and micro-expression recognition.
- **Real-time Video Processing**: Real-time face detection and emotion analysis based on OpenCV.

### User System
- User registration/login with face recognition.
- Personal emotion log recording.
- Diary closed-loop feedback system.

### Intervention System
- **Music Playback**: Automatically plays corresponding music based on emotional state.
- **Voice Comfort**: Plays preset comfort scripts.
- **Kinetic Model**: Intervention decision algorithm based on emotional changes.

### Data Management
- SQLite database storage.
- Emotion logs, system events, music library, comfort scripts management.
- Data visualization and analytics.

## Tech Stack

| Module | Technology |
|------|------|
| Backend | Python, FastAPI, OpenCV, SQLAlchemy |
| Frontend | Vue 3, Element Plus, ECharts, Axios |
| Deep Learning | PyTorch, DeepFace, YOLOv8, HSEmotion |
| Database | SQLite |
| Model Fusion | Scikit-learn (Random Forest) |

## Face Recognition System Architecture

### Overall Pipeline

```
Video Input → Face Detection → Eye Region Detection → Feature Extraction → Emotion Classification → Fusion Decision → Output
                 ↓                   ↓                     ↓
            YOLOv8            Haar Cascade        Global/Local Features
                                                           ↓
                                                   Meta-Learner Fusion
```

### Core Modules

#### 1. Face Detection & Eye Localization
- **Detector**: YOLOv8-Face (for global face bounding box detection).
- **Eye Localization**: Haar Cascade applied strictly to the upper ROI of the detected face, preventing interference from lower facial features (e.g., mouth).

#### 2. Global Feature Extraction (HSEmotion)
- **Backbone**: EfficientNet-B0
- **Input Size**: 48×48 grayscale face or 224×224 RGB
- **Output**: 7-class emotion probability distribution (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Pretraining**: AffectNet dataset + OAHEGA fine-tuning

#### 3. Local Feature Extraction (Eye Region Model)
- **Model Architecture**: Modified ResNet18 (with Dropout for robust 2-class output)
- **Input Size**: 224×224 eye region crop
- **Output**: 2-class probability (Sad, Neutral)
- **Specialty**: Sad micro-expression recognition (inspired by Gorbova et al., 2019)

#### 4. Meta-Learner Fusion Module
- **Meta-Classifier**: Random Forest (max depth 5 to prevent overfitting)
- **Input Features**: 6D feature vector `[p_sad, p_neutral, eye_sad, p_angry, p_happy, p_surprise]`
- **Output**: Final emotion category
- **Training**: Stacking training using soft probabilities to eliminate data distribution drift

## Fusion Decision-Making Principles

### Meta-Learner Fusion Architecture

This system employs a decision-level fusion architecture based on the Stacking strategy, integrating predictions from the global face model (HSEmotion) and local eye region model (ResNet) through a meta-learner, achieving more accurate emotion recognition than single models.

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
    │HSEmotion│              │Eye ResNet│
    └────┬────┘              └────┬────┘
         │                         │
    ┌────▼────────────────────────▼────┐
    │      Feature Concatenation (6D)  │
    │[p_sad, p_neutral, eye_sad,      │
    │  p_angry, p_happy, p_surprise]   │
    └─────────────┬────────────────────┘
                  │
         ┌────────▼────────┐
         │  Meta-Classifier │
         │ (Random Forest)   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Final Output   │
         │  (7 Emotions)    │
         └─────────────────┘
```

### Four Core Innovation Mechanisms (Code Implementation)

#### 1. Occlusion-Aware Adaptive Fallback

**Problem**: When eyes are occluded by hair, glasses, or hands, forcing a geometric crop introduces severe noise.

**Code Implementation** (`eye_feature_extractor.py:extract_eye_region`):
```python
# 🚨 Failed! Trigger debug image saving mechanism
logger.debug("⚠️ Eyes occluded (sunglasses/head down), local expert ineffective, refusing to guess")

# Core change: Never do hard crop! Directly tell upper layer "I can't see"
return None
```

**Upper Layer Handling** (`decision_fusion_detector.py:analyze_emotion`):
```python
# Step 3: Get eye probability & occlusion immunity mechanism
P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)

# 🚨 Sunglasses / closed eyes / no eyes detected -> local expert is "blind", 100% trust global!
if P_eye < 0:  # When extract_eye_region returns None, _get_eye_sad_prob returns -1
    return top_emotion, top_confidence  # Directly return global model result
```

**Design Philosophy**: If a person is wearing sunglasses, the eye expert is "blind". At this point, we should immediately strip the eye expert's voting rights and 100% trust the global model (HSEmotion), because the global model can still see lips, facial muscles, and head posture.

#### 2. Asymmetric Confidence Gating

**Problem**: Meta-classifier may overly override correct predictions from the baseline model on high-confidence samples ("catastrophic overriding").

**Code Implementation** (`decision_fusion_detector.py:analyze_emotion`):
```python
# Step 2: High confidence direct pass & gating interception
if top_confidence >= 80.0:
    return top_emotion, top_confidence  # High confidence direct output, skip fusion

p_sad = all_emotions.get('sad', 0.0)
p_neutral = all_emotions.get('neutral', 0.0)
is_target_top = (top_emotion in self.FUSION_TARGET_EMOTIONS)

# Non-target emotion + medium-high confidence -> no fusion triggered
if not is_target_top:
    if top_confidence > 55.0 or (p_sad < 25.0 and p_neutral < 25.0):
        return top_emotion, top_confidence
```

**Effect**: On high-arousal categories like Happy, the recall rate remains stable at 0.94. The fusion algorithm achieves a non-destructive integration by preserving the baseline's strengths.

#### 3. Dynamic Trigger Boundary Extension

**Problem**: Traditional fixed threshold methods miss potential emotions (e.g., a 26% sadness signal might be genuine sadness).

**Solution**:
- Extend the trigger boundary down to 25%: `sad >= 25%` or `neutral >= 25%` is judged as a hesitant state, activating the meta-learner.
- **Effect**: Neutral recall rate jumped from 0.60 to 0.66 (+6%), accurately correcting the pain point of "expressionless" faces being overly sensitively misclassified as sadness.

#### 4. Domain Adaptation via Soft Probabilities

**Problem**: Simple rule-based fusion (e.g., k=0.5) cannot capture complex non-linear dependencies among multiple classes.

**Training Pipeline**:
1. Generate core 5-class soft probabilities using HSEmotion on the validation set.
2. Generate eye sad probability using the Eye ResNet.
3. Concatenate into a **6D feature vector**, and retrain the Random Forest to eliminate data drift.
4. The meta-classifier implicitly learns the arbitration rules for "when to trust global" vs "when to trust local" through its decision tree structure.

**Effect**: Overall accuracy improved from 71.94% to 72.80% (+0.86%), and Macro-F1 improved from 74.07% to 74.73% (+0.66%).

### Fusion Decision Flow (Complete Code Logic)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Get Global Emotion Distribution (HSEmotion Probs)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │  Step 2: High Conf Gate │
         │  if confidence >= 80%   │──── Yes ────→ Direct Output
         │  if confidence > 55% &  │
         │     not target emotion  │
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 3: Detect Eye      │
         │  if P_eye < 0 (occluded) │──── Yes ────→ Fallback to Global
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 4: Meta-Learner    │
         │  features = [5 global +1 eye] │
         │  if meta_conf < 55%      │──── Yes ────→ Fallback (Abstain)
         │  if not target & < 75%   │
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 5: Accept Meta-   │
         │          Learner Result │
         └─────────────────────────┘
```

## Emotion Categories

The system supports the following 7 basic emotion recognitions:
- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

## Dataset

### OAHEGA Emotion Recognition Dataset

The emotion recognition dataset used in this project mainly contains various emotion categories: Happy, Angry, Sad, Neutral, Surprise, etc.

**Dataset Features:**
- Image Format: RGB cropped face images
- Data Sources: Scraped from Facebook, Instagram, YouTube videos, and public datasets like IMDB and AffectNet.
- *(Note: To align with mainstream emotion recognition benchmarks, the dataset-specific 'Ahegao' class and extremely low-frequency categories were excluded, focusing the evaluation on 5 core high-frequency emotions to validate fusion effectiveness.)*

**Citation:**
```
Kovenko, Volodymyr; Shevchuk, Vitalii (2021), "OAHEGA : EMOTION RECOGNITION DATASET", Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
```

## Experimental Results

### Meta-Learner Fusion Model Performance Evaluation

Rigorous comparative testing results on a large-scale mixed emotion dataset with **8,539 samples**:

**Data Distribution:** Angry (1306), Happy (2000), Neutral (2000), Sad (2000), Surprise (1233)

| Model Architecture | Overall Accuracy | Macro-F1 |
|----------|----------|----------|
| DeepFace Single Model (Classic Baseline) | 30.10% | 35.10% |
| HSEmotion Single Model (SOTA Baseline) | 71.94% | 74.07% |
| **Meta-Learner Fusion (Proposed)** | **72.80%** (+0.86%) | **74.73%** (+0.66%) |

### Fine-Grained Emotion Diagnosis and HCI Value

| Emotion | Breakthrough & HCI Value |
|---------|--------------|
| **Neutral** | Recall jumped from 0.60 to **0.66** (**+6%**), accurately correcting the critical issue where "expressionless" faces are overly sensitively misjudged as sadness. |
| **Angry** | Precision improved from 0.69 to **0.72**. Since Neutral emotions are accurately intercepted, this brings a spillover benefit of suppressing false positives. |
| **Sad** | Precision maintained at a high level of **0.87**. The meta-learner shows "restraint" on ambiguous boundaries, preferring to miss extremely slight sadness rather than making unwarranted interventions, thereby greatly reducing the intrusiveness of the companion AI. |
| **Happy** | Recall is rock-solid at **0.94**. The non-destructive fusion perfectly preserves the baseline model's advantage in high-arousal emotions. |

## Features Highlights

1. **Multi-Modal Fusion**: Combines global face and eye region features to eliminate local noise and improve recognition accuracy.
2. **Configurable Architecture**: Supports switching between multiple detectors to meet different scenario requirements.
3. **Closed-Loop Feedback**: A diary system used to calibrate and optimize emotion recognition parameters.
4. **Real-Time Intervention**: Automatically triggers music and voice interventions based on emotional states.
5. **Data Visualization**: ECharts visualize emotion trends and analytical reports.

## License

MIT License

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface)
- [HSEmotion](https://github.com/HSEmotion/emotion)
- [FER](https://github.com/oarriaga/face_classification)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Academic Paper Reference**:
  - Gorbova, J., Colovic, M., Marjanovic, M., Njegus, A., & Anbarjafari, G. (2019). Going deeper in hidden sadness recognition using spontaneous micro-expressions database. *Springer Science+Business Media*.

---

---
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
| 模型融合 | Scikit-learn (Random Forest) |

## 人脸识别系统架构

### 整体流程

```
视频输入 → 人脸检测 → 眼部区域检测 → 特征提取 → 情绪分类 → 融合决策 → 输出结果
              ↓            ↓             ↓
         YOLOv8      Haar Cascade    全局/局部特征
                                          ↓
                                     元学习器融合
```

### 核心模块

#### 1. 人脸检测与眼部定位
- **检测器**: YOLOv8-Face (全局人脸框检测)
- **眼部定位**: 基于 Haar Cascade 在人脸 ROI 上半区域进行精准裁剪，避免下半脸（如嘴部）干扰

#### 2. 全局特征提取（HSEmotion）
- **骨干网络**: EfficientNet-B0
- **输入尺寸**: 48×48 灰度人脸或 224×224 RGB
- **输出**: 7 类情绪概率分布 (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **预训练**: AffectNet 数据集 + OAHEGA 微调

#### 3. 局部特征提取（眼部模型）
- **模型架构**: 改良版 ResNet18 (引入 Dropout 防过拟合，双分类输出)
- **输入尺寸**: 224×224 眼部区域裁剪
- **输出**: 2 类概率 (Sad, Neutral)
- **专长**: 悲伤微表情识别（参考 Gorbova 等，2019）

#### 4. 元学习器融合模块
- **元分类器**: Random Forest (最大深度 5，防过拟合)
- **输入特征**: 6 维特征向量 `[p_sad, p_neutral, eye_sad, p_angry, p_happy, p_surprise]`
- **输出**: 最终情绪类别
- **训练方式**: 提取软概率进行 Stacking 训练，解决数据分布漂移问题

## 融合决策原理

### Meta-Learner Fusion 架构

本系统采用基于 Stacking 策略的决策级融合架构，通过元学习器整合全局人脸模型（HSEmotion）和局部眼部模型的预测结果，实现比单一模型更准确的情绪识别。

```
┌─────────────────────────────────────────────────────────────┐
│                    输入人脸图像                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐              ┌────▼────┐
    │ 全局模型 │              │ 局部模型 │
    │HSEmotion│              │眼部ResNet│
    └────┬────┘              └────┬────┘
         │                         │
    ┌────▼────────────────────────▼────┐
    │      特征拼接层 (6维)             │
    │[p_sad, p_neutral, eye_sad,      │
    │  p_angry, p_happy, p_surprise]   │
    └─────────────┬────────────────────┘
                  │
         ┌────────▼────────┐
         │  元分类器        │
         │ (Random Forest) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   最终输出       │
         │  (7 类情绪)      │
         └─────────────────┘
```

### 四大核心创新机制（代码实现对应）

#### 1. 感知遮挡的自适应退避机制 (Occlusion-Aware Adaptive Fallback)

**问题**: 眼部被头发、眼镜或手遮挡时，强制裁剪局部特征会引入严重噪声。

**代码实现** (`eye_feature_extractor.py:extract_eye_region`):
```python
# 🚨 失败了！触发 Debug 存图机制
logger.debug("⚠️ 眼睛被遮挡（墨镜/低头），局部专家失效，拒绝瞎猜")

# 核心修改：绝对不要硬裁剪！直接告诉上层"我看不见"
return None
```

**上层处理** (`decision_fusion_detector.py:analyze_emotion`):
```python
# Step 3：获取眼睛概率 & 遮挡免疫机制
P_eye = self._get_eye_sad_prob(frame_bgr, face_rect)

# 🚨 戴了墨镜 / 闭眼 / 没找到眼睛 -> 局部专家"瞎了"，100% 听全局！
if P_eye < 0:  # extract_eye_region 返回 None 时，_get_eye_sad_prob 返回 -1
    return top_emotion, top_confidence  # 直接返回全局模型结果
```

**设计哲学**: 如果一个人戴着墨镜，眼睛专家就"瞎了"，这时候我们应该立刻剥夺眼睛专家的投票权，100% 信任全局模型（HSEmotion），因为全局模型还能看到嘴唇、面部肌肉和头部姿态。

#### 2. 非对称置信度门控 (Asymmetric Confidence Gating)

**问题**: 元分类器可能过度推翻基线模型在高置信度样本上的正确预测（"灾难性推翻"）。

**代码实现** (`decision_fusion_detector.py:analyze_emotion`):
```python
# Step 2：高置信度直通与门控拦截
if top_confidence >= 80.0:
    return top_emotion, top_confidence  # 高置信度直接输出，跳过融合

p_sad = all_emotions.get('sad', 0.0)
p_neutral = all_emotions.get('neutral', 0.0)
is_target_top = (top_emotion in self.FUSION_TARGET_EMOTIONS)

# 非目标情绪 + 中高置信度 -> 不触发融合
if not is_target_top:
    if top_confidence > 55.0 or (p_sad < 25.0 and p_neutral < 25.0):
        return top_emotion, top_confidence
```

**效果**: 在 Happy 等高唤醒度类别上，召回率稳定在 0.94，融合算法实现了"取其精华去其糟粕"的非破坏性融合。

#### 3. 动态触发边界扩展 (Dynamic Trigger Boundary)

**问题**: 传统固定阈值法会漏掉潜在边缘情绪（如 26% 的悲伤信号可能是真悲伤）。

**解决方案**:
- 将触发边界下探至 25%：`sad >= 25%` 或 `neutral >= 25%` 即判定为犹豫状态，唤醒元学习器。
- **效果**: Neutral 召回率从 0.60 跃升至 0.66（+6%），精准纠正"面无表情"被过度敏感地误判为悲伤的痛点。

#### 4. 基于软概率的领域适应重塑 (Domain Adaptation via Soft Probabilities)

**问题**: 简单的权重规则融合（如 k=0.5）无法捕捉多类别的非线性依赖关系。

**训练流程**:
1. 用 HSEmotion 在验证集上生成核心 5 类情绪软概率。
2. 用眼部 ResNet 生成眼部 sad 概率。
3. 拼接为 **6 维特征向量**，重新训练 Random Forest 以消除数据漂移。
4. 元分类器通过决策树结构隐式学习了"何时相信全局"与"何时相信局部"的仲裁规则。

**效果**: 整体准确率从 71.94% 提升至 72.80%（+0.86%），Macro-F1 从 74.07% 提升至 74.73%（+0.66%）。

### 融合决策流程（完整代码逻辑）

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 获取全局情绪分布 (HSEmotion 概率)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │  Step 2: 高置信度门控    │
         │  if confidence >= 80%   │──── Yes ────→ 直通输出
         │  if confidence > 55% &  │
         │     not target emotion  │
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 3: 检测眼部特征    │
         │  if P_eye < 0 (遮挡)     │──── Yes ────→ 退避至全局
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 4: 元学习器预测    │
         │  features = [5 全局 +1 眼部] │
         │  if meta_conf < 55%      │──── Yes ────→ 回退全局 (弃权)
         │  if not target & < 75%   │
         └────────────┬────────────┘
                      No
         ┌────────────▼────────────┐
         │  Step 5: 采纳元学习器结果 │
         └─────────────────────────┘
```

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

本项目使用的情绪识别数据集主要包含多种情绪类别：Happy、Angry、Sad、Neutral、Surprise 等。

**数据集特点：**
- 图像格式：RGB 人脸裁剪图像
- 数据来源：Facebook、Instagram 社交网络爬取，YouTube 视频，以及 IMDB、AffectNet 等公开数据集
- *(注：为对齐主流情绪识别基准，本次评估剔除了数据集特有的 Ahegao 类别与极低频类别，并聚焦于 5 种核心高频情绪进行融合有效性验证。)*

**引用：**
```
Kovenko, Volodymyr; Shevchuk, Vitalii (2021), "OAHEGA : EMOTION RECOGNITION DATASET", Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
```

## 实验结果

### 元学习器融合模型性能评估

在 **8,539 个样本** 的大规模混合情绪数据集上进行的严格对比测试结果：

**数据分布：** Angry (1306), Happy (2000), Neutral (2000), Sad (2000), Surprise (1233)

| 模型架构 | 总体准确率 (Accuracy) | 宏平均 F1 (Macro-F1) |
|----------|----------------------|---------------------|
| DeepFace 单体 (经典基准) | 30.10% | 35.10% |
| HSEmotion 单体 (SOTA 基线) | 71.94% | 74.07% |
| **Meta-Learner Fusion (本文提出)** | **72.80%** (+0.86%) | **74.73%** (+0.66%) |

### 细粒度情绪诊断与产品价值

| 情绪 | 突破点与 HCI 价值 |
|------|----------------|
| **中性 (Neutral)** | 召回率从 0.60 跃升至 **0.66**（提升 **6%**），精准纠正"面无表情"被过度敏感误判为悲伤的痛点。 |
| **愤怒 (Angry)** | 精确率从 0.69 提升至 **0.72**，由于中性情绪被精准拦截，带来了假阳性被抑制的溢出红利。 |
| **悲伤 (Sad)** | 精确率维持 **0.87** 极高水准。元学习器在模棱两可边界上表现出"克制"，宁可漏判极其轻微的悲伤，也绝不无病呻吟，从而极大降低了陪伴型 AI 的打扰感。 |
| **快乐 (Happy)** | 召回率稳如磐石（**0.94**），非破坏性融合完美保留了基线模型的高唤醒度情绪优势。 |

## 项目特色

1. **多模态融合**: 结合全局人脸和眼部区域特征，消除局部噪声提升识别准确率
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
