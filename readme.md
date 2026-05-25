# EmotiSense: Real-time Emotion Detection System with Multi-Model Fusion and Intelligent Intervention

> A multi-modal, real-time emotion recognition and monitoring system that integrates computer vision, affective computing, and intelligent intervention into a unified web-based platform.

---

## Abstract

EmotiSense is a full-stack emotion detection system capable of real-time facial emotion analysis through webcam input. The system employs a hierarchical multi-model fusion architecture combining global face emotion classifiers (DeepFace, HSEmotion, FER) with a specialized eye-region CNN for enhanced sad/neutral discrimination. An Emotion Dynamics Engine models emotional state as a continuous valence trajectory with stress accumulation, while an Advanced Analytics module implements attractor models, RMSSD fluctuation quantification, Kalman filter smoothing, and diary-visual consistency validation. The system automatically triggers context-aware interventions (music, TTS voice comfort, hitokoto quotes) and provides comprehensive user/admin dashboards with ECharts visualizations.

---

## Table of Contents

- [System Architecture](#1-system-architecture)
- [Processing Pipeline](#2-processing-pipeline)
- [Multi-Model Fusion](#3-multi-model-fusion)
- [Emotion Dynamics Engine](#4-emotion-dynamics-engine)
- [Advanced Analytics](#5-advanced-analytics)
- [Intervention System](#6-intervention-system)
- [Explainable AI (XAI)](#7-explainable-ai-xai)
- [Database Schema](#8-database-schema)
- [API Reference](#9-api-reference)
- [Frontend Architecture](#10-frontend-architecture)
- [Installation](#11-installation)
- [Training & Evaluation](#12-training--evaluation)
- [Configuration](#13-configuration)
- [Model Zoo](#14-model-zoo)
- [Experimental Results](#15-experimental-results)
- [Dataset](#16-dataset)
- [Academic References](#17-academic-references)

---

## 1. System Architecture

```
+====================================================================+
|                        Frontend (Vue 3 + Vite)                      |
|                                                                    |
|  +---------------+ +---------------+ +---------------+             |
|  | MonitorMode   | | User Dashboard| | Admin Panel   |             |
|  | (MJPEG Stream)| | (History/Stats)| | (Analytics)  |             |
|  +---------------+ +---------------+ +---------------+             |
|  +---------------+ +---------------+ +---------------+             |
|  | Diary System  | | Calendar Mood | | Resource Mgmt |             |
|  | (CRUD + Tags) | | (Face+Diary)  | | (Music/Script)|             |
|  +---------------+ +---------------+ +---------------+             |
|                                                                    |
|  Tech: Vue 3 (Composition API) | Element Plus | ECharts | Axios    |
+================================|===================================+
                                 | HTTP REST API
+================================|===================================+
|                     Backend (FastAPI + Uvicorn)                     |
|                                                                    |
|  +----------------------------------------------------------------+|
|  |  API Layer (app/api.py) -- ~50 endpoints across 6 namespaces  ||
|  |  /video_feed | /api/status | /api/my/* | /api/admin/*         ||
|  +----------------------------------------------------------------+|
|                                                                    |
|  +----------------------------------------------------------------+|
|  |  Core Processing Pipeline (real-time, per-frame)               ||
|  |                                                                ||
|  |  Camera --> FaceDetector --> EmotionDetector --> Stabilizer   ||
|  |    (OpenCV)    (YOLOv8)     (Fusion/DeepFace)  (Window Vote)  ||
|  |                              |                                 ||
|  |                        DynamicsEngine                          ||
|  |                        (EMA Valence + Distress)                ||
|  |                              |                                 ||
|  |                        EmotionLog (throttled)                  ||
|  |                              |                                 ||
|  |                        Intervention Decision                   ||
|  +----------------------------------------------------------------+|
|                                                                    |
|  +----------------------------------------------------------------+|
|  |  Advanced Analytics Engine (post-hoc, per-user)                ||
|  |  - Attractor Model (personalized emotional baseline)           ||
|  |  - RMSSD (emotion fluctuation index, borrowed from HRV)        ||
|  |  - Kalman Filter (1D, mood trajectory smoothing)               ||
|  |  - Emotion Inertia (lag-1 autocorrelation)                     ||
|  |  - Diary-Visual Consistency Validation (closed-loop)           ||
|  +----------------------------------------------------------------+|
|                                                                    |
|  +----------------------------------------------------------------+|
|  |  Audio Intervention System                                     ||
|  |  - Music: pygame mixer, emotion-tagged, user-specific          ||
|  |  - TTS: edge-tts (zh-CN-XiaoxiaoNeural) with volume ducking   ||
|  |  - Scripts: comfort message library, global + per-user         ||
|  +----------------------------------------------------------------+|
|                                                                    |
|  Data: SQLite (SQLAlchemy ORM)                                     |
+====================================================================+
```

### Directory Structure

```
EmotiSense/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app with lifespan management
│   │   ├── api.py              # All ~50 API endpoints (monolithic router)
│   │   └── database.py         # SQLAlchemy engine + SessionLocal
│   ├── core/
│   │   ├── config.py           # YAML configuration loader (singleton)
│   │   ├── models.py           # ORM models (6 tables)
│   │   ├── detector.py         # FaceDetector + EmotionDetector + find_identity
│   │   ├── stabilizer.py       # EmotionStabilizer (sliding window vote)
│   │   ├── emotion_dynamics.py # EmotionDynamicsEngine (EMA + distress)
│   │   ├── audio_manager.py    # AudioManager (pygame + edge-tts)
│   │   ├── advanced_analyzer.py   # AdvancedEmotionAnalyzer + DiarySentimentAnalyzer
│   │   ├── advanced_detectors.py  # HSEmotionDetector, FERDetector, EyeFusionDetector, EnsembleDetector
│   │   ├── decision_fusion_detector.py  # ImprovedDecisionFusionDetector (gated routing)
│   │   ├── meta_learner_inference.py    # MetaLearnerPrediction (feature extraction + inference)
│   │   ├── eye_feature_extractor.py     # EyeFeatureExtractor (unified extraction)
│   │   └── eye_fusion_detector.py       # EyeFusionDetectorFixed (rule-based fusion)
│   ├── train_meta_learner.py   # Meta-learner training script (Stacking)
│   ├── train_meta_learner_fixed.py
│   ├── evaluate_meta_learner.py
│   ├── evaluate_all_models.py
│   ├── finetune_eye_model.py
│   ├── config.yaml             # Centralized configuration
│   ├── models/                 # Pre-trained ML models
│   ├── weights/                # Trained weights (meta-learner, eye model, etc.)
│   └── assets/                 # Static files (music, backgrounds, TTS output)
├── frontend/
│   ├── src/
│   │   ├── main.js             # App entry (Vue + Element Plus + icons)
│   │   ├── App.vue             # Root (background management)
│   │   ├── router/index.js     # Vue Router with role-based guards
│   │   ├── layouts/            # AdminLayout.vue, UserLayout.vue
│   │   ├── views/
│   │   │   ├── Login.vue
│   │   │   ├── user/           # MonitorMode, UserHome, UserHistory, UserDiary, UserSettings
│   │   │   └── admin/          # UserManager, ResourceManager, Analytics, SystemLogs
│   │   └── styles/             # theme.css (design tokens)
│   └── package.json
└── CLAUDE.md
```

---

## 2. Processing Pipeline

The core processing pipeline executes on every video frame (or every Nth frame based on `frame_skip`):

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Frame Processing Loop                         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  1. Capture Raw Frame (OpenCV VideoCapture)          │
        │     - BGR format, 640x360, 30fps                     │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  2. Face Detection (YOLOv8 + Haar Cascade)           │
        │     - YOLOv8 detects face bounding boxes             │
        │     - Haar Cascade detects eyes within upper 60% ROI │
        │     - EMA smoothing on face bounding box             │
        │     - Returns: rect, has_eyes, eye_coords            │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  3. Emotion Analysis (Configurable Detector)         │
        │     - Adaptive call via inspect.signature:           │
        │       * Full frame + face_rect -> Fusion detectors   │
        │       * Cropped face_img -> DeepFace/HSEmotion       │
        │     - Returns: (emotion_name, confidence_percentage) │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  4. Confidence-Based Dynamic Weighting               │
        │     - Eye-dependent emotions (surprise, fear, sad):  │
        │       * Eyes visible: confidence *= 1.15             │
        │       * Eyes missing: confidence *= 0.80             │
        │     - Mouth-dependent emotions (happy, angry):       │
        │       No adjustment                                  │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  5. Identity Recognition (Optional)                  │
        │     - DeepFace VGG-Face embedding extraction         │
        │     - Cosine distance comparison (threshold = 0.6)   │
        │     - Face DB cached every 10 seconds                │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  6. XAI Visual Overlays (Optional)                   │
        │     - Blue overlay on eye regions (eye-dependent)    │
        │     - Red overlay on mouth region (mouth-dependent)  │
        │     - Label: "XAI: Eye/Mouth Region Activation"      │
        │     - Top label: Name | Emotion | Confidence%        │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  7. Encode to JPEG + Return to Frontend              │
        │     - MJPEG stream via multipart/x-mixed-replace     │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  8. Emotion Stabilizer (if faces detected)           │
        │     - Add prediction to sliding window (size=15)     │
        │     - Majority vote with 60% hysteresis threshold    │
        │     - Returns stable emotion                         │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  9. Emotion Dynamics Engine                          │
        │     - EMA valence update (alpha=0.2)                 │
        │     - Distress accumulation/decay (decay=0.95)       │
        │     - Intervention trigger check (threshold=1.2)     │
        │     - 60s cooldown between interventions             │
        └──────────────────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  10. Database Log (Throttled)                        │
        │      - Only if emotion persisted > 3s                │
        │      - Only if > 5s since last write                 │
        │      - Stores: timestamp, user_id, emotion, score    │
        └─────────────────────────────────────────────────────┘
```

---

## 3. Multi-Model Fusion

### 3.1 Available Detection Backends

EmotiSense implements **7 emotion detection backends** with a factory pattern for easy switching via `config.yaml`:

| Detector | Architecture | Emotion Classes | Speed | Notes |
|----------|-------------|-----------------|-------|-------|
| `deepface` | VGG-Face CNN | 7 | ~200ms | Default, stable and reliable |
| `hsemotion` | EfficientNet-B0 | 7 or 8 | ~60ms | High-speed, best accuracy |
| `fer` | Custom CNN | 7 | ~150ms | Lightweight FER library |
| `ensemble` | Weighted average | 7 | ~250ms | Combines multiple detectors |
| `eye_fusion` | Eye CNN + global features | 3 | ~100ms | Feature-level fusion for sad/neutral |
| `decision_fusion` | Gated routing + weighted fusion | 7 | ~100ms | Decision-level fusion with gating |
| `meta_learner` | Stacking classifier (sklearn) | 2 | ~50ms | Data-driven fusion, requires training |

### 3.2 Detector Factory

```python
# backend/core/detector.py
def create_emotion_detector(config: Config, use_meta_learner: Optional[bool] = None):
    detector_type = config.get('emotion.detector_type', 'deepface')

    if detector_type == 'meta_learner':
        # Auto-uses decision_fusion with use_meta_learner=True
        return create_improved_decision_fusion_detector(config, use_meta_learner=True)

    if detector_type == 'decision_fusion':
        use_meta_learner = use_meta_learner or config.get('emotion.use_meta_learner', False)
        return create_improved_decision_fusion_detector(config, use_meta_learner=use_meta_learner)

    # ... other detectors
```

### 3.3 Face Detection

**Primary**: YOLOv8 with custom fine-tuned model (`yolov8m-face-lindevs.pt`) trained specifically for face detection.

**Fallback**: Standard `yolov8n.pt` (person detection, class 0) if custom model is unavailable.

**Bounding Box Smoothing**: Exponential Moving Average (EMA) to reduce video jitter:
```
smooth_x = last_x * (1 - smoothing_factor) + x * smoothing_factor
```
where `smoothing_factor = 0.3`.

**Eye Detection**: Haar Cascade (`haarcascade_eye.xml`) within the upper 60% of the face ROI to avoid misclassifying mouths as eyes. Parameters: `scaleFactor=1.1`, `minNeighbors=4`, `minSize=(20,20)`.

### 3.4 Identity Recognition

DeepFace VGG-Face embeddings with cosine distance matching:

1. Extract embedding from detected face using `DeepFace.represent()`
2. Compare against cached face database (refreshed every 10 seconds)
3. Cosine distance threshold: 0.6 (tuned for webcam conditions)
4. Returns matched username or "Stranger"

### 3.5 Stacking Meta-Learner

The meta-learner implements **Stacking ensemble**, the industrial standard for model fusion:

**Level 0 (Base Models):**
- HSEmotion: produces 5-class soft probabilities `[P_sad, P_neutral, P_angry, P_happy, P_surprise]`
- Eye CNN (ResNet18): produces `P_eye_sad`

**Level 1 (Meta Classifier):**
- 6D feature vector concatenation
- Trained classifiers: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting
- Best classifier selected by accuracy on stratified test set
- Saved as `weights/meta_learner_fusion_model.pkl`

**Gating Logic** in `ImprovedDecisionFusionDetector.analyze_emotion()`:

1. **High Confidence Pass-Through** (>= 80%): Direct output, skip fusion
2. **Iron Gate** (top emotion not in {sad, neutral}): Direct output, protect happy/angry/surprise
3. **Conditional Activation** (top confidence > 55% AND sad/neutral < 25%): Direct output
4. **Meta-Learner Decision**: 6D feature vector -> sklearn classifier -> sad/neutral prediction
5. **Abstain Protection** (meta confidence < 55%): Fall back to global
6. **Cross-Emotion Protection** (global says surprise, meta says neutral): Require meta confidence > 75% to override

---

## 4. Emotion Dynamics Engine

The `EmotionDynamicsEngine` models emotion as a **continuous valence trajectory** rather than discrete classifications, implementing psychological theories of emotional inertia and mood dynamics.

### 4.1 Valence Mapping

Each discrete emotion maps to a continuous valence value based on psychological research:

| Emotion | Valence | Interpretation |
|---------|---------|---------------|
| happy | +1.0 | Maximum positive affect |
| surprise | +0.3 | Mildly positive (valence-ambiguous) |
| neutral | 0.0 | Emotional baseline |
| sad | -0.6 | Moderate negative affect |
| contempt | -0.7 | Social-evaluative negative |
| disgust | -0.9 | Strong rejection response |
| fear | -0.8 | High-arousal negative |
| angry | -1.0 | Maximum negative affect |

### 4.2 EMA Update Rule

```
valence_ema(t) = ALPHA * (valence_map(emotion) * confidence) + (1 - ALPHA) * valence_ema(t-1)
```

ALPHA = 0.2: The system responds to new detections at 20% weight while maintaining 80% memory of historical mood, creating a smooth emotional trajectory.

### 4.3 Distress Accumulation

Stress (distress) accumulates during negative valence and decays during positive valence:

```
if valence_ema < 0:
    distress(t) = distress(t-1) * DECAY + |valence_ema|
else:
    distress(t) = distress(t-1) * DECAY
```

DECAY = 0.95: Simulates natural emotional recovery -- stress slowly dissipates when not reinforced.

### 4.4 Intervention Trigger

```python
if (time_now - last_intervention > COOLDOWN) and (distress > TRIGGER_THRESHOLD):
    trigger_intervention = True
    distress *= 0.5  # Post-intervention stress reduction
    last_intervention = time_now
```

Parameters: TRIGGER_THRESHOLD = 1.2, COOLDOWN = 60 seconds.

---

## 5. Advanced Analytics

### 5.1 Attractor Model

Inspired by dynamical systems theory in psychology, each user has a personalized **emotion attractor** -- their baseline emotional equilibrium:

- **Attractor (mean)**: `mean(valence_series)` -- the user's typical emotional valence
- **Attractor std**: `std(valence_series)` -- individual variation range

This enables detection of deviations relative to the user's own baseline rather than population averages.

### 5.2 RMSSD (Root Mean Square of Successive Differences)

Borrowed from Heart Rate Variability (HRV) analysis, RMSSD quantifies emotion fluctuation intensity:

```
RMSSD = sqrt(mean((v[i+1] - v[i])^2))
```

**Clinical Interpretation:**
| RMSSD + Valence | Interpretation |
|-----------------|---------------|
| Low RMSSD (< 0.1) + Low valence (< -0.5) | Emotional rigidity / depressive state |
| High RMSSD (> 0.5) | Extreme emotional volatility / anxiety |
| Moderate RMSSD | Healthy emotional flexibility |

**Sessioning Mechanism**: Only consecutive detections within 15 minutes are used for RMSSD calculation, preventing spurious fluctuation measurements across different contexts.

### 5.3 Kalman Filter Smoothing

A 1D Kalman filter extracts the "slow mood trajectory" from noisy momentary detections:

- Process noise: Q = 0.01
- Measurement noise: R = 0.1
- Purpose: Separate genuine mood trends from transient expression noise (blinking, head turns, lighting changes)

### 5.4 Emotion Inertia

Computed as the lag-1 autocorrelation of the valence series, normalized to [0, 1]:

```
inertia = (autocorr + 1) / 2
```

High inertia (> 0.8) indicates the user tends to "get stuck" in an emotional state, resistant to change.

### 5.5 Trend Direction

Linear regression on the most recent N valence values:
- `slope > 0.05`: rising
- `slope < -0.05`: falling
- otherwise: stable

### 5.6 Diary-Visual Consistency Validation

A closed-loop validation system compares subjective diary sentiment with objective visual recognition:

1. **Diary Sentiment Analysis**: Bilingual (Chinese/English) sentiment lexicon matching
   - Positive words: 开心, 快乐, happy, love, etc.
   - Negative words: 难过, 悲伤, sad, angry, etc.

2. **Temporal Matching**: Diary entries matched with visual emotion logs within a configurable tolerance window (default: 2 hours)

3. **Correlation Analysis**: Pearson correlation coefficient between diary valence and matched visual valence

4. **Consistency Classification:**
   - High: r > 0.7 -- visual model is trustworthy
   - Medium: r > 0.3 -- visual model is basically reliable
   - Low: r < 0.3 -- visual model may have systematic bias

5. **Extreme Asymmetry Detection:**
   - **Masking detection**: visual valence < -0.5 (visibly distressed) AND diary valence > 0.2 (claims to be happy)
   - **Severe conflict**: |visual - diary| > 0.7
   - **Action**: Triggers `trigger_questionnaire` flag for PHQ-9 questionnaire

---

## 6. Intervention System

### 6.1 Three-Tier Intervention Strategy

| Risk Level | Trigger Condition | Intervention |
|------------|-------------------|--------------|
| **High** | RMSSD < 0.1 + valence < -0.3 sustained for N records | TTS urgency + calming music |
| **Medium** | valence < -0.5 OR RMSSD > 0.4 | Music + hitokoto quotes |
| **Low** | valence > 0.4 | Positive reinforcement |

### 6.2 Music Intervention

- **Engine**: pygame mixer with loop playback
- **Library**: Emotion-tagged music with user-specific priority
- **Retrieval**: User-exclusive music > Global music > Random selection
- **Volume Management**: Automatically ducks to 10-20% during TTS speech

### 6.3 TTS (Text-to-Speech) Comfort Voice

- **Engine**: edge-tts with `zh-CN-XiaoxiaoNeural` voice
- **Flow**: Generate MP3 -> pause background music -> play TTS -> resume music
- **Coordination**: async/await with `asyncio.sleep(duration)` for proper timing

### 6.4 Comfort Script Library

- **Global scripts**: Shared by all users (initialized with 4 default messages)
- **User-specific scripts**: Personalized messages per user
- **Emotion tags**: sad, angry, happy, etc.
- **Admin management**: Add/delete scripts via admin panel

### 6.5 Hitokoto Integration

Integration with [Hitokoto API](https://hitokoto.cn) for emotionally-contextual quotes:

| Emotion | Category | Example Source |
|---------|----------|---------------|
| happy | literature + humor | 文学 + 抖机灵 |
| sad | poetry + NetEase | 诗词 + 网易云 |
| angry | philosophy | 哲学 |
| neutral | anime + manga | 动画 + 漫画 |
| fear | movies/TV | 影视 |
| surprise | original | 原创 |

2-second timeout ensures the third-party API doesn't slow down the system.

---

## 7. Explainable AI (XAI)

### 7.1 Region-Based Visual Explanations

The system provides visual overlays explaining which facial regions drove the emotion decision:

- **Eye Region Activation** (blue overlay): Applied for surprise, fear, sad -- emotions where periorbital features are diagnostically important
- **Mouth Region Activation** (red overlay): Applied for happy, angry -- emotions where mouth/lip features are key
- Overlay transparency: 40% blend with original frame

### 7.2 Confidence-Based Dynamic Weighting

An implicit attention mechanism adjusts confidence based on feature availability:

```python
eye_dependent_emotions = ["surprise", "fear", "sad", "sadness"]
mouth_dependent_emotions = ["happy", "angry", "disgust"]

if base_emotion in eye_dependent_emotions:
    if has_eyes:
        final_confidence = min(100.0, base_confidence * 1.15)  # Boost
    else:
        final_confidence = base_confidence * 0.80  # Penalty
```

### 7.3 Identity Labels

Each detected face displays: `Name | Emotion | Confidence%`
- Green box for recognized users
- Gray box for strangers

---

## 8. Database Schema

SQLite database with 6 tables managed by SQLAlchemy ORM:

### users
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| username | String (UNIQUE) | Login name |
| password_hash | String | Password (plaintext, should be hashed) |
| role | String | "admin" or "user" |
| face_encoding | JSON | DeepFace VGG-Face embedding (list of floats) |
| avatar | String | Avatar image path |
| created_at | DateTime | Registration time |

### emotion_logs
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| timestamp | DateTime | Detection time |
| user_id | Integer (FK -> users) | Associated user (nullable) |
| is_stranger | Boolean | Whether the person is unrecognized |
| emotion | String | Emotion label (happy, sad, etc.) |
| score | Float | Normalized mood score (0-1) |

### diaries
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| user_id | Integer (FK -> users) | Author |
| title | String | Diary title |
| content | Text | Diary body text |
| emotion | String | Self-reported mood |
| timestamp | DateTime | Entry time (supports backdating) |

### music_library
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| user_id | Integer (FK -> users, NULL=global) | Owner |
| title | String | Display name |
| filepath | String | Relative path to audio file |
| emotion_tag | String | Associated emotion |
| is_active | Boolean | Whether track is enabled |

### comfort_scripts
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| user_id | Integer (FK -> users, NULL=global) | Owner |
| content | Text | Comfort message text |
| emotion_tag | String | Associated emotion |

### system_events
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| timestamp | DateTime | Event time |
| people_count | Integer | Number of people detected |
| vote_result | JSON | Majority vote result (e.g., `{'sad':3, 'happy':1}`) |
| final_mood | String | Aggregated mood |
| action_type | String | "music", "tts", or "none" |
| resource_id | Integer | Associated resource ID |

---

## 9. API Reference

### Core Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/video_feed` | MJPEG real-time video stream | None |
| GET | `/api/status` | Current emotion, mood_score, stress_level | None |
| POST | `/analyze` | Single-frame image analysis | None |
| POST | `/api/login` | Login (returns role + username) | None |
| POST | `/api/register` | User registration | None |

### User Endpoints (`/api/my/*`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/my/stats` | Personal emotion pie chart data |
| GET | `/api/my/history` | Paginated emotion log history |
| GET | `/api/my/history/stats` | Time-aggregated mood chart (day/week/month) |
| GET | `/api/my/diaries` | User diary entries |
| POST | `/api/my/diaries` | Create diary (with backdating support) |
| PUT | `/api/my/diaries/:id` | Update diary |
| DELETE | `/api/my/diaries/:id` | Delete diary |
| GET | `/api/my/calendar_moods` | Calendar mood (face + diary merged, diary takes priority) |
| GET | `/api/my/personalized_quote` | Emotion-targeted quote (hitokoto -> local script fallback) |
| POST | `/api/user/upload_background` | Upload custom background |
| DELETE | `/api/user/upload_background` | Reset background to default |
| GET/POST/DELETE | `/api/user/scripts` | User-specific comfort scripts |
| GET/POST/DELETE | `/api/user/music` | User-specific music library |
| POST | `/api/user/upload_music` | Upload user-specific music |

### Admin Endpoints (`/api/admin/*`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/users` | List all users with face enrollment status |
| POST | `/api/admin/capture_face/:user_id` | Capture face from webcam for enrollment |
| GET | `/api/admin/logs` | System-wide emotion logs (paginated, filterable by username) |
| GET | `/api/admin/analytics/stats` | Dashboard stats (overview, pie, trend, bar, heatmap, radar, top users) |
| GET | `/api/admin/analytics` | Simple analytics (pie + trend, fallback) |
| GET | `/api/admin/analytics/advanced/:user_id` | Advanced analytics (attractor, RMSSD, Kalman, trend, inertia, suggestions) |
| GET | `/api/admin/analytics/comprehensive/:user_id` | AI diagnostic report (valence, RMSSD, risk level, conclusion, suggestions) |
| GET | `/api/admin/analytics/alerts` | Real-time alert feed (intervention events + high-risk users) |
| GET | `/api/admin/analytics/quadrant` | Valence x RMSSD scatter plot for all users |
| POST | `/api/admin/analytics/diary/validate/:user_id` | Diary-visual consistency validation |
| GET | `/api/admin/analytics/intervention/suggest/:user_id` | Intervention recommendations |
| GET | `/api/admin/analytics/interventions/:user_id` | Intervention event timeline |
| GET | `/api/admin/analytics/system-health` | AI system health (confidence distribution, emotion pie, accuracy) |
| GET/POST/DELETE | `/api/admin/scripts` | Global comfort script management |
| GET/POST/DELETE | `/api/admin/music` | Global music library management |

### Debug Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/debug/seed_data` | Generate 50 random emotion log entries |
| GET | `/api/debug/create_test_user` | Create default admin (admin/123456) |

---

## 10. Frontend Architecture

### Technology Stack
- **Vue 3** with Composition API (`<script setup>`)
- **Vue Router 5** with role-based navigation guards
- **Element Plus** UI component library with custom theme CSS variables
- **ECharts 6** for data visualization
- **Axios** for HTTP communication
- **Vite 7** for development and build
- **screenfull** for fullscreen support

### Route Structure

```
/                         -> MonitorMode (real-time webcam, public)
/login                    -> Login page
/admin (requires: admin)  -> AdminLayout
  /admin/users            -> UserManager (CRUD + face capture)
  /admin/resources        -> ResourceManager (music + scripts)
  /admin/analytics        -> Analytics (dashboard with 7+ charts)
  /admin/logs             -> SystemLogs (paginated log viewer)
/user (requires: user)    -> UserLayout
  /user/home              -> UserHome (daily quote, quick stats)
  /user/history           -> UserHistory (emotion timeline)
  /user/diary             -> UserDiary (CRUD + emotion tagging)
  /user/settings          -> UserSettings (profile, background, music)
```

### Auth Model
Simple role-based authentication using `localStorage`:
- Keys: `user`, `role`
- Route guards redirect unauthenticated users to `/login`
- Role mismatch redirects to appropriate dashboard

### Background System
Priority-based background resolution:
1. Custom background from `localStorage` (`custom_bg`)
2. User-specific background from server (`/assets/bg_{username}.jpg`)
3. Default soft gradient fallback

### Theme System
CSS custom properties in `styles/theme.css` for consistent design tokens (colors, border radii, shadows).

---

## 11. Installation

### Prerequisites
- Python 3.10+
- Node.js 20.19+ or 22.12+
- Webcam

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install core dependencies
pip install fastapi uvicorn sqlalchemy opencv-python deepface ultralytics
pip install pygame edge-tts pyyaml python-dotenv httpx
pip install scikit-learn joblib numpy

# Install optional dependencies (for advanced detectors)
pip install torch torchvision  # Eye model + meta-learner
pip install hsemotion          # HSEmotion detector (optional)
pip install fer                # FER detector (optional)

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at `http://127.0.0.1:8000/docs`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### First Run
1. Start the backend server
2. Create the default admin account: visit `http://127.0.0.1:8000/api/debug/create_test_user`
3. Login with admin / 123456
4. Enroll user faces via Admin -> Users -> Capture Face

---

## 12. Training & Evaluation

### Meta-Learner Training

```bash
cd backend
python train_meta_learner.py \
  --dataset-path /path/to/dataset \
  --samples-per-class 200 \
  --output-path weights/meta_learner_fusion_model.pkl
```

**Training process:**
1. Loads images from dataset directory (organized as `Angry/`, `Happy/`, `Neutral/`, `Sad/`, `Surprise/`)
2. For each image: extracts 6D feature vector from HSEmotion + eye CNN
3. Trains 4 classifiers (LR, SVM, RF, GBM) with stratified 80/20 split
4. Selects best classifier by accuracy
5. Saves model with feature names and classifier metadata

### Model Evaluation

```bash
# Evaluate meta-learner on test set
python evaluate_meta_learner.py

# Compare all emotion models side by side
python evaluate_all_models.py
```

### Eye Model Fine-Tuning

```bash
python finetune_eye_model.py
```

Fine-tunes a ResNet18 on eye region images for neutral vs sad binary classification.

### Fixed Meta-Learner Training

```bash
python train_meta_learner_fixed.py
```

Fixed version of the meta-learner training script with improved feature extraction consistency.

---

## 13. Configuration

All configuration is centralized in `backend/config.yaml`:

### Video Capture
```yaml
video:
  camera_index: 0          # Webcam device index
  frame_width: 640         # Capture width
  frame_height: 360        # Capture height
  fps: 30                  # Target frame rate
  frame_skip: 2            # Process every Nth frame
```

### Face Detection
```yaml
face_detection:
  scale_factor: 1.1        # Haar cascade scale factor
  min_neighbors: 3         # Minimum neighbors for face detection
  min_size: [80, 80]       # Minimum face size
  max_size: [300, 300]     # Maximum face size
  smoothing_factor: 0.3    # EMA smoothing factor for bounding box
```

### Emotion Detection
```yaml
emotion:
  detector_type: 'meta_learner'              # Backend selection
  detection_interval: 3.0                    # Seconds between analyses
  high_confidence_threshold: 95              # Percentage
  anger_threshold: 50                        # Minimum angry confidence to report
  max_data_records: 1000                     # Max emotion records in memory
  decision_fusion_k: 0.5                     # Eye model weight in fusion (0.0-1.0)
  use_meta_learner: true                     # Use trained LR vs rule-based fusion
  sad_threshold: 30                          # Sad confidence threshold for eye model
  hsemotion_model: 'enet_b0_8_best_afex'     # HSEmotion model variant
  fer_use_mtcnn: false                       # Use MTCNN for FER (slower, more accurate)
  eye_model_path: 'models/eye_model_finetuned.pth'
```

### Key Configuration Choices

| `detector_type` | When to use |
|-----------------|-------------|
| `deepface` | Default, stable, no additional dependencies |
| `hsemotion` | Best speed + accuracy, requires `hsemotion` package |
| `meta_learner` | **Recommended**: Best sad/neutral discrimination |
| `eye_fusion` | When eye model is available and well-trained |
| `decision_fusion` | When you want rule-based fusion with configurable weights |

---

## 14. Model Zoo

| Model | File | Purpose | Framework |
|-------|------|---------|-----------|
| YOLOv8 Face | `models/yolov8m-face-lindevs.pt` | Face detection (custom fine-tuned) | Ultralytics |
| YOLOv8 Standard | `models/yolov8n.pt` | Fallback person detection | Ultralytics |
| Eye CNN (fine-tuned) | `models/eye_model_finetuned.pth` | Eye region sad/neutral binary classification | PyTorch ResNet18 |
| Eye CNN (paper-style) | `models/paper_style_eye_region_fold1_best.pth` | Alternative eye model | PyTorch ResNet18 |
| Face Landmarker | `models/face_landmarker.task` | Facial landmark detection | MediaPipe |
| Gesture Recognizer | `models/gesture_recognizer.task` | Hand gesture recognition | MediaPipe |
| Meta-Learner | `weights/meta_learner_fusion_model.pkl` | Stacking fusion classifier | scikit-learn |
| Face Alignment | `weights/Alignment_RetinaFace.pth` | Face alignment preprocessing | PyTorch |
| MobileNet | `weights/mobilenetV1X0.25_pretrain.tar` | Lightweight backbone | PyTorch |

### Eye Model Details

The eye region model is a **ResNet18** with modified classification head:
```python
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2)  # Binary: neutral, sad
)
```

**Input**: 224x224 eye region crop (from upper 20%-50% of face)
**Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
**Output**: 2-class softmax probabilities [P_neutral, P_sad]

---

## 15. Experimental Results

### Meta-Learner Fusion Model Performance

Evaluated on **8,539 samples** from a mixed emotion dataset:

**Data Distribution:** Angry (1306), Happy (2000), Neutral (2000), Sad (2000), Surprise (1233)

| Model Architecture | Accuracy | Macro-F1 |
|----------|----------|----------|
| DeepFace Single Model (Classic Baseline) | 30.10% | 35.10% |
| HSEmotion Single Model (SOTA Baseline) | 71.94% | 74.07% |
| **Meta-Learner Fusion (Proposed)** | **72.80%** (+0.86%) | **74.73%** (+0.66%) |

### Fine-Grained Emotion Analysis

| Emotion | Metric | Before | After | Improvement | HCI Value |
|---------|--------|--------|-------|-------------|-----------|
| **Neutral** | Recall | 0.60 | **0.66** | +6% | Corrects "expressionless" misclassified as sad |
| **Angry** | Precision | 0.69 | **0.72** | +3% | Suppresses false positives via better neutral interception |
| **Sad** | Precision | -- | **0.87** | -- | High restraint: avoids unwarranted interventions |
| **Happy** | Recall | -- | **0.94** | -- | Non-destructive fusion preserves high-arousal detection |

### Four Core Innovation Mechanisms

1. **Occlusion-Aware Adaptive Fallback**: When eyes are occluded (sunglasses, head down), the system returns -1.0 for `P_eye`, triggering 100% trust in the global model. This prevents noise from geometric cropping of occluded regions.

2. **Asymmetric Confidence Gating**: High-confidence predictions (>= 80%) bypass fusion entirely, preventing the meta-learner from "catastrophically overriding" correct baseline predictions.

3. **Dynamic Trigger Boundary Extension**: Fusion activates when sad >= 25% OR neutral >= 25%, extending the trigger boundary downward to catch borderline cases that fixed thresholds would miss.

4. **Domain Adaptation via Soft Probabilities**: Instead of hard class labels, the meta-learner trains on soft probability distributions from base models, eliminating data distribution drift between training and inference.

---

## 16. Dataset

### OAHEGA Emotion Recognition Dataset

**Citation:**
```
Kovenko, Volodymyr; Shevchuk, Vitalii (2021), "OAHEGA: EMOTION RECOGNITION DATASET",
Mendeley Data, V2, doi: 10.17632/5ck5zz6f2c.2
```

**Characteristics:**
- RGB cropped face images
- Sources: Facebook, Instagram, YouTube videos, IMDB, AffectNet
- Classes used: Happy, Angry, Sad, Neutral, Surprise
- Note: The dataset-specific 'Ahegao' class and extremely low-frequency categories were excluded to align with mainstream emotion recognition benchmarks.

---

## 17. Academic References

1. **Gorbova et al. (2019)**: "Going deeper in hidden sadness recognition using spontaneous micro-expressions database." *Springer Science+Business Media*. -- Foundation for eye-region sadness analysis.

2. **HSEmotion (EmotiEffLib)**: High-speed emotion recognition using EfficientNet. -- Global face emotion baseline.

3. **DeepFace**: "DeepFace: Closing the Gap to Human-Level Performance in Face Verification." *CVPR 2014*. -- Face embedding and emotion recognition.

4. **HRV Analysis**: RMSSD methodology adapted from Heart Rate Variability literature for emotion fluctuation quantification.

5. **Kalman Filter**: 1D state estimation for mood trajectory smoothing.

6. **Dynamical Systems Theory**: Attractor model for personalized emotional baseline computation.

---

## License

MIT License
