<!-- frontend/src/views/user/MonitorMode.vue -->
<template>
  <div class="mood-diary-app" ref="appRoot">

    <!-- 顶部 Banner：访客模式提示 -->
    <transition name="slide-down">
      <div v-if="isGuest" class="top-banner">
        <div class="banner-inner">
          <el-icon class="banner-icon"><InfoFilled /></el-icon>
          <span class="banner-text">您正在使用访客模式，登录后可同步数据</span>
          <el-button class="banner-btn" size="small" @click="$router.push('/login')">
            立即登录
          </el-button>
        </div>
      </div>
    </transition>

    <!-- 头部导航 -->
    <header class="main-header">
      <div class="header-inner">

        <!-- 左侧：Logo -->
        <div class="logo-group" @click="refreshPage" title="刷新页面">
          <div class="logo-icon-wrapper">
            <span class="logo-icon">💙</span>
          </div>
          <div class="logo-text-group">
            <span class="logo-text">EmotiSense</span>
            <span class="logo-status">情绪实时监测系统</span>
          </div>
        </div>

        <!-- 中间：导航按钮 -->
        <nav class="header-nav">
          <router-link to="/user/home" class="nav-btn">
            <el-icon><HomeFilled /></el-icon>
            <span>我的概览</span>
          </router-link>
          <router-link to="/user/history" class="nav-btn">
            <el-icon><TrendCharts /></el-icon>
            <span>心情足迹</span>
          </router-link>
          <router-link to="/user/settings" class="nav-btn">
            <el-icon><Setting /></el-icon>
            <span>设置</span>
          </router-link>
        </nav>

        <!-- 右侧：用户状态 -->
        <div class="user-actions">
          <template v-if="isGuest">
            <el-button class="btn-login" size="default" @click="$router.push('/login')">
              登录 / 注册
            </el-button>
          </template>
          <template v-else>
            <div class="user-info">
              <div class="user-avatar">
                {{ username.charAt(0).toUpperCase() }}
              </div>
              <span class="user-name">{{ username }}</span>
              <el-divider direction="vertical" />
              <el-button link type="danger" size="small" @click="handleLogout">
                退出
              </el-button>
            </div>
          </template>
        </div>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="main-content">
      <div class="grid-layout">

        <!-- 左侧栏 -->
        <div class="col-left">
          <!-- 1. 日历卡片 -->
          <div class="card calendar-card">
            <div class="card-header">
              <el-icon class="card-icon"><Calendar /></el-icon>
              <h3 class="card-title">情绪分布</h3>
              <span class="card-subtitle">本月</span>
            </div>
            <div class="calendar-grid">
              <div v-for="d in ['日','一','二','三','四','五','六']" :key="d" class="weekday-header">
                {{ d }}
              </div>

              <div
                v-for="i in 30"
                :key="i"
                class="day-cell"
                :class="{ today: i === todayDate }"
              >
                <span class="day-number">{{ i }}</span>
                <div
                  v-if="dailyMoods[i]"
                  class="mood-emoji-wrap"
                  :title="`${i}号：${dailyMoods[i]}`"
                >
                  {{ getMoodEmoji(dailyMoods[i]) }}
                </div>
              </div>
            </div>
          </div>

          <!-- 2. 感知日志 -->
          <div class="card log-card">
            <div class="card-header">
              <div class="pulse-indicator"></div>
              <h3 class="card-title">感知日志</h3>
              <span class="card-subtitle">实时</span>
            </div>
            <div class="log-list">
              <transition-group name="list">
                <div v-for="(log, idx) in recentLogs" :key="log.id || idx" class="log-item">
                  <span class="log-time">{{ log.time }}</span>
                  <span class="log-emotion" :style="{ backgroundColor: getEmotionColor(log.emotion) }">
                    {{ log.emotion }}
                  </span>
                  <span class="log-score">{{ (log.score * 100).toFixed(0) }}%</span>
                </div>
              </transition-group>
              <div v-if="recentLogs.length === 0" class="empty-state">
                <span class="empty-emoji">👋</span>
                <p class="empty-text">等待 AI 接入...</p>
              </div>
            </div>
          </div>
        </div>

        <!-- 右侧栏 -->
        <div class="col-right">
          <!-- 1. 监控大屏 -->
          <div class="card monitor-card">
            <div class="monitor-header">
              <div class="monitor-title-group">
                <div class="pulse-dot"></div>
                <h2 class="monitor-title">AI 视觉感知中</h2>
              </div>
              <transition name="fade" mode="out-in">
                <span
                  :key="currentEmotion"
                  class="emotion-badge"
                  :style="{ backgroundColor: getEmotionColor(currentEmotion) }"
                >
                  {{ getMoodEmoji(currentEmotion.toLowerCase()) }} {{ currentEmotion }}
                </span>
              </transition>
            </div>

            <div class="camera-container">
              <img :src="`${API_BASE}/video_feed`" class="live-feed" alt="实时视频流" />

              <!-- HUD 装饰层 -->
              <div class="hud-overlay">
                <div class="scan-line"></div>
                <div class="corner-tl"></div>
                <div class="corner-tr"></div>
                <div class="corner-bl"></div>
                <div class="corner-br"></div>
              </div>
            </div>

            <div class="control-bar">
              <el-button class="btn-ctrl" size="small" @click="toggleFullscreen">
                <el-icon><FullScreen /></el-icon>
                {{ isFullscreen ? '退出全屏' : '全屏' }}
              </el-button>
            </div>
          </div>

          <!-- 2. 日记功能区 -->
          <div class="card diary-section">
            <div class="diary-header">
              <div class="diary-title-group">
                <el-icon class="diary-icon"><Edit /></el-icon>
                <h3 class="diary-title">我的日记</h3>
              </div>
              <el-button class="btn-write-diary" size="small" @click="goToWriteDiary">
                <el-icon><EditPen /></el-icon>
                <span>记录心情</span>
                <el-icon class="arrow-icon"><ArrowRight /></el-icon>
              </el-button>
            </div>

            <!-- 空状态 -->
            <div v-if="diaries.length === 0" class="diary-empty-state">
              <span class="diary-empty-emoji">📝</span>
              <p class="diary-empty-text">记录下此刻的心情吗？</p>
              <el-button class="btn-start-diary" type="primary" @click="goToWriteDiary">
                开始记录
              </el-button>
            </div>

            <!-- 日记列表 -->
            <div v-else class="diary-list-container">
              <div
                v-for="item in diaries.slice(0, 3)"
                :key="item.id"
                class="diary-item"
                @click="$router.push('/user/diary')"
              >
                <div class="diary-content">
                  <span class="diary-date">{{ formatDate(item.timestamp) }}</span>
                  <p class="diary-text">{{ item.content }}</p>
                </div>
                <span class="diary-emotion-tag" :style="{ backgroundColor: getEmotionColor(item.emotion) }">
                  {{ getMoodEmoji(item.emotion.toLowerCase()) }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 隐藏的播放器 -->
    <audio ref="audioPlayer" hidden></audio>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import screenfull from 'screenfull'
import { ElMessage } from 'element-plus'
import {
  HomeFilled, TrendCharts, Setting, InfoFilled, Calendar,
  FullScreen, Edit, EditPen, ArrowRight, Camera
} from '@element-plus/icons-vue'

const router = useRouter()
const API_BASE = 'http://127.0.0.1:8000'

/* ========= 状态变量 ========= */
const isGuest = ref(true)
const username = ref('')
const currentEmotion = ref('Neutral')
const isIntervening = ref(false)
const recentLogs = ref([])
const appRoot = ref(null)
const isFullscreen = ref(false)
const audioPlayer = ref(null)

let pollingTimer = null

// 日历相关
const todayDate = new Date().getDate()
const dailyMoods = ref({})

// 日记相关
const diaries = ref([])

/* ========= 辅助方法 ========= */
const getMoodEmoji = (mood) => {
  const map = {
    happy: '😊', sad: '😢', angry: '😡', neutral: '😐',
    surprise: '😲', fear: '😨', disgust: '🤢'
  }
  return map[mood?.toLowerCase()] || '😶'
}

const getEmotionColor = (e) => {
  const colors = {
    happy: '#34d399', sad: '#60a5fa', angry: '#f87171',
    neutral: '#7dd3fc', surprise: '#f472b6', fear: '#fb923c', disgust: '#a3e635'
  }
  return colors[e?.toLowerCase()] || '#71717a'
}

const formatDate = (ts) => ts?.replace('T', ' ').split('.')[0]

/* ========= 生命周期 ========= */
onMounted(() => {
  const savedUser = localStorage.getItem('user')
  if (savedUser) {
    username.value = savedUser
    isGuest.value = false
  }

  if (screenfull.isEnabled) {
    screenfull.on('change', () => { isFullscreen.value = screenfull.isFullscreen })
  }

  pollingTimer = setInterval(fetchStatus, 2000)
  fetchDiaries()
  fetchCalendarData()
})

onUnmounted(() => {
  if (screenfull.isEnabled) screenfull.off('change')
  clearInterval(pollingTimer)
})

/* ========= 业务逻辑 ========= */
const fetchStatus = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/status`)
    const backendEmotion = res.data.current_emotion

    if (backendEmotion) {
      currentEmotion.value = backendEmotion
    }

    if (res.data.should_intervene) {
      handleIntervention(res.data.resource)
    }

    if (Math.random() > 0.7) {
      addLogToUI(currentEmotion.value)
    }
  } catch (e) {
    // 静默失败
  }
}

const fetchDiaries = async () => {
  if (isGuest.value) return
  try {
    const res = await axios.get(`${API_BASE}/api/my/diaries?username=${username.value}`)
    diaries.value = res.data
  } catch (e) {
    console.warn('日记数据加载失败')
  }
}

const fetchCalendarData = async () => {
  if (isGuest.value) return
  try {
    dailyMoods.value = {
      [todayDate]: currentEmotion.value.toLowerCase(),
      [todayDate - 1]: 'sad',
      [todayDate - 2]: 'happy'
    }
  } catch (e) {
    console.warn('日历数据失败')
  }
}

const goToWriteDiary = () => {
  router.push({
    path: '/user/diary',
    query: { emotion: currentEmotion.value }
  })
}

const handleLogout = () => {
  localStorage.clear()
  isGuest.value = true
  username.value = ''
  router.push('/login')
}

const refreshPage = () => {
  window.location.reload()
}

const toggleFullscreen = () => {
  if (screenfull.isEnabled) {
    screenfull.toggle(appRoot.value)
  }
}

const addLogToUI = (emotion) => {
  const now = new Date()
  recentLogs.value.unshift({
    id: Date.now(),
    time: `${now.getHours()}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`,
    emotion: emotion,
    score: 0.7 + Math.random() * 0.29
  })
  if (recentLogs.value.length > 10) recentLogs.value.pop()
}

const handleIntervention = (resource) => {
  if (isIntervening.value) return
  isIntervening.value = true

  if (resource.audio_url && audioPlayer.value) {
    audioPlayer.value.src = `${API_BASE}/${resource.audio_url}`
    audioPlayer.value.play()
  }

  setTimeout(() => {
    isIntervening.value = false
  }, 5000)
}
</script>

<style scoped>
/* ===== 主容器 ===== */
.mood-diary-app {
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

/* ===== 顶部 Banner ===== */
.top-banner {
  background: linear-gradient(90deg, #0ea5e9, #0284c7);
  color: white;
  padding: 0;
  overflow: hidden;
}

.banner-inner {
  max-width: 1400px;
  margin: 0 auto;
  padding: 10px 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.banner-icon {
  font-size: 18px;
  animation: pulse-icon 2s infinite;
}

@keyframes pulse-icon {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.1); }
}

.banner-text {
  font-size: 14px;
  font-weight: 500;
}

.banner-btn {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
}

.banner-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* ===== Header ===== */
.main-header {
  height: 64px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(14, 165, 233, 0.1);
  flex-shrink: 0;
  z-index: 50;
  box-shadow: 0 2px 20px rgba(14, 165, 233, 0.05);
}

.header-inner {
  height: 100%;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}

/* Logo */
.logo-group {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  border-radius: 12px;
  transition: all 0.2s;
}

.logo-group:hover {
  background: var(--primary-50);
}

.logo-icon-wrapper {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #e0f2fe, #bae6fd);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15);
}

.logo-icon {
  font-size: 22px;
}

.logo-text-group {
  display: flex;
  flex-direction: column;
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.logo-status {
  font-size: 11px;
  color: var(--text-tertiary);
  margin-top: 2px;
}

/* 导航 */
.header-nav {
  display: flex;
  gap: 8px;
}

.nav-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border-radius: 10px;
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 500;
  text-decoration: none;
  transition: all 0.2s;
}

.nav-btn:hover {
  background: var(--primary-50);
  color: var(--primary-600);
}

.nav-btn.router-link-active {
  background: var(--primary-100);
  color: var(--primary-600);
}

/* 用户区 */
.user-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.btn-login {
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  border: none;
  color: white;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
}

.btn-login:hover {
  box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35);
  transform: translateY(-1px);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.user-avatar {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #38bdf8, #0ea5e9);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
}

.user-name {
  font-size: 14px;
  color: var(--text-primary);
  font-weight: 500;
}

/* ===== 主内容 ===== */
.main-content {
  flex: 1;
  padding: 20px;
  overflow: hidden;
}

.grid-layout {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 20px;
  height: 100%;
}

/* ===== 卡片通用样式 ===== */
.card {
  background: rgba(255, 255, 255, 0.75);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.4);
  box-shadow:
    0 4px 20px rgba(14, 165, 233, 0.08),
    0 1px 3px rgba(14, 165, 233, 0.05);
  padding: 16px;
  display: flex;
  flex-direction: column;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card:hover {
  box-shadow:
    0 8px 30px rgba(14, 165, 233, 0.12),
    0 2px 6px rgba(14, 165, 233, 0.08);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.card-icon {
  font-size: 18px;
  color: var(--primary-500);
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.card-subtitle {
  font-size: 12px;
  color: var(--text-tertiary);
  background: var(--primary-50);
  padding: 2px 8px;
  border-radius: 6px;
}

/* ===== 左侧栏 ===== */
.col-left {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  overflow: hidden;
}

/* 日历 */
.calendar-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 6px;
}

.weekday-header {
  text-align: center;
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 500;
  padding: 4px 0;
}

.day-cell {
  aspect-ratio: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  background: rgba(249, 250, 251, 0.6);
  border-radius: 10px;
  color: var(--text-secondary);
  position: relative;
  transition: all 0.2s;
}

.day-cell:hover {
  background: var(--primary-50);
  transform: scale(1.05);
}

.day-number {
  margin-top: -2px;
  font-size: 10px;
}

.mood-emoji-wrap {
  font-size: 14px;
  line-height: 1;
  margin-top: 2px;
  filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.08));
  animation: popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

@keyframes popIn {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}

.day-cell.today {
  background: var(--primary-100);
  color: var(--primary-700);
  font-weight: 600;
  border: 1px solid var(--primary-200);
}

/* 日志 */
.log-card {
  flex: 1;
}

.log-list {
  flex: 1;
  overflow-y: auto;
}

.log-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 10px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.6);
  transition: all 0.2s;
}

.log-item:hover {
  background: rgba(255, 255, 255, 0.9);
  transform: translateX(4px);
}

.log-time {
  font-size: 12px;
  color: var(--text-tertiary);
  font-family: 'SF Mono', monospace;
}

.log-emotion {
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  color: white;
}

.log-score {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

.list-enter-active,
.list-leave-active {
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.list-enter-from,
.list-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  color: var(--text-tertiary);
}

.empty-emoji {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.6;
}

.empty-text {
  font-size: 14px;
  margin: 0;
}

/* ===== 右侧栏 ===== */
.col-right {
  display: flex;
  flex-direction: row;
  gap: 20px;
  height: 100%;
  overflow: hidden;
}

.col-right > .card {
  flex: 1;
  width: 0;
}

/* 监控卡片 */
.monitor-card {
  padding: 0;
  overflow: hidden;
}

.monitor-header {
  padding: 14px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: white;
  border-bottom: 1px solid var(--border-light);
}

.monitor-title-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.pulse-dot {
  width: 10px;
  height: 10px;
  background: linear-gradient(135deg, #34d399, #10b981);
  border-radius: 50%;
  animation: pulse-dot 2s infinite;
  box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
}

@keyframes pulse-dot {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

.monitor-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.emotion-badge {
  padding: 6px 14px;
  border-radius: 20px;
  color: white;
  font-weight: 600;
  font-size: 13px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  gap: 6px;
}

/* 相机容器 */
.camera-container {
  flex: 1;
  background: #000;
  position: relative;
  overflow: hidden;
  min-height: 300px;
}

.live-feed {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.hud-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.scan-line {
  position: absolute;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.6), transparent);
  animation: scan 3s linear infinite;
}

@keyframes scan {
  0% { top: 0; }
  100% { top: 100%; }
}

.corner-tl, .corner-tr, .corner-bl, .corner-br {
  position: absolute;
  width: 30px;
  height: 30px;
  border: 2px solid rgba(14, 165, 233, 0.4);
}

.corner-tl { top: 20px; left: 20px; border-right: none; border-bottom: none; }
.corner-tr { top: 20px; right: 20px; border-left: none; border-bottom: none; }
.corner-bl { bottom: 20px; left: 20px; border-right: none; border-top: none; }
.corner-br { bottom: 20px; right: 20px; border-left: none; border-top: none; }

/* 控制栏 */
.control-bar {
  padding: 12px 20px;
  background: white;
  border-top: 1px solid var(--border-light);
}

.btn-ctrl {
  background: var(--lavender-gray-100);
  border: none;
  color: var(--text-secondary);
}

.btn-ctrl:hover {
  background: var(--lavender-gray-200);
}

/* 日记卡片 */
.diary-section {
  display: flex;
  flex-direction: column;
}

.diary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.diary-title-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.diary-icon {
  font-size: 18px;
  color: var(--primary-500);
}

.diary-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.btn-write-diary {
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  border: none;
  color: white;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
}

.btn-write-diary:hover {
  box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35);
  transform: translateY(-1px);
}

.arrow-icon {
  transition: transform 0.2s;
}

.btn-write-diary:hover .arrow-icon {
  transform: translateX(3px);
}

/* 日记空状态 */
.diary-empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--primary-50), var(--primary-100));
  border-radius: 12px;
  border: 2px dashed var(--primary-200);
  padding: 40px 20px;
}

.diary-empty-emoji {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.7;
}

.diary-empty-text {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 16px 0;
}

.btn-start-diary {
  border-radius: 10px;
  padding: 10px 24px;
}

/* 日记列表 */
.diary-list-container {
  flex: 1;
  overflow-y: auto;
}

.diary-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 12px;
  margin-bottom: 10px;
  border-left: 4px solid var(--primary-400);
  cursor: pointer;
  transition: all 0.2s;
}

.diary-item:hover {
  background: rgba(255, 255, 255, 0.95);
  transform: translateX(6px);
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.08);
}

.diary-content {
  flex: 1;
  min-width: 0;
}

.diary-date {
  font-size: 12px;
  color: var(--text-tertiary);
}

.diary-text {
  margin: 6px 0 0 0;
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.diary-emotion-tag {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  margin-left: 12px;
}

/* ===== 过渡动画 ===== */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease;
}

.slide-down-enter-from,
.slide-down-leave-to {
  transform: translateY(-100%);
  opacity: 0;
}

.fade-enter-active,
.fade-leave-active {
  transition: all 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: scale(0.9);
}
</style>
