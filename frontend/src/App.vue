<!-- frontend/src/App.vue -->
<template>
  <div id="app-container" :style="bgStyle">
    <!-- 隐藏 img 用来检测背景是否存在 -->
    <img
      v-if="globalBg"
      :src="globalBg"
      @error="handleBgError"
      style="display: none"
    />

    <router-view />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import './styles/theme.css'

const API_BASE = "http://127.0.0.1:8000"
const username = localStorage.getItem('user')

const globalBg = ref('')
const bgStyle = ref({})

/**
 * 更新背景逻辑
 * 优先级：
 * 1 本地 custom_bg
 * 2 服务器用户背景
 * 3 默认柔和渐变
 */
const updateBackground = () => {
  const customBg = localStorage.getItem('custom_bg')

  if (customBg) {
    globalBg.value = customBg
  } else if (username) {
    globalBg.value = `${API_BASE}/assets/bg_${username}.jpg?t=${Date.now()}`
  } else {
    globalBg.value = ''
  }

  applyStyle()
}

const applyStyle = () => {
  if (globalBg.value) {
    bgStyle.value = {
      backgroundImage: `url(${globalBg.value})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundAttachment: 'fixed',
      backgroundRepeat: 'no-repeat',
      minHeight: '100vh',
      transition: 'background-image 0.5s ease'
    }
  } else {
    // 默认柔和渐变背景
    bgStyle.value = {
      background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #fafafc 100%)',
      minHeight: '100vh'
    }
  }
}

/**
 * 图片不存在 → 自动降级默认背景
 */
const handleBgError = () => {
  globalBg.value = ''
  applyStyle()
}

onMounted(() => {
  updateBackground()
  window.addEventListener('bg-changed', updateBackground)
})

onUnmounted(() => {
  window.removeEventListener('bg-changed', updateBackground)
})
</script>

<style>
/* 全局重置 - 由 theme.css 处理 */
body {
  margin: 0;
  padding: 0;
  font-family:
    -apple-system,
    'Helvetica Neue',
    Helvetica,
    'PingFang SC',
    'Hiragino Sans GB',
    'Microsoft YaHei',
    'Noto Sans SC',
    Arial,
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app-container {
  min-height: 100vh;
  width: 100vw;
}

/* Element Plus 全局样式优化 */
.el-button {
  border-radius: var(--radius-md) !important;
}

.el-card {
  border-radius: var(--radius-lg) !important;
}

.el-input__wrapper {
  border-radius: var(--radius-md) !important;
}

.el-message {
  border-radius: var(--radius-md) !important;
}
</style>
