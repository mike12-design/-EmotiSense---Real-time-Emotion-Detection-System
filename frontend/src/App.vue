<!-- frontend/src/App.vue -->
<template>
  <div id="app-container" class="app-dark">
    <router-view />
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import './styles/theme.css'

onMounted(() => {})
onUnmounted(() => {})
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

.app-dark {
  background:
    radial-gradient(ellipse 80% 55% at 50% 28%, rgba(56, 189, 248, 0.14) 0%, rgba(0, 0, 0, 0) 62%),
    radial-gradient(ellipse 70% 60% at 18% 72%, rgba(14, 165, 233, 0.1) 0%, rgba(0, 0, 0, 0) 58%),
    radial-gradient(ellipse 80% 80% at 85% 75%, rgba(34, 211, 238, 0.06) 0%, rgba(0, 0, 0, 0) 55%),
    linear-gradient(135deg, #050607 0%, #070a0f 35%, #06080b 100%);
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
