<!-- frontend/src/layouts/AdminLayout.vue -->
<template>
  <el-container class="admin-layout">
    <!-- 左侧边栏 -->
    <el-aside width="240px" class="aside">
      <div class="logo-section">
        <div class="logo-icon">💙</div>
        <div class="logo-text">
          <span class="logo-title">EmotiSense</span>
          <span class="logo-subtitle">管理系统</span>
        </div>
      </div>

      <el-menu
        :default-active="$route.path"
        router
        class="side-menu"
        background-color="transparent"
        text-color="#a1a1aa"
        active-text-color="#38bdf8"
      >
        <el-menu-item index="/admin/users" class="menu-item">
          <el-icon class="menu-icon"><User /></el-icon>
          <span>人员管理</span>
        </el-menu-item>
        <el-menu-item index="/admin/analytics" class="menu-item">
          <el-icon class="menu-icon"><DataAnalysis /></el-icon>
          <span>数据分析</span>
        </el-menu-item>
        <el-menu-item index="/admin/resources" class="menu-item">
          <el-icon class="menu-icon"><Files /></el-icon>
          <span>资源管理</span>
        </el-menu-item>
        <el-menu-item index="/admin/logs" class="menu-item">
          <el-icon class="menu-icon"><Document /></el-icon>
          <span>系统日志</span>
        </el-menu-item>
      </el-menu>

      <!-- 底部返回按钮 -->
      <div class="back-btn" @click="router.push('/')">
        <el-icon><Monitor /></el-icon>
        <span>返回监测页面</span>
      </div>
    </el-aside>

    <el-container class="main-container">
      <!-- 顶部导航 -->
      <el-header class="header">
        <div class="header-left">
          <div class="breadcrumb">
            <el-icon><House /></el-icon>
            <span class="breadcrumb-separator">/</span>
            <span class="breadcrumb-current">{{ currentPathName }}</span>
          </div>
        </div>

        <div class="header-right">
          <div class="user-profile">
            <div class="user-avatar">
              <el-icon><UserFilled /></el-icon>
            </div>
            <div class="user-info">
              <span class="user-name">管理员</span>
              <span class="user-role">Administrator</span>
            </div>
          </div>

          <el-divider direction="vertical" class="divider" />

          <div class="logout-btn" @click="handleLogout">
            <el-icon><SwitchButton /></el-icon>
            <span>退出</span>
          </div>
        </div>
      </el-header>

      <!-- 内容区 -->
      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Odometer, User, DataAnalysis, Files, Document,
  Monitor, SwitchButton, UserFilled, House
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const currentPathName = computed(() => {
  const map = {
    '/admin/dashboard': '仪表盘',
    '/admin/users': '人员管理',
    '/admin/analytics': '数据分析',
    '/admin/resources': '资源管理',
    '/admin/logs': '系统日志'
  }
  return map[route.path] || '未知'
})

const handleLogout = () => {
  localStorage.clear()
  router.push('/login')
}
</script>

<style scoped>
/* ===== 主布局 ===== */
.admin-layout {
  height: 100vh;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
}

/* ===== 左侧边栏 ===== */
.aside {
  background: linear-gradient(180deg, #0c4a6e 0%, #075985 100%);
  color: #fff;
  display: flex;
  flex-direction: column;
  box-shadow: 4px 0 24px rgba(14, 165, 233, 0.08);
}

/* Logo 区 */
.logo-section {
  height: 80px;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.logo-icon {
  font-size: 32px;
  filter: drop-shadow(0 4px 8px rgba(14, 165, 233, 0.3));
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.logo-title {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, #38bdf8, #0ea5e9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
}

.logo-subtitle {
  font-size: 11px;
  color: #a1a1aa;
  margin-top: 2px;
  letter-spacing: 1px;
}

/* 菜单 */
.side-menu {
  flex: 1;
  border-right: none;
  padding: 16px 12px;
}

.menu-item {
  margin-bottom: 8px;
  border-radius: 12px;
  height: 48px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.menu-item:hover {
  background: rgba(14, 165, 233, 0.08);
  color: #7dd3fc;
}

.menu-item.is-active {
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(2, 132, 199, 0.15));
  color: #38bdf8;
  box-shadow: 0 2px 12px rgba(14, 165, 233, 0.15);
}

.menu-icon {
  font-size: 20px;
}

.menu-item span {
  font-size: 14px;
  font-weight: 500;
}

/* 底部返回按钮 */
.back-btn {
  margin: 16px;
  padding: 12px 16px;
  background: rgba(14, 165, 233, 0.15);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #7dd3fc;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.25s;
  border: 1px solid rgba(14, 165, 233, 0.2);
}

.back-btn:hover {
  background: rgba(14, 165, 233, 0.25);
  color: #fff;
  transform: translateY(-2px);
}

/* ===== 主容器 ===== */
.main-container {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ===== 顶部 Header ===== */
.header {
  height: 64px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(139, 92, 246, 0.08);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 2px 16px rgba(139, 92, 246, 0.04);
}

.header-left {
  display: flex;
  align-items: center;
}

.breadcrumb {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: var(--text-secondary);
}

.breadcrumb .el-icon {
  font-size: 18px;
  color: var(--primary-400);
}

.breadcrumb-separator {
  color: var(--text-disabled);
}

.breadcrumb-current {
  font-weight: 600;
  color: var(--text-primary);
}

/* ===== 右侧用户区 ===== */
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 14px;
  background: var(--primary-50);
  border-radius: 12px;
  border: 1px solid var(--primary-100);
}

.user-avatar {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, var(--primary-400), var(--primary-500));
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.user-avatar .el-icon {
  font-size: 20px;
}

.user-info {
  display: flex;
  flex-direction: column;
}

.user-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.user-role {
  font-size: 11px;
  color: var(--text-tertiary);
  margin-top: 2px;
}

.divider {
  height: 24px;
  border-left-color: var(--border-light);
}

.logout-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border-radius: 10px;
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.logout-btn:hover {
  background: #fef0f0;
  color: #ef4444;
}

/* ===== 主内容区 ===== */
.main-content {
  flex: 1;
  background: transparent;
  padding: 20px;
  overflow-y: auto;
}
</style>
