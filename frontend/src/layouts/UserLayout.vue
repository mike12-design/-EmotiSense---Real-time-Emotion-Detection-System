<!-- frontend/src/layouts/UserLayout.vue -->
<template>
  <el-container class="user-layout">
    <el-header class="user-header">
      <!-- 【修改点】添加 @click 跳转事件，并增加 pointer 手势样式 -->
      <div class="logo-area" @click="router.push('/')" title="返回监控大屏">
        <span class="logo-icon">🌿</span>
        <span class="logo-text">EmotiSense | 个人空间</span>
      </div>
      
      <div class="nav-menu">
        <el-menu mode="horizontal" :default-active="$route.path" router :ellipsis="false">
          <el-menu-item index="/user/home">我的概览</el-menu-item>
          <el-menu-item index="/user/history">心情足迹</el-menu-item>
          <el-menu-item index="/user/diary">我的日记</el-menu-item>
          <el-menu-item index="/user/settings">设置</el-menu-item>
        </el-menu>
      </div>

      <div class="user-actions">
        <span class="username">Hi, {{ username }}</span>
        <el-button link type="danger" @click="handleLogout">退出</el-button>
      </div>
    </el-header>

    <el-main class="user-content">
      <div class="container">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter(); // 确保这里已经定义了 router
const username = ref(localStorage.getItem('user') || 'User');

const handleLogout = () => {
  localStorage.clear();
  router.push('/login');
};
</script>

<style scoped>
.user-layout { min-height: 100vh; background-color: #f5f7fa; }
.user-header {
  background: white;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 40px;
  z-index: 10;
}

/* 【修改点】增加鼠标悬停效果，让用户知道这是可以点击的 */
.logo-area { 
  display: flex; 
  align-items: center; 
  cursor: pointer; /* 鼠标变小手 */
  transition: opacity 0.2s;
}
.logo-area:hover {
  opacity: 0.7; /* 悬停时稍微透明 */
}

.logo-text { font-size: 18px; font-weight: 600; color: #0ea5e9; margin-left: 10px; }
.user-content {
  padding: 0;
}
.container {
  width: 100%;        
}
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>