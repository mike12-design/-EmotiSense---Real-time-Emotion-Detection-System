<!-- frontend/src/views/Login.vue -->
<template>
  <div class="login-container">
    <!-- 装饰性背景元素 -->
    <div class="bg-decoration">
      <div class="bubble bubble-1"></div>
      <div class="bubble bubble-2"></div>
      <div class="bubble bubble-3"></div>
      <div class="bubble bubble-4"></div>
    </div>

    <div class="login-box animate-fade-in-up">
      <!-- Logo 区域 -->
      <div class="logo-section">
        <div class="logo-icon animate-float">
          <span class="emoji">💙</span>
        </div>
        <h1 class="title">EmotiSense</h1>
        <p class="subtitle">实时情绪感知系统</p>
      </div>

      <!-- 表单区域 -->
      <el-form :model="form" class="login-form" size="large">
        <el-form-item>
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            :prefix-icon="User"
            class="soft-input"
          />
        </el-form-item>

        <el-form-item>
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            :prefix-icon="Lock"
            show-password
            class="soft-input"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <!-- 注册模式下多一个确认密码 -->
        <el-form-item v-if="!isLogin">
          <el-input
            v-model="form.confirmPassword"
            type="password"
            placeholder="请再次输入密码"
            :prefix-icon="Lock"
            show-password
            class="soft-input"
          />
        </el-form-item>

        <!-- 提交按钮 -->
        <el-button
          type="primary"
          class="login-btn animate-scale-in"
          @click="handleSubmit"
          :loading="loading"
          size="large"
        >
          <span class="btn-content">
            {{ isLogin ? ' welcomes 回来' : '开启情绪之旅' }}
          </span>
        </el-button>
      </el-form>

      <!-- 切换模式 -->
      <div class="mode-switch">
        <span class="switch-text">
          {{ isLogin ? '还没有账号？' : '已有账号？' }}
        </span>
        <el-link
          class="switch-link"
          type="primary"
          @click="toggleMode"
          :underline="false"
        >
          {{ isLogin ? '立即注册' : '返回登录' }}
        </el-link>
      </div>

      <!-- 登录提示 -->
      <div v-if="isLogin" class="tips">
        <el-divider direction="horizontal" class="tips-divider">
          <span class="divider-text">提示</span>
        </el-divider>
        <p class="tips-text">
          <el-icon class="tips-icon"><InfoFilled /></el-icon>
          管理员账号：<code>admin</code> / <code>123456</code>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { User, Lock, InfoFilled } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import axios from 'axios';

const router = useRouter();
const loading = ref(false);
const isLogin = ref(true);

const form = ref({
  username: '',
  password: '',
  confirmPassword: ''
});

const API_BASE = 'http://127.0.0.1:8000';

const toggleMode = () => {
  isLogin.value = !isLogin.value;
  form.value = {
    username: '',
    password: '',
    confirmPassword: ''
  };
};

const handleSubmit = async () => {
  if (!form.value.username || !form.value.password) {
    return ElMessage.warning({
      message: '请填写完整信息哦～',
      type: 'warning'
    });
  }

  loading.value = true;

  try {
    if (isLogin.value) {
      // 登录
      const res = await axios.post(`${API_BASE}/api/login`, {
        username: form.value.username,
        password: form.value.password
      });

      if (res.data.success) {
        localStorage.setItem('role', res.data.role);
        localStorage.setItem('user', res.data.username);

        ElMessage.success({
          message: `欢迎回来，${res.data.username}！`,
          type: 'success',
          duration: 1500
        });

        // 根据角色跳转
        setTimeout(() => {
          router.push(res.data.role === 'admin' ? '/admin/users' : '/user/home');
        }, 800);
      } else {
        ElMessage.error({
          message: res.data.message || '登录失败，请检查账号密码',
          type: 'error'
        });
      }
    } else {
      // 注册
      if (form.value.password !== form.value.confirmPassword) {
        loading.value = false;
        return ElMessage.error({
          message: '两次密码输入不一致哦～',
          type: 'error'
        });
      }

      const res = await axios.post(`${API_BASE}/api/register`, {
        username: form.value.username,
        password: form.value.password
      });

      if (res.data.success) {
        ElMessage.success({
          message: '注册成功！请登录',
          type: 'success',
          duration: 1500
        });
        isLogin.value = true;
        form.value = {
          username: '',
          password: '',
          confirmPassword: ''
        };
      } else {
        ElMessage.error({
          message: res.data.message || '注册失败',
          type: 'error'
        });
      }
    }
  } catch (err) {
    ElMessage.error({
      message: '服务器连接失败，请检查后端服务是否启动',
      type: 'error'
    });
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
/* ===== 主容器 ===== */
.login-container {
  height: 100vh;
  width: 100vw;
  display: flex;
  justify-content: center;
  align-items: center;
  position: fixed;
  top: 0;
  left: 0;
  overflow: hidden;

  /* 柔和渐变背景 */
  background: linear-gradient(135deg,
    #f0f9ff 0%,
    #e0f2fe 25%,
    #bae6fd 50%,
    #e0f2fe 75%,
    #fafafc 100%
  );
  background-size: 200% 200%;
  animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

/* ===== 背景装饰气泡 ===== */
.bg-decoration {
  position: absolute;
  width: 100%;
  height: 100%;
  pointer-events: none;
  overflow: hidden;
}

.bubble {
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(
    135deg,
    rgba(14, 165, 233, 0.15),
    rgba(56, 189, 248, 0.1)
  );
  animation: floatUp infinite ease-in-out;
}

.bubble-1 {
  width: 200px;
  height: 200px;
  left: 10%;
  bottom: -200px;
  animation-delay: 0s;
  animation-duration: 8s;
}

.bubble-2 {
  width: 150px;
  height: 150px;
  left: 70%;
  bottom: -150px;
  animation-delay: 2s;
  animation-duration: 10s;
}

.bubble-3 {
  width: 100px;
  height: 100px;
  left: 40%;
  bottom: -100px;
  animation-delay: 4s;
  animation-duration: 12s;
}

.bubble-4 {
  width: 180px;
  height: 180px;
  left: 85%;
  bottom: -180px;
  animation-delay: 1s;
  animation-duration: 9s;
}

@keyframes floatUp {
  0%, 100% {
    transform: translateY(0) rotate(0deg);
    opacity: 0.6;
  }
  50% {
    transform: translateY(-100vh) rotate(180deg);
    opacity: 0.3;
  }
}

/* ===== 登录卡片 ===== */
.login-box {
  width: 420px;
  padding: 48px 40px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow:
    0 20px 60px rgba(14, 165, 233, 0.1),
    0 0 0 1px rgba(14, 165, 233, 0.05);
  position: relative;
  z-index: 1;
}

/* ===== Logo 区域 ===== */
.logo-section {
  text-align: center;
  margin-bottom: 40px;
}

.logo-icon {
  width: 80px;
  height: 80px;
  margin: 0 auto 20px;
  background: linear-gradient(135deg, #e0f2fe, #bae6fd);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 24px rgba(14, 165, 233, 0.15);
}

.logo-icon .emoji {
  font-size: 40px;
}

.title {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px 0;
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
}

.subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  margin: 0;
  font-weight: 400;
}

/* ===== 表单样式 ===== */
.login-form {
  margin-top: 32px;
}

:deep(.el-form-item) {
  margin-bottom: 20px;
}

:deep(.el-input__wrapper) {
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 14px;
  box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.08) inset;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

:deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.2) inset;
}

:deep(.el-input__wrapper.is-focus) {
  box-shadow:
    0 0 0 1px rgba(14, 165, 233, 0.4) inset,
    0 4px 12px rgba(14, 165, 233, 0.1);
}

:deep(.el-input__inner) {
  font-size: 15px;
  color: var(--text-primary);
}

:deep(.el-input__prefix) {
  color: var(--primary-400);
}

/* ===== 按钮样式 ===== */
.login-btn {
  width: 100%;
  height: 52px;
  margin-top: 24px;
  border-radius: 14px;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  border: none;
  box-shadow:
    0 4px 16px rgba(14, 165, 233, 0.3),
    0 2px 8px rgba(14, 165, 233, 0.2);
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.login-btn:hover {
  background: linear-gradient(135deg, #0284c7, #0369a1);
  transform: translateY(-2px);
  box-shadow:
    0 8px 24px rgba(14, 165, 233, 0.4),
    0 4px 12px rgba(14, 165, 233, 0.25);
}

.login-btn:active {
  transform: translateY(0);
}

.btn-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ===== 模式切换 ===== */
.mode-switch {
  margin-top: 28px;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.switch-text {
  font-size: 14px;
  color: var(--text-secondary);
}

.switch-link {
  font-size: 14px;
  font-weight: 600;
  color: var(--primary-500);
  cursor: pointer;
  transition: all 0.2s;
}

.switch-link:hover {
  color: var(--primary-600);
}

/* ===== 提示区域 ===== */
.tips {
  margin-top: 24px;
  text-align: center;
}

.tips-divider {
  margin: 0 0 12px 0;
}

:deep(.el-divider__text) {
  font-size: 12px;
  color: var(--text-tertiary);
  font-weight: 500;
}

.tips-text {
  font-size: 13px;
  color: var(--text-tertiary);
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.tips-icon {
  font-size: 14px;
  color: var(--primary-400);
}

.tips-text code {
  background: var(--primary-50);
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 12px;
  color: var(--primary-600);
  font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
}
</style>
