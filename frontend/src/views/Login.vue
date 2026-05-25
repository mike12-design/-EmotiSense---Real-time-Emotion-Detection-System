<!-- frontend/src/views/Login.vue -->
<template>
  <div class="login-page">
    <div class="login-bg" aria-hidden="true"></div>

    <main class="login-main">
      <section class="login-left" aria-label="intro">
        <div class="left-inner">
          <div class="left-wave" aria-hidden="true">
            <span class="bar b1"></span>
            <span class="bar b2"></span>
            <span class="bar b3"></span>
            <span class="bar b4"></span>
            <span class="bar b5"></span>
            <span class="bar b6"></span>
            <span class="bar b7"></span>
            <span class="bar b8"></span>
            <span class="bar b9"></span>
            <span class="bar b10"></span>
            <span class="bar b11"></span>
            <span class="bar b12"></span>
            <span class="bar b13"></span>
            <span class="bar b14"></span>
            <span class="bar b15"></span>
            <span class="bar b16"></span>
            <span class="bar b17"></span>
            <span class="bar b18"></span>
            <span class="bar b19"></span>
            <span class="bar b20"></span>
            <span class="bar b21"></span>
            <span class="bar b22"></span>
            <span class="bar b23"></span>
            <span class="bar b24"></span>
          </div>
        </div>
      </section>

      <section class="login-panel" :class="{ 'is-register': !isLogin }" aria-label="login panel">
        <div class="panel-title">
          <p class="panel-sub">欢迎回来</p>
        </div>

        <el-form :model="form" class="panel-form" size="large">
          <el-form-item>
            <el-input
              v-model="form.username"
              placeholder="用户名"
              :prefix-icon="User"
              class="panel-input"
            />
          </el-form-item>

          <el-form-item>
            <el-input
              v-model="form.password"
              type="password"
              placeholder="密码"
              :prefix-icon="Lock"
              show-password
              class="panel-input"
              @keyup.enter="handleSubmit"
            />
          </el-form-item>

          <el-form-item v-if="!isLogin">
            <el-input
              v-model="form.confirmPassword"
              type="password"
              placeholder="确认密码"
              :prefix-icon="Lock"
              show-password
              class="panel-input"
              @keyup.enter="handleSubmit"
            />
          </el-form-item>

          <el-button
            type="primary"
            class="panel-btn"
            @click="handleSubmit"
            :loading="loading"
            size="large"
          >
            {{ isLogin ? '登录' : '注册' }}
          </el-button>
        </el-form>

        <div class="panel-footer">
          <div class="mode-switch">
            <span class="switch-text">{{ isLogin ? '还没有账号？' : '已有账号？' }}</span>
            <el-link class="switch-link" type="primary" @click="toggleMode" :underline="false">
              {{ isLogin ? '立即注册' : '返回登录' }}
            </el-link>
          </div>

        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { User, Lock } from '@element-plus/icons-vue';
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
/* ================================================================
   Design Language: "Breath" -- 呼吸感，统一亮色风格
   ================================================================ */

.login-page {
  height: 100vh;
  width: 100vw;
  position: fixed;
  inset: 0;
  overflow: hidden;
  background: #f2f6fa;
  color: #1e293b;
}

.login-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 50% at 30% 0%, rgba(56,189,248,.06) 0%, transparent 55%),
    radial-gradient(ellipse 60% 60% at 80% 90%, rgba(14,165,233,.04) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 50% 50%, rgba(255,255,255,.5) 0%, transparent 70%);
}

.login-bg::after {
  display: none;
}

.login-bg::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: 0.35;
  background:
    radial-gradient(circle at 12% 64%, rgba(34, 197, 94, 0.08) 0%, transparent 46%),
    radial-gradient(circle at 22% 72%, rgba(96, 165, 250, 0.07) 0%, transparent 52%),
    radial-gradient(circle at 33% 78%, rgba(248, 113, 113, 0.06) 0%, transparent 48%),
    radial-gradient(circle at 44% 72%, rgba(244, 114, 182, 0.06) 0%, transparent 52%),
    radial-gradient(circle at 56% 78%, rgba(251, 146, 60, 0.06) 0%, transparent 50%),
    radial-gradient(circle at 70% 74%, rgba(163, 230, 53, 0.05) 0%, transparent 52%);
  mix-blend-mode: screen;
  animation: emotionDrift 10s ease-in-out infinite;
}

@keyframes emotionDrift {
  0% {
    transform: translate3d(0, 0, 0) scale(1);
    opacity: 0.28;
  }
  50% {
    transform: translate3d(-10px, -8px, 0) scale(1.03);
    opacity: 0.42;
  }
  100% {
    transform: translate3d(0, 0, 0) scale(1);
    opacity: 0.30;
  }
}

.login-main {
  position: relative;
  z-index: 2;
  height: 100vh;
  display: grid;
  grid-template-columns: minmax(520px, 1.55fr) minmax(360px, 440px);
  align-items: center;
  gap: 18px;
  padding: 0 56px 48px;
}

.login-left {
  justify-self: start;
  width: 100%;
}

.left-inner {
  width: 100%;
  padding-left: 6px;
}

.left-wave {
  margin-top: 0;
  height: 200px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 6px;
  opacity: 0.45;
}

.bar {
  width: clamp(10px, 1.25vw, 16px);
  flex: 1 1 0;
  height: 100px;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(14,165,233,0.55), rgba(56,189,248,0.12));
  box-shadow: 0 10px 30px rgba(14,165,233,0.08);
  transform-origin: center bottom;
  transform: scaleY(var(--wave-min, 0.2));
  animation: waveScale 7.2s ease-in-out infinite;
  will-change: transform;
}

@media (prefers-reduced-motion: reduce) {
  .bar {
    animation: none;
    transform: scaleY(0.35);
  }
}

/* Sea-wave feel: long periods + phase shifts, smooth group swells */
.b1 { animation-duration: 20s; animation-delay: -0s; }
.b2 { animation-duration: 20s; animation-delay: -0.45s; }
.b3 { animation-duration: 20s; animation-delay: -0.9s; }
.b4 { animation-duration: 20s; animation-delay: -1.35s; }
.b5 { animation-duration: 20s; animation-delay: -1.8s; }
.b6 { animation-duration: 20s; animation-delay: -2.25s; }
.b7 { animation-duration: 20s; animation-delay: -2.7s; }
.b8 { animation-duration: 20s; animation-delay: -3.15s; }
.b9 { animation-duration: 20s; animation-delay: -3.6s; }
.b10 { animation-duration: 20s; animation-delay: -4.05s; }
.b11 { animation-duration: 20s; animation-delay: -4.5s; }
.b12 { animation-duration: 20s; animation-delay: -4.95s; }
.b13 { animation-duration: 20s; animation-delay: -5.4s; }
.b14 { animation-duration: 20s; animation-delay: -5.85s; }
.b15 { animation-duration: 20s; animation-delay: -6.3s; }
.b16 { animation-duration: 20s; animation-delay: -6.75s; }
.b17 { animation-duration: 20s; animation-delay: -7.2s; }
.b18 { animation-duration: 20s; animation-delay: -7.65s; }
.b19 { animation-duration: 20s; animation-delay: -8.1s; }
.b20 { animation-duration: 20s; animation-delay: -8.55s; }
.b21 { animation-duration: 20s; animation-delay: -9.0s; }
.b22 { animation-duration: 20s; animation-delay: -9.45s; }
.b23 { animation-duration: 20s; animation-delay: -9.9s; }
.b24 { animation-duration: 20s; animation-delay: -10.35s; }

/* Occasional group swell (emotion peak): center + right groups */
.b8, .b9, .b10, .b11, .b12, .b13, .b14, .b15 {
  animation-name: waveScale, peakBoost;
  animation-duration: inherit, 46s;
  animation-timing-function: ease-in-out, ease-in-out;
  animation-iteration-count: infinite, infinite;
  animation-delay: inherit, -12s;
}

.b16, .b17, .b18, .b19 {
  animation-name: waveScale, peakBoost;
  animation-duration: inherit, 58s;
  animation-timing-function: ease-in-out, ease-in-out;
  animation-iteration-count: infinite, infinite;
  animation-delay: inherit, -26s;
}

@keyframes waveScale {
  0%, 100% {
    transform: scaleY(0.12);
    opacity: 0.35;
  }
  50% {
    transform: scaleY(0.92);
    opacity: 0.72;
  }
}

@keyframes peakBoost {
  0%, 78%, 100% {
    filter: brightness(1);
  }
  84% {
    filter: brightness(1.06);
  }
  90% {
    filter: brightness(1.15);
  }
  95% {
    filter: brightness(1.04);
  }
}

.login-panel {
  width: min(440px, calc(100vw - 64px));
  padding: 34px 32px 26px;
  background: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.8);
  border-radius: 20px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.8),
    0 2px 16px rgba(14, 165, 233, 0.06),
    0 1px 2px rgba(0, 0, 0, 0.04);
  margin-left: 0;
}

.panel-title {
  margin-bottom: 18px;
}

.panel-sub {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #1e293b;
}

.panel-form {
  margin-top: 18px;
}

:deep(.el-form-item) {
  margin-bottom: 14px;
}

:deep(.panel-input .el-input__wrapper) {
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255, 255, 255, 0.8);
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.08) inset;
  transition: box-shadow 180ms ease, background 180ms ease;
}

:deep(.panel-input .el-input__wrapper:hover) {
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 0 0 1px rgba(14, 165, 233, 0.3) inset;
}

:deep(.panel-input .el-input__wrapper.is-focus) {
  background: #fff;
  box-shadow:
    0 0 0 1px rgba(14, 165, 233, 0.5) inset,
    0 4px 16px rgba(14, 165, 233, 0.08);
}

:deep(.panel-input .el-input__inner) {
  color: #1e293b;
}

:deep(.panel-input .el-input__prefix) {
  color: #94a3b8;
}

.panel-btn {
  margin-top: 12px;
  width: 100%;
  height: 44px;
  border: none;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.06em;
  background: linear-gradient(135deg, #38bdf8, #0ea5e9);
  box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25);
}

.panel-btn:hover {
  box-shadow: 0 6px 24px rgba(14, 165, 233, 0.35);
  transform: translateY(-1px);
}

.panel-footer {
  margin-top: 16px;
}

.mode-switch {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.switch-text {
  font-size: 13px;
  color: #64748b;
}

.switch-link {
  font-size: 13px;
  font-weight: 700;
}

@media (max-width: 720px) {
  .login-main {
    padding: 0 16px 28px;
    grid-template-columns: 1fr;
    gap: 18px;
    justify-items: center;
  }

  .login-left {
    display: none;
  }
}
</style>
