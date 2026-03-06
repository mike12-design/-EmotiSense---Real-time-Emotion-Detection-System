<!-- frontend/src/views/admin/UserManager.vue -->
<template>
  <div class="user-manager">
    <!-- 页面标题区 -->
    <div class="page-header animate-fade-in-down">
      <div class="title-group">
        <el-icon class="page-icon"><UserFilled /></el-icon>
        <div>
          <h1 class="page-title">人员管理</h1>
          <p class="page-subtitle">管理系统注册用户及人脸特征数据</p>
        </div>
      </div>
      <el-button class="btn-add" type="primary">
        <el-icon><Plus /></el-icon>
        <span>新增用户</span>
      </el-button>
    </div>

    <!-- 数据卡片 -->
    <el-card class="data-card animate-fade-in" shadow="hover">
      <template #header>
        <div class="card-header">
          <div class="header-left">
            <el-icon class="card-icon"><User /></el-icon>
            <span class="card-title">注册用户列表</span>
            <el-tag class="count-tag" type="info">
              共 {{ users.length }} 人
            </el-tag>
          </div>
          <div class="header-right">
            <el-input
              v-model="searchQuery"
              placeholder="搜索用户名..."
              class="search-input"
              :prefix-icon="Search"
              clearable
            />
          </div>
        </div>
      </template>

      <el-table
        :data="filteredUsers"
        stripe
        style="width: 100%"
        v-loading="loading"
        :header-cell-style="{ background: 'var(--primary-50)', color: 'var(--text-primary)' }"
        empty-text="暂无用户数据"
      >
        <el-table-column prop="id" label="ID" width="80" align="center" />
        <el-table-column prop="username" label="用户名" min-width="150" align="center">
          <template #default="scope">
            <div class="username-cell">
              <div class="user-avatar-small">{{ scope.row.username.charAt(0).toUpperCase() }}</div>
              <span class="username-text">{{ scope.row.username }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="role" label="角色" width="120" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.role === 'admin' ? 'danger' : 'primary'"
              size="small"
              round
            >
              {{ scope.row.role === 'admin' ? '管理员' : '普通用户' }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="人脸数据" width="140" align="center">
          <template #default="scope">
            <el-tag
              :type="scope.row.has_face ? 'success' : 'info'"
              size="small"
              effect="light"
              round
            >
              <el-icon v-if="scope.row.has_face"><CircleCheck /></el-icon>
              <el-icon v-else><CircleClose /></el-icon>
              {{ scope.row.has_face ? '已录入' : '待采集' }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="220" fixed="right" align="center">
          <template #default="scope">
            <div class="action-buttons">
              <el-button
                size="small"
                type="primary"
                plain
                @click="openCapture(scope.row)"
                :disabled="!scope.row.id"
              >
                <el-icon><Camera /></el-icon>
                采集
              </el-button>
              <el-button
                size="small"
                type="danger"
                plain
                @click="handleDelete(scope.row)"
              >
                <el-icon><Delete /></el-icon>
                删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 人脸采集弹窗 -->
    <el-dialog
      v-model="captureVisible"
      title="人脸特征采集"
      width="520px"
      :close-on-click-modal="false"
      class="capture-dialog"
    >
      <div class="capture-container">
        <div class="video-wrapper">
          <img
            v-if="captureVisible"
            src="http://127.0.0.1:8000/video_feed"
            class="preview-video"
            alt="实时视频流"
          />
          <div class="video-overlay">
            <div class="corner-tl"></div>
            <div class="corner-tr"></div>
            <div class="corner-bl"></div>
            <div class="corner-br"></div>
          </div>
        </div>
        <div class="capture-tips">
          <el-icon class="tip-icon"><InfoFilled /></el-icon>
          <span>请确保用户正对摄像头，且光线充足</span>
        </div>
      </div>
      <template #footer>
        <div class="dialog-footer">
          <el-button @click="captureVisible = false" size="default">
            取消
          </el-button>
          <el-button
            type="primary"
            :loading="isCapturing"
            @click="handleCapture"
            size="default"
          >
            {{ isCapturing ? '采集中...' : '立即捕捉特征' }}
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  User, UserFilled, Plus, Search, CircleCheck, CircleClose,
  Camera, Delete, InfoFilled
} from '@element-plus/icons-vue'

const API_BASE = "http://127.0.0.1:8000"

const users = ref([])
const loading = ref(false)
const searchQuery = ref('')

const captureVisible = ref(false)
const currentUserId = ref(null)
const isCapturing = ref(false)

// 过滤后的用户列表
const filteredUsers = computed(() => {
  if (!searchQuery.value) return users.value
  return users.value.filter(user =>
    user.username?.toLowerCase().includes(searchQuery.value.toLowerCase())
  )
})

const fetchUsers = async () => {
  loading.value = true
  try {
    const res = await axios.get(`${API_BASE}/api/admin/users`)
    users.value = res.data.users || []
  } catch (e) {
    console.error(e)
    ElMessage.error("获取用户列表失败")
  } finally {
    loading.value = false
  }
}

const openCapture = (user) => {
  currentUserId.value = user.id
  captureVisible.value = true
}

const handleCapture = async () => {
  isCapturing.value = true
  try {
    const res = await axios.post(
      `${API_BASE}/api/admin/capture_face/${currentUserId.value}`
    )
    if (res.data.success) {
      ElMessage.success({
        message: '人脸特征采集成功！',
        type: 'success'
      })
      captureVisible.value = false
      fetchUsers()
    } else {
      ElMessage.error(res.data.message)
    }
  } catch (err) {
    ElMessage.error('连接后端失败')
  } finally {
    isCapturing.value = false
  }
}

const handleDelete = (user) => {
  ElMessageBox.confirm(
    `确定要删除用户 "${user.username}" 吗？此操作不可恢复。`,
    '删除确认',
    {
      confirmButtonText: '确定删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    try {
      // 这里调用删除 API
      ElMessage.success('删除成功')
      fetchUsers()
    } catch (e) {
      ElMessage.error('删除失败')
    }
  }).catch(() => {})
}

onMounted(fetchUsers)
</script>

<style scoped>
/* ===== 页面容器 ===== */
.user-manager {
  padding: 0;
  min-height: 100%;
}

/* ===== 页面标题 ===== */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 20px 24px;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.4);
  box-shadow: 0 4px 20px rgba(14, 165, 233, 0.08);
}

.title-group {
  display: flex;
  align-items: center;
  gap: 16px;
}

.page-icon {
  font-size: 36px;
  color: var(--primary-500);
  background: var(--primary-50);
  padding: 12px;
  border-radius: 14px;
}

.page-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0;
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  margin: 4px 0 0 0;
}

.btn-add {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  border: none;
  padding: 12px 24px;
  font-size: 14px;
  font-weight: 600;
  box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3);
}

.btn-add:hover {
  box-shadow: 0 6px 24px rgba(14, 165, 233, 0.4);
  transform: translateY(-2px);
}

/* ===== 数据卡片 ===== */
.data-card {
  border-radius: 16px;
  border: 1px solid var(--border-light);
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
}

:deep(.el-card__header) {
  background: rgba(255, 255, 255, 0.5);
  border-bottom: 1px solid var(--border-light);
  padding: 16px 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.card-icon {
  font-size: 20px;
  color: var(--primary-500);
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.count-tag {
  margin-left: 8px;
}

.search-input {
  width: 280px;
}

/* ===== 表格样式 ===== */
:deep(.el-table) {
  --el-table-tr-bg-color: transparent;
  --el-table-header-bg-color: var(--primary-50);
  --el-table-text-color: var(--text-primary);
  --el-table-header-text-color: var(--text-primary);
  --el-table-border-color: var(--border-light);
  --el-table-fixed-box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
}

:deep(.el-table__row:hover) {
  background: var(--primary-50) !important;
}

:deep(.el-table th) {
  font-weight: 600;
  font-size: 14px;
}

:deep(.el-table td) {
  padding: 14px 0;
}

/* 用户名单元格 */
.username-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}

.user-avatar-small {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, var(--primary-400), var(--primary-500));
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
}

.username-text {
  font-weight: 500;
  color: var(--text-primary);
}

/* 操作按钮 */
.action-buttons {
  display: flex;
  gap: 8px;
  justify-content: center;
}

/* ===== 采集弹窗 ===== */
.capture-dialog :deep(.el-dialog__header) {
  background: linear-gradient(135deg, var(--primary-50), var(--sky-50));
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-light);
}

.capture-dialog :deep(.el-dialog__title) {
  font-size: 18px;
  font-weight: 600;
}

.capture-dialog :deep(.el-dialog__body) {
  padding: 24px;
}

.capture-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.video-wrapper {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(14, 165, 233, 0.15);
}

.preview-video {
  width: 100%;
  display: block;
  border-radius: 16px;
}

.video-overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.video-overlay .corner-tl,
.video-overlay .corner-tr,
.video-overlay .corner-bl,
.video-overlay .corner-br {
  position: absolute;
  width: 40px;
  height: 40px;
  border: 3px solid rgba(14, 165, 233, 0.6);
}

.video-overlay .corner-tl { top: 16px; left: 16px; border-right: none; border-bottom: none; border-radius: 8px 0 0 0; }
.video-overlay .corner-tr { top: 16px; right: 16px; border-left: none; border-bottom: none; border-radius: 0 8px 0 0; }
.video-overlay .corner-bl { bottom: 16px; left: 16px; border-right: none; border-top: none; border-radius: 0 0 0 8px; }
.video-overlay .corner-br { bottom: 16px; right: 16px; border-left: none; border-top: none; border-radius: 0 0 8px 0; }

.capture-tips {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--primary-50);
  border-radius: 12px;
  color: var(--text-secondary);
  font-size: 14px;
}

.tip-icon {
  font-size: 18px;
  color: var(--primary-500);
}

.dialog-footer {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}

/* ===== 动画 ===== */
.animate-fade-in-down {
  animation: fadeInDown 0.4s ease-out;
}

.animate-fade-in {
  animation: fadeIn 0.5s ease-out 0.1s both;
}

@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
