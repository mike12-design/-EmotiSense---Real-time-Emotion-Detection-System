<template>
  <div class="user-diary">
    <!-- 1. 日记管理卡片 -->
    <el-card class="mb-6 shadow-sm" style="border-radius: 16px;">
      <template #header>
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-2">
            <el-icon color="#4f46e5"><Edit /></el-icon>
            <span class="text-lg font-bold">我的日记 ({{ filteredDiaries.length }} 条)</span>
          </div>
          <div class="flex gap-2">
            <el-button type="success" plain :icon="Calendar" @click="openDialog('patch')">补打卡</el-button>
            <el-button type="primary" :icon="Plus" @click="openDialog('create')">新日记</el-button>
          </div>
        </div>
      </template>

      <!-- 搜索与筛选 -->
      <div class="filter-bar flex gap-2 mb-4">
        <el-input
          v-model="searchKeyword"
          placeholder="搜索日记内容..."
          :prefix-icon="Search"
          clearable
          style="width: 220px"
        />
        <el-date-picker
          v-model="searchDate"
          type="date"
          placeholder="按日期筛选"
          value-format="YYYY-MM-DD"
          style="width: 160px"
        />
      </div>

      <!-- 列表区 -->
      <div v-loading="loading">
        <!-- 空状态 -->
        <div v-if="filteredDiaries.length === 0" class="empty-state">
          <p class="text-gray-500 mb-4">没有找到相关日记</p>
          <el-button type="primary" @click="openDialog('create')">写一篇</el-button>
        </div>

        <!-- 日记卡片列表 -->
        <div v-else class="grid gap-4">
          <el-card v-for="item in filteredDiaries" :key="item.id" shadow="hover" class="diary-card">
            <div class="flex justify-between items-start">
              <div class="flex-1">
                <div class="flex items-center gap-2 mb-2">
                  <el-tag :type="getEmotionTag(item.emotion)" size="small" effect="dark">{{ item.emotion }}</el-tag>
                  <span class="text-gray-400 text-sm">{{ formatDate(item.timestamp) }}</span>
                </div>
                <div class="text-gray-800 font-medium whitespace-pre-wrap">{{ item.content }}</div>
              </div>

              <!-- 修改与删除操作图标 -->
              <div class="flex flex-col gap-2 ml-4">
                <el-button type="primary" link :icon="Edit" @click="handleEdit(item)"></el-button>
                <el-button type="danger" link :icon="Delete" @click="handleDelete(item.id)"></el-button>
              </div>
            </div>
          </el-card>
        </div>
      </div>
    </el-card>

    <!-- 写日记/编辑日记 弹窗 -->
    <Teleport to="body">
      <transition name="dialog">
        <div
          v-if="showDialog"
          class="dialog-overlay"
          @click.self="showDialog = false"
          @keydown.escape="showDialog = false"
        >
          <div
            class="dialog-panel"
            role="dialog"
            aria-modal="true"
            :aria-label="dialogTitle"
          >
            <!-- 头部 -->
            <div class="dialog-header">
              <div class="dialog-header-icon">
                <span class="text-2xl">{{ mode === 'create' ? '✍️' : mode === 'patch' ? '📅' : '✏️' }}</span>
              </div>
              <div class="dialog-header-text">
                <h2 class="dialog-title">{{ dialogTitle }}</h2>
                <p class="dialog-subtitle">
                  {{ mode === 'create' ? '记录此刻的心情' : mode === 'patch' ? '补写过去的心情' : '修改你的记录' }}
                </p>
              </div>
              <button class="dialog-close" @click="showDialog = false" aria-label="关闭">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M5 5l10 10M15 5L5 15" />
                </svg>
              </button>
            </div>

            <!-- 内容区 -->
            <div class="dialog-body">
              <!-- 时间 -->
              <div class="field-group">
                <label class="field-label">
                  <svg class="field-label-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>
                  记录时间
                </label>
                <el-date-picker
                  v-model="form.timestamp"
                  type="datetime"
                  placeholder="选择时间"
                  style="width: 100%"
                  :disabled="mode === 'create'"
                  class="field-datepicker"
                />
                <span v-if="mode === 'create'" class="field-hint">默认为当前时间</span>
              </div>

              <!-- 内容 -->
              <div class="field-group">
                <label class="field-label">
                  <svg class="field-label-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                  你想说点什么？
                </label>
                <div class="textarea-wrapper">
                  <textarea
                    ref="contentRef"
                    v-model="form.content"
                    class="field-textarea"
                    :rows="6"
                    placeholder="写下你现在的感受..."
                    maxlength="500"
                    @input="contentLength = form.content.length"
                  ></textarea>
                  <span class="textarea-counter">{{ contentLength }}/500</span>
                </div>
              </div>

              <!-- 心情状态 -->
              <div class="field-group">
                <label class="field-label">
                  <svg class="field-label-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                  心情状态
                </label>
                <div class="emotion-grid" role="radiogroup" :aria-label="'选择心情'">
                  <button
                    v-for="emo in emotions"
                    :key="emo.value"
                    class="emotion-card"
                    :class="{ 'emotion-card--active': form.emotion === emo.value }"
                    :style="{ '--emo-accent': emo.color, '--emo-accent-bg': emo.bg }"
                    role="radio"
                    :aria-checked="form.emotion === emo.value"
                    :aria-label="emo.label"
                    @click="form.emotion = emo.value"
                  >
                    <span class="emotion-card-emoji">{{ emo.emoji }}</span>
                    <span class="emotion-card-label">{{ emo.label }}</span>
                  </button>
                </div>
              </div>
            </div>

            <!-- 底部 -->
            <div class="dialog-footer">
              <button class="btn-cancel" @click="showDialog = false">取消</button>
              <button
                class="btn-submit"
                :disabled="submitting"
                @click="submitDiary"
              >
                <svg v-if="submitting" class="btn-spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M12 2a10 10 0 109.95 11" />
                </svg>
                {{ submitting ? '保存中...' : '保存' }}
              </button>
            </div>
          </div>
        </div>
      </transition>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import { Calendar, Plus, Search, Edit, Delete } from '@element-plus/icons-vue';
import axios from 'axios';
import { ElMessage, ElMessageBox } from 'element-plus';

// --- 状态变量 ---
const diaries = ref([]);
const loading = ref(false);
const showDialog = ref(false);
const submitting = ref(false);
const username = localStorage.getItem('user');

// 搜索与筛选
const searchKeyword = ref('');
const searchDate = ref(null);
const contentLength = ref(0);

const emotions = [
  { value: 'Happy',   label: '开心', emoji: '😊', color: '#34d399', bg: 'rgba(52,211,153,0.12)' },
  { value: 'Neutral', label: '平静', emoji: '😐', color: '#7dd3fc', bg: 'rgba(125,211,252,0.12)' },
  { value: 'Sad',     label: '难过', emoji: '😢', color: '#60a5fa', bg: 'rgba(96,165,250,0.12)' },
  { value: 'Angry',   label: '生气', emoji: '😡', color: '#f87171', bg: 'rgba(248,113,113,0.12)' },
];

// 表单相关
const mode = ref('create'); // 'create' | 'patch' (补卡) | 'edit'
const currentId = ref(null); // 编辑时的 ID
const form = ref({
  content: '',
  emotion: 'Neutral',
  timestamp: new Date()
});

// --- 计算属性：前端搜索过滤 ---
const filteredDiaries = computed(() => {
  return diaries.value.filter(item => {
    // 1. 内容搜索
    const matchContent = item.content.includes(searchKeyword.value);
    // 2. 日期筛选 (比较 YYYY-MM-DD)
    let matchDate = true;
    if (searchDate.value) {
      matchDate = item.timestamp.startsWith(searchDate.value);
    }
    return matchContent && matchDate;
  });
});

const dialogTitle = computed(() => {
  if (mode.value === 'edit') return '编辑日记';
  if (mode.value === 'patch') return '补写日记';
  return '新日记';
});

// --- API 操作 ---

// 1. 获取日记
const fetchDiaries = async () => {
  if (!username) return;
  loading.value = true;
  try {
    const res = await axios.get(`http://127.0.0.1:8000/api/my/diaries?username=${username}`);
    diaries.value = res.data;
  } catch (e) {
    console.error("加载日记失败");
  } finally {
    loading.value = false;
  }
};

// 2. 打开弹窗 (三种模式)
const openDialog = (actionType) => {
  mode.value = actionType;
  showDialog.value = true;
  
  if (actionType === 'create') {
    form.value.timestamp = new Date(); // 默认现在
  } else if (actionType === 'patch') {
    form.value.timestamp = ''; // 补卡让用户自己选，或者默认昨天
  }
};

// 3. 点击编辑
const handleEdit = (item) => {
  mode.value = 'edit';
  currentId.value = item.id;
  // 填充表单
  form.value = {
    content: item.content,
    emotion: item.emotion,
    timestamp: item.timestamp // ISO 字符串可以直接被 Element Plus 解析
  };
  showDialog.value = true;
};

// 4. 点击删除
const handleDelete = (id) => {
  ElMessageBox.confirm(
    '确定要删除这条日记吗？此操作无法撤销。',
    '警告',
    { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' }
  ).then(async () => {
    try {
      await axios.delete(`http://127.0.0.1:8000/api/my/diaries/${id}`);
      ElMessage.success("已删除");
      fetchDiaries();
    } catch (e) {
      ElMessage.error("删除失败");
    }
  }).catch(() => {});
};

// 5. 提交表单 (新增 或 修改)
const submitDiary = async () => {
  if (!form.value.content) return ElMessage.warning("内容不能为空");
  if (!form.value.timestamp) return ElMessage.warning("请选择时间");

  submitting.value = true;
  try {
    const payload = {
      username: username,
      content: form.value.content,
      emotion: form.value.emotion,
      timestamp: form.value.timestamp // 传递时间给后端
    };

    if (mode.value === 'edit') {
      // 修改接口
      await axios.put(`http://127.0.0.1:8000/api/my/diaries/${currentId.value}`, payload);
      ElMessage.success("修改成功");
    } else {
      // 新增接口 (包括补卡)
      await axios.post(`http://127.0.0.1:8000/api/my/diaries`, payload);
      ElMessage.success(mode.value === 'patch' ? "补卡成功" : "发布成功");
    }
    
    showDialog.value = false;
    resetForm();
    fetchDiaries();
  } catch (e) {
    ElMessage.error("操作失败");
  } finally {
    submitting.value = false;
  }
};

const resetForm = () => {
  form.value = { content: '', emotion: 'Neutral', timestamp: new Date() };
  currentId.value = null;
  contentLength.value = 0;
};

// --- 工具函数 ---
const formatDate = (ts) => {
  if (!ts) return '';
  return ts.replace('T', ' ').split('.')[0];
};

const getEmotionTag = (e) => {
  const map = { Happy: 'success', Sad: 'warning', Angry: 'danger', Neutral: 'info' };
  return map[e] || 'info';
};

onMounted(fetchDiaries);
</script>

<style scoped>
/* ===== 页面布局 ===== */
.user-diary {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 16px;
}

.empty-state {
  text-align: center;
  padding: 48px 16px;
  border: 2px dashed #e5e7eb;
  border-radius: 12px;
}

.diary-card {
  border-left: 5px solid #4f46e5;
  transition: transform 0.2s;
  border-radius: 12px;
}
.diary-card:hover {
  transform: translateY(-2px);
}

/* ===== 弹窗覆盖层 ===== */
.dialog-overlay {
  position: fixed;
  inset: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 23, 42, 0.5);
  backdrop-filter: blur(4px);
}

/* ===== 弹窗面板 ===== */
.dialog-panel {
  width: 100%;
  max-width: 520px;
  margin: 24px;
  max-height: 90vh;
  overflow-y: auto;
  background: #fff;
  border-radius: 20px;
  box-shadow:
    0 25px 50px -12px rgba(0, 0, 0, 0.25),
    0 0 0 1px rgba(14, 165, 233, 0.08);
}

/* ===== 头部 ===== */
.dialog-header {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 24px 24px 0;
}

.dialog-header-icon {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  background: linear-gradient(135deg, #e0f2fe, #bae6fd);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.dialog-header-text {
  flex: 1;
  min-width: 0;
}

.dialog-title {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  color: #0f172a;
  line-height: 1.3;
}

.dialog-subtitle {
  margin: 4px 0 0;
  font-size: 13px;
  color: #94a3b8;
}

.dialog-close {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  border: none;
  background: #f1f5f9;
  color: #64748b;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all 0.15s;
}
.dialog-close:hover {
  background: #e2e8f0;
  color: #0f172a;
}
.dialog-close:focus-visible {
  outline: 2px solid #0ea5e9;
  outline-offset: 2px;
}

/* ===== 内容区 ===== */
.dialog-body {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.field-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  color: #475569;
}

.field-label-icon {
  color: #94a3b8;
  flex-shrink: 0;
}

.field-hint {
  font-size: 12px;
  color: #94a3b8;
  margin-top: 2px;
}

/* 日期选择器 */
.field-datepicker :deep(.el-input__wrapper) {
  border-radius: 10px;
  box-shadow: 0 0 0 1px #e2e8f0;
  transition: box-shadow 0.15s;
}
.field-datepicker :deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px #cbd5e1;
}

/* 文本域 */
.textarea-wrapper {
  position: relative;
}

.field-textarea {
  width: 100%;
  min-height: 140px;
  padding: 14px 16px;
  font-size: 15px;
  font-family: inherit;
  line-height: 1.7;
  color: #1e293b;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  resize: vertical;
  transition: border-color 0.15s, box-shadow 0.15s;
  box-sizing: border-box;
}
.field-textarea::placeholder {
  color: #94a3b8;
}
.field-textarea:focus {
  outline: none;
  border-color: #0ea5e9;
  box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
  background: #fff;
}

.textarea-counter {
  position: absolute;
  bottom: 8px;
  right: 12px;
  font-size: 11px;
  color: #94a3b8;
  pointer-events: none;
}

/* ===== 情绪卡片 ===== */
.emotion-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}

.emotion-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 14px 8px;
  border-radius: 14px;
  border: 2px solid transparent;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  font-family: inherit;
}
.emotion-card:hover {
  background: var(--emo-accent-bg, #f1f5f9);
  transform: translateY(-2px);
}
.emotion-card:focus-visible {
  outline: 2px solid #0ea5e9;
  outline-offset: 2px;
}

.emotion-card--active {
  border-color: var(--emo-accent, #0ea5e9);
  background: var(--emo-accent-bg, rgba(14,165,233,0.1));
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.emotion-card-emoji {
  font-size: 28px;
  line-height: 1;
}

.emotion-card-label {
  font-size: 12px;
  font-weight: 600;
  color: #475569;
}

/* ===== 底部按钮 ===== */
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 0 24px 24px;
}

.btn-cancel {
  padding: 10px 20px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: #fff;
  color: #475569;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: inherit;
}
.btn-cancel:hover {
  background: #f1f5f9;
  border-color: #cbd5e1;
}
.btn-cancel:focus-visible {
  outline: 2px solid #0ea5e9;
  outline-offset: 2px;
}

.btn-submit {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 10px 24px;
  border-radius: 12px;
  border: none;
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  color: #fff;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  font-family: inherit;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
}
.btn-submit:hover {
  box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
  transform: translateY(-1px);
}
.btn-submit:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}
.btn-submit:focus-visible {
  outline: 2px solid #0ea5e9;
  outline-offset: 2px;
}

.btn-spinner {
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ===== 过渡动画 ===== */
.dialog-enter-active {
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.dialog-leave-active {
  transition: all 0.15s ease-in;
}

.dialog-enter-from {
  opacity: 0;
}
.dialog-enter-from .dialog-panel {
  transform: scale(0.95) translateY(10px);
  opacity: 0;
}

.dialog-leave-to {
  opacity: 0;
}
.dialog-leave-to .dialog-panel {
  transform: scale(0.95) translateY(10px);
  opacity: 0;
}

/* ===== Flex 工具类 ===== */
.flex { display: flex; }
.flex-col { flex-direction: column; }
.justify-between { justify-content: space-between; }
.items-center { align-items: center; }
.items-start { align-items: flex-start; }
.flex-1 { flex: 1; }
.gap-2 { gap: 0.5rem; }
.gap-4 { gap: 1rem; }
.mb-2 { margin-bottom: 0.5rem; }
.mb-4 { margin-bottom: 1rem; }
.mb-6 { margin-bottom: 1.5rem; }
.ml-4 { margin-left: 1rem; }
.grid { display: grid; }
.whitespace-pre-wrap { white-space: pre-wrap; }
</style>