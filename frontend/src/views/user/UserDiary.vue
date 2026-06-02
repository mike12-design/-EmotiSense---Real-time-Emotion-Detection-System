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
            <!-- 装饰顶边 -->
            <div class="dialog-accent-bar"></div>

            <!-- 头部 -->
            <div class="dialog-header">
              <div class="dialog-header-row">
                <h2 class="dialog-title">{{ dialogTitle }}</h2>
                <button class="dialog-close" @click="showDialog = false" aria-label="关闭">
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                    <path d="M4 4l10 10M14 4L4 14" />
                  </svg>
                </button>
              </div>
              <p class="dialog-subtitle">
                {{ mode === 'create' ? '记录此刻的心情，留给未来的自己' : mode === 'patch' ? '补写一段遗漏的时光' : '重新审视那一瞬间' }}
              </p>
            </div>

            <!-- 内容区 -->
            <div class="dialog-body">
              <!-- 心情选择（提到最前 — 先定调再写） -->
              <div class="emotion-strip">
                <label class="field-label">此刻心情</label>
                <div class="emotion-row" role="radiogroup" aria-label="选择心情">
                  <button
                    v-for="emo in emotions"
                    :key="emo.value"
                    class="emotion-chip"
                    :class="{ 'emotion-chip--active': form.emotion === emo.value }"
                    :style="{ '--chip-color': emo.color, '--chip-bg': emo.bg }"
                    role="radio"
                    :aria-checked="form.emotion === emo.value"
                    :aria-label="emo.label"
                    @click="form.emotion = emo.value"
                  >
                    {{ emo.emoji }} {{ emo.label }}
                  </button>
                </div>
              </div>

              <!-- 内容区 -->
              <div class="textarea-wrapper">
                <textarea
                  ref="contentRef"
                  v-model="form.content"
                  class="field-textarea"
                  :rows="10"
                  :placeholder="form.emotion === 'Happy' ? '今天发生了什么开心的事？☀️' : form.emotion === 'Sad' ? '没关系，把不开心写下来会好很多...🍃' : form.emotion === 'Angry' ? '深呼吸，把情绪倒进文字里...🌊' : '写下你现在的感受...✨'"
                  maxlength="500"
                  @input="contentLength = form.content.length"
                ></textarea>
                <span class="textarea-counter">{{ contentLength }}<span class="counter-max">/500</span></span>
              </div>

              <!-- 时间（仅补卡/编辑时显示） -->
              <div v-if="mode !== 'create'" class="time-row">
                <svg class="time-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <el-date-picker
                  v-model="form.timestamp"
                  type="datetime"
                  placeholder="选择记录时间"
                  class="field-datepicker"
                />
              </div>
            </div>

            <!-- 底部 -->
            <div class="dialog-footer">
              <button class="btn-cancel" @click="showDialog = false">取消</button>
              <button
                class="btn-submit"
                :disabled="submitting || !form.content.trim()"
                @click="submitDiary"
              >
                <svg v-if="submitting" class="btn-spinner" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                  <path d="M12 2a10 10 0 109.95 11" />
                </svg>
                <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
                {{ submitting ? '保存中...' : '保存日记' }}
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
  { value: 'Happy',   label: '开心', emoji: '😊', color: '#F59E0B', bg: 'rgba(245,158,11,0.1)' },
  { value: 'Neutral', label: '平静', emoji: '😐', color: '#78716C', bg: 'rgba(120,113,108,0.08)' },
  { value: 'Sad',     label: '难过', emoji: '😢', color: '#6366F1', bg: 'rgba(99,102,241,0.1)' },
  { value: 'Angry',   label: '生气', emoji: '😡', color: '#EF4444', bg: 'rgba(239,68,68,0.1)' },
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
  background: rgba(69, 39, 16, 0.45);
  backdrop-filter: blur(6px);
}

/* ===== 弹窗面板 ===== */
.dialog-panel {
  width: 100%;
  max-width: 520px;
  margin: 24px;
  max-height: 92vh;
  overflow-y: auto;
  background: #FFFCF7;
  border-radius: 24px;
  box-shadow:
    0 4px 0 0 rgba(146, 64, 14, 0.06),
    0 0 0 1px rgba(146, 64, 14, 0.08),
    0 25px 60px -12px rgba(69, 39, 16, 0.25);
}

/* ===== 装饰顶边 ===== */
.dialog-accent-bar {
  height: 4px;
  background: linear-gradient(90deg, #92400E, #D97706, #6366F1, #8B5CF6);
  border-radius: 24px 24px 0 0;
}

/* ===== 头部 ===== */
.dialog-header {
  padding: 24px 28px 0;
}

.dialog-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.dialog-title {
  margin: 0;
  font-family: 'Caveat', 'Ma Shan Zheng', cursive;
  font-size: 32px;
  font-weight: 600;
  color: #451A03;
  letter-spacing: 0.02em;
  line-height: 1.2;
}

.dialog-subtitle {
  margin: 6px 0 0;
  font-family: 'Quicksand', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: #A16207;
  letter-spacing: 0.01em;
}

.dialog-close {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  background: rgba(146, 64, 14, 0.06);
  color: #78716C;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all 0.2s;
}
.dialog-close:hover {
  background: rgba(146, 64, 14, 0.12);
  color: #451A03;
  transform: rotate(90deg);
}
.dialog-close:focus-visible {
  outline: 2px solid #6366F1;
  outline-offset: 2px;
}

/* ===== 内容区 ===== */
.dialog-body {
  padding: 24px 28px 8px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.field-label {
  display: block;
  font-family: 'Quicksand', sans-serif;
  font-size: 12px;
  font-weight: 700;
  color: #78716C;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 10px;
}

/* ===== 心情芯片组 ===== */
.emotion-strip {
  /* nothing extra needed */
}

.emotion-row {
  display: flex;
  gap: 8px;
}

.emotion-chip {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 10px 6px;
  border-radius: 40px;
  border: 1.5px solid #E7D7CC;
  background: #FFFCF7;
  font-family: 'Quicksand', sans-serif;
  font-size: 13px;
  font-weight: 600;
  color: #78716C;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
  user-select: none;
}
.emotion-chip:hover {
  border-color: var(--chip-color, #6366F1);
  color: var(--chip-color, #6366F1);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--chip-bg, rgba(99,102,241,0.15));
}
.emotion-chip:focus-visible {
  outline: 2px solid #6366F1;
  outline-offset: 2px;
}

.emotion-chip--active {
  background: var(--chip-bg, rgba(99,102,241,0.1));
  border-color: var(--chip-color, #6366F1);
  color: var(--chip-color, #6366F1);
  box-shadow:
    0 0 0 3px var(--chip-bg, rgba(99,102,241,0.12)),
    0 2px 8px rgba(0,0,0,0.06);
}

/* ===== 文本域 ===== */
.textarea-wrapper {
  position: relative;
}

.field-textarea {
  width: 100%;
  min-height: 240px;
  padding: 20px 22px;
  font-family: 'Quicksand', sans-serif;
  font-size: 15px;
  font-weight: 500;
  line-height: 1.85;
  color: #451A03;
  background: #FFFBF3;
  border: 1.5px solid #E7D7CC;
  border-radius: 18px;
  resize: none;
  outline: none;
  transition: border-color 0.25s, box-shadow 0.25s, background 0.25s;
  box-sizing: border-box;
}
.field-textarea::placeholder {
  color: #C4B5A5;
  font-weight: 500;
}
.field-textarea:focus {
  border-color: #6366F1;
  background: #FFFCF7;
  box-shadow:
    0 0 0 4px rgba(99, 102, 241, 0.08),
    inset 0 1px 3px rgba(99, 102, 241, 0.04);
}

.textarea-counter {
  position: absolute;
  bottom: 14px;
  right: 16px;
  font-family: 'Quicksand', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: #451A03;
  pointer-events: none;
}
.counter-max {
  color: #C4B5A5;
  font-weight: 500;
}

/* ===== 时间行 ===== */
.time-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: #FFFBF3;
  border: 1.5px dashed #E7D7CC;
  border-radius: 14px;
}

.time-icon {
  color: #A16207;
  flex-shrink: 0;
}

.field-datepicker {
  flex: 1;
}
.field-datepicker :deep(.el-input__wrapper) {
  border-radius: 10px;
  background: transparent;
  box-shadow: none;
  border: none;
  padding: 0 8px;
}
.field-datepicker :deep(.el-input__wrapper:hover) {
  box-shadow: none;
}
.field-datepicker :deep(.el-input__inner) {
  font-family: 'Quicksand', sans-serif;
  font-size: 13px;
  color: #451A03;
}
.field-datepicker :deep(.el-input__inner::placeholder) {
  color: #C4B5A5;
}

/* ===== 底部按钮 ===== */
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 20px 28px 28px;
}

.btn-cancel {
  padding: 12px 22px;
  border-radius: 40px;
  border: 1.5px solid #E7D7CC;
  background: transparent;
  color: #78716C;
  font-family: 'Quicksand', sans-serif;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-cancel:hover {
  background: rgba(146, 64, 14, 0.04);
  border-color: #D4C4B0;
  color: #451A03;
}
.btn-cancel:focus-visible {
  outline: 2px solid #6366F1;
  outline-offset: 2px;
}

.btn-submit {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 12px 28px;
  border-radius: 40px;
  border: none;
  background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
  color: #fff;
  font-family: 'Quicksand', sans-serif;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
}
.btn-submit:hover:not(:disabled) {
  box-shadow: 0 8px 28px rgba(99, 102, 241, 0.4);
  transform: translateY(-2px);
}
.btn-submit:active:not(:disabled) {
  transform: scale(0.96);
}
.btn-submit:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
.btn-submit:focus-visible {
  outline: 2px solid #6366F1;
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
  transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.dialog-leave-active {
  transition: all 0.2s ease-in;
}

.dialog-enter-from {
  opacity: 0;
}
.dialog-enter-from .dialog-panel {
  transform: scale(0.92) translateY(24px);
  opacity: 0;
}

.dialog-leave-to {
  opacity: 0;
}
.dialog-leave-to .dialog-panel {
  transform: scale(0.95) translateY(8px);
  opacity: 0;
}

@media (prefers-reduced-motion: reduce) {
  .dialog-enter-active,
  .dialog-leave-active {
    transition: opacity 0.15s;
  }
  .dialog-enter-from .dialog-panel,
  .dialog-leave-to .dialog-panel {
    transform: none;
  }
  .dialog-close:hover {
    transform: none;
  }
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