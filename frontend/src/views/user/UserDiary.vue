<template>
  <div class="mt-6">
    <!-- 头部：标题与操作栏 -->
    <div class="header-section mb-6">
      <h2 class="text-xl font-semibold text-gray-800">
        我的日记 ({{ filteredDiaries.length }} 条)
      </h2>
      
      <!-- 搜索与筛选区 -->
      <div class="filter-bar flex gap-2 mt-2">
        <el-input 
          v-model="searchKeyword" 
          placeholder="搜索日记内容..." 
          :prefix-icon="Search"
          clearable
          style="width: 200px"
        />
        <el-date-picker
          v-model="searchDate"
          type="date"
          placeholder="按日期筛选"
          value-format="YYYY-MM-DD"
          style="width: 150px"
        />
      </div>

      <!-- 操作按钮 -->
      <div class="action-btns flex space-x-2 mt-2">
        <el-button type="success" plain :icon="Calendar" @click="openDialog('patch')">补打卡</el-button>
        <el-button type="primary" :icon="Plus" @click="openDialog('create')">新日记</el-button>
      </div>
    </div>

    <!-- 列表区 -->
    <div v-loading="loading">
      <!-- 空状态 -->
      <div v-if="filteredDiaries.length === 0" class="bg-white rounded-lg shadow-md p-8 text-center border-dashed border-2 border-gray-200">
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

    <!-- 写日记/编辑日记 弹窗 -->
    <el-dialog 
      v-model="showDialog" 
      :title="dialogTitle" 
      width="500px"
      @close="resetForm"
    >
      <el-form label-position="top">
        <!-- 补打卡或编辑时，允许修改时间 -->
        <el-form-item label="记录时间">
          <el-date-picker
            v-model="form.timestamp"
            type="datetime"
            placeholder="选择时间"
            style="width: 100%"
            :disabled="mode === 'create'" 
          />
          <span v-if="mode === 'create'" class="text-xs text-gray-400">默认为当前时间</span>
        </el-form-item>

        <el-form-item label="你想说点什么？">
          <el-input
            v-model="form.content"
            type="textarea"
            :rows="4"
            placeholder="写下你现在的感受..."
          />
        </el-form-item>
        
        <el-form-item label="心情状态">
          <el-radio-group v-model="form.emotion">
            <el-radio-button label="Happy">😊 开心</el-radio-button>
            <el-radio-button label="Neutral">😐 平静</el-radio-button>
            <el-radio-button label="Sad">😢 难过</el-radio-button>
            <el-radio-button label="Angry">😡 生气</el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" @click="submitDiary" :loading="submitting">保存</el-button>
      </template>
    </el-dialog>
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
.header-section {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
}
.diary-card {
  border-left: 5px solid #4f46e5;
  transition: transform 0.2s;
}
.diary-card:hover {
  transform: translateY(-2px);
}
/* 简单的 Flex 工具类 */
.flex { display: flex; }
.flex-col { flex-direction: column; }
.justify-between { justify-content: space-between; }
.items-center { align-items: center; }
.items-start { align-items: flex-start; }
.flex-1 { flex: 1; }
.gap-2 { gap: 0.5rem; }
.gap-4 { gap: 1rem; }
.mb-2 { margin-bottom: 0.5rem; }
.ml-4 { margin-left: 1rem; }
.grid { display: grid; }
.whitespace-pre-wrap { white-space: pre-wrap; }
</style>