<!-- frontend/src/views/admin/SystemLogs.vue -->
<template>
  <div class="logs-container">
    <el-card shadow="never" class="logs-card">
      <!-- 头部：标题 + 筛选器 -->
      <template #header>
        <div class="header-box">
          <div class="left-panel">
            <el-icon class="icon-title" :size="20"><DataAnalysis /></el-icon>
            <span class="title">用户情绪表现日志</span>
          </div>
          
          <!-- 用户筛选器 -->
          <div class="filter-box">
            <span class="label">按用户筛选：</span>
            <el-select 
              v-model="selectedUser" 
              placeholder="查看全部用户" 
              clearable 
              filterable
              @change="handleFilterChange"
              style="width: 200px"
            >
              <el-option
                v-for="user in userList"
                :key="user.id"
                :label="user.username"
                :value="user.username"
              >
                <span style="float: left">{{ user.username }}</span>
                <span style="float: right; color: #8492a6; font-size: 13px">
                  {{ user.role === 'admin' ? '管理员' : '用户' }}
                </span>
              </el-option>
            </el-select>
            
            <el-button 
              type="primary" 
              icon="Refresh" 
              circle 
              plain 
              class="ml-2"
              @click="refreshData" 
            />
          </div>
        </div>
      </template>

      <!-- 数据表格 -->
      <el-table 
        :data="logs" 
        v-loading="loading" 
        stripe 
        style="width: 100%"
        :header-cell-style="{ background: '#f5f7fa', color: '#606266' }"
      >
        
        <!-- 1. 捕获时间 -->
        <el-table-column prop="timestamp" label="捕获时间" width="200">
          <template #default="scope">
            <div class="time-cell">
              <el-icon><Clock /></el-icon>
              <span>{{ formatTime(scope.row.timestamp) }}</span>
            </div>
          </template>
        </el-table-column>

        <!-- 2. 识别用户 -->
<!-- 2. 识别用户 -->
<el-table-column label="识别用户" width="180">
  <template #default="scope">
    <div class="user-cell">
      <span class="username">{{ scope.row.username }}</span>
    </div>
  </template>
</el-table-column>

        <!-- 3. 情绪状态 -->
        <el-table-column prop="emotion" label="情绪状态" width="160">
          <template #default="scope">
            <el-tag :type="getEmotionType(scope.row.emotion)" effect="light" round>
              <span class="emoji">{{ getEmoji(scope.row.emotion) }}</span>
              {{ scope.row.emotion.toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>

        <!-- 4. 情绪强度 (置信度) -->
        <el-table-column label="情绪强度 (AI置信度)" min-width="200">
          <template #default="scope">
            <div class="score-wrapper">
               <el-progress
                :percentage="Math.round((scope.row.score || 0) * 100)"
                :stroke-width="10"
                :color="getScoreColor(scope.row.score)"
                :format="percentage => `${percentage}%`"
              />
            </div>
          </template>
        </el-table-column>
      </el-table>

      <!-- 底部：分页 -->
      <div class="pagination-container">
        <el-pagination
          background
          layout="total, prev, pager, next, jumper"
          :total="total"
          :page-size="pageSize"
          v-model:current-page="currentPage"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { ElMessage } from 'element-plus';
import { 
  DataAnalysis, 
  UserFilled, 
  Clock, 
  Refresh 
} from '@element-plus/icons-vue';

// --- 状态定义 ---
const API_BASE = "http://127.0.0.1:8000";
const logs = ref([]);
const userList = ref([]); // 用户下拉列表
const total = ref(0);
const loading = ref(false);
const currentPage = ref(1);
const pageSize = ref(10);
const selectedUser = ref(''); // 当前筛选的用户名

// --- API 请求 ---

/* 1. 获取所有用户列表 (用于筛选) */
const fetchUserList = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/admin/users`);
    userList.value = res.data;
  } catch (err) {
    console.error('获取用户列表失败', err);
  }
};

/* 2. 获取日志数据 (支持分页和筛选) */
const fetchLogs = async (page = 1) => {
  loading.value = true;
  try {
    const res = await axios.get(`${API_BASE}/api/admin/logs`, {
      params: { 
        page: page,
        page_size: pageSize.value,
        username: selectedUser.value // 传空字符串则显示所有注册用户
      }
    });
    logs.value = res.data.data || [];
    total.value = res.data.total || 0;
    currentPage.value = page;
  } catch (err) {
    ElMessage.error('无法加载日志数据');
  } finally {
    loading.value = false;
  }
};

// --- 事件处理 ---

const handleFilterChange = () => {
  // 筛选条件变化时，重置到第一页
  fetchLogs(1);
};

const handlePageChange = (page) => {
  fetchLogs(page);
};

const refreshData = () => {
  fetchLogs(currentPage.value);
  ElMessage.success('数据已刷新');
};

// --- 视觉辅助函数 ---

const formatTime = (ts) => {
  if (!ts) return '';
  return ts.replace('T', ' ').substring(0, 19);
};

const getEmoji = (m) => {
  const map = { happy: '😊', sad: '😢', angry: '😡', neutral: '😐', fear: '😨', surprise: '😲' };
  return map[m?.toLowerCase()] || '😶';
};

const getEmotionType = (emotion) => {
  const map = {
    'happy': 'success', 'sad': 'primary', 'angry': 'danger', 
    'neutral': 'info', 'surprise': 'warning', 'fear': 'danger'
  };
  return map[emotion?.toLowerCase()] || 'info';
};

const getScoreColor = (score) => {
  // 强度越高颜色越深/越明显
  if (score > 0.8) return '#67C23A'; // 高置信度 - 绿
  if (score > 0.5) return '#E6A23C'; // 中置信度 - 黄
  return '#909399';                  // 低置信度 - 灰
};

// 根据用户名生成一个固定的背景色，让头像更好看
const stringToColor = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
  return '#' + '00000'.substring(0, 6 - c.length) + c;
};

// --- 生命周期 ---
onMounted(() => {
  fetchUserList();
  fetchLogs(1);
});
</script>

<style scoped>
.logs-container {
  padding: 0;
}

.logs-card {
  border-radius: 8px;
  border: 1px solid #ebeef5;
}

/* 头部样式 */
.header-box { 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
}
.left-panel {
  display: flex;
  align-items: center;
  gap: 8px;
}
.icon-title { color: #409EFF; }
.title { font-size: 16px; font-weight: bold; color: #303133; }

/* 筛选区 */
.filter-box { display: flex; align-items: center; }
.label { font-size: 14px; color: #606266; margin-right: 10px; }
.ml-2 { margin-left: 10px; }

/* 表格内容样式 */
.time-cell {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #606266;
  font-family: monospace; /* 等宽字体显示时间更好看 */
}

.user-cell { display: flex; align-items: center; }
.username { font-weight: 600; color: #303133; }

.emoji { margin-right: 4px; font-size: 14px; }

.score-wrapper {
  max-width: 250px;
}

/* 分页 */
.pagination-container { 
  margin-top: 25px; 
  display: flex; 
  justify-content: flex-end; 
  padding-right: 10px;
}
</style>