<template>
  <div class="logs-page">

    <!-- 顶部：用户选择器 + 统计 -->
    <div class="top-bar">
      <div class="user-select-area">
        <el-icon class="select-icon"><UserFilled /></el-icon>
        <span class="select-label">查看用户：</span>
        <el-select
          v-model="selectedUser"
          placeholder="选择用户"
          filterable
          size="large"
          style="width: 220px"
          @change="onUserChange"
        >
          <el-option label="全部用户（汇总）" value="" />
          <el-option
            v-for="u in userList"
            :key="u.id"
            :label="u.username"
            :value="u.username"
          />
        </el-select>
      </div>
      <div class="top-actions">
        <el-radio-group v-model="daysFilter" size="small" @change="onFilterChange">
          <el-radio-button value="1">今天</el-radio-button>
          <el-radio-button value="7">近7天</el-radio-button>
          <el-radio-button value="30">近30天</el-radio-button>
          <el-radio-button value="">全部时间</el-radio-button>
        </el-radio-group>
        <el-button size="small" @click="refreshData" circle><el-icon><Refresh /></el-icon></el-button>
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon" style="background:#ecf5ff;color:#409eff"><DataAnalysis /></div>
        <div class="stat-body">
          <div class="stat-value">{{ stats.total_count }}</div>
          <div class="stat-label">日志总数</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background:#f0f9eb;color:#67c23a"><Calendar /></div>
        <div class="stat-body">
          <div class="stat-value">{{ stats.today_count }}</div>
          <div class="stat-label">今日记录</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background:#fdf6ec;color:#e6a23c"><UserFilled /></div>
        <div class="stat-body">
          <div class="stat-value">{{ selectedUser ? 1 : stats.active_users }}</div>
          <div class="stat-label">{{ selectedUser ? '当前用户' : '活跃用户数' }}</div>
        </div>
      </div>
      <div class="stat-card dominant">
        <div class="stat-icon" :style="{ background: emotionColorBg(stats.dominant_emotion), color: emotionColor(stats.dominant_emotion) }">
          <span class="big-emoji">{{ getEmoji(stats.dominant_emotion) }}</span>
        </div>
        <div class="stat-body">
          <div class="stat-value">{{ emotionLabel(stats.dominant_emotion) }}</div>
          <div class="stat-label">主要情绪</div>
        </div>
      </div>
    </div>

    <!-- 内容区 -->
    <div class="content-row">
      <!-- 日志表格 -->
      <div class="table-panel">
        <el-card shadow="never">
          <template #header>
            <div class="panel-header">
              <span class="panel-title">
                <el-icon><List /></el-icon>
                {{ selectedUser ? `${selectedUser} 的情绪日志` : '全部用户情绪日志' }}
              </span>
              <div class="filter-bar">
                <el-select v-model="selectedEmotion" placeholder="情绪筛选" clearable size="small" style="width:120px" @change="onFilterChange">
                  <el-option v-for="e in emotions" :key="e.value" :label="e.label" :value="e.value" />
                </el-select>
              </div>
            </div>
          </template>

          <el-table :data="logs" v-loading="loading" stripe size="small" empty-text="暂无日志">
            <el-table-column label="时间" width="170">
              <template #default="{ row }">
                <span class="time-text">{{ formatTime(row.timestamp) }}</span>
              </template>
            </el-table-column>
            <el-table-column v-if="!selectedUser" label="用户" width="120">
              <template #default="{ row }">
                <div class="user-chip">
                  <span class="user-avatar-dot" :style="{ background: stringToColor(row.username) }"></span>
                  {{ row.username }}
                </div>
              </template>
            </el-table-column>
            <el-table-column label="情绪" width="110">
              <template #default="{ row }">
                <span class="emotion-tag" :style="{ background: emotionColorBg(row.emotion), color: emotionColor(row.emotion) }">
                  {{ getEmoji(row.emotion) }} {{ emotionLabel(row.emotion) }}
                </span>
              </template>
            </el-table-column>
            <el-table-column label="置信度" min-width="180">
              <template #default="{ row }">
                <div class="score-cell">
                  <span class="score-num">{{ scorePercent(row.score) }}%</span>
                  <div class="score-bar-track">
                    <div class="score-bar-fill" :style="{ width: scorePercent(row.score) + '%' }"></div>
                  </div>
                </div>
              </template>
            </el-table-column>
          </el-table>

          <div class="pagination-wrap">
            <el-pagination background layout="total, prev, pager, next" :total="total" :page-size="pageSize" v-model:current-page="currentPage" @current-change="fetchLogs" />
          </div>
        </el-card>
      </div>

      <!-- 情绪分布 -->
      <div class="side-panel">
        <el-card shadow="never">
          <template #header>
            <span class="panel-title"><el-icon><PieChart /></el-icon> 情绪分布</span>
          </template>
          <div class="dist-list" v-if="stats.emotion_distribution?.length">
            <div v-for="item in stats.emotion_distribution" :key="item.emotion" class="dist-item">
              <div class="dist-header">
                <span>{{ getEmoji(item.emotion) }} {{ emotionLabel(item.emotion) }}</span>
                <span class="dist-count">{{ item.count }}次</span>
              </div>
              <div class="dist-bar-track">
                <div class="dist-bar-fill" :style="{ width: distPercent(item.count) + '%', background: emotionColor(item.emotion) }"></div>
              </div>
            </div>
          </div>
          <div v-else class="empty-dist">暂无数据</div>
        </el-card>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { DataAnalysis, Calendar, UserFilled, TrendCharts, List, PieChart, Refresh } from '@element-plus/icons-vue'

const API_BASE = 'http://127.0.0.1:8000'

const logs = ref([])
const userList = ref([])
const total = ref(0)
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const selectedUser = ref('')
const selectedEmotion = ref('')
const daysFilter = ref('7')

const stats = reactive({
  total_count: 0, today_count: 0, active_users: 0,
  dominant_emotion: '', emotion_distribution: []
})

const emotions = [
  { label: '😊 开心', value: 'happy' }, { label: '😢 难过', value: 'sad' },
  { label: '😡 愤怒', value: 'angry' }, { label: '😐 平静', value: 'neutral' },
  { label: '😨 恐惧', value: 'fear' }, { label: '😲 惊讶', value: 'surprise' }
]

const fetchUserList = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/admin/users`)
    const users = res.data.users || []
    userList.value = users.filter(u => u.role !== 'admin')
    if (users.length && !selectedUser.value) {
      selectedUser.value = users[0].username
    }
  } catch { /* ignore */ }
}

const fetchLogs = async (page = 1) => {
  loading.value = true
  try {
    const res = await axios.get(`${API_BASE}/api/admin/logs`, {
      params: {
        page, page_size: pageSize.value,
        username: selectedUser.value || undefined,
        emotion: selectedEmotion.value || undefined,
        days: daysFilter.value || undefined
      }
    })
    logs.value = res.data.data || []
    total.value = res.data.total || 0
    currentPage.value = page
  } catch { ElMessage.error('加载日志失败') }
  finally { loading.value = false }
}

const fetchStats = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/admin/logs/stats`, {
      params: {
        username: selectedUser.value || undefined,
        days: daysFilter.value || undefined
      }
    })
    Object.assign(stats, res.data)
  } catch { /* ignore */ }
}

const onUserChange = () => { fetchLogs(1); fetchStats() }
const onFilterChange = () => { fetchLogs(1); fetchStats() }
const refreshData = () => { fetchLogs(currentPage.value); fetchStats(); ElMessage.success('已刷新') }

const formatTime = ts => ts ? ts.replace('T', ' ').substring(0, 19) : ''

const getEmoji = m => ({ happy:'😊', sad:'😢', angry:'😡', neutral:'😐', fear:'😨', surprise:'😲' })[m?.toLowerCase()] || '😶'
const emotionLabel = m => ({ happy:'开心', sad:'难过', angry:'愤怒', neutral:'平静', fear:'恐惧', surprise:'惊讶' })[m?.toLowerCase()] || m || '--'
const emotionColor = m => ({ happy:'#67c23a', sad:'#409eff', angry:'#f56c6c', neutral:'#909399', fear:'#e6a23c', surprise:'#9b59b6' })[m?.toLowerCase()] || '#909399'
const emotionColorBg = m => ({ happy:'#f0f9eb', sad:'#ecf5ff', angry:'#fef0f0', neutral:'#f4f4f5', fear:'#fdf6ec', surprise:'#f3e8ff' })[m?.toLowerCase()] || '#f4f4f5'

const scorePercent = s => Math.round((s || 0) * 100)

const distMax = () => Math.max(...(stats.emotion_distribution || []).map(d => d.count), 1)
const distPercent = count => Math.round((count / distMax()) * 100)

const stringToColor = str => {
  let hash = 0
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash)
  return '#' + ('00000' + (hash & 0x00FFFFFF).toString(16)).slice(-6)
}

onMounted(async () => {
  await fetchUserList()
  fetchLogs(1)
  fetchStats()
})
</script>

<style scoped>
.logs-page { padding: 0; }

/* ===== 顶部栏 ===== */
.top-bar {
  display: flex; justify-content: space-between; align-items: center;
  background: #fff; border-radius: 12px; padding: 16px 24px;
  margin-bottom: 20px; border: 1px solid #ebeef5;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.user-select-area { display: flex; align-items: center; gap: 10px; }
.select-icon { font-size: 20px; color: #409eff; }
.select-label { font-size: 15px; font-weight: 600; color: #303133; }
.top-actions { display: flex; align-items: center; gap: 10px; }

/* ===== 统计卡片 ===== */
.stats-row { display: flex; gap: 16px; margin-bottom: 20px; }
.stat-card {
  flex: 1; background: #fff; border-radius: 12px; padding: 18px 20px;
  display: flex; align-items: center; gap: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #ebeef5;
  transition: transform .2s, box-shadow .2s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.1); }
.stat-icon {
  width: 48px; height: 48px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center; font-size: 22px; flex-shrink: 0;
}
.big-emoji { font-size: 24px; line-height: 1; }
.stat-value { font-size: 24px; font-weight: 700; color: #303133; line-height: 1.2; }
.stat-label { font-size: 13px; color: #909399; margin-top: 2px; }

/* ===== 内容区 ===== */
.content-row { display: flex; gap: 20px; align-items: flex-start; }
.table-panel { flex: 1; min-width: 0; }
.side-panel { width: 260px; flex-shrink: 0; }

:deep(.el-card__header) { padding: 14px 20px; }
.panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
.panel-title { font-size: 15px; font-weight: 600; color: #303133; display: flex; align-items: center; gap: 6px; }
.filter-bar { display: flex; align-items: center; gap: 8px; }

/* ===== 表格 ===== */
.time-text { font-family: monospace; color: #606266; font-size: 13px; }
.user-chip { display: flex; align-items: center; gap: 6px; font-weight: 500; }
.user-avatar-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.emotion-tag { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 500; }
.score-cell { display: flex; align-items: center; gap: 10px; }
.score-num { font-size: 13px; font-weight: 600; color: #303133; min-width: 38px; }
.score-bar-track { flex: 1; height: 6px; border-radius: 3px; background: #f0f0f0; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #60a5fa, #3b82f6); transition: width .4s ease; }

.pagination-wrap { margin-top: 16px; display: flex; justify-content: flex-end; }

/* ===== 情绪分布 ===== */
.dist-list { display: flex; flex-direction: column; gap: 14px; }
.dist-header { display: flex; justify-content: space-between; font-size: 13px; color: #606266; margin-bottom: 4px; }
.dist-count { font-weight: 600; color: #303133; }
.dist-bar-track { height: 8px; border-radius: 4px; background: #f0f0f0; overflow: hidden; }
.dist-bar-fill { height: 100%; border-radius: 4px; transition: width .4s ease; }
.empty-dist { text-align: center; color: #909399; padding: 30px 0; font-size: 13px; }
</style>
