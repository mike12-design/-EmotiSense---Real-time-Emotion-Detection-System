<template>
  <div class="user-home">
    <!-- 1. 欢迎卡片 -->
    <el-card class="welcome-card" shadow="hover">
      <div class="welcome-content">
        <div class="text-section">
          <h1>{{ greeting }}，{{ username }} ☀️</h1>
          
          <!-- 话术展示区 -->
          <div class="quote-box">
            <transition name="fade-transform" mode="out-in">
              <!-- 使用 :key 绑定 quote 内容，确保内容变化时触发过渡动画 -->
              <div :key="quote" class="quote-wrap">
                <el-tag 
                  size="small" 
                  :type="quoteSource === 'hitokoto' ? 'success' : 'warning'" 
                  effect="plain"
                  class="source-tag"
                >
                  {{ quoteSource === 'hitokoto' ? '一言' : '心语' }}
                </el-tag>
                <span class="quote-text">“{{ quote || '正在为你寻觅动人的文字...' }}”</span>
              </div>
            </transition>
            
            <!-- “换一个” 按钮 -->
            <el-button 
              type="primary" 
              link 
              :icon="Refresh" 
              class="refresh-btn" 
              :loading="quoteLoading"
              @click="fetchMixedQuote"
            >
              换一个
            </el-button>
          </div>
        </div>

        <div class="stats-badge">
          <div class="badge-num">{{ totalRecords }}</div>
          <div class="badge-label">本月已记录</div>
        </div>
      </div>
    </el-card>

    <el-row :gutter="20" class="mt-20">
      <!-- 2. 左侧：情绪比重 -->
      <el-col :span="10">
        <el-card header="本月心情画像" shadow="hover">
          <div ref="pieChartRef" style="height: 350px;"></div>
        </el-card>
      </el-col>

      <!-- 3. 右侧：打卡日历 -->
      <el-col :span="14">
        <el-card header="心情日历" shadow="hover">
          <el-calendar v-model="currentDate">
            <template #date-cell="{ data }">
              <div class="calendar-item">
                <span :class="{ 'is-selected': data.isSelected }">{{ data.day.split('-')[2] }}</span>
                <div v-if="getDayMood(data.day)" class="mood-emoji">
                   {{ getMoodEmoji(getDayMood(data.day)) }}
                </div>
                <div v-if="getDayMood(data.day)" class="mood-dot" :style="{ backgroundColor: getMoodColor(getDayMood(data.day)) }"></div>
              </div>
            </template>
          </el-calendar>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, onUnmounted, nextTick } from 'vue';
import { Refresh } from '@element-plus/icons-vue'; // 引入刷新图标
import * as echarts from 'echarts';
import axios from 'axios';

// --- 状态定义 ---
const username = localStorage.getItem('user') || 'User';
const currentDate = ref(new Date());
const totalRecords = ref(0);
const pieChartRef = ref(null);
const moodData = ref({});
const quote = ref('');
const quoteSource = ref('local'); 
const quoteLoading = ref(false);
let myChart = null;

const API_BASE = 'http://127.0.0.1:8000';

// 情绪映射
const emotionMap = {
  happy:   { color: '#67C23A', emoji: '😄', label: '开心' },
  sad:     { color: '#409EFF', emoji: '😢', label: '难过' },
  angry:   { color: '#F56C6C', emoji: '😡', label: '生气' },
  neutral: { color: '#909399', emoji: '😐', label: '平静' },
  fear:    { color: '#303133', emoji: '😨', label: '恐惧' },
  surprise:{ color: '#E6A23C', emoji: '😲', label: '惊讶' }
};

// --- 计算属性 ---
const greeting = computed(() => {
  const h = new Date().getHours();
  return h < 6 ? '凌晨好' : h < 12 ? '早安' : h < 18 ? '午安' : '晚安';
});

// --- 核心方法：换一个 (1:1 混合) ---
const fetchMixedQuote = async () => {
  if (quoteLoading.value) return;
  quoteLoading.value = true;

  // 1:1 概率决定去向
  const isHitokotoTurn = Math.random() > 0.5; 
  
  try {
    let res;
    if (isHitokotoTurn) {
      // 尝试获取个性化一言 (后端逻辑：根据最近心情推荐)
      res = await axios.get(`${API_BASE}/api/my/personalized_quote`, { 
        params: { username },
        timeout: 2500 // 一言接口响应慢时快速跳入 catch
      });
    } else {
      // 尝试获取本地话术
      res = await axios.get(`${API_BASE}/api/admin/scripts/daily`);
    }

    quote.value = res.data.content;
    quoteSource.value = res.data.source; // 后端需返回 source 字段
  } catch (e) {
    console.warn("API请求失败，切换本地兜底");
    // 失败重试本地话术
    try {
      const fallback = await axios.get(`${API_BASE}/api/admin/scripts/daily`);
      quote.value = fallback.data.content;
      quoteSource.value = 'local';
    } catch (err) {
      quote.value = "生活总是充满希望，记得给自己一个微笑。";
      quoteSource.value = 'local';
    }
  } finally {
    // 稍微延迟关闭 loading 效果，防止动画太快看不清
    setTimeout(() => { quoteLoading.value = false; }, 300);
  }
};

// 数据加载
const fetchData = async () => {
  try {
    const [calRes, statsRes] = await Promise.all([
      axios.get(`${API_BASE}/api/my/calendar_moods?username=${username}`),
      axios.get(`${API_BASE}/api/my/stats?username=${username}`)
    ]);
    moodData.value = calRes.data;
    totalRecords.value = Object.keys(moodData.value).length;
    await nextTick();
    initPieChart(statsRes.data.pie_data);
  } catch (e) { console.error(e); }
};

// 辅助
const getDayMood = (d) => moodData.value[d];
const getMoodColor = (m) => emotionMap[m?.toLowerCase()]?.color || '#eee';
const getMoodEmoji = (m) => emotionMap[m?.toLowerCase()]?.emoji || '';

const initPieChart = (data) => {
  if (!pieChartRef.value) return;
  if (myChart) myChart.dispose();
  myChart = echarts.init(pieChartRef.value);
  const chartData = data.map(item => ({
    name: emotionMap[item.name.toLowerCase()]?.label || item.name,
    value: item.value,
    itemStyle: { color: getMoodColor(item.name) }
  }));
  myChart.setOption({
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie', radius: ['45%', '70%'], avoidLabelOverlap: false,
      itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
      label: { show: false }, data: chartData
    }]
  });
};

const handleResize = () => myChart && myChart.resize();

onMounted(() => {
  fetchMixedQuote();
  fetchData();
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  if (myChart) myChart.dispose();
  window.removeEventListener('resize', handleResize);
});
</script>

<style scoped>
.user-home { padding: 20px; }
.mt-20 { margin-top: 20px; }

/* 欢迎卡片：清新渐变 */
.welcome-card { 
  background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
  border: none; border-radius: 16px;
}
.welcome-content { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; }

.text-section h1 { margin: 0 0 15px 0; font-size: 28px; color: #2c3e50; }

.quote-box { display: flex; align-items: center; gap: 15px; min-height: 40px; }
.quote-wrap { display: flex; align-items: center; }
.quote-text { font-size: 16px; color: #5d6d7e; font-style: italic; }
.source-tag { margin-right: 10px; font-style: normal; }

.refresh-btn { font-size: 14px; color: #409EFF; transition: all 0.3s; }
.refresh-btn:hover { transform: rotate(180deg); }

.stats-badge { 
  text-align: center; background: white; padding: 15px 25px; 
  border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.badge-num { font-size: 32px; font-weight: bold; color: #409EFF; }
.badge-label { font-size: 12px; color: #909399; }

/* 动画效果：淡入、位移 */
.fade-transform-enter-active,
.fade-transform-leave-active {
  transition: all 0.4s ease;
}
.fade-transform-enter-from {
  opacity: 0;
  transform: translateX(-20px);
}
.fade-transform-leave-to {
  opacity: 0;
  transform: translateX(20px);
}

/* 日历单元格 */
.calendar-item { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.mood-emoji { font-size: 22px; margin: 4px 0; }
.mood-dot { width: 6px; height: 6px; border-radius: 50%; }
.is-selected { background: #409EFF; color: #fff; padding: 0 5px; border-radius: 4px; }

:deep(.el-calendar-table .el-calendar-day) { height: 75px !important; }
</style>