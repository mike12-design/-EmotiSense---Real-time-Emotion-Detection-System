<template>
  <div class="user-history p-4">
    <!-- 1. 趋势分析图 -->
    <el-card class="mb-6 shadow-sm" style="border-radius: 16px;">
      <template #header>
        <div class="flex justify-between items-center">
          <div class="flex items-center gap-2">
            <el-icon color="#4f46e5"><TrendCharts /></el-icon>
            <span class="text-lg font-bold">情绪曲线</span>
          </div>
          <el-radio-group v-model="timeRange" size="small" @change="fetchChartData">
            <el-radio-button label="day">今日</el-radio-button>
            <el-radio-button label="week">本周</el-radio-button>
            <el-radio-button label="month">本月</el-radio-button>
          </el-radio-group>
        </div>
      </template>
      
      <!-- 图表容器 -->
      <div v-loading="chartLoading">
        <div ref="trendChartRef" style="height: 350px; width: 100%;"></div>
      </div>
    </el-card>

    <!-- 2. 数据列表 -->
    <el-card class="shadow-sm" style="border-radius: 16px;">
      <template #header>
        <span class="font-bold">历史明细</span>
      </template>
      <el-table :data="logs" stripe style="width: 100%">
        <el-table-column prop="timestamp" label="记录时间" width="180">
          <template #default="scope">
            {{ formatTime(scope.row.timestamp) }}
          </template>
        </el-table-column>
        
        <el-table-column label="心情状态" width="150">
          <template #default="scope">
            <el-tag :type="getEmotionTag(scope.row.emotion)" effect="dark" round>
              {{ getEmoji(scope.row.emotion) }} {{ scope.row.emotion }}
            </el-tag>
          </template>
        </el-table-column>
        
        <!-- 注意：列表里的 score 是置信度(0-1)，图表里的是插值分数(0-100) -->
        <el-table-column label="AI置信度">
          <template #default="scope">
            <el-progress 
              :percentage="Math.round(scope.row.score * 100)" 
              :color="scoreColors"
              :stroke-width="10"
            />
          </template>
        </el-table-column>
      </el-table>
      
      <div class="flex justify-end mt-4">
        <el-pagination 
          background 
          layout="prev, pager, next" 
          :total="total" 
          @current-change="fetchHistory"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import axios from 'axios';
import * as echarts from 'echarts';
import { TrendCharts } from '@element-plus/icons-vue';

const username = localStorage.getItem('user') || 'admin';
const timeRange = ref('day');
const chartLoading = ref(false);
const trendChartRef = ref(null);
let myChart = null;

const logs = ref([]);
const total = ref(0);

// 列表进度条颜色
const scoreColors = [
  { color: '#f56c6c', percentage: 40 },
  { color: '#e6a23c', percentage: 70 },
  { color: '#67c23a', percentage: 100 }
];

// 初始化图表
const initChart = (data) => {
  if (!myChart) {
    myChart = echarts.init(trendChartRef.value);
  }
  
  const option = {
    // 1. 颜色线性渐变映射 (红 -> 黄 -> 绿)
    visualMap: {
      show: false,
      dimension: 1,
      min: 0,
      max: 100,
      inRange: {
        color: ['#F56C6C', '#E6A23C', '#67C23A'] 
      }
    },
    tooltip: { 
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      formatter: (params) => {
        let val = params[0];
        let mood = '😐';
        if (val.data >= 80) mood = '😆';
        else if (val.data >= 60) mood = '🙂';
        else if (val.data <= 40) mood = '😢';
        
        return `<div style="font-size:12px;color:#666">${val.name}</div>
                <div style="font-weight:bold;font-size:14px;margin-top:4px">
                  ${mood} 心情指数: ${val.data}
                </div>`;
      }
    },
    grid: { top: '12%', left: '5%', right: '4%', bottom: '5%', containLabel: true },
    xAxis: {
      type: 'category',
      data: data.labels,
      axisLine: { lineStyle: { color: '#eee' } },
      axisLabel: { color: '#999' },
      boundaryGap: false // 让折线从头画到尾
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      interval: 20, // 强制分为 5 个刻度区间
      splitLine: { lineStyle: { type: 'dashed', color: '#f0f0f0' } },
      axisLabel: {
        color: '#888',
        fontSize: 16, // Emoji 稍微大一点
        formatter: (value) => {
          if (value >= 90) return '😆'; // 狂喜
          if (value >= 70) return '🙂'; // 开心
          if (value === 50) return '😐'; // 平静
          if (value <= 30) return '😢'; // 难过
          if (value <= 10) return '🤬'; // 愤怒
          return ''; 
        }
      }
    },
    series: [{
      name: '情绪指数',
      data: data.scores,
      type: 'line',
      smooth: 0.5, // 0.5 的平滑度比较自然
      lineStyle: { width: 4 },
      symbol: 'circle',
      symbolSize: 8,
      // 添加区域填充，让图表更丰满
      areaStyle: {
        opacity: 0.2
      }
    }]
  };
  
  myChart.setOption(option, true);
};

const fetchChartData = async () => {
  chartLoading.value = true;
  try {
    const res = await axios.get(`http://127.0.0.1:8000/api/my/history/stats`, {
      params: { username: username, range_type: timeRange.value }
    });
    initChart(res.data);
  } catch (e) {
    console.error("图表数据获取失败", e);
  } finally {
    chartLoading.value = false;
  }
};

const fetchHistory = async (page = 1) => {
  try {
    const res = await axios.get(`http://127.0.0.1:8000/api/my/history`, {
      params: { username: username, page: page }
    });
    logs.value = res.data.data;
    total.value = res.data.total;
  } catch (e) {
    console.error("历史记录获取失败", e);
  }
};

// 工具函数
const getEmoji = (m) => {
  const map = { happy: '😊', sad: '😢', angry: '😡', neutral: '😐', fear: '😨', surprise: '😲' };
  return map[m?.toLowerCase()] || '😶';
};

const getEmotionTag = (e) => {
  const map = { happy: 'success', sad: 'primary', angry: 'danger', neutral: 'info', fear: 'warning' };
  return map[e?.toLowerCase()] || 'info';
};

const formatTime = (ts) => {
  if (!ts) return '';
  return ts.replace('T', ' ').substring(0, 19);
};

const handleResize = () => myChart && myChart.resize();

onMounted(() => {
  fetchChartData();
  fetchHistory(1);
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
  if (myChart) myChart.dispose();
});
</script>

<style scoped>
.user-history { max-width: 1000px; margin: 0 auto; }
.flex { display: flex; }
.justify-between { justify-content: space-between; }
.items-center { align-items: center; }
.gap-2 { gap: 0.5rem; }
.mt-4 { margin-top: 1rem; }
.p-4 { padding: 1rem; }
.mb-6 { margin-bottom: 1.5rem; }
</style>