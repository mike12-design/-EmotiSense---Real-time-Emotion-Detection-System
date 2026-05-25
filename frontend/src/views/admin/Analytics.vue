<!--
  EmotiSense 管理后台 - 数据分析页面

  功能模块：
  1. 高危干预预警台（实时警报）
  2. 个体情绪动态轨迹图（卡尔曼滤波可视化）
  3. 干预效果事件轴
  4. 多模态日记 - 视觉冲突监控板
  5. AI 系统健康度看板
-->
<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue';
import axios from 'axios';
import { ElMessage, ElEmpty, ElTag } from 'element-plus';
import {
  TrendCharts, DataAnalysis, User, Warning, CircleCheck,
  Headset, Mic, Timer, DataLine,
  Document, VideoPlay, ChatLineRound, View
} from '@element-plus/icons-vue';

import * as echarts from 'echarts';

// 显式注册图表组件
import { use } from 'echarts/core';
import { LineChart, BarChart, PieChart, ScatterChart, GaugeChart } from 'echarts/charts';
import {
  TitleComponent, TooltipComponent, LegendComponent, GridComponent,
  VisualMapComponent, DatasetComponent,
  DataZoomComponent, MarkLineComponent, MarkPointComponent,
  TimelineComponent
} from 'echarts/components';

use([
  LineChart, BarChart, PieChart, ScatterChart, GaugeChart,
  TitleComponent, TooltipComponent, LegendComponent, GridComponent,
  VisualMapComponent, DatasetComponent,
  DataZoomComponent, MarkLineComponent, MarkPointComponent,
  TimelineComponent
]);

// ============ 响应式数据 ============
const loading = ref(false);
const timeRange = ref('7d');
const selectedUserId = ref(null);
const userList = ref([]);
const currentUser = ref(null);
const advancedStats = ref(null);

// 综合诊断报告数据
const comprehensiveReport = ref(null);

// 引入需要用到的图标


// 模块 1：高危预警台数据
const alertFeed = ref([]);
const alertLimit = ref(50);
const alertDateRange = ref([]);
const alertTotal = ref(0);

// 模块 2：情绪轨迹图
const trajectoryData = ref({
  scatterPoints: [],
  smoothedLine: [],
  attractor: 0,
  std: 1
});


// 模块 5：AI 系统健康度
const systemHealth = ref({
  emotionPieData: []
});

// DOM 引用
const trajectoryChartRef = ref(null);
const emotionPieChartRef = ref(null);

// 图表实例缓存
let trajectoryChart = null;
let emotionPieChart = null;
let isUnmounted = false;
let resizeTimer = null;

// 定时刷新
let alertRefreshTimer = null;
const ALERT_REFRESH_INTERVAL = 5000; // 5 秒刷新一次警报

const API_BASE = 'http://127.0.0.1:8000/api';

const nonAdminUsers = computed(() => userList.value.filter(u => u.role !== 'admin'));

// ============ 获取当前登录用户 ============
const fetchCurrentUser = async () => {
  const username = localStorage.getItem('user');
  if (!username) {
    ElMessage.error('未找到登录信息，请重新登录');
    return;
  }

  try {
    const res = await axios.get(`${API_BASE}/admin/users`);
    const users = res.data.users || [];
    const admin = users.find(u => u.username === username);
    if (admin) {
      currentUser.value = admin;
      selectedUserId.value = admin.id;
    }
  } catch (e) {
    console.error('获取当前用户失败:', e);
  }
};

// ============ 获取用户列表 ============
const fetchUserList = async () => {
  try {
    const res = await axios.get(`${API_BASE}/admin/users`);
    userList.value = res.data.users || [];
  } catch (e) {
    console.error('用户列表加载失败:', e);
  }
};

// ============ 获取高危预警数据 ============
const fetchAlertFeed = async () => {
  try {
    const params = { limit: alertLimit.value };
    if (alertDateRange.value?.length === 2) {
      const fmt = d => {
        const p = n => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
      };
      params.start_date = fmt(alertDateRange.value[0]);
      params.end_date = fmt(alertDateRange.value[1]);
    }
    const res = await axios.get(`${API_BASE}/admin/analytics/alerts`, { params });
    alertFeed.value = res.data.alerts || [];
    alertTotal.value = res.data.total || alertFeed.value.length;
  } catch (e) {
    console.error('警报数据加载失败:', e);
  }
};

const onAlertFilterChange = () => { fetchAlertFeed(); };

const formatTimestamp = (ts) => {
  if (!ts) return '--';
  const d = new Date(ts);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

// ============ 获取情绪轨迹数据 ============
const fetchTrajectoryData = async () => {
  if (!selectedUserId.value) return;

  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const res = await axios.get(`${API_BASE}/admin/analytics/advanced/${selectedUserId.value}?days=${days}`);

    // 转换为轨迹图数据
    const valenceHistory = res.data.valence_history || [];
    trajectoryData.value = {
      scatterPoints: valenceHistory.map((p, i) => [i, p.value]),
      smoothedLine: res.data.smoothed_valence || [],
      attractor: res.data.attractor || 0,
      std: res.data.attractor_std || 1
    };

    advancedStats.value = res.data;
    renderTrajectoryChart();
  } catch (e) {
    console.error('轨迹数据加载失败:', e);
  }
};
// ============ 获取综合诊断报告 ============
const fetchComprehensiveReport = async () => {
  if (!selectedUserId.value) return;
  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const res = await axios.get(`${API_BASE}/admin/analytics/comprehensive/${selectedUserId.value}?days=${days}`);
    comprehensiveReport.value = res.data;
  } catch (e) {
    console.error('综合报告加载失败:', e);
  }
};

// ============ 获取 AI 系统健康度数据 ============
const fetchSystemHealth = async () => {
  if (!selectedUserId.value) return;

  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const res = await axios.get(`${API_BASE}/admin/analytics/system-health`, {
      params: { user_id: selectedUserId.value, days }
    });
    systemHealth.value = res.data;
    renderEmotionPieChart(systemHealth.value.emotionPieData || []);
  } catch (e) {
    console.error('系统健康度加载失败:', e);
  }
};

// ============ 获取高级分析数据 ============
const fetchAdvancedAnalytics = async () => {
  if (!selectedUserId.value) return;

  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const url = `${API_BASE}/admin/analytics/advanced/${selectedUserId.value}?days=${days}`;
    const res = await axios.get(url);

    advancedStats.value = res.data;
  } catch (e) {
    console.error('高级分析数据加载失败:', e);
  }
};

// ============ 模块 1：情绪轨迹图 ============
const renderTrajectoryChart = () => {
  if (!trajectoryChartRef.value) return;
  if (!trajectoryChart) trajectoryChart = echarts.init(trajectoryChartRef.value);

  const { scatterPoints, smoothedLine, attractor, std } = trajectoryData.value;

  // 生成索引序列
  const indices = scatterPoints.map((_, i) => i);

  // 计算±2σ 边界
  const upperBand = smoothedLine.map(v => attractor + 2 * std);
  const lowerBand = smoothedLine.map(v => attractor - 2 * std);

  trajectoryChart.setOption({
    title: {
      text: '情绪动态轨迹（卡尔曼滤波）',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'bold' }
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        let result = `<b>时间点 ${params[0].axisValue}</b><br/>`;
        params.forEach(p => {
          result += `${p.seriesName}: ${p.value[1] || p.value}<br/>`;
        });
        return result;
      }
    },
    legend: {
      data: ['原始检测', '平滑曲线', '吸引子基线', '+2σ', '-2σ'],
      top: 40
    },
    xAxis: {
      type: 'category',
      data: indices,
      name: '时间序列'
    },
    yAxis: {
      type: 'value',
      name: '效价值',
      min: -1.5,
      max: 1.5
    },
    dataZoom: [{
      type: 'slider',
      start: 0,
      end: 100
    }],
    series: [
      // 原始散点
      {
        name: '原始检测',
        type: 'scatter',
        data: scatterPoints.map((p, i) => [indices[i], p[1]]),
        itemStyle: { color: '#ccc' },
        symbolSize: 6
      },
      // 平滑曲线
      {
        name: '平滑曲线',
        type: 'line',
        data: smoothedLine.map((v, i) => [indices[i], v]),
        itemStyle: { color: '#5f27cd' },
        lineStyle: { width: 3 },
        smooth: true
      },
      // 吸引子基线
      {
        name: '吸引子基线',
        type: 'line',
        data: indices.map(i => [i, attractor]),
        itemStyle: { color: '#1dd1a1' },
        lineStyle: { type: 'dashed', width: 2 }
      },
      // +2σ 边界
      {
        name: '+2σ',
        type: 'line',
        data: indices.map(i => [i, attractor + 2 * std]),
        itemStyle: { color: '#e6a23c' },
        lineStyle: { type: 'dotted', width: 1 },
        areaStyle: {
          color: 'rgba(230, 162, 60, 0.1)'
        }
      },
      // -2σ 边界
      {
        name: '-2σ',
        type: 'line',
        data: indices.map(i => [i, attractor - 2 * std]),
        itemStyle: { color: '#e6a23c' },
        lineStyle: { type: 'dotted', width: 1 },
        areaStyle: {
          color: 'rgba(230, 162, 60, 0.1)'
        }
      }
    ]
  });
};


// ============ 模块 5：情绪类别占比 ============
const renderEmotionPieChart = (pieData) => {
  if (!emotionPieChartRef.value) return;
  if (!emotionPieChart) emotionPieChart = echarts.init(emotionPieChartRef.value);

  const colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd', '#ff9ff3', '#c8d6e5'];
  emotionPieChart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { orient: 'vertical', left: 'left', top: 'middle' },
    color: colors,
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      center: ['55%', '50%'],
      data: pieData || []
    }]
  });
};

// 窗口缩放处理
const handleResize = () => {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (trajectoryChart && typeof trajectoryChart.resize === 'function') trajectoryChart.resize();
    if (emotionPieChart && typeof emotionPieChart.resize === 'function') emotionPieChart.resize();
  }, 100);
};

// 用户选择
const handleUserSelect = (userId) => {
  // 清理旧图表
  if (trajectoryChart && typeof trajectoryChart.dispose === 'function') trajectoryChart.dispose();
  if (emotionPieChart && typeof emotionPieChart.dispose === 'function') emotionPieChart.dispose();

  trajectoryChart = null;
  emotionPieChart = null;

  if (resizeTimer) clearTimeout(resizeTimer);

  advancedStats.value = null;
  selectedUserId.value = userId;
  fetchAlertFeed();

  fetchTrajectoryData();
  fetchSystemHealth();
};

// 时间范围选择
const handleTimeRangeChange = () => {
  fetchAlertFeed();

  fetchTrajectoryData();
  fetchSystemHealth();
};

// ============ 高级状态辅助函数 ============
const getAttractorClass = (value) => {
  if (value > 0.3) return 'positive';
  if (value < -0.3) return 'negative';
  return 'neutral';
};

const getRmssdClass = (value) => {
  if (value > 0.3) return 'high';
  if (value < 0.1) return 'low';
  return 'medium';
};

const getDeviationClass = (value) => {
  if (value > 2) return 'danger';
  if (value > 1.5) return 'warning';
  return 'normal';
};

const getInterventionText = (type) => {
  const map = {
    'tts_urgency': '⚠️ 情绪急救 - 需要立即关注',
    'tts': '建议语音安抚',
    'music': '建议音乐干预',
    'none': '状态良好'
  };
  return map[type] || type;
};

const getRiskLevelTag = (level) => {
  const map = { 'high': 'danger', 'medium': 'warning', 'low': 'success' };
  return map[level] || 'info';
};

const getTrendText = (direction) => {
  const map = {
    'rising': '📈 情绪上升',
    'falling': '📉 情绪下降',
    'stable': '➡️ 情绪稳定'
  };
  return map[direction] || '数据不足';
};

const getAlertIcon = (level) => {
  if (level === 'high') return '🚨';
  if (level === 'medium') return '⚠️';
  return 'ℹ️';
};

// 生命周期
onMounted(async () => {
  // 默认近3天
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 3);
  alertDateRange.value = [start, end];

  await fetchCurrentUser();
  fetchUserList();
  fetchAlertFeed();

  fetchTrajectoryData();
  fetchSystemHealth();
  window.addEventListener('resize', handleResize);

  // 启动警报定时刷新
  alertRefreshTimer = setInterval(fetchAlertFeed, ALERT_REFRESH_INTERVAL);
});

onUnmounted(() => {
  isUnmounted = true;

  if (alertRefreshTimer) clearInterval(alertRefreshTimer);
  window.removeEventListener('resize', handleResize);
  if (resizeTimer) clearTimeout(resizeTimer);

  // 清理所有图表实例
  if (trajectoryChart && typeof trajectoryChart.dispose === 'function') trajectoryChart.dispose();
  if (emotionPieChart && typeof emotionPieChart.dispose === 'function') emotionPieChart.dispose();
});
</script>

<template>
  <div class="analytics-container">
    <!-- 左侧：用户列表 -->
    <el-card class="user-list-card" shadow="never">
      <template #header>
        <div class="card-header">
          <el-icon><User /></el-icon>
          <span>用户列表</span>
        </div>
      </template>
      <el-menu :default-active="String(selectedUserId)" class="user-menu" @select="handleUserSelect">
        <el-menu-item v-for="user in nonAdminUsers" :key="user.id" :index="String(user.id)">
          <el-avatar :size="28" :style="{ backgroundColor: user.role === 'admin' ? '#f56c6c' : '#409EFF' }">
            {{ user.username.charAt(0).toUpperCase() }}
          </el-avatar>
          <span class="user-name">{{ user.username }}</span>
          <el-tag v-if="user.role === 'admin'" size="small" type="danger">管</el-tag>
        </el-menu-item>
      </el-menu>
    </el-card>

    <!-- 右侧：分析面板 -->
    <div class="analysis-panel">
      <!-- 顶部标题栏 -->
      <el-card class="header-card" shadow="never">
        <div class="header-content">
          <h2 v-if="selectedUserId">
            {{ userList.find(u => u.id === selectedUserId)?.username }} 用户情绪分析
          </h2>
          <h2 v-else>请选择用户</h2>
          <el-radio-group v-model="timeRange" size="small" @change="handleTimeRangeChange">
            <el-radio-button value="24h">24 小时</el-radio-button>
            <el-radio-button value="7d">7 天</el-radio-button>
            <el-radio-button value="30d">30 天</el-radio-button>
          </el-radio-group>
        </div>
      </el-card>
      <!-- 🌟 模块 0：AI 综合诊断报告 -->
      <el-row :gutter="16" class="module-row" v-if="comprehensiveReport && comprehensiveReport.has_data">
        <el-col :span="24">
          <el-card class="diagnostic-card" shadow="hover" :class="comprehensiveReport.risk_level">
            <!-- 头部：诊断结论 -->
            <div class="diagnostic-header">
              <div class="title-area">
                <span class="emoji-icon">🧠</span>
                <h3>AI 综合诊断与干预建议</h3>
              </div>
              <el-tag 
                effect="dark" 
                :type="comprehensiveReport.risk_level === 'high' ? 'danger' : comprehensiveReport.risk_level === 'medium' ? 'warning' : 'success'"
                size="large"
                class="diagnosis-tag"
              >
                {{ comprehensiveReport.conclusion }}
              </el-tag>
            </div>
            
            <!-- 主体：分析总结 & 指标 -->
            <div class="diagnostic-body">
              <div class="summary-text">
                <p>{{ comprehensiveReport.summary }}</p>
                <div class="metrics-tags">
                  <el-tag size="small" type="info" plain>平均效价 (Valence): {{ comprehensiveReport.metrics.avg_valence }}</el-tag>
                  <el-tag size="small" type="info" plain>波动率 (RMSSD): {{ comprehensiveReport.metrics.rmssd }}</el-tag>
                </div>
              </div>

              <!-- 智能建议列表 -->
              <div class="suggestions-area">
                <h4 class="sub-title">💡 智能干预建议：</h4>
                <div class="suggestion-list">
                  <div 
                    v-for="(sug, index) in comprehensiveReport.suggestions" 
                    :key="index" 
                    class="suggestion-item"
                  >
                    <!-- 动态图标盒子 -->
                    <div class="icon-box" :class="sug.type">
                      <el-icon v-if="sug.icon === 'Document'"><Document /></el-icon>
                      <el-icon v-else-if="sug.icon === 'Headset'"><Headset /></el-icon>
                      <el-icon v-else-if="sug.icon === 'Mic'"><Mic /></el-icon>
                      <el-icon v-else-if="sug.icon === 'VideoPlay'"><VideoPlay /></el-icon>
                      <el-icon v-else-if="sug.icon === 'ChatLineRound'"><ChatLineRound /></el-icon>
                      <el-icon v-else-if="sug.icon === 'View'"><View /></el-icon>
                    </div>
                    
                    <!-- 建议文字内容 -->
                    <div class="suggestion-content">
                      <span class="sug-title">{{ sug.title }}</span>
                      <span class="sug-desc">{{ sug.desc }}</span>
                    </div>

                    <!-- 执行动作按钮 (根据风险/类型可选展示) -->
                    <div class="sug-action" v-if="sug.type === 'medical' || sug.type === 'music'">
                      <el-button type="primary" size="small" plain @click="ElMessage.success('干预指令已发送至用户端')">
                        执行
                      </el-button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 模块 1：高危干预预警台 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="alert-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Warning /></el-icon>
                <span>高危干预预警台</span>
                <el-tag size="small" type="danger" v-if="alertFeed.length > 0">
                  {{ alertTotal }} 条警报
                </el-tag>
              </div>
            </template>

            <!-- 时间筛选栏 -->
            <div class="alert-filter-bar">
              <span class="filter-label">时间范围：</span>
              <el-date-picker
                v-model="alertDateRange"
                type="daterange"
                range-separator="至"
                start-placeholder="开始日期"
                end-placeholder="结束日期"
                size="small"
                style="width: 280px"
                @change="onAlertFilterChange"
              />
              <span class="filter-hint">最多显示 {{ alertLimit }} 条</span>
            </div>

            <div class="alert-feed">
              <div v-if="alertFeed.length === 0" class="no-alerts">
                <el-empty description="暂无警报" :image-size="60" />
              </div>
              <div v-else class="alert-list">
                <div
                  v-for="(alert, index) in alertFeed"
                  :key="index"
                  class="alert-item"
                  :class="alert.risk_level"
                >
                  <div class="alert-time">
                    <el-icon><Timer /></el-icon>
                    {{ formatTimestamp(alert.timestamp) }}
                  </div>
                  <div class="alert-content">
                    <span class="alert-icon">{{ getAlertIcon(alert.risk_level) }}</span>
                    <span class="alert-text">
                      用户 <b>{{ alert.username }}</b>
                      {{ alert.condition }}
                    </span>
                  </div>
                  <div class="alert-action">
                    <el-tag size="small" :type="getRiskLevelTag(alert.risk_level)">
                      {{ alert.risk_level === 'high' ? '高风险' : alert.risk_level === 'medium' ? '中风险' : '低风险' }}
                    </el-tag>
                    <span class="intervention-badge" v-if="alert.intervention">
                      <el-icon><Headset /></el-icon>
                      {{ alert.intervention }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 模块 2：情绪动态轨迹图 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="module-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><DataLine /></el-icon>
                <span>情绪动态轨迹（卡尔曼滤波）</span>
              </div>
            </template>
            <div ref="trajectoryChartRef" class="trajectory-chart"></div>
            <div class="chart-legend">
              <div class="legend-item">
                <span class="dot scatter"></span>
                <span>原始检测</span>
              </div>
              <div class="legend-item">
                <span class="dot line"></span>
                <span>平滑曲线</span>
              </div>
              <div class="legend-item">
                <span class="dot attractor"></span>
                <span>吸引子基线</span>
              </div>
              <div class="legend-item">
                <span class="dot band"></span>
                <span>±2σ 安全范围</span>
              </div>
            </div>
            <el-divider />
            <div class="trajectory-guide">
              <h4>图表解读</h4>
              <p><strong>Y 轴（效价值）：</strong>范围 -1 到 +1，正值表示积极情绪，负值表示消极情绪。</p>
              <p><strong>灰色散点：</strong>每次摄像头检测的原始情绪值，存在噪声波动。</p>
              <p><strong>紫色平滑曲线：</strong>经卡尔曼滤波去噪后的情绪趋势，反映真实情绪走向。</p>
              <p><strong>绿色吸引子基线：</strong>统计周期内用户情绪的"稳态"锚点，情绪长期围绕此基线波动。</p>
              <p><strong>±2σ 橙色边界：</strong>正常波动范围。曲线持续超出此边界意味着情绪显著偏离常态，需关注。</p>
              <p class="guide-tip">简单判断：曲线在绿色线附近小幅摆动 = 情绪稳定；曲线持续低于绿色线且触碰下边界 = 情绪低落需干预。</p>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 模块 5：情绪类别占比 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="module-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><CircleCheck /></el-icon>
                <span>情绪类别占比</span>
              </div>
            </template>
            <div ref="emotionPieChartRef" class="emotion-pie-chart"></div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<style scoped>
/* =======================================
   🌟 AI 综合诊断卡片专属样式 
======================================= */
.diagnostic-card {
  margin-bottom: 16px;
  border-left: 5px solid #409EFF; /* 默认蓝色边框 */
  background: linear-gradient(to right, rgba(64, 158, 255, 0.05), #ffffff 40%);
  transition: all 0.3s ease;
}

/* 动态风险等级变色 */
.diagnostic-card.high {
  border-left-color: #f56c6c;
  background: linear-gradient(to right, rgba(245, 108, 108, 0.1), #ffffff 40%);
}
.diagnostic-card.medium {
  border-left-color: #e6a23c;
  background: linear-gradient(to right, rgba(230, 162, 60, 0.1), #ffffff 40%);
}
.diagnostic-card.low {
  border-left-color: #67c23a;
  background: linear-gradient(to right, rgba(103, 194, 58, 0.08), #ffffff 40%);
}

.diagnostic-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  border-bottom: 1px dashed #ebeef5;
  padding-bottom: 12px;
}

.title-area {
  display: flex;
  align-items: center;
  gap: 10px;
}

.title-area h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
  font-weight: 600;
}

.emoji-icon {
  font-size: 24px;
}

.diagnosis-tag {
  font-size: 14px;
  font-weight: bold;
  padding: 8px 16px;
  border-radius: 6px;
}

.diagnostic-body {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.summary-text p {
  margin: 0 0 12px 0;
  font-size: 15px;
  color: #606266;
  line-height: 1.6;
  font-weight: 500;
}

.metrics-tags {
  display: flex;
  gap: 12px;
}

/* 建议列表区域 */
.suggestions-area {
  background: #f8f9fa;
  padding: 16px;
  border-radius: 8px;
}

.sub-title {
  margin: 0 0 16px 0;
  font-size: 15px;
  color: #303133;
  font-weight: 600;
}

.suggestion-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.suggestion-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  background: #fff;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #e4e7ed;
  transition: all 0.2s;
}

.suggestion-item:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
  border-color: #dcdfe6;
}

/* 动态图标盒子颜色 */
.icon-box {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  flex-shrink: 0;
}
.icon-box.medical { background: #fef0f0; color: #f56c6c; }
.icon-box.music { background: #ecf5ff; color: #409eff; }
.icon-box.tts { background: #fdf6ec; color: #e6a23c; }
.icon-box.action { background: #f0f9eb; color: #67c23a; }
.icon-box.hitokoto { background: #f4f4f5; color: #909399; }
.icon-box.monitor { background: #ecf5ff; color: #409eff; }

.suggestion-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.sug-title {
  font-weight: 600;
  font-size: 14px;
  color: #303133;
}

.sug-desc {
  font-size: 13px;
  color: #909399;
  line-height: 1.4;
}

.sug-action {
  margin-left: auto;
  align-self: center;
}
.analytics-container {
  display: flex;
  gap: 16px;
  padding: 16px;
  min-height: calc(100vh - 84px);
  box-sizing: border-box;
}

.user-list-card {
  width: 240px;
  flex-shrink: 0;
  height: fit-content;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.user-menu {
  border: none;
}

.user-menu .el-menu-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
}

.user-name {
  flex: 1;
  font-weight: 500;
}

.analysis-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
}

.header-card {
  --el-card-padding: 16px 20px;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h2 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

/* 模块通用样式 */
.module-row {
  margin-bottom: 0;
}

.module-card {
  margin-bottom: 16px;
}

/* 警报卡片样式 */
.alert-card {
  margin-bottom: 16px;
}

.alert-filter-bar {
  display: flex; align-items: center; gap: 12px;
  padding: 8px 0 14px 0; border-bottom: 1px solid #ebeef5; margin-bottom: 12px;
}
.filter-label { font-size: 13px; color: #606266; }
.filter-hint { font-size: 12px; color: #909399; margin-left: auto; }

.alert-feed {
  max-height: 320px;
  overflow-y: auto;
}

.alert-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.alert-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border-radius: 8px;
  border-left: 4px solid #ccc;
  background: #f8f9fa;
  transition: all 0.2s;
}

.alert-item:hover {
  transform: translateX(4px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.alert-item.high {
  border-left-color: #f56c6c;
  background: linear-gradient(to right, rgba(245, 108, 108, 0.1), transparent);
}

.alert-item.medium {
  border-left-color: #e6a23c;
  background: linear-gradient(to right, rgba(230, 162, 60, 0.1), transparent);
}

.alert-item.low {
  border-left-color: #67c23a;
}

.alert-time {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #909399;
}

.alert-content {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.alert-icon {
  font-size: 18px;
}

.alert-text b {
  color: #303133;
}

.alert-action {
  display: flex;
  align-items: center;
  gap: 8px;
}

.intervention-badge {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #606266;
  padding: 2px 8px;
  background: #f0f2f5;
  border-radius: 4px;
}

.no-alerts {
  text-align: center;
  padding: 40px 0;
}

/* 轨迹图 */
.trajectory-chart {
  height: 400px;
  width: 100%;
}

.chart-legend {
  display: flex;
  justify-content: center;
  gap: 24px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 4px;
  margin-top: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #606266;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.dot.scatter {
  background: #ccc;
}

.dot.line {
  background: #5f27cd;
}

.dot.attractor {
  background: #1dd1a1;
  border: 2px dashed #1dd1a1;
}

.dot.band {
  background: rgba(230, 162, 60, 0.3);
}

/* 轨迹图表解读指引 */
.trajectory-guide h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #303133;
}

.trajectory-guide {
  font-size: 13px;
  color: #606266;
  line-height: 1.8;
  padding: 8px 0;
}

.trajectory-guide p {
  margin: 6px 0;
}

.trajectory-guide .guide-tip {
  background: #ecf5ff;
  border-left: 3px solid #409eff;
  padding: 8px 12px;
  border-radius: 0 6px 6px 0;
  color: #303133;
  margin-top: 12px;
}

/* 情绪类别占比 */
.emotion-pie-chart {
  height: 300px;
  width: 100%;
}

/* 滚动条样式 */
.user-list-card::-webkit-scrollbar,
.alert-feed::-webkit-scrollbar {
  width: 6px;
}

.user-list-card::-webkit-scrollbar-thumb,
.alert-feed::-webkit-scrollbar-thumb {
  background: #dcdfe6;
  border-radius: 3px;
}

.user-list-card::-webkit-scrollbar-thumb:hover,
.alert-feed::-webkit-scrollbar-thumb:hover {
  background: #c0c4cc;
}

/* 响应式设计 */
@media (max-width: 1400px) {
  .analytics-container {
    flex-direction: column;
  }

  .user-list-card {
    width: 100%;
    max-height: 200px;
  }

}

@media (max-width: 1200px) {
  .module-row .el-col {
    width: 100%;
  }
}
</style>
