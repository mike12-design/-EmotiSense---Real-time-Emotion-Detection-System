<!--
  EmotiSense 管理后台 - 数据分析页面

  功能模块：
  1. 高危干预预警台（实时警报 + 四象限散点图）
  2. 个体情绪动态轨迹图（卡尔曼滤波可视化）
  3. 干预效果事件轴
  4. 多模态日记 - 视觉冲突监控板
  5. AI 系统健康度看板
-->
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import axios from 'axios';
import { ElMessage, ElEmpty, ElTag } from 'element-plus';
import {
  TrendCharts, DataAnalysis, User, Warning, CircleCheck,
  Headset, Mic, Notebook, Timer, DataLine,
  Document, VideoPlay, ChatLineRound, View // 👈 这 4 个是给建议模块用的
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
const quadrantData = ref([]);

// 模块 2：情绪轨迹图
const trajectoryData = ref({
  scatterPoints: [],
  smoothedLine: [],
  attractor: 0,
  std: 1
});

// 模块 3：干预事件轴
const interventionEvents = ref([]);

// 模块 4：多模态冲突监控（增加 triggerQuestionnaire 字段）
const diaryVisualConflict = ref({
  visualEmotion: 0,
  diaryEmotion: 0,
  trustWeight: 0.7,
  conflictHistory: [],
  triggerQuestionnaire: false // ✨ 新增：默认不触发
});

// 模块 5：AI 系统健康度
const systemHealth = ref({
  confidenceDistribution: [],
  emotionPieData: [],
  modelAccuracy: 0
});

// DOM 引用
const quadrantChartRef = ref(null);
const trajectoryChartRef = ref(null);
const eventTimelineChartRef = ref(null);
const conflictChartRef = ref(null);
const confidenceChartRef = ref(null);
const emotionPieChartRef = ref(null);

// 图表实例缓存
let quadrantChart = null;
let trajectoryChart = null;
let eventTimelineChart = null;
let conflictChart = null;
let confidenceChart = null;
let emotionPieChart = null;
let isUnmounted = false;
let resizeTimer = null;

// 定时刷新
let alertRefreshTimer = null;
const ALERT_REFRESH_INTERVAL = 5000; // 5 秒刷新一次警报

const API_BASE = 'http://127.0.0.1:8000/api';

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
    const res = await axios.get(`${API_BASE}/admin/analytics/alerts`);
    alertFeed.value = res.data.alerts || [];
  } catch (e) {
    // 静默失败，不影响主功能
    console.error('警报数据加载失败:', e);
  }
};

// ============ 获取四象限数据 ============
const fetchQuadrantData = async () => {
  try {
    const res = await axios.get(`${API_BASE}/admin/analytics/quadrant`);
    quadrantData.value = res.data.users || [];
    renderQuadrantChart();
  } catch (e) {
    console.error('四象限数据加载失败:', e);
  }
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
// ============ 获取干预事件数据 ============
const fetchInterventionEvents = async () => {
  if (!selectedUserId.value) return;

  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const res = await axios.get(`${API_BASE}/admin/analytics/interventions/${selectedUserId.value}?days=${days}`);
    interventionEvents.value = res.data.events || [];
    renderEventTimeline();
  } catch (e) {
    console.error('干预事件加载失败:', e);
  }
};

// ============ 获取多模态冲突数据 ============
const fetchConflictData = async () => {
  if (!selectedUserId.value) return;

  try {
    const days = timeRange.value === '24h' ? 1 : timeRange.value === '7d' ? 7 : 30;
    const res = await axios.post(
      `${API_BASE}/admin/analytics/diary/validate/${selectedUserId.value}`,
      null,
      { params: { days } }
    );

    diaryVisualConflict.value = {
      visualEmotion: res.data.visual_avg || 0,
      diaryEmotion: res.data.diary_avg || 0,
      trustWeight: res.data.trust_weight || 0.7,
      conflictHistory: res.data.conflict_history || [],
      // ✨ 新增：接收后端判断的"撒谎/伪装"信号
      triggerQuestionnaire: res.data.trigger_questionnaire || false
    };

    renderConflictChart();

    // ✨ 可选：如果检测到撒谎，自动弹出轻提示
    if (diaryVisualConflict.value.triggerQuestionnaire) {
      ElMessage.warning(`检测到用户 ${selectedUserId.value} 存在"微笑抑郁"特征（表情与内心严重冲突），建议推送 PHQ-9 量表。`);
    }

  } catch (e) {
    console.error('冲突数据加载失败:', e);
  }
};

// ============ 获取 AI 系统健康度数据 ============
const fetchSystemHealth = async () => {
  try {
    const res = await axios.get(`${API_BASE}/admin/analytics/system-health`);
    systemHealth.value = res.data;
    renderConfidenceChart();
    renderAiHealthEmotionPieChart(systemHealth.value.emotionPieData || []);
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

// ============ 模块 1：四象限散点图 ============
const renderQuadrantChart = () => {
  if (!quadrantChartRef.value) return;
  if (!quadrantChart) quadrantChart = echarts.init(quadrantChartRef.value);

  // 准备数据
  const scatterData = quadrantData.value.map(u => [
    u.valence || 0,  // X 轴：效价
    u.rmssd || 0,    // Y 轴：波动率
    u.username,      // 用户名
    u.risk_level || 'low'  // 风险等级
  ]);

  // 风险颜色映射
  const riskColors = {
    high: '#f56c6c',    // 红
    medium: '#e6a23c',  // 橙
    low: '#67c23a'      // 绿
  };

  quadrantChart.setOption({
    title: {
      text: '用户情绪四象限分布',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'bold' }
    },
    tooltip: {
      formatter: (params) => {
        const riskText = params.data[3] === 'high' ? '高风险' : params.data[3] === 'medium' ? '中风险' : '低风险';
        return `<b>${params.data[2]}</b><br/>效价：${params.data[0]}<br/>波动率：${params.data[1]}<br/>风险：${riskText}`;
      }
    },
    xAxis: {
      name: '效价 (Valence)',
      nameLocation: 'middle',
      nameGap: 30,
      min: -1.2,
      max: 1.2,
      splitLine: { show: true, lineStyle: { type: 'dashed' } }
    },
    yAxis: {
      name: '波动率 (RMSSD)',
      nameLocation: 'middle',
      nameGap: 40,
      min: 0,
      max: 1,
      splitLine: { show: true, lineStyle: { type: 'dashed' } }
    },
    series: [{
      type: 'scatter',
      data: scatterData,
      symbolSize: (data) => {
        // 高风险用户显示更大的点
        return data[3] === 'high' ? 20 : data[3] === 'medium' ? 15 : 12;
      },
      itemStyle: {
        color: (params) => riskColors[params.data[3]] || '#999'
      },
      label: {
        show: true,
        formatter: (params) => params.data[2],
        position: 'top',
        fontSize: 10
      }
    }],
    // 区域标注
    markArea: {
      itemStyle: {
        color: [
          ['rgba(245, 108, 108, 0.1)', 'rgba(245, 108, 108, 0.05)'],  // 左下：抑郁重灾区
          ['rgba(230, 162, 60, 0.1)', 'rgba(230, 162, 60, 0.05)']    // 左上：崩溃边缘区
        ]
      },
      data: [
        [{
          name: '抑郁重灾区\n(深陷悲伤)',
          xAxis: -1.2,
          yAxis: 0
        }, {
          xAxis: 0,
          yAxis: 0.3
        }],
        [{
          name: '崩溃边缘区\n(情绪不稳定)',
          xAxis: -1.2,
          yAxis: 0.3
        }, {
          xAxis: 0,
          yAxis: 1
        }]
      ]
    }
  });
};

// ============ 模块 2：情绪轨迹图 ============
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

// ============ 模块 3：干预事件轴 ============
const renderEventTimeline = () => {
  if (!eventTimelineChartRef.value) return;
  if (!eventTimelineChart) eventTimelineChart = echarts.init(eventTimelineChartRef.value);

  // 准备数据
  const events = interventionEvents.value.map((e, i) => ({
    time: i,
    type: e.type,
    effect: e.effect || 0
  }));

  const musicEvents = events.filter(e => e.type === 'music').map(e => [e.time, 0.5]);
  const ttsEvents = events.filter(e => e.type === 'tts').map(e => [e.time, 0.8]);

  eventTimelineChart.setOption({
    title: {
      text: '干预效果时间轴',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'bold' }
    },
    tooltip: {
      formatter: (params) => {
        if (params.seriesName === '音乐干预') {
          return `🎵 音乐干预<br/>时间点：${params.data[0]}<br/>效果：${params.data[1] * 100}%`;
        } else {
          return `🎤 语音安抚<br/>时间点：${params.data[0]}<br/>效果：${params.data[1] * 100}%`;
        }
      }
    },
    xAxis: {
      type: 'category',
      data: events.map((_, i) => i),
      name: '时间序列'
    },
    yAxis: {
      type: 'value',
      name: '干预强度',
      min: 0,
      max: 1
    },
    series: [
      {
        name: '音乐干预',
        type: 'scatter',
        data: musicEvents,
        symbol: 'diamond',
        symbolSize: 20,
        itemStyle: { color: '#48dbfb' }
      },
      {
        name: '语音安抚',
        type: 'scatter',
        data: ttsEvents,
        symbol: 'circle',
        symbolSize: 20,
        itemStyle: { color: '#feca57' }
      }
    ]
  });
};

// ============ 模块 4：冲突对比图 ============
const renderConflictChart = () => {
  if (!conflictChartRef.value) return;
  if (!conflictChart) conflictChart = echarts.init(conflictChartRef.value);

  const { visualEmotion, diaryEmotion, trustWeight } = diaryVisualConflict.value;

  conflictChart.setOption({
    title: {
      text: '视觉 vs 日记 情绪对比',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'bold' }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    xAxis: {
      type: 'category',
      data: ['视觉识别', '日记情感']
    },
    yAxis: {
      type: 'value',
      name: '效价值',
      min: -1,
      max: 1
    },
    series: [{
      type: 'bar',
      data: [
        {
          value: visualEmotion,
          itemStyle: { color: visualEmotion > 0 ? '#67c23a' : '#f56c6c' }
        },
        {
          value: diaryEmotion,
          itemStyle: { color: diaryEmotion > 0 ? '#67c23a' : '#f56c6c' }
        }
      ],
      label: {
        show: true,
        formatter: '{c}',
        position: 'top'
      }
    }]
  });
};

// ============ 模块 5：AI 健康度图表 ============
const renderConfidenceChart = () => {
  if (!confidenceChartRef.value) return;
  if (!confidenceChart) confidenceChart = echarts.init(confidenceChartRef.value);

  const distribution = systemHealth.value.confidenceDistribution || [];

  confidenceChart.setOption({
    title: {
      text: '识别置信度分布',
      left: 'center',
      textStyle: { fontSize: 14 }
    },
    xAxis: {
      type: 'category',
      data: ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
      name: '置信度区间'
    },
    yAxis: {
      type: 'value',
      name: '次数'
    },
    series: [{
      type: 'bar',
      data: distribution,
      itemStyle: {
        color: (params) => {
          const colors = ['#f56c6c', '#e6a23c', '#909399', '#409eff', '#67c23a'];
          return colors[params.dataIndex];
        }
      }
    }]
  });
};

const renderAiHealthEmotionPieChart = (pieData) => {
  if (!emotionPieChartRef.value) return;
  if (!emotionPieChart) emotionPieChart = echarts.init(emotionPieChartRef.value);

  const colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd', '#ff9ff3', '#c8d6e5'];
  emotionPieChart.setOption({
    title: { text: '情绪类别占比', left: 'center', textStyle: { fontSize: 14 } },
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
    if (quadrantChart && typeof quadrantChart.resize === 'function') quadrantChart.resize();
    if (trajectoryChart && typeof trajectoryChart.resize === 'function') trajectoryChart.resize();
    if (eventTimelineChart && typeof eventTimelineChart.resize === 'function') eventTimelineChart.resize();
    if (conflictChart && typeof conflictChart.resize === 'function') conflictChart.resize();
    if (confidenceChart && typeof confidenceChart.resize === 'function') confidenceChart.resize();
    if (emotionPieChart && typeof emotionPieChart.resize === 'function') emotionPieChart.resize();
  }, 100);
};

// ✨ 推送 PHQ-9 问卷处理函数
const handlePushQuestionnaire = () => {
  ElMessage.success('指令已下发：PHQ-9 问卷已推送至用户端 App。');
  // 这里可以调用后端接口真正发消息
  // axios.post(`${API_BASE}/admin/push-phq9/${selectedUserId.value}`)
};

// ✨ 查看日记详情处理函数
const handleViewDiaryDetail = () => {
  // 跳转到日记管理页面或打开对话框
  ElMessage.info('即将跳转至日记详情页面');
  // router.push(`/admin/diaries?user=${selectedUserId.value}`)
};

// 用户选择
const handleUserSelect = (userId) => {
  // 清理旧图表
  if (quadrantChart && typeof quadrantChart.dispose === 'function') quadrantChart.dispose();
  if (trajectoryChart && typeof trajectoryChart.dispose === 'function') trajectoryChart.dispose();
  if (eventTimelineChart && typeof eventTimelineChart.dispose === 'function') eventTimelineChart.dispose();
  if (conflictChart && typeof conflictChart.dispose === 'function') conflictChart.dispose();
  if (confidenceChart && typeof confidenceChart.dispose === 'function') confidenceChart.dispose();
  if (emotionPieChart && typeof emotionPieChart.dispose === 'function') emotionPieChart.dispose();

  quadrantChart = null;
  trajectoryChart = null;
  eventTimelineChart = null;
  conflictChart = null;
  confidenceChart = null;
  emotionPieChart = null;

  if (resizeTimer) clearTimeout(resizeTimer);

  advancedStats.value = null;
  selectedUserId.value = userId;
  fetchAlertFeed();
  fetchQuadrantData();
  fetchTrajectoryData();
  fetchInterventionEvents();
  fetchConflictData();
  fetchSystemHealth();
};

// 时间范围选择
const handleTimeRangeChange = () => {
  fetchAlertFeed();
  fetchQuadrantData();
  fetchTrajectoryData();
  fetchInterventionEvents();
  fetchConflictData();
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
onMounted(() => {
  fetchCurrentUser();
  fetchUserList();
  fetchAlertFeed();
  fetchQuadrantData();
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
  if (quadrantChart && typeof quadrantChart.dispose === 'function') quadrantChart.dispose();
  if (trajectoryChart && typeof trajectoryChart.dispose === 'function') trajectoryChart.dispose();
  if (eventTimelineChart && typeof eventTimelineChart.dispose === 'function') eventTimelineChart.dispose();
  if (conflictChart && typeof conflictChart.dispose === 'function') conflictChart.dispose();
  if (confidenceChart && typeof confidenceChart.dispose === 'function') confidenceChart.dispose();
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
        <el-menu-item v-for="user in userList" :key="user.id" :index="String(user.id)">
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
        <!-- 实时警报滚动条 -->
        <el-col :span="12">
          <el-card class="alert-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Warning /></el-icon>
                <span>高危干预预警台</span>
                <el-tag size="small" type="danger" v-if="alertFeed.length > 0">
                  {{ alertFeed.length }} 条警报
                </el-tag>
              </div>
            </template>
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
                    {{ alert.timestamp }}
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

        <!-- 四象限散点图 -->
        <el-col :span="12">
          <el-card class="quadrant-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><TrendCharts /></el-icon>
                <span>用户情绪四象限</span>
              </div>
            </template>
            <div ref="quadrantChartRef" class="quadrant-chart"></div>
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
          </el-card>
        </el-col>
      </el-row>

      <!-- 模块 3：干预效果事件轴 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="module-card" shadow="hover" v-if="interventionEvents.length > 0">
            <template #header>
              <div class="card-header">
                <el-icon><Headset /></el-icon>
                <span>干预效果时间轴</span>
              </div>
            </template>
            <div ref="eventTimelineChartRef" class="timeline-chart"></div>
          </el-card>
        </el-col>
      </el-row>

  <!-- 模块 4：多模态冲突监控 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="module-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Notebook /></el-icon>
                <span>日记 - 视觉冲突监控</span>
              </div>
            </template>
            <div ref="conflictChartRef" class="conflict-chart"></div>
            
            <div class="conflict-analysis">
              <!-- 原有：信任度进度条 -->
              <div class="analysis-item">
                <span class="label">AI 视觉信任权重:</span>
                <el-progress
                  :percentage="Math.round(diaryVisualConflict.trustWeight * 100)"
                  :color="diaryVisualConflict.trustWeight > 0.7 ? '#67c23a' : '#e6a23c'"
                  style="width: 200px"
                ></el-progress>
              </div>

              <!-- ✨ 新增：微笑抑郁/撒谎检测 专用警告块 -->
              <template v-if="diaryVisualConflict.triggerQuestionnaire">
                <transition name="el-zoom-in-top" appear>
                  <div class="masking-alert">
                    <el-alert
                      title="⚠️ 触发【微笑抑郁】预警机制"
                      type="error"
                      description="检测到'表情积极'但'日记消极'的严重倒置（Masking Behavior）。系统判定视觉数据失效，建议立即介入。"
                      :closable="false"
                      show-icon
                      effect="dark"
                    ></el-alert>
                    <div class="action-row">
                      <el-button type="danger" size="small" plain @click="handlePushQuestionnaire">
                        立即推送 PHQ-9 抑郁筛查量表
                      </el-button>
                      <el-button type="info" size="small" plain @click="handleViewDiaryDetail">
                        查看日记详情
                      </el-button>
                    </div>
                  </div>
                </transition>
              </template>

              <!-- 原有：普通冲突提示（保留作为轻度提示） -->
              <template v-else-if="Math.abs(diaryVisualConflict.visualEmotion - diaryVisualConflict.diaryEmotion) > 0.5">
                <div class="analysis-item">
                  <el-alert
                    title="注意：视觉表情与文字情感存在偏差"
                    type="warning"
                    :closable="false"
                    show-icon
                  ></el-alert>
                </div>
              </template>
              
            </div> <!-- 这里是 conflict-analysis 的闭合 -->
          </el-card> <!-- 这里是 el-card 的闭合 -->
        </el-col> <!-- 这里是 el-col 的闭合 -->
      </el-row> <!-- 这里是 el-row 的闭合 -->

      <!-- 模块 5：AI 系统健康度 -->
      <el-row :gutter="16" class="module-row">
        <el-col :span="24">
          <el-card class="module-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><CircleCheck /></el-icon>
                <span>AI 系统健康度</span>
              </div>
            </template>
            <div class="health-metrics">
              <div ref="confidenceChartRef" class="confidence-chart"></div>
              <div ref="emotionPieChartRef" class="emotion-pie-chart"></div>
            </div>
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

.alert-feed {
  height: 300px;
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

/* 四象限图表 */
.quadrant-card {
  margin-bottom: 16px;
}

.quadrant-chart {
  height: 300px;
  width: 100%;
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

/* 时间轴图表 */
.timeline-chart {
  height: 200px;
  width: 100%;
}

/* 冲突图表 */
.conflict-chart {
  height: 250px;
  width: 100%;
}

.conflict-analysis {
  margin-top: 16px;
}

/* ✨ 微笑抑郁预警专用样式 */
.masking-alert {
  margin-top: 16px;
  border: 1px solid #fde2e2;
  padding: 10px;
  border-radius: 4px;
  background: #fef0f0;
}

.action-row {
  margin-top: 10px;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

/* AI 健康度 */
.health-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.confidence-chart,
.emotion-pie-chart {
  height: 250px;
  width: 100%;
  grid-column: span 2;
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

  .health-metrics {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 1200px) {
  .module-row .el-col {
    width: 100%;
  }
}
</style>
