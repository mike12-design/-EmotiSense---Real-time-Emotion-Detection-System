import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

const API_BASE = 'http://127.0.0.1:8000'

export const useSensingStore = defineStore('sensing', () => {
  /* ========= 状态 ========= */
  const sensing = ref(false)
  const currentEmotion = ref('Neutral')
  const isIntervening = ref(false)
  const recentLogs = ref([])
  const feedKey = ref(0)
  const feedUrl = ref('')
  const feedReady = ref(false)
  const overlayClosed = ref(false)

  let pollingTimer = null
  let audioPlayer = null

  /* ========= 辅助方法 ========= */
  const getEmotionColor = (e) => {
    const colors = {
      happy: '#34d399', sad: '#60a5fa', angry: '#f87171',
      neutral: '#7dd3fc', surprise: '#f472b6', fear: '#38bdf8', disgust: '#a3e635'
    }
    return colors[e?.toLowerCase()] || '#71717a'
  }

  const getMoodLabel = (mood) => {
    const map = {
      happy: '开心', sad: '悲伤', angry: '愤怒', neutral: '平静',
      surprise: '惊讶', fear: '恐惧', disgust: '厌恶'
    }
    return map[mood?.toLowerCase()] || '未知'
  }

  const setAudioPlayer = (el) => {
    audioPlayer = el
  }

  /* ========= 核心逻辑 ========= */
  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/status`)
      if (res.data.current_emotion) {
        currentEmotion.value = res.data.current_emotion
      }
      if (res.data.should_intervene) {
        handleIntervention(res.data.resource)
      }
      if (Math.random() > 0.7) {
        addLogToUI(currentEmotion.value)
      }
    } catch (_) {
      // 静默失败
    }
  }

  // 递归 setTimeout：等上一次请求完成后再等 500ms 发下一次，避免请求堆积
  const schedulePoll = () => {
    pollingTimer = setTimeout(async () => {
      await fetchStatus()
      if (sensing.value) schedulePoll()
    }, 500)
  }

  const startSensing = () => {
    overlayClosed.value = false
    sensing.value = true
    feedKey.value = Date.now()
    feedUrl.value = `${API_BASE}/video_feed?t=${feedKey.value}`
    feedReady.value = true
    schedulePoll()
  }

  const stopSensing = () => {
    overlayClosed.value = false
    sensing.value = false
    clearTimeout(pollingTimer)
    pollingTimer = null
    axios.post(`${API_BASE}/api/camera/stop`).catch(() => {})
    feedReady.value = false
  }

  const addLogToUI = (emotion) => {
    const now = new Date()
    recentLogs.value.unshift({
      id: Date.now(),
      time: `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`,
      emotion,
      score: 0.7 + Math.random() * 0.29
    })
    if (recentLogs.value.length > 10) recentLogs.value.pop()
  }

  const handleIntervention = (resource) => {
    if (isIntervening.value) return
    isIntervening.value = true

    if (resource?.audio_url && audioPlayer) {
      audioPlayer.src = `${API_BASE}/${resource.audio_url}`
      audioPlayer.play()
    }

    setTimeout(() => {
      isIntervening.value = false
    }, 5000)
  }

  const stopCamera = () => {
    axios.post(`${API_BASE}/api/camera/stop`).catch(() => {})
  }

  const releaseCamera = () => {
    axios.post(`${API_BASE}/api/camera/release`).catch(() => {})
  }

  const cleanup = () => {
    stopSensing()
  }

  return {
    sensing, currentEmotion, isIntervening, recentLogs,
    feedKey, feedUrl, feedReady, overlayClosed,
    getEmotionColor, getMoodLabel, setAudioPlayer,
    startSensing, stopSensing, fetchStatus, addLogToUI,
    stopCamera, releaseCamera, cleanup
  }
})
