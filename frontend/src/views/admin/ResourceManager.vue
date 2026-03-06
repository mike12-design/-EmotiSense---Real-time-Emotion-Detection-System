<template>
  <div class="resource-container">
    <el-container style="height: 85vh; border: 1px solid #eee; border-radius: 8px;">
      
      <!-- 左侧：用户与系统列表 -->
      <el-aside width="250px" style="background-color: #fcfcfc; border-right: 1px solid #eee;">
        <div class="user-list-header">
          <span><el-icon><Menu /></el-icon> 资源作用域</span>
          <el-button link type="primary" icon="Refresh" @click="fetchUsers" circle></el-button>
        </div>
        
        <el-menu :default-active="activeUserIndex" class="user-menu">
          <el-menu-item 
            v-for="(item, index) in userList" 
            :key="item.id" 
            :index="index.toString()"
            @click="handleUserSelect(item)"
          >
            <div class="user-item">
              <el-avatar 
                v-if="item.isGlobal" 
                :size="24" 
                style="background-color: #409EFF"
              >
                <el-icon><Setting /></el-icon>
              </el-avatar>
              <el-avatar 
                v-else 
                :size="24" 
                :style="{ backgroundColor: stringToColor(item.username) }"
              >
                {{ item.username.charAt(0).toUpperCase() }}
              </el-avatar>
              
              <span class="username" :class="{'global-text': item.isGlobal}">
                {{ item.displayName }}
              </span>
              <el-tag size="small" v-if="item.role==='admin'" type="danger" effect="plain">管</el-tag>
            </div>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <!-- 右侧：资源详情 -->
      <el-main style="padding: 20px; background-color: #fff;">
        <div v-if="currentUser" class="resource-panel">
          <div class="panel-header">
            <h2>{{ currentUser.isGlobal ? '🌐 系统全局共享配置' : `👤 ${currentUser.username} 的专属配置` }}</h2>
            <el-tag :type="currentUser.isGlobal ? 'danger' : 'info'">
              {{ currentUser.isGlobal ? '影响所有未配置专属资源的用户' : '仅对此用户生效' }}
            </el-tag>
          </div>

          <el-tabs v-model="activeTab" class="resource-tabs">
            
            <!-- Tab 1: 个性化背景 (全局没有背景) -->
            <el-tab-pane label="🖼️ 背景图片" name="background" :disabled="currentUser.isGlobal">
              <!-- 省略背景配置代码，保持原样 -->
              <div v-if="currentUser.isGlobal" class="empty-state">
                <el-empty description="全局配置不支持设置背景图片，请在左侧选择具体用户" />
              </div>
              <el-row :gutter="20" v-else>
                <el-col :span="12">
                  <div class="bg-preview-card">
                    <p class="section-title">当前背景</p>
                    <div class="image-box">
                      <img :src="currentUserBgUrl" @error="handleImgError" v-if="!bgLoadError" key="bg-image"/>
                      <div v-else class="no-bg">
                        <el-icon :size="40" color="#ddd"><Picture /></el-icon>
                        <p>该用户使用的是系统默认背景</p>
                      </div>
                    </div>
                  </div>
                </el-col>
                <el-col :span="12">
                  <div class="bg-action-card">
                    <p class="section-title">管理操作</p>
                    <el-alert title="强制为该用户更改专属背景" type="warning" :closable="false" class="mb-4" />
                    <el-upload
                      class="upload-demo"
                      action="#"
                      :http-request="handleBgUpload"
                      :show-file-list="false"
                      accept=".jpg,.jpeg,.png"
                    >
                      <el-button type="primary" icon="Upload">上传/覆盖背景</el-button>
                    </el-upload>
                  </div>
                </el-col>
              </el-row>
            </el-tab-pane>

            <!-- Tab 2: 话术库 -->
            <el-tab-pane :label="`💬 安慰话术库 ${currentUser.isGlobal ? '(全局)' : '(专属)'}`" name="scripts">
              <div class="flex-header mb-4">
                <el-alert 
                  :title="currentUser.isGlobal ? '全局话术库：当用户没有专属话术时，将使用这里的句子兜底。' : `专属话术库：触发情绪时，优先对 ${currentUser.username} 播放这里的句子。`" 
                  :type="currentUser.isGlobal ? 'error' : 'success'" 
                  show-icon :closable="false" style="width: 70%" 
                />
                <el-button type="primary" @click="openAddScript" icon="Plus">添加话术</el-button>
              </div>

              <el-table :data="scripts" stripe height="450" style="width: 100%" border>
                <el-table-column prop="emotion_tag" label="情绪" width="100">
                  <template #default="scope">
                    <el-tag :type="getEmotionType(scope.row.emotion_tag)">{{ scope.row.emotion_tag }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column prop="content" label="话术内容" show-overflow-tooltip />
                <el-table-column label="操作" width="100" align="center">
                  <template #default="scope">
                    <el-button type="danger" link icon="Delete" @click="deleteScript(scope.row.id)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>

            <!-- 💡 核心修改点：Tab 3 音乐干预列表重构 -->
            <el-tab-pane :label="`🎵 音乐库 ${currentUser.isGlobal ? '(全局)' : '(专属)'}`" name="music">
              <div class="flex-header mb-4">
                <el-alert 
                  :title="currentUser.isGlobal ? '全局音乐库：当用户未配置对应情绪的专属音乐时，将随机播放这里的音乐。' : `专属音乐库：触发情绪时优先随机播放这里的音乐，未配置则使用全局。`" 
                  :type="currentUser.isGlobal ? 'error' : 'success'" 
                  show-icon :closable="false" style="width: 70%"
                />
                <el-button type="success" @click="openAddMusic" icon="Upload">上传音乐</el-button>
              </div>

              <el-table :data="musicList" border stripe height="450">
                <el-table-column prop="emotion_tag" label="情绪场景" width="150" align="center">
                  <template #default="scope">
                    <el-tag effect="dark" :type="getEmotionType(scope.row.emotion_tag)">
                      {{ (scope.row.emotion_tag || '').toUpperCase() }}
                    </el-tag>
                  </template>
                </el-table-column>
                
                <el-table-column prop="title" label="音乐名称" show-overflow-tooltip>
                  <template #default="scope">
                    <el-icon><Headset /></el-icon> {{ scope.row.title }}
                  </template>
                </el-table-column>

                <el-table-column label="操作" align="center" width="120">
                  <template #default="scope">
                    <el-button type="danger" link icon="Delete" @click="deleteMusic(scope.row.id)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-tab-pane>

          </el-tabs>
        </div>
      </el-main>
    </el-container>

    <!-- 添加话术弹窗 -->
    <el-dialog v-model="scriptDialogVisible" :title="`新增 ${currentUser?.isGlobal ? '全局' : '专属'} 话术`" width="450px" center>
      <el-form :model="scriptForm" label-position="top">
        <el-form-item label="触发情绪">
          <el-radio-group v-model="scriptForm.emotion_tag" size="large">
            <el-radio-button label="sad">😢 悲伤</el-radio-button>
            <el-radio-button label="angry">😡 愤怒</el-radio-button>
            <el-radio-button label="happy">😊 开心</el-radio-button>
            <el-radio-button label="neutral">😐 平静</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="话术内容">
          <el-input 
            v-model="scriptForm.content" 
            type="textarea" :rows="4" 
            placeholder="请输入话术内容..." 
            maxlength="100" show-word-limit
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="scriptDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitScript">确认添加</el-button>
      </template>
    </el-dialog>

    <!-- 💡 核心修改点：新增上传音乐弹窗 -->
    <el-dialog v-model="musicDialogVisible" :title="`上传 ${currentUser?.isGlobal ? '全局' : '专属'} 音乐`" width="450px" center>
      <el-form :model="musicForm" label-position="top">
        <el-form-item label="触发情绪场景">
          <el-radio-group v-model="musicForm.emotion_tag" size="large">
            <el-radio-button label="sad">😢 悲伤</el-radio-button>
            <el-radio-button label="angry">😡 愤怒</el-radio-button>
            <el-radio-button label="happy">😊 开心</el-radio-button>
            <el-radio-button label="neutral">😐 平静</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="选择MP3文件">
          <el-upload
            ref="musicUploadRef"
            action="#"
            :auto-upload="false"
            :limit="1"
            accept=".mp3"
            :on-change="handleMusicFileChange"
            :on-remove="() => musicForm.file = null"
          >
            <template #trigger>
              <el-button type="primary">选择文件</el-button>
            </template>
            <template #tip>
              <div class="el-upload__tip">只能上传 MP3 格式文件</div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="musicDialogVisible = false">取消</el-button>
        <el-button type="success" @click="submitMusicUpload" :disabled="!musicForm.file">确认上传</el-button>
      </template>
    </el-dialog>

  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Menu, Setting, Picture, Upload, Plus, Delete, Headset } from '@element-plus/icons-vue'

const API_BASE = "http://127.0.0.1:8000"

const userList = ref([])
const currentUser = ref(null)
const activeUserIndex = ref("0")
const activeTab = ref('scripts') 
const bgLoadError = ref(false)
const timeStamp = ref(Date.now())

const scripts = ref([])
const scriptDialogVisible = ref(false)
const scriptForm = ref({ emotion_tag: 'sad', content: '' })

// 💡 核心修改点：音乐列表变为动态数组，不再固定4个
const musicList = ref([])
const musicDialogVisible = ref(false)
const musicForm = ref({ emotion_tag: 'happy', file: null })
const musicUploadRef = ref(null)

const targetIdentifier = computed(() => {
  return currentUser.value?.isGlobal ? 'global' : currentUser.value?.username
})

// 修复：正确映射情绪颜色
const getEmotionType = (e) => {
  const map = { happy: 'success', sad: 'primary', angry: 'danger', neutral: 'info' }
  return map[e?.toLowerCase()] || 'info'
}

// 获取用户及系统列表
const fetchUsers = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/admin/users`)

    const globalConfig = { 
      id: 'global_001', 
      username: 'global', 
      displayName: '全局系统共享资源', 
      isGlobal: true 
    }

    const normalUsers = res.data.users.map(u => ({
      ...u,
      displayName: u.username,
      isGlobal: false
    }))
    userList.value = [globalConfig, ...normalUsers]



    if (!currentUser.value && userList.value.length > 0) {
      handleUserSelect(userList.value)
    }
  } catch (e) {
    ElMessage.error('用户列表加载失败')
  }
}

const handleUserSelect = (user) => {
  currentUser.value = user
  bgLoadError.value = false
  timeStamp.value = Date.now()
  
  if (user.isGlobal && activeTab.value === 'background') {
    activeTab.value = 'scripts'
  }
  
  fetchScripts()
  fetchMusic() 
}

// 获取话术
const fetchScripts = async () => {
  if (!currentUser.value) return
  try {
    const res = await axios.get(`${API_BASE}/api/admin/scripts`, {
      params: { target_user: targetIdentifier.value }
    })
    scripts.value = res.data
  } catch (e) { console.error(e) }
}

const openAddScript = () => {
  scriptForm.value = { emotion_tag: 'sad', content: '' }
  scriptDialogVisible.value = true
}

const submitScript = async () => {
  if (!scriptForm.value.content) return ElMessage.warning("请输入内容")
  try {
    const payload = {
      ...scriptForm.value,
      target_user: targetIdentifier.value 
    }
    await axios.post(`${API_BASE}/api/admin/scripts`, payload)
    ElMessage.success('添加成功')
    scriptDialogVisible.value = false
    fetchScripts()
  } catch (e) { ElMessage.error('添加失败') }
}

const deleteScript = async (id) => {
  try {
    await axios.delete(`${API_BASE}/api/admin/scripts/${id}?target_user=${targetIdentifier.value}`)
    ElMessage.success('已删除')
    fetchScripts()
  } catch (e) { ElMessage.error('删除失败') }
}

// 💡 核心修改点：获取音乐列表逻辑调整为获取数据库记录数组
const fetchMusic = async () => {
  if (!currentUser.value) return
  try {
    const res = await axios.get(`${API_BASE}/api/admin/music`, {
      params: { target_user: targetIdentifier.value }
    })
    musicList.value = res.data ||[]
  } catch (e) {
    console.error('获取音乐列表失败', e)
  }
}

// 💡 核心修改点：打开上传音乐弹窗
const openAddMusic = () => {
  musicForm.value = { emotion_tag: 'happy', file: null }
  if (musicUploadRef.value) {
    musicUploadRef.value.clearFiles()
  }
  musicDialogVisible.value = true
}

// 💡 核心修改点：处理文件选择
const handleMusicFileChange = (file) => {
  musicForm.value.file = file.raw
}

// 💡 核心修改点：提交音乐表单（不再是自动上传，而是点击按钮统一提交）
const submitMusicUpload = async () => {
  if (!musicForm.value.file) return ElMessage.warning("请选择MP3文件")
  
  const formData = new FormData()
  formData.append('file', musicForm.value.file)
  formData.append('emotion', musicForm.value.emotion_tag)
  formData.append('target_user', targetIdentifier.value) 
  
  try {
    await axios.post(`${API_BASE}/api/admin/upload_music`, formData)
    ElMessage.success(`${musicForm.value.emotion_tag} 音乐上传成功`)
    musicDialogVisible.value = false
    fetchMusic() // 重新拉取最新列表
  } catch (e) { 
    ElMessage.error('上传失败') 
  }
}

// 💡 核心修改点：新增删除音乐的方法
const deleteMusic = async (id) => {
  ElMessageBox.confirm('确定要删除这首音乐吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      // 假设你的后端删除接口为 DELETE /api/admin/music/{id}
      await axios.delete(`${API_BASE}/api/admin/music/${id}?target_user=${targetIdentifier.value}`)
      ElMessage.success('删除成功')
      fetchMusic()
    } catch (e) {
      ElMessage.error('删除失败')
    }
  }).catch(() => {})
}

// 背景相关代码
const currentUserBgUrl = computed(() => `${API_BASE}/assets/bg_${currentUser.value?.username}.jpg?t=${timeStamp.value}`)
const handleImgError = () => bgLoadError.value = true

const handleBgUpload = async (options) => {
  const formData = new FormData()
  formData.append('file', options.file)
  formData.append('username', currentUser.value.username)
  try {
    await axios.post(`${API_BASE}/api/user/upload_background`, formData)
    ElMessage.success(`已更新用户的背景`)
    bgLoadError.value = false
    timeStamp.value = Date.now()
  } catch (e) { ElMessage.error('上传失败') }
}

const stringToColor = (str) => {
  let hash = 0
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash)
  const c = (hash & 0x00FFFFFF).toString(16).toUpperCase()
  return '#' + '00000'.substring(0, 6 - c.length) + c
}

onMounted(() => fetchUsers())
</script>

<style scoped>
.resource-container { padding: 0; }
.user-list-header { 
  padding: 15px 20px; font-weight: bold; border-bottom: 1px solid #eee;
  display: flex; justify-content: space-between; align-items: center; color: #606266;
}
.user-menu { border-right: none; }
.user-item { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
.username { font-weight: 500; color: #303133; overflow: hidden; text-overflow: ellipsis; }
.global-text { font-weight: 800; color: #409EFF; }

.panel-header { border-bottom: 1px solid #eee; padding-bottom: 15px; margin-bottom: 20px; }
.panel-header h2 { margin: 0 0 5px 0; font-size: 20px; }
.bg-preview-card, .bg-action-card { border: 1px solid #ebeef5; border-radius: 8px; padding: 20px; height: 300px; }
.section-title { font-weight: bold; margin-bottom: 15px; color: #606266; }
.image-box { width: 100%; height: 200px; background: #f5f7fa; border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 2px dashed #e4e7ed; }
.image-box img { width: 100%; height: 100%; object-fit: cover; }
.no-bg { text-align: center; color: #909399; }
.mb-4 { margin-bottom: 16px; }

/* 新增 header 通用 Flex 布局 */
.flex-header { display: flex; justify-content: space-between; align-items: center; }

.empty-state { display: flex; align-items: center; justify-content: center; height: 100%; flex-direction: column; padding: 50px 0;}
.el-aside::-webkit-scrollbar { width: 4px; }
.el-aside::-webkit-scrollbar-thumb { background: #e0e3e9; border-radius: 2px; }
</style>