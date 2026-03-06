<template>
  <div class="settings-container">
    
    <!-- 第一行：安全与人脸 (原有功能) -->
    <el-row :gutter="20">
      <!-- 密码修改 -->
      <el-col :span="10">
        <el-card header="安全设置" shadow="hover">
          <el-form label-position="top">
            <el-form-item label="旧密码"><el-input type="password" placeholder="请输入旧密码" /></el-form-item>
            <el-form-item label="新密码"><el-input type="password" placeholder="请输入新密码" /></el-form-item>
            <el-button type="primary" class="w-100">保存修改</el-button>
          </el-form>
        </el-card>
      </el-col>

      <!-- 人脸录入 -->
      <el-col :span="14">
        <el-card header="生物特征管理" shadow="hover">
          <div class="face-section" v-loading="loading">
            <div v-if="hasFace">
              <div class="face-status">
                <el-icon :size="60" color="#67C23A"><CircleCheckFilled /></el-icon>
                <h3 class="success-text">人脸特征已录入</h3>
                <p class="sub-text">系统可以识别您的面部表情</p>
              </div>
              <el-divider />
              <div class="face-actions">
                <p class="tip">如果您更换了眼镜、发型，或者觉得识别不准确，可以重新录入。</p>
                <el-button type="warning" plain @click="openCaptureDialog">更新/重新录入</el-button>
              </div>
            </div>
            <div v-else>
              <div class="face-status">
                <el-icon :size="60" color="#909399"><WarningFilled /></el-icon>
                <h3 class="info-text">尚未录入人脸</h3>
                <p class="sub-text">录入后，系统将自动记录您的情绪变化</p>
              </div>
              <el-divider />
              <div class="face-actions">
                <p class="tip">请正对摄像头，保持光线充足。</p>
                <el-button type="primary" size="large" @click="openCaptureDialog">立即录入人脸</el-button>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    <!-- ⭐⭐⭐ 第三行：个性化设置 ⭐⭐⭐ -->
<el-row :gutter="20" class="mt-20">
  <el-col :span="24">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon class="mr-2"><Picture /></el-icon>全局背景设置</span>
        </div>
      </template>
      
      <div class="bg-settings-box">
        <!-- 预览 -->
        <div class="preview-area">
          <div 
            class="preview-box" 
            :style="{ backgroundImage: `url(${currentBg})` }"
          >
            <span v-if="!currentBg" class="no-bg-text">暂无背景</span>
          </div>
          <p class="text-center text-gray">当前背景预览</p>
        </div>

        <!-- 操作 -->
        <div class="action-area">
          <el-alert 
            title="设置后将应用到全站背景" 
            type="success" 
            :closable="false" 
            class="mb-3"
          />
          
          <el-upload
            action="#"
            :http-request="handleBgUpload"
            :show-file-list="false"
            accept=".jpg,.jpeg,.png"
          >
            <el-button type="primary">上传新背景图片</el-button>
          </el-upload>

          <el-button type="danger" plain class="mt-3" @click="clearBackground">
            恢复默认背景
          </el-button>
        </div>
      </div>
    </el-card>
  </el-col>
</el-row>


    <!-- 第二行：资源管理 (新增模块) -->
    <el-row :gutter="20" class="mt-20">
      
      <!-- 1. 音乐上传模块 -->
    <!-- 1. 音乐上传与管理 -->
<el-col :span="12">
  <el-card shadow="hover">
    <template #header>
      <div class="card-header flex justify-between">
        <span><el-icon class="mr-2"><Headset /></el-icon>我的专属音乐库</span>
        <el-button type="primary" size="small" @click="fetchUserMusic" circle>
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
    </template>

    <!-- 上传 -->
    <el-form :inline="true" size="small">
      <el-form-item label="情绪">
        <el-select v-model="musicEmotion" style="width: 120px">
          <el-option label="😊 开心" value="happy" />
          <el-option label="😢 难过" value="sad" />
          <el-option label="😡 愤怒" value="angry" />
          <el-option label="😐 平静" value="neutral" />
        </el-select>
      </el-form-item>

      <el-form-item>
        <el-upload
          action="#"
          :http-request="handleMusicUpload"
          :show-file-list="false"
          accept=".mp3"
          :disabled="!musicEmotion"
        >
          <el-button type="primary">上传音乐</el-button>
        </el-upload>
      </el-form-item>
    </el-form>

    <!-- 列表 -->
    <el-table :data="userMusicList" height="300" stripe border size="small">
      <el-table-column prop="emotion_tag" label="情绪" width="90">
        <template #default="scope">
          <el-tag :type="getEmotionTagType(scope.row.emotion_tag)" size="small">
            {{ scope.row.emotion_tag }}
          </el-tag>
        </template>
      </el-table-column>

      <el-table-column prop="title" label="歌名" />

      <el-table-column label="操作" width="60" align="center">
        <template #default="scope">
          <el-button type="danger" link @click="handleDeleteMusic(scope.row.id)">
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</el-col>

      <!-- 2. 话术管理模块 -->
      <el-col :span="14">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header flex justify-between">
              <span><el-icon class="mr-2"><ChatDotRound /></el-icon>安慰话术库</span>
              <el-button type="primary" size="small" @click="fetchScripts" circle><el-icon><Refresh /></el-icon></el-button>
            </div>
          </template>

          <!-- 添加新话术表单 -->
          <div class="add-script-box">
            <el-input 
              v-model="newScriptContent" 
              placeholder="输入新的安慰话术内容..." 
              class="mb-2"
              clearable
            >
              <template #prepend>
                <el-select v-model="newScriptEmotion" placeholder="情绪" style="width: 100px">
                  <el-option label="😊 开心" value="happy" />
                  <el-option label="😢 难过" value="sad" />
                  <el-option label="😡 愤怒" value="angry" />
                  <el-option label="😐 平静" value="neutral" />
                </el-select>
              </template>
              <template #append>
                <el-button @click="handleAddScript" :loading="scriptSubmitting">添加</el-button>
              </template>
            </el-input>
          </div>

          <!-- 话术列表 -->
          <div class="script-list-container">
            <el-table :data="scriptList" style="width: 100%" height="250" stripe>
              <el-table-column prop="emotion_tag" label="情绪" width="100">
                <template #default="scope">
                  <el-tag :type="getEmotionTagType(scope.row.emotion_tag)">
                    {{ scope.row.emotion_tag }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="content" label="话术内容" show-overflow-tooltip />
              <el-table-column label="操作" width="80" align="center">
                <template #default="scope">
                  <el-button 
                    type="danger" 
                    link 
                    icon="Delete" 
                    @click="handleDeleteScript(scope.row.id)"
                  ></el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 录入弹窗 (原有) -->
    <el-dialog 
      v-model="dialogVisible" 
      title="人脸特征采集" 
      width="500px" 
      center 
      :close-on-click-modal="false"
      destroy-on-close
    >
      <div class="camera-box">
        <img :src="videoUrl" style="width: 100%; border-radius: 8px; display: block;" />
        <div class="scan-line"></div>
      </div>
      <p style="text-align: center; color: #666; margin-top: 10px;">请摘下口罩，正视前方</p>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="capturing" @click="confirmCapture">
          {{ capturing ? '正在分析...' : '立即拍照录入' }}
        </el-button>
      </template>
    </el-dialog>

  </div>
</template>

<script setup>

import { Picture } from '@element-plus/icons-vue';
const userMusicList = ref([])
import { ref, onMounted } from 'vue';
import { 
  CircleCheckFilled, WarningFilled, UploadFilled, Headset, 
  ChatDotRound, Refresh, Delete 
} from '@element-plus/icons-vue';
import axios from 'axios';
import { ElMessage, ElMessageBox } from 'element-plus';

// ===== 基础配置 =====
const API_BASE = "http://127.0.0.1:8000";
const username = localStorage.getItem('user') || 'admin';
const currentBg = ref(localStorage.getItem('custom_bg') || '');

// ===== 人脸识别相关变量 (原有) =====
const dialogVisible = ref(false);
const capturing = ref(false);
const loading = ref(false);
const userId = ref(null);
const hasFace = ref(false);
const videoUrl = ref('');

// ===== 音乐模块变量 =====
const musicEmotion = ref(''); 

// ===== 话术模块变量 =====
const scriptList = ref([]);
const newScriptContent = ref('');
const newScriptEmotion = ref('sad'); // 默认选中难过，因为这个最常用
const scriptSubmitting = ref(false);
// ===== 背景上传 =====
const handleBgUpload = async (options) => {
  const formData = new FormData();
  formData.append("file", options.file);
  formData.append("username", username);

  try {
    const res = await axios.post(`${API_BASE}/api/user/upload_background`, formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });

    if (res.data.success) {
      const newUrl = res.data.url;
      currentBg.value = newUrl;
      localStorage.setItem('custom_bg', newUrl);
      window.dispatchEvent(new Event('bg-changed'));
      ElMessage.success("背景设置成功！");
    }
  } catch {
    ElMessage.error("上传失败");
  }
};

// Settings.vue 中的修改
const clearBackground = async () => {
  try {
    // 1. 调用后端接口删除服务器上的图片文件
    await axios.delete(`${API_BASE}/api/user/upload_background`, {
      params: { username: username }
    });

    // 2. 清理前端预览状态和本地持久化缓存
    currentBg.value = '';
    localStorage.removeItem('custom_bg');

    // 3. 💡 发送全局事件，通知 App.vue 立即更新背景
    window.dispatchEvent(new Event('bg-changed'));

    ElMessage.success("已恢复默认背景");
  } catch (e) {
    ElMessage.error("恢复默认背景失败");
  }
};

// ==========================================
// 1. 用户信息与人脸逻辑 (原有)
// ==========================================
const fetchUserInfo = async () => {
  loading.value = true;
  try {
    const res = await axios.get(`${API_BASE}/api/my/stats?username=${username}`);
    userId.value = res.data.user_id;
    hasFace.value = res.data.has_face;
  } catch (error) {
    ElMessage.error("获取用户信息失败");
  } finally {
    loading.value = false;
  }
};

const openCaptureDialog = () => {
  videoUrl.value = `${API_BASE}/video_feed?t=${new Date().getTime()}`;
  dialogVisible.value = true;
};

const confirmCapture = async () => {
  if (!userId.value) return;
  capturing.value = true;
  try {
    const res = await axios.post(`${API_BASE}/api/admin/capture_face/${userId.value}`);
    if (res.data.success) {
      ElMessage.success(res.data.message);
      dialogVisible.value = false;
      hasFace.value = true;
    } else {
      ElMessage.error(res.data.message);
    }
  } catch (e) {
    ElMessage.error("录入失败，请检查摄像头");
  } finally {
    capturing.value = false;
  }
};

// ==========================================
// 2. 音乐上传逻辑 (新增)
// ==========================================
const handleMusicUpload = async (options) => {
  const formData = new FormData()
  formData.append("file", options.file)
  formData.append("emotion", musicEmotion.value)
  formData.append("username", username)

  try {
    await axios.post(`${API_BASE}/api/user/upload_music`, formData)
    ElMessage.success("上传成功")
    fetchUserMusic()
  } catch {
    ElMessage.error("上传失败")
  }
}


// ==========================================
// 3. 话术管理逻辑 (新增)
// ==========================================
const fetchScripts = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/user/scripts`, {
      params: { username: username }});
    scriptList.value = res.data;
  } catch (error) {
    console.error("加载话术失败", error);
  }
};

const handleAddScript = async () => {
  if (!newScriptContent.value) return ElMessage.warning("请输入话术内容");
  
  scriptSubmitting.value = true;
  try {
     await axios.post(`${API_BASE}/api/user/scripts`, {
      content: newScriptContent.value,
      emotion_tag: newScriptEmotion.value,
      username: username // 提交当前用户
    });
    ElMessage.success("添加成功");
    newScriptContent.value = ""; // 清空输入
    fetchScripts(); // 刷新列表
  } catch (e) {
    ElMessage.error("添加失败");
  } finally {
    scriptSubmitting.value = false;
  }
};

const handleDeleteScript = async (id) => {
  try {
    await axios.delete(`${API_BASE}/api/user/scripts/${id}`, {
      params: { username: username } 
    });
    ElMessage.success("已删除");
    fetchScripts();
  } catch (e) {
    ElMessage.error("删除失败");
  }
};

const getEmotionTagType = (emotion) => {
  const map = { happy: 'success', sad: 'info', angry: 'danger', neutral: 'warning' };
  return map[emotion.toLowerCase()] || '';
};
const fetchUserMusic = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/user/music`, {
      params: { username }
    })
    userMusicList.value = res.data
  } catch {
    ElMessage.error("加载音乐失败")
  }
}


const handleDeleteMusic = async (id) => {
  ElMessageBox.confirm('确定删除吗？', '提示').then(async () => {
    await axios.delete(`${API_BASE}/api/user/music/${id}`, {
      params: { username }
    })
    fetchUserMusic()
    ElMessage.success("已删除")
  })
}


// ==========================================
// 生命周期
// ==========================================
onMounted(() => {
  fetchUserInfo()
  fetchScripts()
  fetchUserMusic()
});
</script>

<style scoped>

.bg-settings-box {
  display: flex;
  gap: 30px;
  align-items: center;
}
.preview-area {
  width: 300px;
}
.preview-box {
  width: 100%;
  height: 170px;
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  background-size: cover;
  background-position: center;
  background-color: #f5f7fa;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 10px;
}
.no-bg-text {
  color: #909399;
  font-size: 14px;
}
.text-gray {
  color: #606266;
  font-size: 12px;
}
.action-area {
  flex: 1;
}
.mt-3 { margin-top: 12px; }

.settings-container { padding: 20px; }
.face-section { text-align: center; padding: 10px; min-height: 200px; }
.face-status h3 { margin: 10px 0 5px; font-weight: 600; }
.success-text { color: #67C23A; }
.info-text { color: #909399; }
.sub-text { color: #909399; font-size: 14px; margin: 0; }
.tip { color: #606266; font-size: 13px; margin-bottom: 20px; line-height: 1.5; }
.camera-box { position: relative; overflow: hidden; border-radius: 8px; border: 2px solid #eee; }

/* 扫描线动画 */
.scan-line {
  position: absolute;
  top: 0; left: 0; width: 100%; height: 2px;
  background: rgba(64, 158, 255, 0.8);
  box-shadow: 0 0 4px rgba(64, 158, 255, 0.8);
  animation: scan 2s infinite linear;
}
@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

/* 间距辅助类 */
.mt-20 { margin-top: 20px; }
.mb-2 { margin-bottom: 8px; }
.mb-3 { margin-bottom: 12px; }
.mr-2 { margin-right: 8px; }
.text-center { text-align: center; }

/* 头部图标对齐 */
.card-header { display: flex; align-items: center; font-weight: bold; }
.flex { display: flex; }
.justify-between { justify-content: space-between; }
</style>