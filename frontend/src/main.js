// frontend/src/main.js
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

// 样式
import './assets/main.css'

// Element Plus
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 创建应用
const app = createApp(App)

// 注册 Element Plus
app.use(ElementPlus)

// 注册图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 使用路由
app.use(router)

// 挂载
try {
  app.mount('#app')
  console.log('[main] app mounted successfully')
} catch (e) {
  console.error('[main] failed to mount app', e)
}
