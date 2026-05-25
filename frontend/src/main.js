// frontend/src/main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

// 样式
import './assets/main.css'

// Element Plus
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// 创建应用
const app = createApp(App)
const pinia = createPinia()

// 注册 Element Plus（中文）
app.use(ElementPlus, { locale: zhCn })

// 注册图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 使用路由和 Pinia
app.use(router)
app.use(pinia)

// 挂载
try {
  app.mount('#app')
  console.log('[main] app mounted successfully')
} catch (e) {
  console.error('[main] failed to mount app', e)
}
