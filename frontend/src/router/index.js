import { createRouter, createWebHistory } from 'vue-router'

// 1. 导入 Layouts
import AdminLayout from '../layouts/AdminLayout.vue'
import UserLayout from '../layouts/UserLayout.vue'

// 2. 导入 Admin 视图组件
import UserManager from '../views/admin/UserManager.vue'
import ResourceManager from '../views/admin/ResourceManager.vue'
import Analytics from '../views/admin/Analytics.vue'
import SystemLogs from '../views/admin/SystemLogs.vue'

// 3. 导入 Login 组件
import Login from '../views/Login.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'monitor',
      component: () => import('../views/user/MonitorMode.vue')
    },

    // 管理后台
    {
      path: '/admin',
      component: AdminLayout,
      redirect: '/admin/users', // ⭐ 修改1：将默认跳转改为 /admin/users
      meta: { requiresAuth: true, role: 'admin' },
      children: [
        { path: 'users', name: 'users', component: UserManager },
        { path: 'resources', name: 'resources', component: ResourceManager },
        { path: 'analytics', name: 'analytics', component: Analytics },
        { path: 'logs', name: 'logs', component: SystemLogs }
      ]
    },

    // 普通用户前台
    {
      path: '/user',
      component: UserLayout,
      redirect: '/user/home',
      meta: { requiresAuth: true, role: 'user' },
      children: [
        { path: 'home', component: () => import('../views/user/UserHome.vue') },
        { path: 'history', component: () => import('../views/user/UserHistory.vue') },
        { path: 'settings', component: () => import('../views/user/UserSettings.vue') },
        { 
          path: 'diary', 
          name: 'UserDiary',
          component: () => import('../views/user/UserDiary.vue') 
        }
      ]
    },

    // 登录页
    {
      path: '/login',
      name: 'Login',
      component: Login
    }
  ]
})

// --- 路由守卫 ---
router.beforeEach((to, from, next) => {
  const role = localStorage.getItem('role'); 

  if (to.matched.some(record => record.meta.requiresAuth)) {
    if (!role) {
      next('/login');
    } else if (to.meta.role && to.meta.role !== role) {
      // ⭐ 修改2：将管理员的重定向目标改为 /admin/users
      next(role === 'admin' ? '/admin/users' : '/user/home');
    } else {
      next();
    }
  } else {
    next();
  }
});

export default router