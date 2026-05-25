import { expect, test, Page } from '@playwright/test'

async function mockUserHomeApis(page: Page, username: string) {
  await page.route('**/api/my/personalized_quote*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ content: '今天也要温柔对待自己。', source: 'hitokoto' }),
    })
  })

  await page.route('**/api/admin/scripts/daily*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ content: '慢一点也没关系。', source: 'local' }),
    })
  })

  await page.route('**/api/my/calendar_moods*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ '2026-05-10': 'happy' }),
    })
  })

  await page.route('**/api/my/stats*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        user_id: 1,
        has_face: false,
        pie_data: [
          { name: 'happy', value: 3 },
          { name: 'neutral', value: 2 },
        ],
      }),
    })
  })

  await page.addInitScript(([name]) => {
    window.localStorage.setItem('user', name)
    window.localStorage.setItem('role', 'user')
  }, [username])
}

async function mockAdminUsers(page: Page) {
  await page.route('**/api/admin/users*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        users: [
          { id: 1, username: 'admin', role: 'admin', has_face: true },
          { id: 2, username: 'alice', role: 'user', has_face: false },
        ],
      }),
    })
  })
}

test.describe('登录模块', () => {
  test('输入正确的普通用户账号密码后跳转到用户首页', async ({ page }) => {
    await mockUserHomeApis(page, 'alice')

    await page.route('**/api/login', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, role: 'user', username: 'alice' }),
      })
    })

    await page.goto('/login')
    await page.getByPlaceholder('请输入用户名').fill('alice')
    await page.getByPlaceholder('请输入密码').fill('secret123')
    await page.getByRole('button').filter({ hasText: '回来' }).click()

    await expect(page).toHaveURL(/\/user\/home$/)
    await expect(page.getByText('Hi, alice')).toBeVisible()
  })

  test('输入正确的管理员账号密码后跳转到管理员首页', async ({ page }) => {
    await mockAdminUsers(page)

    await page.route('**/api/login', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, role: 'admin', username: 'admin' }),
      })
    })

    await page.goto('/login')
    await page.getByPlaceholder('请输入用户名').fill('admin')
    await page.getByPlaceholder('请输入密码').fill('123456')
    await page.getByRole('button').filter({ hasText: '回来' }).click()

    await expect(page).toHaveURL(/\/admin\/users$/)
    await expect(page.getByRole('heading', { name: '人员管理' })).toBeVisible()
    await expect(page.getByText('注册用户列表')).toBeVisible()
  })

  test('输入错误密码时显示 Element Plus 错误提示', async ({ page }) => {
    await page.route('**/api/login', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: false, message: '用户名或密码错误' }),
      })
    })

    await page.goto('/login')
    await page.getByPlaceholder('请输入用户名').fill('alice')
    await page.getByPlaceholder('请输入密码').fill('wrong-password')
    await page.getByRole('button').filter({ hasText: '回来' }).click()

    await expect(page.getByText('用户名或密码错误')).toBeVisible()
    await expect(page).toHaveURL(/\/login$/)
  })

  test('账号密码为空点击登录时触发表单校验拦截', async ({ page }) => {
    let loginCalled = false

    await page.route('**/api/login', async (route) => {
      loginCalled = true
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, role: 'user', username: 'alice' }),
      })
    })

    await page.goto('/login')
    await page.getByRole('button').filter({ hasText: '回来' }).click()

    await expect(page.getByText('请填写完整信息哦～')).toBeVisible()
    await expect(page).toHaveURL(/\/login$/)
    expect(loginCalled).toBeFalsy()
  })

  test('未登录状态直接访问受限页面时重定向回登录页', async ({ page }) => {
    await page.goto('/admin')

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByText('管理员账号：')).toBeVisible()
  })
})
