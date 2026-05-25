import { expect, test } from '@playwright/test'

test.describe('用户管理自动化测试', () => {
  test('普通用户访问用户管理页时被路由守卫拦截', async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('user', 'alice')
      window.localStorage.setItem('role', 'user')
    })

    await page.goto('/admin/users')

    await expect(page).toHaveURL(/\/user\/home$/)
  })

  test.describe('管理员场景', () => {
    test.beforeEach(async ({ page }) => {
      await page.addInitScript(() => {
        window.localStorage.setItem('user', 'admin')
        window.localStorage.setItem('role', 'admin')
      })

      await page.route('**/api/admin/users*', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            users: [
              { id: 1, username: 'admin', role: 'admin', has_face: true },
              { id: 2, username: 'alice', role: 'user', has_face: false },
              { id: 3, username: 'bob', role: 'user', has_face: false },
            ],
          }),
        })
      })

      await page.route('**/api/admin/capture_face/*', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ success: true, message: '人脸特征采集成功！' }),
        })
      })
    })

    test('管理员进入页面后成功加载用户表格', async ({ page }) => {
      await page.goto('/admin/users')

      await expect(page.getByRole('heading', { name: '人员管理' })).toBeVisible()
      await expect(page.getByText('admin', { exact: true })).toBeVisible()
      await expect(page.getByText('alice', { exact: true })).toBeVisible()
    })

    test('搜索用户名时表格正确过滤', async ({ page }) => {
      await page.goto('/admin/users')

      await page.getByPlaceholder('搜索用户名...').fill('bob')

      await expect(page.getByText('bob')).toBeVisible()
      await expect(page.getByText('alice')).not.toBeVisible()
    })

    test('点击删除时显示成功提示', async ({ page }) => {
      await page.goto('/admin/users')

      await page.getByRole('button', { name: '删除' }).first().click()
      await page.getByRole('button', { name: '确定删除' }).click()

      await expect(page.getByText('删除成功')).toBeVisible()
    })

    test('模拟人脸录入成功流程', async ({ page }) => {
      await page.goto('/admin/users')

      await page.getByRole('button', { name: '采集' }).nth(1).click()
      await page.getByRole('button', { name: '立即捕捉特征' }).click()

      await expect(page.getByText('人脸特征采集成功！')).toBeVisible()
    })
  })
})
