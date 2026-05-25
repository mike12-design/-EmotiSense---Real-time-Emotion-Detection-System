import { expect, test } from '@playwright/test'

const TEST_MP3 = {
  name: 'calm-track.mp3',
  mimeType: 'audio/mpeg',
  buffer: Buffer.from('ID3'),
}

const TEST_TXT = {
  name: 'invalid.txt',
  mimeType: 'text/plain',
  buffer: Buffer.from('not-an-audio-file'),
}

test.describe('资源管理功能', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('user', 'alice')
      window.localStorage.setItem('role', 'user')
    })

    await page.route('**/api/my/stats*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ user_id: 2, has_face: false }),
      })
    })

    await page.route('**/api/user/scripts*', async (route) => {
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { id: 1, emotion_tag: 'sad', content: '别着急，一切都会好起来。' },
          ]),
        })
        return
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      })
    })

    await page.route('**/api/user/music*', async (route) => {
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { id: 11, emotion_tag: 'happy', title: 'sunny.mp3' },
          ]),
        })
        return
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      })
    })

    await page.route('**/api/user/upload_music', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      })
    })

    await page.route('**/api/user/upload_background', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ success: false, message: '仅支持图片格式' }),
      })
    })
  })

  test('登录后进入个人设置页并渲染音乐库和话术库', async ({ page }) => {
    await page.goto('/user/settings')

    await expect(page.getByText('我的专属音乐库')).toBeVisible()
    await expect(page.getByText('安慰话术库')).toBeVisible()
    await expect(page.getByText('sunny.mp3')).toBeVisible()
    await expect(page.getByText('别着急，一切都会好起来。')).toBeVisible()
  })

  test('上传合法音频后触发上传成功流程', async ({ page }) => {
    await page.goto('/user/settings')

    await page.locator('.el-select').first().click()
    await page.locator('.el-select-dropdown:visible').getByText('😊 开心').first().click()

    const musicUploadInput = page.locator('input[type="file"]').nth(1)
    await musicUploadInput.setInputFiles(TEST_MP3)

    await expect(page.getByText('上传成功')).toBeVisible()
  })

  test('删除音乐项时确认弹窗后触发删除', async ({ page }) => {
    await page.goto('/user/settings')

    await page.locator('.el-table').getByRole('button', { name: '删除' }).first().click()
    await expect(page.getByText('确定删除吗？')).toBeVisible()
    await page.locator('.el-message-box__btns .el-button--primary').click()

    await expect(page.getByText('sunny.mp3')).toBeVisible()
  })

  test('添加话术后触发列表刷新', async ({ page }) => {
    await page.goto('/user/settings')

    await page.getByPlaceholder('输入新的安慰话术内容...').fill('你已经做得很好了。')
    await page.getByRole('button', { name: '添加' }).click()

    await expect(page.getByText('添加成功')).toBeVisible()
  })

  test('上传 txt 文件时显示错误提示', async ({ page }) => {
    await page.goto('/user/settings')

    const backgroundUploadInput = page.locator('input[type="file"]').first()
    await backgroundUploadInput.setInputFiles(TEST_TXT)

    await expect(page.getByText('上传失败')).toBeVisible()
  })
})
