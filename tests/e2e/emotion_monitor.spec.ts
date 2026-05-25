import { expect, test } from '@playwright/test'

test.describe('实时监控页面', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('user', 'alice')
      window.localStorage.setItem('role', 'user')
    })

    await page.route('**/api/my/diaries*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { id: 1, content: '今天状态一般。', emotion: 'sad', timestamp: '2026-05-10T08:00:00' },
        ]),
      })
    })

    await page.route('**/api/status*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          current_emotion: 'sad',
          mood_score: 0.2,
          stress_level: 0.9,
          should_intervene: true,
          resource: {
            text: '别难过，我在这里陪你。',
            audio_url: 'assets/music/sad.mp3',
          },
        }),
      })
    })
  })

  test('页面渲染正常并显示视频区域和侧边内容', async ({ page }) => {
    await page.goto('/')

    await expect(page.getByText('AI 视觉感知中')).toBeVisible()
    await expect(page.locator('img.live-feed')).toBeVisible()
    await expect(page.getByText('情绪分布')).toBeVisible()
    await expect(page.getByText('感知日志')).toBeVisible()
  })

  test('Mock 负面情绪数据后触发智能干预 UI 状态', async ({ page }) => {
    await page.goto('/')

    await expect(page.getByText('😢 sad')).toBeVisible()
    await expect(page.locator('audio')).toHaveJSProperty('src', 'http://127.0.0.1:8000/assets/music/sad.mp3')
    await expect(page.getByText('我的日记')).toBeVisible()
  })

  test('模拟摄像头权限失败时出现异常提示', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {
          getUserMedia: async () => {
            throw new Error('Permission denied')
          },
        },
      })
    })

    await page.goto('/')

    await expect(page.locator('img.live-feed')).toBeVisible()
    await expect(page.getByText('AI 视觉感知中')).toBeVisible()
  })
})
