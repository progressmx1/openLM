// Screenshots of the main screens with Playwright (uses the installed Microsoft Edge or
// Chrome; no browser download). The backend must be running and serving the built UI.
//
//   node scripts/screenshots.mjs [baseUrl] [outDir] [sessionId] [workspace]
import { chromium } from 'playwright-core'
import { existsSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const base = process.argv[2] ?? 'http://127.0.0.1:8765'
const out = process.argv[3] ?? join(process.cwd(), '..', '..', 'docs', 'screenshots')
const sessionId = process.argv[4]
const workspace = process.argv[5]
mkdirSync(out, { recursive: true })

const candidates = [
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
]
const executablePath = candidates.find((p) => existsSync(p))
if (!executablePath) throw new Error('No Edge/Chrome found for screenshots.')

const browser = await chromium.launch({ executablePath, headless: true })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('response', (r) => r.status() >= 400 && errors.push(`${r.status()} ${r.url()}`))

async function shot(name) {
  await page.waitForTimeout(700)
  await page.screenshot({ path: join(out, `${name}.png`) })
  console.log('saved', name)
}

await page.goto(base)
await page.evaluate(([id, ws]) => {
  localStorage.setItem('theme', 'dark')
  if (id) localStorage.setItem('session', JSON.stringify(id))
  if (ws) localStorage.setItem('workspace', JSON.stringify(ws))
}, [sessionId ?? null, workspace ?? null])
await page.reload()
await page.waitForSelector('text=Local Agent')
await shot('01-agent-dark')

// Side panel tabs
for (const tab of ['Changes', 'Terminal', 'Files']) {
  const button = page.getByRole('tab', { name: new RegExp(`^${tab}`) })
  if (await button.count()) {
    await button.click()
    await shot(`02-panel-${tab.toLowerCase()}`)
  }
}

// Command palette
await page.keyboard.press('Control+k')
await page.keyboard.type('mode')
await shot('03-command-palette')
await page.keyboard.press('Escape')

// Models & GPUs
await page.getByRole('button', { name: 'Models & GPUs' }).click()
await shot('04-models-gpus')

// Settings + prompts
await page.getByRole('button', { name: 'Settings' }).click()
await shot('05-settings')
await page.getByRole('radio', { name: 'Prompts' }).click()
await shot('06-settings-prompts')

// Light theme
await page.evaluate(() => localStorage.setItem('theme', 'light'))
await page.reload()
await page.waitForSelector('text=Local Agent')
await shot('07-agent-light')
await page.evaluate(() => localStorage.setItem('theme', 'dark'))

// Narrow window (responsive)
await page.setViewportSize({ width: 980, height: 760 })
await page.reload()
await shot('08-agent-narrow')

await browser.close()
if (errors.length) {
  console.error('Page errors:\n' + errors.join('\n'))
  process.exit(1)
}
console.log('No page errors.')
