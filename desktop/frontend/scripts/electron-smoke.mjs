// Launch the real Electron shell (which starts the Python backend), attach over the
// Chrome DevTools Protocol, check the UI rendered, take a screenshot, and quit.
//
//   node scripts/electron-smoke.mjs <screenshot.png> [port] [dataDir]
// Uses LOCAL_AI_DATA_DIR=<dataDir> so the real data/ folder is untouched.
import { chromium } from 'playwright-core'
import { spawn } from 'node:child_process'
import { join, resolve } from 'node:path'

const out = process.argv[2] ?? 'electron.png'
const port = process.argv[3] ?? '8777'
const dataDir = process.argv[4]
const debugPort = 9333
const shellDir = resolve(process.cwd(), '..', 'electron')
const exe = join(shellDir, 'node_modules', 'electron', 'dist', 'electron.exe')

const env = { ...process.env, LOCAL_AI_PORT: port }
if (dataDir) env.LOCAL_AI_DATA_DIR = dataDir
const child = spawn(exe, [`--remote-debugging-port=${debugPort}`, shellDir], { env, stdio: ['ignore', 'pipe', 'pipe'] })
let logs = ''
child.stdout.on('data', (d) => (logs += d))
child.stderr.on('data', (d) => (logs += d))

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let browser
for (let i = 0; i < 90 && !browser; i++) {
  try {
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${debugPort}`)
  } catch {
    await sleep(1000)
  }
}
if (!browser) {
  child.kill()
  throw new Error('Electron did not expose a debugging endpoint.\n' + logs.slice(-2000))
}
let page
for (let i = 0; i < 90 && !page; i++) {
  page = browser.contexts().flatMap((c) => c.pages()).find((p) => p.url().startsWith('http://127.0.0.1'))
  if (!page) await sleep(1000)
}
if (!page) throw new Error('Electron window never loaded the app.\n' + logs.slice(-2000))
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
await page.waitForSelector('text=Local Agent', { timeout: 60_000 })
await page.waitForTimeout(1500)
const title = await page.title()
const health = await page.evaluate(async () => (await fetch('/api/health')).json())
await page.screenshot({ path: out })
console.log(JSON.stringify({ title, url: page.url(), health, errors }))
await browser.close().catch(() => undefined)
child.kill()
await sleep(2000)
if (errors.length) process.exit(1)
