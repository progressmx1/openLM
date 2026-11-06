/**
 * Electron main process.
 *
 * Responsibilities:
 *  1. Spawn the Python FastAPI backend (uvicorn) on a localhost-only port.
 *  2. Wait for the backend to become healthy.
 *  3. Open a BrowserWindow pointed at the backend, which serves the built React
 *     UI (desktop/frontend/dist) and all API routes.
 *
 * The frontend must be built before launching (see scripts/start.ps1).
 */
const { app, BrowserWindow, shell, ipcMain, dialog } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')

const PORT = Number(process.env.LOCAL_AI_PORT || 8765)
const HOST = '127.0.0.1'
const URL = `http://${HOST}:${PORT}`

let backendProcess = null

function projectRoot() {
  // desktop/electron/main.js -> project root is two levels up.
  return path.resolve(__dirname, '..', '..')
}

function pythonCommand() {
  return process.env.LOCAL_AI_PYTHON || 'python'
}

function startBackend() {
  const root = projectRoot()
  const backendDir = path.join(root, 'backend')
  const python = pythonCommand()
  const args = [
    '-m', 'uvicorn', 'app.main:app',
    '--host', HOST,
    '--port', String(PORT),
  ]
  backendProcess = spawn(python, args, {
    cwd: backendDir,
    env: { ...process.env, PYTHONPATH: backendDir },
    stdio: 'inherit',
    windowsHide: true,
  })
  backendProcess.on('error', (err) => {
    console.error('Failed to start Python backend:', err.message)
  })
  backendProcess.on('exit', (code) => {
    backendProcess = null
    if (code !== 0 && code !== null) {
      console.error(`Backend exited with code ${code}`)
    }
  })
}

function waitForBackend(timeoutMs) {
  const started = Date.now()
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get(`${URL}/api/health`, (res) => {
        res.resume()
        resolve()
      })
      req.on('error', () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error('Backend did not become healthy in time'))
        } else {
          setTimeout(attempt, 300)
        }
      })
    }
    attempt()
  })
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 940,
    minHeight: 600,
    backgroundColor: '#0e0f11',
    title: 'Local Agent',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })
  win.removeMenu()
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
  await win.loadURL(URL)
}

// Native folder picker for "Open workspace" (the renderer has no filesystem access).
ipcMain.handle('pick-folder', async (event) => {
  const win = BrowserWindow.fromWebContents(event.sender)
  const result = await dialog.showOpenDialog(win, { properties: ['openDirectory', 'createDirectory'] })
  return result.canceled || result.filePaths.length === 0 ? null : result.filePaths[0]
})

app.whenReady().then(async () => {
  startBackend()
  try {
    await waitForBackend(30000)
    await createWindow()
  } catch (err) {
    console.error(err.message)
    app.quit()
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  if (backendProcess) {
    try {
      backendProcess.kill()
    } catch {
      /* already gone */
    }
  }
})
