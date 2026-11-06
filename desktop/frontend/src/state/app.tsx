import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { AppSettings, DevicesResponse, Model, Telemetry } from '../api/types'

export type Theme = 'dark' | 'light' | 'system'
export interface Toast {
  id: number
  tone: 'ok' | 'err' | 'info' | 'warn'
  text: string
}

interface AppData {
  connected: boolean
  models: Model[]
  devices: DevicesResponse | null
  telemetry: Telemetry | null
  settings: AppSettings | null
  theme: Theme
  setTheme: (t: Theme) => void
  refreshModels: () => Promise<void>
  refreshDevices: (rescan?: boolean) => Promise<void>
  saveSettings: (patch: Partial<AppSettings>) => Promise<void>
  toast: (text: string, tone?: Toast['tone']) => void
  toasts: Toast[]
  dismissToast: (id: number) => void
}

const Ctx = createContext<AppData | null>(null)

export function useApp(): AppData {
  const value = useContext(Ctx)
  if (!value) throw new Error('useApp outside provider')
  return value
}

function applyTheme(theme: Theme) {
  const dark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
  document.documentElement.classList.toggle('dark', dark)
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(true)
  const [models, setModels] = useState<Model[]>([])
  const [devices, setDevices] = useState<DevicesResponse | null>(null)
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      return (localStorage.getItem('theme') as Theme) || 'dark'
    } catch {
      return 'dark'
    }
  })
  const nextToast = useRef(1)

  const toast = useCallback((text: string, tone: Toast['tone'] = 'info') => {
    const id = nextToast.current++
    setToasts((t) => [...t.slice(-3), { id, tone, text }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), tone === 'err' ? 9000 : 4500)
  }, [])

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.models())
      setConnected(true)
    } catch {
      setConnected(false)
    }
  }, [])

  const refreshDevices = useCallback(async (rescan = false) => {
    try {
      setDevices(await api.devices(rescan))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }, [toast])

  const saveSettings = useCallback(async (patch: Partial<AppSettings>) => {
    try {
      setSettings(await api.updateSettings(patch))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }, [toast])

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t)
    try {
      localStorage.setItem('theme', t)
    } catch {
      /* ignore */
    }
    applyTheme(t)
  }, [])

  useEffect(() => applyTheme(theme), [theme])

  useEffect(() => {
    void refreshModels()
    void refreshDevices()
    api.getSettings().then(setSettings).catch(() => setConnected(false))
  }, [refreshModels, refreshDevices])

  // Models: poll faster while one is loading so the UI picks up "ready"/"crashed".
  const anyBusy = models.some((m) => m.status?.state === 'loading')
  useEffect(() => {
    const id = setInterval(refreshModels, anyBusy ? 1500 : 5000)
    return () => clearInterval(id)
  }, [refreshModels, anyBusy])

  // Live GPU telemetry for the status bar and the GPU manager.
  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const t = await api.telemetry()
        if (!stop) setTelemetry(t)
      } catch {
        /* backend unreachable; connection state comes from model polling */
      }
    }
    void tick()
    const id = setInterval(tick, 2000)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [])

  const value = useMemo<AppData>(
    () => ({
      connected, models, devices, telemetry, settings, theme, setTheme, refreshModels, refreshDevices, saveSettings, toast, toasts,
      dismissToast: (id: number) => setToasts((t) => t.filter((x) => x.id !== id)),
    }),
    [connected, models, devices, telemetry, settings, theme, setTheme, refreshModels, refreshDevices, saveSettings, toast, toasts],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
