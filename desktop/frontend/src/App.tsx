import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api/client'
import type { AgentEvent, Mode, Session, Surface } from './api/types'
import { IconCheck, IconInfo, IconWarning, IconX } from './components/icons'
import { cx } from './components/ui'
import { AgentView, MODES } from './features/agent/AgentView'
import { ChatView } from './features/chat/ChatView'
import type { ComposerHandle } from './features/agent/Composer'
import { ModelsView } from './features/models/ModelsView'
import { CommandPalette, type PaletteAction } from './features/palette/CommandPalette'
import { RightPanel, type PanelTab } from './features/panels/RightPanel'
import { SettingsView } from './features/settings/SettingsView'
import { Sidebar, type View } from './features/sidebar/Sidebar'
import { WorkspacePicker } from './features/sidebar/WorkspacePicker'
import { StatusBar } from './features/status/StatusBar'
import { modelLabel } from './lib/format'
import type { ToolItem } from './lib/transcript'
import { AppProvider, useApp } from './state/app'
import { useSessionStream, useSessions } from './state/session'

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    /* ignore */
  }
}

function Shell() {
  const app = useApp()
  const { models, toast, theme, setTheme } = app
  const [view, setView] = useState<View>('agent')
  const [workspace, setWorkspace] = useState<string | null>(() => load('workspace', null))
  const [surface, setSurface] = useState<Surface>(() => load('surface', 'code'))
  // Each side remembers its own open session.
  const [codeSessionId, setCodeSessionId] = useState<string | null>(() => load('session', null))
  const [chatSessionId, setChatSessionId] = useState<string | null>(() => load('chatSession', null))
  const sessionId = surface === 'chat' ? chatSessionId : codeSessionId
  const setSessionId = useCallback((id: string | null) => (surface === 'chat' ? setChatSessionId(id) : setCodeSessionId(id)), [surface])
  const [session, setSession] = useState<Session | null>(null)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(() => load('sidebar', true))
  const [panelOpen, setPanelOpen] = useState(() => load('panel', true))
  const [panelTab, setPanelTab] = useState<PanelTab>('plan')
  const [changesVersion, setChangesVersion] = useState(0)
  const [commands, setCommands] = useState<Record<string, string>>({})
  const composer = useRef<ComposerHandle>(null)
  // Below ~820 px the sidebar becomes an overlay (closed by default; Ctrl+B opens it).
  const [narrow, setNarrow] = useState(() => window.innerWidth < 820)
  useEffect(() => {
    const onResize = () => {
      const isNarrow = window.innerWidth < 820
      setNarrow((was) => {
        if (isNarrow && !was) setSidebarOpen(false)
        if (!isNarrow && was) setSidebarOpen(true)
        return isNarrow
      })
    }
    onResize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  const { sessions, refresh: refreshSessions } = useSessions(workspace, surface)

  useEffect(() => save('workspace', workspace), [workspace])
  useEffect(() => save('session', codeSessionId), [codeSessionId])
  useEffect(() => save('chatSession', chatSessionId), [chatSessionId])
  useEffect(() => save('surface', surface), [surface])
  useEffect(() => save('sidebar', sidebarOpen), [sidebarOpen])
  useEffect(() => save('panel', panelOpen), [panelOpen])
  useEffect(() => {
    api.meta().then((m) => setCommands(m.commands)).catch(() => undefined)
  }, [])

  // Keep the selected session valid for the current workspace.
  useEffect(() => {
    if (!sessionId) return setSession(null)
    api.session(sessionId).then((s) => {
      const wrongSide = (surface === 'chat') !== (s.mode === 'chat')
      const wrongFolder = surface === 'code' && workspace && s.workspace.toLowerCase() !== workspace.toLowerCase()
      if (wrongSide || wrongFolder) setSessionId(null)
      else setSession(s)
    }).catch(() => setSessionId(null))
  }, [sessionId, workspace, surface, setSessionId])

  const onEvent = useCallback((e: AgentEvent) => {
    if (e.type === 'turn_start' || e.type === 'turn_end' || (e.type === 'tool' && e.status === 'awaiting_approval')) void refreshSessions()
    if (e.type === 'tool_update' && (e.data as Record<string, unknown> | undefined)?.diff) setChangesVersion((v) => v + 1)
    if (e.type === 'notice' && (typeof e.mode === 'string' || typeof e.model_id === 'string') && sessionId) {
      api.session(sessionId).then(setSession).catch(() => undefined)
    }
    if (e.type === 'turn_end') {
      setChangesVersion((v) => v + 1)
      if (document.hidden && e.reason === 'completed') document.title = '● Local Agent'
    }
  }, [refreshSessions, sessionId])
  useEffect(() => {
    const reset = () => !document.hidden && (document.title = 'Local Agent')
    document.addEventListener('visibilitychange', reset)
    return () => document.removeEventListener('visibilitychange', reset)
  }, [])

  const { transcript, loading } = useSessionStream(sessionId, onEvent)
  const terminal = useMemo(() => transcript.items.filter((i): i is ToolItem => i.kind === 'tool' && i.name === 'run_command'), [transcript.items])
  const model = models.find((m) => m.id === session?.model_id) ?? null

  const openWorkspace = async (path: string) => {
    const opened = await api.openWorkspace(path)
    setWorkspace(opened.path)
    setSessionId(null)
    setView('agent')
  }

  const newSession = useCallback(async (): Promise<Session | null> => {
    if (surface === 'code' && !workspace) {
      setPickerOpen(true)
      return null
    }
    try {
      const preferred = models.find((m) => m.loaded) ?? models.find((m) => m.id === session?.model_id) ?? null
      const created = surface === 'chat'
        ? await api.createSession('', preferred?.id ?? null, 'chat')
        : await api.createSession(workspace!, preferred?.id ?? null, session && session.mode !== 'chat' ? session.mode : 'auto')
      setSessionId(created.id)
      setSession(created)
      setView('agent')
      void refreshSessions()
      setTimeout(() => composer.current?.focus(), 0)
      return created
    } catch (e) {
      toast((e as Error).message, 'err')
      return null
    }
  }, [surface, workspace, models, session, refreshSessions, toast, setSessionId])

  const setMode = useCallback(async (mode: Mode) => {
    if (!session || session.mode === 'chat') return
    try {
      setSession(await api.updateSession(session.id, { mode }))
      void refreshSessions()
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }, [session, refreshSessions, toast])

  const deleteSession = async (id: string) => {
    if (!window.confirm(surface === 'chat' ? 'Delete this chat?' : 'Delete this session and its transcript? Files on disk are not changed.')) return
    await api.deleteSession(id)
    if (id === sessionId) setSessionId(null)
    void refreshSessions()
  }

  // Keyboard-first: global shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((v) => !v)
      } else if (mod && e.key.toLowerCase() === 'n') {
        e.preventDefault()
        void newSession()
      } else if (mod && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        setSidebarOpen((v) => !v)
      } else if (mod && e.key.toLowerCase() === 'j') {
        e.preventDefault()
        setPanelOpen((v) => !v)
      } else if (mod && ['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault()
        void setMode(MODES[Number(e.key) - 1].value)
      } else if (mod && e.key === 'l') {
        e.preventDefault()
        composer.current?.focus()
      } else if (!mod && view === 'agent' && session && !paletteOpen && !pickerOpen) {
        // Enter / Esc act on the first pending approval when focus is not in a text field.
        const target = e.target as HTMLElement
        if (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT' || target.tagName === 'SELECT') return
        const pending = transcript.items.find((i): i is ToolItem => i.kind === 'tool' && i.status === 'awaiting_approval')
        if (!pending) return
        if (e.key === 'Enter' && target.tagName !== 'BUTTON') {
          e.preventDefault()
          void api.decide(session.id, pending.callId, true)
        } else if (e.key === 'Escape') {
          e.preventDefault()
          void api.decide(session.id, pending.callId, false)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [newSession, setMode, view, session, transcript.items, paletteOpen, pickerOpen])

  const actions: PaletteAction[] = useMemo(() => {
    const list: PaletteAction[] = [
      { id: 'new', group: 'Session', label: surface === 'chat' ? 'New chat' : 'New session', hint: 'Ctrl N', run: () => void newSession() },
      { id: 'surface', group: 'Go to', label: surface === 'chat' ? 'Switch to Code' : 'Switch to Chat', run: () => { setSurface(surface === 'chat' ? 'code' : 'chat'); setView('agent') } },
      { id: 'open', group: 'Workspace', label: 'Open folder…', run: () => setPickerOpen(true) },
      { id: 'models', group: 'Go to', label: 'Models & GPUs', run: () => setView('models') },
      { id: 'settings', group: 'Go to', label: 'Settings', run: () => setView('settings') },
      { id: 'agent', group: 'Go to', label: 'Agent', run: () => setView('agent') },
      { id: 'sidebar', group: 'View', label: 'Toggle sidebar', hint: 'Ctrl B', run: () => setSidebarOpen((v) => !v) },
      { id: 'panel', group: 'View', label: 'Toggle side panel', hint: 'Ctrl J', run: () => setPanelOpen((v) => !v) },
      { id: 'theme', group: 'View', label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme', run: () => setTheme(theme === 'dark' ? 'light' : 'dark') },
      ...(['plan', 'changes', 'files', 'terminal'] as PanelTab[]).map((t) => ({
        id: `tab-${t}`, group: 'Panel', label: `Show ${t}`, run: () => { setPanelOpen(true); setPanelTab(t) },
      })),
      ...MODES.map((m, i) => ({ id: `mode-${m.value}`, group: 'Mode', label: `${m.label} — ${m.title}`, hint: `Ctrl ${i + 1}`, run: () => void setMode(m.value) })),
      ...Object.keys(commands).map((c) => ({ id: `cmd-${c}`, group: 'Command', label: c, run: () => { setView('agent'); composer.current?.setText(c === '/model' || c === '/mode' ? `${c} ` : c); composer.current?.focus() } })),
      ...sessions.map((s) => ({ id: `s-${s.id}`, group: 'Sessions', label: s.title, run: () => { setSessionId(s.id); setView('agent') } })),
      ...models.map((m) => ({
        id: `m-${m.id}`, group: 'Model', label: `${session ? 'Use' : 'Select'} ${modelLabel(m)}${m.loaded ? '' : ' (not loaded)'}`,
        run: () => session && void api.updateSession(session.id, { model_id: m.id }).then(setSession),
      })),
    ]
    if (session) list.push({ id: 'interrupt', group: 'Session', label: 'Stop the agent', run: () => void api.interrupt(session.id) })
    return list
  }, [newSession, theme, setTheme, setMode, commands, sessions, models, session, surface, setSessionId])

  return (
    <div className="flex h-full flex-col">
      <div className="flex min-h-0 flex-1">
        {sidebarOpen && narrow && <div className="fixed inset-0 z-30 bg-black/40" onClick={() => setSidebarOpen(false)} aria-hidden="true" />}
        {sidebarOpen && (
          <div className={narrow ? 'fixed inset-y-0 left-0 z-40 w-64 shadow-2xl' : 'w-60 shrink-0'}>
            <Sidebar
              surface={surface}
              onSurface={(next) => { setSurface(next); setView('agent') }}
              workspace={workspace}
              sessions={sessions}
              activeId={sessionId}
              view={view}
              onPickWorkspace={() => setPickerOpen(true)}
              onNewSession={() => void newSession()}
              onSelect={(id) => { setSessionId(id); setView('agent'); if (narrow) setSidebarOpen(false) }}
              onDelete={(id) => void deleteSession(id)}
              onView={setView}
            />
          </div>
        )}
        <main className="min-w-0 flex-1">
          {view === 'agent' && surface === 'chat' && (
            <ChatView
              ref={composer}
              session={session}
              transcript={transcript}
              loading={loading}
              commands={commands}
              onNewChat={newSession}
              onSessionChange={(s) => { setSession(s); void refreshSessions() }}
              onToggleSidebar={() => setSidebarOpen((v) => !v)}
              onOpenModels={() => setView('models')}
            />
          )}
          {view === 'agent' && surface === 'code' && (
            <AgentView
              ref={composer}
              workspace={workspace}
              session={session}
              transcript={transcript}
              loading={loading}
              commands={commands}
              panelOpen={panelOpen}
              onTogglePanel={() => setPanelOpen((v) => !v)}
              onToggleSidebar={() => setSidebarOpen((v) => !v)}
              onPickWorkspace={() => setPickerOpen(true)}
              onNewSession={newSession}
              onSessionChange={(s) => { setSession(s); void refreshSessions() }}
              onOpenModels={() => setView('models')}
            />
          )}
          {view === 'models' && <ModelsView />}
          {view === 'settings' && <SettingsView />}
        </main>
        {view === 'agent' && surface === 'code' && panelOpen && workspace && (
          <div className="hidden w-[380px] shrink-0 lg:block xl:w-[420px]">
            <RightPanel
              tab={panelTab}
              setTab={setPanelTab}
              sessionId={sessionId}
              workspace={workspace}
              todos={transcript.todos.length ? transcript.todos : session?.todos ?? []}
              terminal={terminal}
              changesVersion={changesVersion}
            />
          </div>
        )}
      </div>
      <StatusBar model={model} usage={transcript.usage} onOpenModels={() => setView('models')} />
      <WorkspacePicker open={pickerOpen} onClose={() => setPickerOpen(false)} onOpen={openWorkspace} />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} actions={actions} />
      <Toasts />
    </div>
  )
}

function Toasts() {
  const { toasts, dismissToast } = useApp()
  return (
    <div className="pointer-events-none fixed bottom-10 right-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.tone === 'err' ? 'alert' : 'status'}
          className={cx(
            'pointer-events-auto flex items-start gap-2 rounded-lg border bg-raised px-3 py-2 text-xs shadow-xl animate-fade-in',
            t.tone === 'err' ? 'border-err/40' : t.tone === 'ok' ? 'border-ok/40' : t.tone === 'warn' ? 'border-warn/40' : 'border-line',
          )}
        >
          <span className={cx('mt-px', t.tone === 'err' ? 'text-err' : t.tone === 'ok' ? 'text-ok' : t.tone === 'warn' ? 'text-warn' : 'text-info')}>
            {t.tone === 'ok' ? <IconCheck size={14} /> : t.tone === 'err' || t.tone === 'warn' ? <IconWarning size={14} /> : <IconInfo size={14} />}
          </span>
          <span className="min-w-0 flex-1 leading-relaxed text-fg">{t.text}</span>
          <button aria-label="Dismiss" onClick={() => dismissToast(t.id)} className="text-faint hover:text-fg"><IconX size={12} /></button>
        </div>
      ))}
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  )
}
