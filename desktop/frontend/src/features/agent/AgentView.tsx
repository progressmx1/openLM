import { forwardRef, useState } from 'react'
import { api } from '../../api/client'
import type { Mode, Model, Session } from '../../api/types'
import { IconFolder, IconPanelRight, IconPlay, IconSidebar, IconSpark, IconWarning } from '../../components/icons'
import { Button, Callout, EmptyState, IconButton, Kbd, Segmented, Spinner } from '../../components/ui'
import { WebToggle } from '../../components/WebToggle'
import { basename, modelLabel } from '../../lib/format'
import type { ToolItem, TranscriptState } from '../../lib/transcript'
import { useApp } from '../../state/app'
import { Composer, type ComposerHandle } from './Composer'
import { Transcript } from './Transcript'

export const MODES: { value: Exclude<Mode, 'chat'>; label: string; title: string }[] = [
  { value: 'ask', label: 'Ask', title: 'Read-only: answers questions about the code' },
  { value: 'plan', label: 'Plan', title: 'Investigates and proposes a plan; no edits' },
  { value: 'agent', label: 'Agent', title: 'Edits files and runs commands after your approval' },
  { value: 'auto', label: 'Auto', title: 'Fully autonomous: edits and commands run without asking; only destructive commands ask' },
]

const STARTERS = [
  'Explain how this project is structured',
  'Find and fix the failing tests',
  'Add a README section that explains how to run it',
  '/init',
]

export const AgentView = forwardRef<ComposerHandle, {
  workspace: string | null
  session: Session | null
  transcript: TranscriptState
  loading: boolean
  commands: Record<string, string>
  panelOpen: boolean
  onTogglePanel: () => void
  onToggleSidebar: () => void
  onPickWorkspace: () => void
  onNewSession: () => Promise<Session | null>
  onSessionChange: (s: Session) => void
  onOpenModels: () => void
}>(function AgentView(props, ref) {
  const { workspace, session, transcript, loading, commands, panelOpen, onTogglePanel, onToggleSidebar, onPickWorkspace, onNewSession, onSessionChange, onOpenModels } = props
  const { models, toast, refreshModels } = useApp()
  const [loadingModel, setLoadingModel] = useState(false)
  const model = models.find((m) => m.id === session?.model_id) ?? null
  const loadedModels = models.filter((m) => m.loaded)

  const send = async (text: string): Promise<boolean> => {
    let target = session
    if (!target) target = await onNewSession()
    if (!target) return false
    try {
      await api.send(target.id, text)
      return true
    } catch (e) {
      toast((e as Error).message, 'err')
      return false
    }
  }

  const setMode = async (mode: Mode) => {
    if (!session) return
    try {
      onSessionChange(await api.updateSession(session.id, { mode }))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }

  const setModel = async (id: string) => {
    if (!session) return
    try {
      onSessionChange(await api.updateSession(session.id, { model_id: id }))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }

  const quickLoad = async (m: Model) => {
    setLoadingModel(true)
    try {
      const status = await api.loadModel(m.id, {})
      toast(status.placement_check?.message ?? 'Model loaded.', status.placement_check?.level === 'ok' ? 'ok' : 'warn')
    } catch (e) {
      toast((e as Error).message, 'err')
    } finally {
      setLoadingModel(false)
      void refreshModels()
    }
  }

  const pending = transcript.items.filter((i): i is ToolItem => i.kind === 'tool' && i.status === 'awaiting_approval')
  const onDecide = async (tool: ToolItem, approve: boolean, remember = false) => {
    if (!session) return
    try {
      await api.decide(session.id, tool.callId, approve, remember)
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }

  if (!workspace) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState icon={<IconFolder size={36} />} title="Open a project folder to start" action={<Button variant="primary" onClick={onPickWorkspace}>Open folder…</Button>}>
          The agent works inside one folder at a time. It can read, edit and run code there, and asks before doing anything risky.
        </EmptyState>
      </div>
    )
  }

  const blocked = !model ? 'Pick a model for this session first.' : !model.loaded ? 'Load the model to start.' : undefined
  return (
    <div className="flex h-full min-w-0 flex-col">
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-line px-4">
        <IconButton label="Toggle sidebar (Ctrl+B)" onClick={onToggleSidebar}>
          <IconSidebar size={15} />
        </IconButton>
        <div className="min-w-0 flex-1 truncate text-[13px] font-medium max-sm:hidden" title={session?.title}>
          {session?.title ?? 'New session'} <span className="text-faint">· {basename(workspace)}</span>
        </div>
        <WebToggle />
        <Segmented<Mode> label="Mode" size="sm" value={session?.mode ?? 'auto'} onChange={(m) => void setMode(m)} options={MODES} />
        <select
          value={session?.model_id ?? ''}
          onChange={(e) => void setModel(e.target.value)}
          disabled={!session}
          aria-label="Model for this session"
          className="h-7 max-w-[220px] truncate rounded-md border border-line bg-raised px-2 text-xs focus:outline-none"
        >
          <option value="" disabled>Choose a model…</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.loaded ? '● ' : '○ '}{modelLabel(m)}</option>
          ))}
        </select>
        <IconButton label={panelOpen ? 'Hide side panel (Ctrl+J)' : 'Show side panel (Ctrl+J)'} onClick={onTogglePanel}>
          <IconPanelRight size={15} />
        </IconButton>
      </header>

      {model && !model.loaded && model.status?.state !== 'loading' && (
        <div className="px-4 pt-3">
          <Callout tone={model.status?.state === 'crashed' ? 'err' : 'info'} icon={<IconWarning size={14} />}>
            <div className="flex items-center gap-3">
              <span className="flex-1">{model.status?.state === 'crashed' ? model.status.error : `${modelLabel(model)} is not loaded.`}</span>
              <Button size="sm" variant="primary" onClick={() => void quickLoad(model)} disabled={loadingModel}>
                {loadingModel ? <Spinner size={12} /> : <IconPlay size={12} />} Load with saved GPU choice
              </Button>
              <Button size="sm" variant="ghost" onClick={onOpenModels}>GPU options…</Button>
            </div>
          </Callout>
        </div>
      )}
      {!model && session && (
        <div className="px-4 pt-3">
          <Callout tone="info" icon={<IconWarning size={14} />}>
            {loadedModels.length ? 'Choose a model in the header to start.' : <>No model is loaded. <button className="text-accent underline" onClick={onOpenModels}>Open Models & GPUs</button> to load one.</>}
          </Callout>
        </div>
      )}

      {loading ? (
        <div className="flex flex-1 items-center justify-center"><Spinner /></div>
      ) : (
        <Transcript
          items={transcript.items}
          running={transcript.running}
          onDecide={onDecide}
          empty={
            <div className="mt-[10vh] flex flex-col items-center gap-4 text-center">
              <IconSpark size={28} className="text-accent" />
              <div>
                <div className="text-base font-semibold">What should we build in {basename(workspace)}?</div>
                <div className="mt-1 text-xs text-muted">Runs entirely on your GPU. Edits show as diffs; risky actions wait for you.</div>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    onClick={() => (ref && typeof ref !== 'function' ? ref.current?.setText(s) : undefined)}
                    className="rounded-full border border-line bg-raised px-3 py-1 text-xs text-muted hover:border-accent/50 hover:text-fg"
                  >
                    {s}
                  </button>
                ))}
              </div>
              <div className="text-2xs text-faint"><Kbd>Ctrl K</Kbd> commands · <Kbd>Ctrl 1–4</Kbd> modes · <Kbd>Ctrl J</Kbd> side panel</div>
            </div>
          }
        />
      )}

      {pending.length > 1 && <div className="mx-auto -mt-1 mb-1 text-2xs text-warn">{pending.length} actions are waiting for your approval</div>}
      <Composer
        ref={ref}
        disabled={false}
        running={transcript.running}
        commands={commands}
        onSend={send}
        onStop={() => session && void api.interrupt(session.id)}
        placeholder={blocked && session ? `${blocked} (/ commands still work)` : session?.mode === 'ask' ? 'Ask about the code…' : session?.mode === 'plan' ? 'Describe what you want planned…' : 'Describe a task, or type / for commands…'}
      />
    </div>
  )
})
