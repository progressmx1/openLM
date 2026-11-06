import { forwardRef, useState } from 'react'
import { api } from '../../api/client'
import type { Model, Session } from '../../api/types'
import { IconPlay, IconSidebar, IconSpark, IconWarning } from '../../components/icons'
import { Button, Callout, IconButton, Kbd, Spinner } from '../../components/ui'
import { WebToggle } from '../../components/WebToggle'
import { modelLabel, isCloudModel } from '../../lib/format'
import type { ToolItem, TranscriptState } from '../../lib/transcript'
import { useApp } from '../../state/app'
import { Composer, type ComposerHandle } from '../agent/Composer'
import { Transcript } from '../agent/Transcript'

const STARTERS = [
  'Explain how GPUs speed up LLM inference',
  'Help me write a polite follow-up email',
  'Compare SQLite and PostgreSQL for a small app',
  'What happened in AI this week?',
]

/** Plain chat: no workspace and no file or command tools; web lookups when Web is on. */
export const ChatView = forwardRef<ComposerHandle, {
  session: Session | null
  transcript: TranscriptState
  loading: boolean
  commands: Record<string, string>
  onNewChat: () => Promise<Session | null>
  onSessionChange: (s: Session) => void
  onToggleSidebar: () => void
  onOpenModels: () => void
}>(function ChatView({ session, transcript, loading, commands, onNewChat, onSessionChange, onToggleSidebar, onOpenModels }, ref) {
  const { models, toast, refreshModels } = useApp()
  const [loadingModel, setLoadingModel] = useState(false)
  const model = models.find((m) => m.id === session?.model_id) ?? null
  const chatCommands = Object.fromEntries(Object.entries(commands).filter(([c]) => ['/help', '/clear', '/compact', '/model'].includes(c)))

  const send = async (text: string): Promise<boolean> => {
    const target = session ?? (await onNewChat())
    if (!target) return false
    try {
      await api.send(target.id, text)
      return true
    } catch (e) {
      toast((e as Error).message, 'err')
      return false
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

  const onDecide = async (tool: ToolItem, approve: boolean, remember = false) => {
    if (session) await api.decide(session.id, tool.callId, approve, remember).catch((e) => toast((e as Error).message, 'err'))
  }

  return (
    <div className="flex h-full min-w-0 flex-col">
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-line px-4">
        <IconButton label="Toggle sidebar (Ctrl+B)" onClick={onToggleSidebar}>
          <IconSidebar size={15} />
        </IconButton>
        <div className="min-w-0 flex-1 truncate text-[13px] font-medium" title={session?.title}>{session?.title ?? 'New chat'}</div>
        <WebToggle />
        <select
          value={session?.model_id ?? ''}
          onChange={(e) => void setModel(e.target.value)}
          disabled={!session}
          aria-label="Model for this chat"
          className="h-7 max-w-[240px] truncate rounded-md border border-line bg-raised px-2 text-xs focus:outline-none"
        >
          <option value="" disabled>Choose a model…</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.loaded ? '● ' : '○ '}{modelLabel(m)}</option>
          ))}
        </select>
      </header>

      {model && !model.loaded && model.status?.state !== 'loading' && (
        <div className="mx-auto w-full max-w-3xl px-6 pt-3">
          <Callout tone={model.status?.state === 'crashed' ? 'err' : 'info'} icon={<IconWarning size={14} />}>
            <div className="flex items-center gap-3">
              <span className="flex-1">{model.status?.state === 'crashed' ? model.status.error : `${modelLabel(model)} is not loaded.`}</span>
              <Button size="sm" variant="primary" onClick={() => void quickLoad(model)} disabled={loadingModel}>
                {loadingModel ? <Spinner size={12} /> : <IconPlay size={12} />} Load
              </Button>
              <Button size="sm" variant="ghost" onClick={onOpenModels}>GPU options…</Button>
            </div>
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
            <div className="mt-[12vh] flex flex-col items-center gap-4 text-center">
              <IconSpark size={28} className="text-accent" />
              <div>
                <div className="text-base font-semibold">What's on your mind?</div>
                <div className="mt-1 text-xs text-muted">{model && isCloudModel(model) ? `Chat with ${modelLabel(model)}; messages go to the provider.` : 'Private chat with a model on your own GPU.'} Turn on Web to let it look things up.</div>
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
              <div className="text-2xs text-faint"><Kbd>Ctrl N</Kbd> new chat · <Kbd>Ctrl K</Kbd> commands</div>
            </div>
          }
        />
      )}

      <Composer
        ref={ref}
        disabled={false}
        running={transcript.running}
        commands={chatCommands}
        onSend={send}
        onStop={() => session && void api.interrupt(session.id)}
        placeholder={!model ? 'Pick a model in the header, then ask anything…' : 'Message…'}
      />
    </div>
  )
})
