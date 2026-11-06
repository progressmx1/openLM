import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { Change, Todo, WorkspaceEntry } from '../../api/types'
import { CodeView } from '../../components/CodeView'
import { DiffView } from '../../components/DiffView'
import { IconCheck, IconChevron, IconFile, IconFolder, IconRefresh, IconUndo, IconX } from '../../components/icons'
import { Badge, Button, EmptyState, IconButton, Spinner, cx } from '../../components/ui'
import { formatBytes } from '../../lib/format'
import type { ToolItem } from '../../lib/transcript'
import { useApp } from '../../state/app'

export type PanelTab = 'plan' | 'changes' | 'files' | 'terminal'

export function RightPanel({ tab, setTab, sessionId, workspace, todos, terminal, changesVersion }: {
  tab: PanelTab
  setTab: (t: PanelTab) => void
  sessionId: string | null
  workspace: string | null
  todos: Todo[]
  terminal: ToolItem[]
  changesVersion: number
}) {
  const tabs: { id: PanelTab; label: string; count?: number }[] = [
    { id: 'plan', label: 'Plan', count: todos.length || undefined },
    { id: 'changes', label: 'Changes' },
    { id: 'files', label: 'Files' },
    { id: 'terminal', label: 'Terminal', count: terminal.length || undefined },
  ]
  return (
    <aside className="flex h-full min-w-0 flex-col border-l border-line bg-panel" aria-label="Session panels">
      <div role="tablist" className="flex h-10 shrink-0 items-center gap-0.5 border-b border-line px-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={cx('flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium', tab === t.id ? 'bg-raised text-fg' : 'text-muted hover:text-fg')}
          >
            {t.label}
            {t.count != null && <span className="rounded bg-sunken px-1 text-2xs text-faint">{t.count}</span>}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === 'plan' && <PlanPanel todos={todos} />}
        {tab === 'changes' && <ChangesPanel sessionId={sessionId} version={changesVersion} />}
        {tab === 'files' && <FilesPanel workspace={workspace} />}
        {tab === 'terminal' && <TerminalPanel tools={terminal} />}
      </div>
    </aside>
  )
}

function PlanPanel({ todos }: { todos: Todo[] }) {
  if (!todos.length) {
    return <EmptyState title="No plan yet">The agent writes its task list here with the todo tool as it works. Plan mode always produces one.</EmptyState>
  }
  const done = todos.filter((t) => t.status === 'completed').length
  return (
    <div className="p-3">
      <div className="mb-2 flex items-center justify-between text-2xs text-muted">
        <span>{done} of {todos.length} done</span>
      </div>
      <ol className="flex flex-col gap-1">
        {todos.map((t, i) => (
          <li key={i} className="flex items-start gap-2 rounded-md px-2 py-1.5 text-xs hover:bg-raised">
            <span
              className={cx(
                'mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border',
                t.status === 'completed' ? 'border-ok bg-ok text-bg' : t.status === 'in_progress' ? 'border-accent' : 'border-line',
              )}
            >
              {t.status === 'completed' && <IconCheck size={10} strokeWidth={3} />}
              {t.status === 'in_progress' && <span className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse2" />}
            </span>
            <span className={cx('leading-relaxed', t.status === 'completed' ? 'text-muted line-through' : 'text-fg')}>{t.content}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function ChangesPanel({ sessionId, version }: { sessionId: string | null; version: number }) {
  const { toast } = useApp()
  const [changes, setChanges] = useState<Change[] | null>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const load = useCallback(async () => {
    if (!sessionId) return setChanges([])
    try {
      setChanges(await api.changes(sessionId))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }, [sessionId, toast])
  useEffect(() => {
    void load()
  }, [load, version])

  if (!sessionId) return <EmptyState title="No session">Open a session to review the agent's edits.</EmptyState>
  if (changes === null) return <div className="flex justify-center p-6"><Spinner /></div>
  if (!changes.length) return <EmptyState title="No edits yet">Every file the agent creates or edits appears here with its diff. You can keep or revert each one.</EmptyState>

  const decide = async (c: Change, accept: boolean) => {
    try {
      await api.decideFile(sessionId, c.turn, c.path, accept)
      toast(accept ? `Kept ${c.path}` : `Reverted ${c.path}`, accept ? 'ok' : 'info')
      await load()
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }
  const undo = async () => {
    try {
      const { reverted } = await api.undo(sessionId)
      toast(reverted.length ? `Reverted ${reverted.join(', ')}` : 'Nothing to undo', 'info')
      await load()
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }
  const turns = [...new Set(changes.map((c) => c.turn))].sort((a, b) => b - a)
  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center gap-2">
        <span className="text-2xs text-muted">{changes.filter((c) => c.status !== 'rejected').length} file change(s)</span>
        <IconButton label="Refresh" className="ml-auto" onClick={() => void load()}><IconRefresh size={14} /></IconButton>
        <Button size="sm" onClick={() => void undo()} title="Revert every edit from the latest turn"><IconUndo size={13} /> Undo last turn</Button>
      </div>
      {turns.map((turn) => (
        <section key={turn} className="flex flex-col gap-1.5">
          <h3 className="text-2xs font-semibold uppercase tracking-wide text-faint">Turn {turn}</h3>
          {changes.filter((c) => c.turn === turn).map((c) => {
            const key = `${c.turn}:${c.path}`
            return (
              <div key={key} className={cx('overflow-hidden rounded-md border border-line bg-bg', c.status === 'rejected' && 'opacity-60')}>
                <div className="flex items-center gap-2 px-2 py-1.5">
                  <button className="flex min-w-0 flex-1 items-center gap-1.5 text-left" onClick={() => setOpen((o) => ({ ...o, [key]: !o[key] }))} aria-expanded={!!open[key]}>
                    <IconChevron size={12} className={cx('shrink-0 text-faint transition-transform', open[key] && 'rotate-90')} />
                    <span className="truncate font-mono text-xs">{c.path}</span>
                  </button>
                  <span className="font-mono text-2xs"><span className="text-ok">+{c.added}</span> <span className="text-err">−{c.removed}</span></span>
                  {c.created && <Badge tone="info">new</Badge>}
                  {c.status === 'applied' ? (
                    <>
                      <IconButton label={`Revert ${c.path}`} onClick={() => void decide(c, false)}><IconX size={13} /></IconButton>
                      <IconButton label={`Keep ${c.path}`} onClick={() => void decide(c, true)}><IconCheck size={13} /></IconButton>
                    </>
                  ) : (
                    <Badge tone={c.status === 'accepted' ? 'ok' : 'warn'}>{c.status === 'accepted' ? 'kept' : 'reverted'}</Badge>
                  )}
                </div>
                {open[key] && <div className="border-t border-line"><DiffView diff={c.diff} path={c.path} /></div>}
              </div>
            )
          })}
        </section>
      ))}
    </div>
  )
}

function FilesPanel({ workspace }: { workspace: string | null }) {
  const [viewing, setViewing] = useState<{ path: string; content: string; binary: boolean } | null>(null)
  const { toast } = useApp()
  if (!workspace) return <EmptyState title="No workspace">Open a folder to browse its files.</EmptyState>
  const open = async (path: string) => {
    try {
      setViewing(await api.file(path))
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }
  if (viewing) {
    const rel = viewing.path.slice(workspace.length).replace(/^[\\/]/, '')
    return (
      <div className="flex h-full flex-col">
        <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
          <button className="text-xs text-accent hover:underline" onClick={() => setViewing(null)}>← Files</button>
          <span className="truncate font-mono text-xs text-muted">{rel}</span>
        </div>
        <div className="min-h-0 flex-1 overflow-auto py-2">
          {viewing.binary ? <div className="p-3 text-xs text-muted">Binary file.</div> : <CodeView code={viewing.content} path={viewing.path} />}
        </div>
      </div>
    )
  }
  return (
    <div className="py-1">
      <TreeNode path={workspace} name={workspace} depth={0} initiallyOpen onOpenFile={open} />
    </div>
  )
}

function TreeNode({ path, name, depth, initiallyOpen, onOpenFile }: { path: string; name: string; depth: number; initiallyOpen?: boolean; onOpenFile: (p: string) => void }) {
  const [open, setOpen] = useState(!!initiallyOpen)
  const [entries, setEntries] = useState<WorkspaceEntry[] | null>(null)
  useEffect(() => {
    if (open && entries === null) api.tree(path).then((r) => setEntries(r.entries)).catch(() => setEntries([]))
  }, [open, entries, path])
  return (
    <div>
      {depth > 0 && (
        <button
          className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-xs hover:bg-raised"
          style={{ paddingLeft: depth * 12 }}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          <IconChevron size={11} className={cx('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
          <IconFolder size={13} className="shrink-0 text-accent/80" />
          <span className="truncate">{name}</span>
        </button>
      )}
      {open && entries === null && <div className="py-1 text-2xs text-faint" style={{ paddingLeft: depth * 12 + 20 }}>Loading…</div>}
      {open && entries?.map((e) =>
        e.dir ? (
          <TreeNode key={e.path} path={e.path} name={e.name} depth={depth + 1} onOpenFile={onOpenFile} />
        ) : (
          <button
            key={e.path}
            onClick={() => onOpenFile(e.path)}
            className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-xs hover:bg-raised"
            style={{ paddingLeft: (depth + 1) * 12 + 11 }}
          >
            <IconFile size={13} className="shrink-0 text-faint" />
            <span className="truncate">{e.name}</span>
            <span className="ml-auto text-2xs text-faint">{formatBytes(e.size)}</span>
          </button>
        ),
      )}
      {open && entries?.length === 0 && depth === 0 && <EmptyState title="Empty folder">The agent can create files here.</EmptyState>}
    </div>
  )
}

function TerminalPanel({ tools }: { tools: ToolItem[] }) {
  if (!tools.length) return <EmptyState title="No commands yet">Commands the agent runs (tests, builds, scripts) show up here with their full output.</EmptyState>
  return (
    <div className="flex flex-col gap-2 p-2 font-mono text-[12px]">
      {tools.map((t) => {
        const code = t.data?.exit_code as number | undefined
        const out = t.status === 'running' ? t.live : t.output ?? ''
        return (
          <div key={t.key} className="overflow-hidden rounded-md border border-line bg-sunken">
            <div className="flex items-center gap-2 border-b border-line/70 px-2 py-1">
              <span className="text-accent">PS&gt;</span>
              <span className="min-w-0 flex-1 truncate text-fg">{t.command ?? String(t.args.command ?? '')}</span>
              {t.status === 'running' ? <Spinner size={11} className="text-accent" /> : code != null ? (
                <span className={code === 0 ? 'text-ok' : 'text-err'}>exit {code}</span>
              ) : (
                <span className="text-warn">{t.status.replace('_', ' ')}</span>
              )}
            </div>
            {out && <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words px-2 py-1.5 text-fg/85">{out}</pre>}
          </div>
        )
      })}
    </div>
  )
}
