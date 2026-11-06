import { useEffect, useRef, useState } from 'react'
import { DiffView } from '../../components/DiffView'
import {
  IconCheck, IconChevron, IconEdit, IconFile, IconGit, IconGlobe, IconList, IconSearch, IconTerminal, IconWarning, IconX,
} from '../../components/icons'
import { Badge, Button, Kbd, Spinner, cx } from '../../components/ui'
import type { ToolItem } from '../../lib/transcript'

const ICONS: Record<string, typeof IconFile> = {
  read_file: IconFile, list_dir: IconList, glob: IconSearch, grep: IconSearch, write_file: IconEdit, edit_file: IconEdit,
  run_command: IconTerminal, git_status: IconGit, git_diff: IconGit, git_log: IconGit, git_commit: IconGit,
  web_search: IconGlobe, web_fetch: IconGlobe, todo_write: IconList,
}

function title(t: ToolItem): { verb: string; target: string } {
  const a = t.args as Record<string, string>
  switch (t.name) {
    case 'read_file': return { verb: 'Read', target: a.path }
    case 'list_dir': return { verb: 'List', target: a.path || '.' }
    case 'glob': return { verb: 'Find files', target: a.pattern }
    case 'grep': return { verb: 'Search', target: `${a.pattern}${a.glob ? `  in ${a.glob}` : ''}` }
    case 'write_file': return { verb: t.created === false ? 'Rewrite' : 'Create', target: t.path || a.path }
    case 'edit_file': return { verb: 'Edit', target: t.path || a.path }
    case 'run_command': return { verb: 'Run', target: a.command }
    case 'git_commit': return { verb: 'Commit', target: a.message }
    case 'todo_write': return { verb: 'Update plan', target: '' }
    case 'web_search': return { verb: 'Web search', target: a.query }
    case 'web_fetch': return { verb: 'Fetch', target: a.url }
    default: return { verb: t.name.replace(/_/g, ' '), target: '' }
  }
}

const STATUS: Record<string, { tone: 'ok' | 'err' | 'warn' | 'neutral' | 'accent'; label: string }> = {
  success: { tone: 'ok', label: 'done' },
  error: { tone: 'err', label: 'failed' },
  denied: { tone: 'warn', label: 'denied' },
  cancelled: { tone: 'warn', label: 'stopped' },
  timeout: { tone: 'err', label: 'timed out' },
  running: { tone: 'accent', label: 'running' },
  awaiting_approval: { tone: 'warn', label: 'needs approval' },
}

export function ToolCard({ tool, onDecide, focused }: { tool: ToolItem; onDecide: (approve: boolean, remember?: boolean) => void; focused: boolean }) {
  const Icon = ICONS[tool.name] ?? IconFile
  const { verb, target } = title(tool)
  const waiting = tool.status === 'awaiting_approval'
  const isEdit = tool.toolKind === 'edit'
  const isCommand = tool.name === 'run_command'
  const [open, setOpen] = useState<boolean>(waiting || isEdit || isCommand)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    if (waiting) setOpen(true)
  }, [waiting])
  const status = STATUS[tool.status] ?? { tone: 'neutral' as const, label: tool.status }
  const exitCode = tool.data?.exit_code as number | undefined
  const output = tool.status === 'running' ? tool.live : tool.output ?? ''
  const showOutput = !isEdit && !!output && tool.name !== 'todo_write'

  const decide = async (approve: boolean, remember = false) => {
    setBusy(true)
    try {
      await onDecide(approve, remember)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className={cx(
        'animate-fade-in overflow-hidden rounded-lg border bg-panel',
        waiting ? (tool.risk === 'destructive' ? 'border-err/50 ring-1 ring-err/25' : 'border-warn/50 ring-1 ring-warn/20') : 'border-line',
      )}
      data-tool={tool.name}
    >
      <button
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-raised/60"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <IconChevron size={12} className={cx('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
        <Icon size={14} className="shrink-0 text-muted" />
        <span className="shrink-0 text-xs font-medium text-fg">{verb}</span>
        <span className={cx('min-w-0 truncate text-xs text-muted', (isCommand || tool.path) && 'font-mono')}>{target}</span>
        <span className="ml-auto flex shrink-0 items-center gap-1.5">
          {isEdit && tool.added != null && (
            <span className="font-mono text-2xs">
              <span className="text-ok">+{tool.added}</span> <span className="text-err">−{tool.removed}</span>
            </span>
          )}
          {exitCode != null && <span className="font-mono text-2xs text-faint">exit {exitCode}</span>}
          {tool.status === 'running' ? <Spinner size={12} className="text-accent" /> : <Badge tone={status.tone}>{status.label}</Badge>}
        </span>
      </button>

      {open && (
        <div className="border-t border-line">
          {tool.diff && <DiffView diff={tool.diff} path={tool.path} maxHeight={waiting ? 460 : 320} />}
          {isCommand && (
            <div className="border-b border-line/70 bg-sunken px-3 py-1.5 font-mono text-[12px] text-fg">
              <span className="select-none text-accent">PS&gt; </span>
              {tool.command ?? String(tool.args.command ?? '')}
            </div>
          )}
          {showOutput && <OutputBlock text={output} live={tool.status === 'running'} />}
          {!showOutput && !tool.diff && tool.output && (
            <div className="px-3 py-2 text-xs text-muted">{tool.output}</div>
          )}
          {isEdit && tool.output && tool.status !== 'success' && (
            <div className="border-t border-line px-3 py-2 text-xs text-err">{tool.output}</div>
          )}
        </div>
      )}

      {waiting && (
        <div className={cx('flex flex-wrap items-center gap-2 border-t px-3 py-2', tool.risk === 'destructive' ? 'border-err/30 bg-err/5' : 'border-warn/30 bg-warn/5')}>
          {tool.risk === 'destructive' ? (
            <span className="flex items-center gap-1.5 text-xs text-err">
              <IconWarning size={14} /> Potentially destructive command. Check it carefully.
            </span>
          ) : (
            <span className="text-xs text-muted">{isEdit ? 'Apply this change?' : 'Run this?'}</span>
          )}
          <div className="ml-auto flex flex-wrap items-center justify-end gap-1.5 [&>button]:whitespace-nowrap">
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => decide(false)}>
              <IconX size={13} /> Deny {focused && <Kbd>Esc</Kbd>}
            </Button>
            {tool.prefix && tool.risk !== 'destructive' && (
              <Button size="sm" variant="secondary" disabled={busy} onClick={() => decide(true, true)} title="Approve and stop asking for commands that start with this">
                Always allow <code className="max-w-[160px] truncate font-mono text-2xs">{tool.prefix}</code>
              </Button>
            )}
            <Button size="sm" variant={tool.risk === 'destructive' ? 'danger' : 'primary'} disabled={busy} onClick={() => decide(true)} autoFocus={focused}>
              <IconCheck size={13} /> {isEdit ? 'Apply' : 'Approve'} {focused && <Kbd>Enter</Kbd>}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function OutputBlock({ text, live }: { text: string; live: boolean }) {
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (live && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [text, live])
  const trimmed = text.length > 12000 ? `…\n${text.slice(-12000)}` : text
  return (
    <pre ref={ref} className="max-h-72 overflow-auto whitespace-pre-wrap break-words bg-sunken px-3 py-2 font-mono text-[12px] leading-[1.5] text-fg/90">
      {trimmed}
      {live && <span className="caret" />}
    </pre>
  )
}
