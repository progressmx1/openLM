import type { Session, Surface } from '../../api/types'
import { IconChevronDown, IconFolder, IconGpu, IconPlus, IconSettings, IconTrash } from '../../components/icons'
import { Kbd, Segmented, Spinner, cx } from '../../components/ui'
import { basename, relativeTime } from '../../lib/format'

export type View = 'agent' | 'models' | 'settings'

export function Sidebar({ surface, onSurface, workspace, sessions, activeId, view, onPickWorkspace, onNewSession, onSelect, onDelete, onView }: {
  surface: Surface
  onSurface: (s: Surface) => void
  workspace: string | null
  sessions: Session[]
  activeId: string | null
  view: View
  onPickWorkspace: () => void
  onNewSession: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onView: (v: View) => void
}) {
  return (
    <nav className="flex h-full flex-col border-r border-line bg-panel" aria-label="Sessions">
      <div className="flex h-10 shrink-0 items-center gap-2 border-b border-line px-3">
        <div className="flex h-5 w-5 items-center justify-center rounded-md bg-accent text-[10px] font-bold text-accent-fg">LA</div>
        <span className="text-[13px] font-semibold">Local Agent</span>
      </div>

      <div className="mx-2 mt-2">
        <Segmented<Surface>
          label="Chat or code"
          value={surface}
          onChange={onSurface}
          options={[
            { value: 'chat', label: 'Chat', title: 'Just talk — no project folder needed' },
            { value: 'code', label: 'Code', title: 'Coding agent working inside a project folder' },
          ]}
        />
      </div>

      {surface === 'code' && (
        <button
          onClick={onPickWorkspace}
          className="mx-2 mt-2 flex items-center gap-2 rounded-md border border-line bg-raised px-2.5 py-2 text-left hover:border-accent/50"
          title={workspace ?? 'Open a folder'}
        >
          <IconFolder size={15} className="shrink-0 text-accent" />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium">{workspace ? basename(workspace) : 'Open a folder…'}</span>
            {workspace && <span className="block truncate text-2xs text-faint">{workspace}</span>}
          </span>
          <IconChevronDown size={13} className="shrink-0 text-faint" />
        </button>
      )}

      <button
        onClick={onNewSession}
        disabled={surface === 'code' && !workspace}
        className="mx-2 mt-2 flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-muted hover:bg-raised hover:text-fg disabled:opacity-40"
      >
        <IconPlus size={14} /> {surface === 'chat' ? 'New chat' : 'New session'} <span className="ml-auto"><Kbd>Ctrl N</Kbd></span>
      </button>

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {surface === 'code' && workspace && sessions.length === 0 && <div className="px-2.5 py-3 text-2xs text-faint">No sessions in this folder yet.</div>}
        {surface === 'chat' && sessions.length === 0 && <div className="px-2.5 py-3 text-2xs text-faint">No chats yet.</div>}
        <ul className="flex flex-col gap-0.5">
          {sessions.map((s) => (
            <li key={s.id} className="group relative">
              <button
                onClick={() => onSelect(s.id)}
                aria-current={s.id === activeId && view === 'agent' ? 'page' : undefined}
                className={cx(
                  'flex w-full flex-col rounded-md px-2.5 py-1.5 text-left',
                  s.id === activeId && view === 'agent' ? 'bg-raised text-fg' : 'text-muted hover:bg-raised/60 hover:text-fg',
                )}
              >
                <span className="flex items-center gap-1.5">
                  {s.state !== 'idle' && (s.state === 'awaiting_approval' ? <span className="h-1.5 w-1.5 rounded-full bg-warn" title="Waiting for approval" /> : <Spinner size={10} className="text-accent" />)}
                  <span className="truncate pr-5 text-xs">{s.title}</span>
                </span>
                <span className="text-2xs text-faint">{s.mode === 'chat' ? '' : `${s.mode} · `}{relativeTime(s.updated_at)}</span>
              </button>
              <button
                aria-label={`Delete session ${s.title}`}
                onClick={() => onDelete(s.id)}
                className="absolute right-1.5 top-1.5 hidden rounded p-1 text-faint hover:bg-sunken hover:text-err group-hover:block focus:block"
              >
                <IconTrash size={12} />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex flex-col gap-0.5 border-t border-line p-2">
        <NavButton active={view === 'models'} onClick={() => onView('models')} icon={<IconGpu size={15} />} label="Models & GPUs" />
        <NavButton active={view === 'settings'} onClick={() => onView('settings')} icon={<IconSettings size={15} />} label="Settings" />
      </div>
    </nav>
  )
}

function NavButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={cx('flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs', active ? 'bg-raised text-fg' : 'text-muted hover:bg-raised/60 hover:text-fg')}
    >
      {icon} {label}
    </button>
  )
}
