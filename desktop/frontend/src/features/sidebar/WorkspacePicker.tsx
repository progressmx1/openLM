import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { RecentWorkspace } from '../../api/types'
import { FolderBrowser, cleanPath } from '../../components/FolderBrowser'
import { IconFolder } from '../../components/icons'
import { Button, Modal } from '../../components/ui'
import { relativeTime } from '../../lib/format'

export function WorkspacePicker({ open, onClose, onOpen }: { open: boolean; onClose: () => void; onOpen: (path: string) => Promise<void> }) {
  const [recent, setRecent] = useState<RecentWorkspace[]>([])
  const [path, setPath] = useState('')
  const [browseFrom, setBrowseFrom] = useState<string | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const native = typeof window.desktop?.pickFolder === 'function'

  useEffect(() => {
    if (!open) return
    setError(null)
    api.recentWorkspaces().then(setRecent).catch(() => setRecent([]))
  }, [open])

  const go = async (p: string) => {
    const cleaned = cleanPath(p)
    if (!cleaned) return
    setError(null)
    try {
      await onOpen(cleaned)
      onClose()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Open a workspace folder" width="max-w-2xl">
      <div className="flex flex-col gap-3 p-4">
        <p className="text-xs leading-relaxed text-muted">
          Pick the project folder below, or paste its path. The agent can read and edit files only inside the folder you open.
        </p>
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void go(path) }}>
          <input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder={'C:\\Users\\you\\projects\\my-app'}
            aria-label="Folder path"
            className="h-8 flex-1 rounded-md border border-line bg-raised px-2.5 font-mono text-xs placeholder:text-faint focus:border-accent/60 focus:outline-none"
          />
          <Button type="button" onClick={() => setBrowseFrom(cleanPath(path) || undefined)} title="Show the folders inside this path">Go</Button>
          {native && (
            <Button
              type="button"
              onClick={async () => {
                const picked = await window.desktop!.pickFolder!()
                if (picked) void go(picked)
              }}
            >
              Browse…
            </Button>
          )}
          <Button type="submit" variant="primary" disabled={!cleanPath(path)}>Open</Button>
        </form>
        {error && <div role="alert" className="rounded-md border border-err/30 bg-err/8 px-3 py-2 text-xs text-fg">{error}</div>}

        {open && (
          <FolderBrowser start={browseFrom} folderAction="Open this folder" onPickFolder={(p) => void go(p)} onPathChange={setPath} onError={setError} />
        )}

        {recent.length > 0 && (
          <div>
            <div className="mb-1 text-2xs font-semibold uppercase tracking-wide text-faint">Recent</div>
            <ul className="flex max-h-40 flex-col gap-0.5 overflow-y-auto">
              {recent.map((w) => (
                <li key={w.path}>
                  <button
                    disabled={!w.exists}
                    onClick={() => void go(w.path)}
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-raised disabled:opacity-40"
                  >
                    <IconFolder size={14} className="shrink-0 text-accent/80" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs">{w.name}</span>
                      <span className="block truncate text-2xs text-faint">{w.path}</span>
                    </span>
                    <span className="text-2xs text-faint">{w.exists ? relativeTime(w.last_used) : 'missing'}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Modal>
  )
}
