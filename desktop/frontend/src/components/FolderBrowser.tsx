import { useCallback, useEffect, useState } from 'react'
import { api, type BrowseListing } from '../api/client'
import { formatBytes } from '../lib/format'
import { IconChevron, IconFile, IconFolder, IconUndo } from './icons'
import { Button, Spinner } from './ui'

/** Explorer's "Copy as path" adds quotes; pasted text often has stray spaces. */
export const cleanPath = (p: string) => p.trim().replace(/^["']+|["']+$/g, '').trim()

/**
 * A file-explorer style browser served by the backend, so it works in a normal browser as
 * well as in Electron. Shows folders, plus files with `fileExt` when given.
 */
export function FolderBrowser({ start, fileExt, folderAction, onPickFolder, onPickFile, onPathChange, onError }: {
  start?: string
  fileExt?: string
  folderAction: string
  onPickFolder: (path: string) => void
  onPickFile?: (path: string) => void
  onPathChange?: (path: string) => void
  onError: (message: string | null) => void
}) {
  const [listing, setListing] = useState<BrowseListing | null>(null)
  const [busy, setBusy] = useState(false)

  const browse = useCallback(async (target?: string) => {
    setBusy(true)
    onError(null)
    try {
      const result = await api.browse(target ? cleanPath(target) : undefined, fileExt)
      setListing(result)
      if (result.path) onPathChange?.(result.path)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [fileExt, onError, onPathChange])

  useEffect(() => {
    void browse(start)
    // Only on mount / when the starting folder changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start])

  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <div className="flex items-center gap-2 border-b border-line bg-sunken px-2 py-1.5">
        <Button size="sm" variant="ghost" disabled={!listing?.path || busy} onClick={() => void browse(listing?.parent ?? undefined)} title="Up one folder">
          <IconUndo size={13} /> Up
        </Button>
        <span className="min-w-0 flex-1 truncate font-mono text-2xs text-muted" title={listing?.path ?? ''}>
          {listing?.path ?? 'This PC'}
        </span>
        {busy && <Spinner size={12} />}
        {listing?.path && (
          <Button size="sm" variant={onPickFile ? 'secondary' : 'primary'} onClick={() => onPickFolder(listing.path!)}>
            {folderAction}
          </Button>
        )}
      </div>
      <ul className="max-h-72 overflow-y-auto py-1" aria-label="Folders and files">
        {listing?.dirs.map((d) => (
          <li key={d.path}>
            <button
              onClick={() => void browse(d.path)}
              onDoubleClick={() => !onPickFile && onPickFolder(d.path)}
              className="flex w-full items-center gap-2 px-3 py-1 text-left text-xs hover:bg-raised"
              title="Open this folder in the browser"
            >
              <IconFolder size={14} className="shrink-0 text-accent/80" />
              <span className="min-w-0 flex-1 truncate">{d.name}</span>
              <IconChevron size={12} className="shrink-0 text-faint" />
            </button>
          </li>
        ))}
        {listing?.files.map((f) => (
          <li key={f.path}>
            <button
              onClick={() => onPickFile?.(f.path)}
              className="flex w-full items-center gap-2 px-3 py-1 text-left text-xs hover:bg-accent/10"
              title="Add this model"
            >
              <IconFile size={14} className="shrink-0 text-ok" />
              <span className="min-w-0 flex-1 truncate font-medium">{f.name}</span>
              <span className="shrink-0 text-2xs text-faint">{formatBytes(f.size)}</span>
              <span className="shrink-0 text-2xs text-accent">Add</span>
            </button>
          </li>
        ))}
        {listing && listing.dirs.length === 0 && listing.files.length === 0 && (
          <li className="px-3 py-2 text-xs text-faint">{fileExt ? `No subfolders or ${fileExt} files here.` : 'No subfolders.'}</li>
        )}
      </ul>
    </div>
  )
}
