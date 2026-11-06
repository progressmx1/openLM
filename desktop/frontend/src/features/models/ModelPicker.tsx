import { useState } from 'react'
import { api } from '../../api/client'
import { FolderBrowser, cleanPath } from '../../components/FolderBrowser'
import { Button, Modal } from '../../components/ui'

/**
 * Add models by browsing the disk: click a .gguf file to register it, or scan the folder
 * you are in for every .gguf inside it. Paths can also be pasted (quotes are fine).
 */
export function ModelPicker({ open, onClose, onAdded }: { open: boolean; onClose: () => void; onAdded: (message: string) => void }) {
  const [path, setPath] = useState('')
  const [browseFrom, setBrowseFrom] = useState<string | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<string>) => {
    setBusy(true)
    setError(null)
    try {
      onAdded(await action())
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const addFile = (file: string) =>
    run(async () => {
      const model = await api.register(cleanPath(file))
      return `Added ${model.display_name || model.name}.`
    })
  const scanFolder = (folder: string) =>
    run(async () => {
      const r = await api.discover([cleanPath(folder)])
      return r.found ? `Found ${r.found} model file(s) in ${cleanPath(folder)}, ${r.added} new.` : `No .gguf files in ${cleanPath(folder)}.`
    })
  const submit = () => {
    const p = cleanPath(path)
    if (!p) return
    if (p.toLowerCase().endsWith('.gguf')) void addFile(p)
    else void scanFolder(p)
  }

  return (
    <Modal open={open} onClose={onClose} title="Add models" width="max-w-2xl">
      <div className="flex flex-col gap-3 p-4">
        <p className="text-xs leading-relaxed text-muted">
          Browse to a <code>.gguf</code> file and click it to add it, or open a folder and scan it for every model inside (including subfolders).
          Model files are only read, never moved or changed.
        </p>
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); submit() }}>
          <input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder={'C:\\models  or  C:\\models\\my-model.gguf'}
            aria-label="Model file or folder path"
            className="h-8 flex-1 rounded-md border border-line bg-raised px-2.5 font-mono text-xs placeholder:text-faint focus:border-accent/60 focus:outline-none"
          />
          <Button type="button" onClick={() => setBrowseFrom(cleanPath(path) || undefined)} disabled={cleanPath(path).toLowerCase().endsWith('.gguf')}>Go</Button>
          <Button type="submit" variant="primary" disabled={!cleanPath(path) || busy}>
            {cleanPath(path).toLowerCase().endsWith('.gguf') ? 'Add file' : 'Scan folder'}
          </Button>
        </form>
        {error && <div role="alert" className="rounded-md border border-err/30 bg-err/8 px-3 py-2 text-xs text-fg">{error}</div>}
        {open && (
          <FolderBrowser
            start={browseFrom}
            fileExt="gguf"
            folderAction="Scan this folder"
            onPickFolder={(p) => void scanFolder(p)}
            onPickFile={(p) => void addFile(p)}
            onPathChange={setPath}
            onError={setError}
          />
        )}
      </div>
    </Modal>
  )
}
