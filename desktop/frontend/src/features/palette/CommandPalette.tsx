import { useEffect, useMemo, useRef, useState } from 'react'
import { IconSearch } from '../../components/icons'
import { Kbd, cx } from '../../components/ui'

export interface PaletteAction {
  id: string
  label: string
  group: string
  hint?: string
  run: () => void
}

function score(label: string, query: string): number {
  if (!query) return 1
  const l = label.toLowerCase()
  const q = query.toLowerCase()
  if (l.includes(q)) return 100 - l.indexOf(q)
  // Subsequence match ("nwses" → "New session").
  let i = 0
  for (const ch of l) if (ch === q[i]) i++
  return i === q.length ? 10 : 0
}

export function CommandPalette({ open, onClose, actions }: { open: boolean; onClose: () => void; actions: PaletteAction[] }) {
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setActive(0)
      setTimeout(() => input.current?.focus(), 0)
    }
  }, [open])

  const results = useMemo(
    () => actions.map((a) => ({ a, s: score(`${a.group} ${a.label}`, query) })).filter((x) => x.s > 0).sort((x, y) => y.s - x.s).map((x) => x.a),
    [actions, query],
  )
  useEffect(() => setActive(0), [query])
  useEffect(() => {
    list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [active])

  if (!open) return null
  const run = (a: PaletteAction) => {
    onClose()
    a.run()
  }
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 pt-[14vh] backdrop-blur-[2px] animate-fade-in" onMouseDown={onClose}>
      <div role="dialog" aria-label="Command palette" className="w-full max-w-xl overflow-hidden rounded-xl border border-line bg-panel shadow-2xl" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-line px-3">
          <IconSearch size={15} className="text-faint" />
          <input
            ref={input}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setActive((i) => Math.min(i + 1, results.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setActive((i) => Math.max(i - 1, 0))
              } else if (e.key === 'Enter' && results[active]) {
                e.preventDefault()
                run(results[active])
              } else if (e.key === 'Escape') {
                e.preventDefault()
                onClose()
              }
            }}
            placeholder="Type a command…"
            aria-label="Search commands"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            className="h-11 flex-1 bg-transparent text-sm placeholder:text-faint focus:outline-none"
          />
          <Kbd>Esc</Kbd>
        </div>
        <ul id="palette-list" ref={list} role="listbox" className="max-h-80 overflow-y-auto p-1.5">
          {results.map((a, i) => (
            <li
              key={a.id}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onClick={() => run(a)}
              className={cx('flex cursor-pointer items-center gap-3 rounded-md px-2.5 py-1.5 text-[13px]', i === active ? 'bg-accent/12 text-fg' : 'text-fg/90')}
            >
              <span className="w-20 shrink-0 text-2xs uppercase tracking-wide text-faint">{a.group}</span>
              <span className="flex-1 truncate">{a.label}</span>
              {a.hint && <Kbd>{a.hint}</Kbd>}
            </li>
          ))}
          {!results.length && <li className="px-3 py-6 text-center text-xs text-muted">No matching command.</li>}
        </ul>
      </div>
    </div>
  )
}
