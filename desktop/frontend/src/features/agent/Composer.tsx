import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import { IconSend, IconStop } from '../../components/icons'
import { Kbd, cx } from '../../components/ui'

export interface ComposerHandle {
  focus: () => void
  setText: (text: string) => void
}

export const Composer = forwardRef<ComposerHandle, {
  disabled: boolean
  disabledReason?: string
  running: boolean
  commands: Record<string, string>
  onSend: (text: string) => Promise<boolean>
  onStop: () => void
  placeholder: string
}>(function Composer({ disabled, disabledReason, running, commands, onSend, onStop, placeholder }, ref) {
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [active, setActive] = useState(0)
  const area = useRef<HTMLTextAreaElement>(null)
  const history = useRef<string[]>([])
  const historyIndex = useRef(-1)

  useImperativeHandle(ref, () => ({ focus: () => area.current?.focus(), setText: (t) => setText(t) }))

  useEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 260)}px`
  }, [text])

  const suggestions = useMemo(() => {
    if (!text.startsWith('/') || text.includes(' ') || text.includes('\n')) return []
    return Object.entries(commands).filter(([name]) => name.startsWith(text.toLowerCase()))
  }, [text, commands])

  useEffect(() => setActive(0), [suggestions.length])

  const submit = async () => {
    const value = text.trim()
    if (!value || disabled || sending) return
    setSending(true)
    const ok = await onSend(value)
    setSending(false)
    if (ok) {
      history.current = [value, ...history.current.filter((h) => h !== value)].slice(0, 50)
      historyIndex.current = -1
      setText('')
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (suggestions.length) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault()
        setActive((i) => (i + (e.key === 'ArrowDown' ? 1 : suggestions.length - 1)) % suggestions.length)
        return
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey && suggestions[active][0] !== text)) {
        e.preventDefault()
        const name = suggestions[active][0]
        setText(['/model', '/mode'].includes(name) ? `${name} ` : name)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      void submit()
    } else if (e.key === 'ArrowUp' && !text && history.current.length) {
      e.preventDefault()
      historyIndex.current = Math.min(historyIndex.current + 1, history.current.length - 1)
      setText(history.current[historyIndex.current])
    } else if (e.key === 'Escape' && running) {
      e.preventDefault()
      onStop()
    }
  }

  return (
    <div className="relative mx-auto w-full max-w-3xl px-6 pb-4">
      {suggestions.length > 0 && (
        <ul role="listbox" aria-label="Commands" className="absolute bottom-full left-6 right-6 mb-2 overflow-hidden rounded-lg border border-line bg-raised shadow-xl">
          {suggestions.map(([name, help], i) => (
            <li
              key={name}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => {
                e.preventDefault()
                setText(['/model', '/mode'].includes(name) ? `${name} ` : name)
                area.current?.focus()
              }}
              className={cx('flex cursor-pointer items-baseline gap-3 px-3 py-1.5 text-xs', i === active ? 'bg-accent/12' : 'hover:bg-sunken')}
            >
              <span className="font-mono text-fg">{name}</span>
              <span className="truncate text-muted">{help}</span>
            </li>
          ))}
        </ul>
      )}
      <div className={cx('rounded-xl border bg-raised shadow-sm transition-colors focus-within:border-accent/60', disabled ? 'border-line opacity-70' : 'border-line')}>
        <textarea
          ref={area}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          disabled={disabled}
          aria-label="Message the agent"
          placeholder={disabled ? disabledReason : placeholder}
          className="block max-h-[260px] w-full resize-none bg-transparent px-4 pt-3 text-[13.5px] leading-relaxed text-fg placeholder:text-faint focus:outline-none"
        />
        <div className="flex items-center gap-2 px-3 pb-2 pt-1">
          <span className="text-2xs text-faint">
            <Kbd>Enter</Kbd> send · <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> newline · <Kbd>/</Kbd> commands
          </span>
          <div className="ml-auto">
            {running ? (
              <button onClick={onStop} aria-label="Stop the agent" title="Stop (Esc)" className="flex h-8 items-center gap-1.5 rounded-lg bg-err/15 px-3 text-xs font-medium text-err hover:bg-err/25">
                <IconStop size={14} /> Stop
              </button>
            ) : (
              <button
                onClick={() => void submit()}
                disabled={disabled || !text.trim() || sending}
                aria-label="Send"
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-fg transition-opacity hover:opacity-90 disabled:opacity-30"
              >
                <IconSend size={15} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})
