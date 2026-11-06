import { memo, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Markdown } from '../../components/Markdown'
import { IconBrain, IconChevron, IconInfo, IconWarning } from '../../components/icons'
import { Spinner, cx } from '../../components/ui'
import type { Item, ToolItem } from '../../lib/transcript'
import { ToolCard } from './ToolCard'

export function Transcript({ items, running, onDecide, empty }: {
  items: Item[]
  running: boolean
  onDecide: (tool: ToolItem, approve: boolean, remember?: boolean) => void
  empty: React.ReactNode
}) {
  const scroller = useRef<HTMLDivElement>(null)
  const [stick, setStick] = useState(true)

  // Follow the bottom while streaming unless the user scrolled up.
  useLayoutEffect(() => {
    const el = scroller.current
    if (el && stick) el.scrollTop = el.scrollHeight
  }, [items, stick])

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    const onScroll = () => setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 80)
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  const firstPending = items.find((i): i is ToolItem => i.kind === 'tool' && i.status === 'awaiting_approval')
  const last = items[items.length - 1]
  const thinking = running && (!last || last.kind === 'user' || (last.kind === 'tool' && last.status !== 'awaiting_approval') || last.kind === 'notice')

  return (
    <div ref={scroller} className="relative min-h-0 flex-1 overflow-y-auto" aria-live="polite">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 px-6 pb-10 pt-6">
        {items.length === 0 && empty}
        {items.map((item) => (
          <Row key={item.key} item={item} focused={firstPending?.key === item.key} onDecide={onDecide} />
        ))}
        {thinking && (
          <div className="flex items-center gap-2 pl-1 text-xs text-muted">
            <Spinner size={12} className="text-accent" /> Working…
          </div>
        )}
      </div>
      {!stick && (
        <button
          onClick={() => {
            setStick(true)
            scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
          }}
          className="sticky bottom-3 left-1/2 mx-auto block -translate-x-1/2 rounded-full border border-line bg-raised px-3 py-1 text-xs text-muted shadow-lg hover:text-fg"
        >
          Jump to latest
        </button>
      )}
    </div>
  )
}

const Row = memo(function Row({ item, focused, onDecide }: { item: Item; focused: boolean; onDecide: (tool: ToolItem, approve: boolean, remember?: boolean) => void }) {
  switch (item.kind) {
    case 'user':
      return (
        <div className="flex justify-end animate-fade-in">
          <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent/12 px-4 py-2.5 text-[13.5px] leading-relaxed text-fg">
            {item.text}
          </div>
        </div>
      )
    case 'assistant':
      return <AssistantBlock content={item.content} reasoning={item.reasoning} streaming={item.streaming} toolDraft={item.toolDraft} />
    case 'tool':
      return <ToolCard tool={item} focused={focused} onDecide={(approve, remember) => onDecide(item, approve, remember)} />
    case 'notice':
      return (
        <div className="flex items-start gap-2 rounded-md px-1 text-xs text-muted animate-fade-in">
          <IconInfo size={13} className="mt-0.5 shrink-0 text-info" />
          <Markdown content={item.text} className="text-xs" />
        </div>
      )
    case 'error':
      return (
        <div role="alert" className="flex items-start gap-2 rounded-md border border-err/30 bg-err/8 px-3 py-2 text-xs text-fg">
          <IconWarning size={14} className="mt-0.5 shrink-0 text-err" />
          <span>{item.text}</span>
        </div>
      )
    case 'turn_end':
      if (item.reason === 'completed') return <div className="h-1" aria-hidden="true" />
      return (
        <div className="flex items-center gap-2 text-2xs uppercase tracking-wide text-faint">
          <span className="h-px flex-1 bg-line" />
          {item.reason === 'interrupted' ? 'Stopped' : item.reason === 'step_budget' ? 'Step budget reached' : 'Ended with an error'}
          <span className="h-px flex-1 bg-line" />
        </div>
      )
  }
})

function AssistantBlock({ content, reasoning, streaming, toolDraft }: { content: string; reasoning: string; streaming: boolean; toolDraft?: { name: string; chars: number } }) {
  const [open, setOpen] = useState(false)
  const thinkingNow = streaming && !content && !!reasoning
  return (
    <div className="flex flex-col gap-1.5 animate-fade-in">
      {reasoning && (
        <div className="rounded-lg border border-line/70 bg-panel/60">
          <button
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-muted hover:text-fg"
          >
            <IconChevron size={12} className={cx('transition-transform', open && 'rotate-90')} />
            <IconBrain size={13} />
            {thinkingNow ? <span className="animate-pulse2">Thinking…</span> : <span>Reasoning</span>}
            <span className="ml-auto text-2xs text-faint">{reasoning.length.toLocaleString()} chars</span>
          </button>
          {open && (
            <div className="max-h-80 overflow-auto whitespace-pre-wrap border-t border-line/70 px-3 py-2 text-xs leading-relaxed text-muted">
              {reasoning}
            </div>
          )}
        </div>
      )}
      {content && (
        <div className={cx('text-[13.5px] text-fg', streaming && 'caret')}>
          <Markdown content={content} />
        </div>
      )}
      {streaming && toolDraft && (
        <div className="flex items-center gap-2 text-xs text-muted">
          <Spinner size={12} className="text-accent" />
          Preparing {toolDraft.name ? <code className="font-mono">{toolDraft.name}</code> : 'a tool call'} · {toolDraft.chars.toLocaleString()} chars
        </div>
      )}
    </div>
  )
}
