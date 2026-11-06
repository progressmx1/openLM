import type { AgentEvent, Todo } from '../api/types'

export type ToolStatus = 'awaiting_approval' | 'running' | 'success' | 'error' | 'denied' | 'cancelled' | 'timeout'

export interface ToolItem {
  kind: 'tool'
  key: string
  callId: string
  turn: number
  name: string
  args: Record<string, unknown>
  toolKind: string
  risk?: string
  status: ToolStatus
  diff?: string
  path?: string
  added?: number
  removed?: number
  created?: boolean
  command?: string
  prefix?: string | null
  output?: string
  data?: Record<string, unknown>
  live: string
}

export type Item =
  | { kind: 'user'; key: string; turn: number; text: string }
  | { kind: 'assistant'; key: string; turn: number; content: string; reasoning: string; streaming: boolean; toolDraft?: { name: string; chars: number } }
  | ToolItem
  | { kind: 'notice'; key: string; turn: number; text: string }
  | { kind: 'error'; key: string; turn: number; text: string }
  | { kind: 'turn_end'; key: string; turn: number; reason: string }

export interface Usage {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  context_used?: number | null
  context_max?: number | null
  tokens_per_second?: number | null
  prompt_tokens_per_second?: number | null
}

export interface TranscriptState {
  items: Item[]
  lastSeq: number
  running: boolean
  todos: Todo[]
  usage: Usage | null
  turn: number
}

export const emptyTranscript = (): TranscriptState => ({ items: [], lastSeq: 0, running: false, todos: [], usage: null, turn: 0 })

/** Pure reducer: fold one event into the transcript (used for history replay and live events). */
export function applyEvent(state: TranscriptState, e: AgentEvent): TranscriptState {
  if (e.seq <= state.lastSeq) return state
  const items = state.items.slice()
  const next: TranscriptState = { ...state, items, lastSeq: e.seq, turn: Math.max(state.turn, e.turn) }
  const key = `e${e.seq}`
  const last = items[items.length - 1]
  const str = (v: unknown) => (typeof v === 'string' ? v : '')

  switch (e.type) {
    case 'turn_start':
      next.running = true
      break
    case 'user':
      items.push({ kind: 'user', key, turn: e.turn, text: str(e.text) })
      break
    case 'delta': {
      const channel = str(e.channel)
      let target = last && last.kind === 'assistant' && last.streaming ? last : null
      if (!target) {
        target = { kind: 'assistant', key, turn: e.turn, content: '', reasoning: '', streaming: true }
        items.push(target)
      } else {
        target = { ...target }
        items[items.length - 1] = target
      }
      if (channel === 'content') target.content += str(e.text)
      else if (channel === 'reasoning') target.reasoning += str(e.text)
      else if (channel === 'tool') target.toolDraft = { name: str(e.name), chars: Number(e.chars) || 0 }
      break
    }
    case 'assistant': {
      const final = { kind: 'assistant' as const, key, turn: e.turn, content: str(e.content), reasoning: str(e.reasoning), streaming: false }
      if (last && last.kind === 'assistant' && last.streaming) items[items.length - 1] = { ...final, key: last.key }
      else items.push(final)
      break
    }
    case 'tool': {
      // A streaming block that only held a tool-call draft disappears once the card arrives.
      if (last && last.kind === 'assistant' && last.streaming) {
        if (!last.content.trim() && !last.reasoning.trim()) items.pop()
        else items[items.length - 1] = { ...last, streaming: false, toolDraft: undefined }
      }
      items.push({
        kind: 'tool',
        key,
        callId: str(e.call_id),
        turn: e.turn,
        name: str(e.name),
        args: (e.args as Record<string, unknown>) ?? {},
        toolKind: str(e.kind),
        risk: str(e.risk) || undefined,
        status: (str(e.status) || 'running') as ToolStatus,
        diff: str(e.diff) || undefined,
        path: str(e.path) || undefined,
        added: e.added as number | undefined,
        removed: e.removed as number | undefined,
        created: e.created as boolean | undefined,
        command: str(e.command) || undefined,
        prefix: (e.prefix as string | null | undefined) ?? null,
        live: '',
      })
      break
    }
    case 'tool_update':
    case 'tool_output': {
      const index = findTool(items, str(e.call_id))
      if (index < 0) break
      const tool = { ...(items[index] as ToolItem) }
      if (e.type === 'tool_output') tool.live = (tool.live + str(e.text)).slice(-20000)
      else {
        tool.status = str(e.status) as ToolStatus
        if (e.output !== undefined) tool.output = str(e.output)
        if (e.data) tool.data = e.data as Record<string, unknown>
        const diff = (e.data as Record<string, unknown> | undefined)?.diff
        if (typeof diff === 'string' && diff) tool.diff = diff
      }
      items[index] = tool
      break
    }
    case 'todos':
      next.todos = (e.todos as Todo[]) ?? []
      break
    case 'usage':
      next.usage = e as unknown as Usage
      break
    case 'notice':
      items.push({ kind: 'notice', key, turn: e.turn, text: str(e.text) })
      break
    case 'error':
      items.push({ kind: 'error', key, turn: e.turn, text: str(e.message) })
      break
    case 'turn_end':
      next.running = false
      if (last && last.kind === 'assistant' && last.streaming) items[items.length - 1] = { ...last, streaming: false }
      items.push({ kind: 'turn_end', key, turn: e.turn, reason: str(e.reason) })
      break
  }
  return next
}

function findTool(items: Item[], callId: string): number {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i]
    if (item.kind === 'tool' && item.callId === callId) return i
  }
  return -1
}

export function pendingApprovals(state: TranscriptState): ToolItem[] {
  return state.items.filter((i): i is ToolItem => i.kind === 'tool' && i.status === 'awaiting_approval')
}
