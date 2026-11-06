import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api, subscribeSession } from '../api/client'
import type { AgentEvent, Session } from '../api/types'
import { applyEvent, emptyTranscript, type TranscriptState } from '../lib/transcript'

type Action = { type: 'reset' } | { type: 'events'; events: AgentEvent[] }

function reducer(state: TranscriptState, action: Action): TranscriptState {
  if (action.type === 'reset') return emptyTranscript()
  let next = state
  for (const e of action.events) next = applyEvent(next, e)
  return next
}

/** Loads a session's history, then follows its live event stream. */
export function useSessionStream(sessionId: string | null, onEvent?: (e: AgentEvent) => void) {
  const [transcript, dispatch] = useReducer(reducer, undefined, emptyTranscript)
  const [loading, setLoading] = useState(false)
  const [live, setLive] = useState(false)
  const handler = useRef(onEvent)
  handler.current = onEvent

  useEffect(() => {
    dispatch({ type: 'reset' })
    if (!sessionId) return
    let cancelled = false
    let unsubscribe: (() => void) | null = null
    let queue: AgentEvent[] = []
    let frame = 0
    // Batch bursts of streamed tokens into one render per animation frame.
    const flush = () => {
      frame = 0
      const batch = queue
      queue = []
      if (batch.length) dispatch({ type: 'events', events: batch })
    }
    setLoading(true)
    api
      .events(sessionId)
      .then((history) => {
        if (cancelled) return
        dispatch({ type: 'events', events: history })
        setLoading(false)
        const after = history.length ? history[history.length - 1].seq : 0
        unsubscribe = subscribeSession(
          sessionId,
          after,
          (e) => {
            queue.push(e)
            handler.current?.(e)
            if (!frame) frame = requestAnimationFrame(flush)
          },
          setLive,
        )
      })
      .catch(() => setLoading(false))
    return () => {
      cancelled = true
      if (frame) cancelAnimationFrame(frame)
      unsubscribe?.()
    }
  }, [sessionId])

  return { transcript, loading, live }
}

export function useSessions(workspace: string | null, kind: 'chat' | 'code') {
  const [sessions, setSessions] = useState<Session[]>([])
  const refresh = useCallback(async () => {
    if (kind === 'code' && !workspace) {
      setSessions([])
      return
    }
    try {
      setSessions(await api.sessions(kind === 'code' ? workspace ?? undefined : undefined, kind))
    } catch {
      /* offline */
    }
  }, [workspace, kind])
  useEffect(() => {
    void refresh()
  }, [refresh])
  return { sessions, refresh, setSessions }
}
