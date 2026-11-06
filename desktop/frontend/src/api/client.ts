import type {
  AgentEvent,
  AppSettings,
  Change,
  DevicesResponse,
  Mode,
  Model,
  ModelStatus,
  PlacementRequest,
  PromptFile,
  Provider,
  ProviderModel,
  RecentWorkspace,
  Session,
  Telemetry,
  WorkspaceEntry,
} from './types'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export interface BrowseListing {
  path: string | null
  parent: string | null
  dirs: { name: string; path: string }[]
  files: { name: string; path: string; size: number }[]
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError('Cannot reach the local backend. Is it running? (scripts\\dev-backend.ps1)', 0)
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* not JSON */
    }
    throw new ApiError(detail || `Request failed (${response.status})`, response.status)
  }
  return (await response.json()) as T
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const api = {
  health: () => request<{ status: string; version: string; loaded_models: string[] }>('/api/health'),
  diagnostics: () => request<Record<string, unknown>>('/api/system/diagnostics'),
  devices: (refresh = false) => request<DevicesResponse>(`/api/system/devices${refresh ? '?refresh=true' : ''}`),
  telemetry: () => request<Telemetry>('/api/system/gpu-telemetry'),
  getSettings: () => request<AppSettings>('/api/settings'),
  updateSettings: (settings: Partial<AppSettings>) =>
    request<AppSettings>('/api/settings', { method: 'PUT', body: JSON.stringify({ settings }) }),

  // models
  models: () => request<Model[]>('/api/models'),
  discover: (locations: string[]) =>
    request<{ found: number; added: number; updated: number }>('/api/models/discover', json({ locations })),
  register: (path: string) => request<Model>('/api/models/register', json({ path })),
  loadModel: (id: string, options: { placement?: PlacementRequest; n_ctx?: number; n_gpu_layers?: number }) =>
    request<ModelStatus>(`/api/models/${id}/load`, json(options)),
  unloadModel: (id: string) => request<ModelStatus>(`/api/models/${id}/unload`, { method: 'POST' }),
  removeModel: (id: string) => request<{ deleted: boolean }>(`/api/models/${id}`, { method: 'DELETE' }),
  providers: () => request<Provider[]>('/api/providers'),
  updateProvider: (id: string, body: { api_key?: string; base_url?: string }) =>
    request<Provider>(`/api/providers/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteProviderKey: (id: string) => request<Provider>(`/api/providers/${id}/key`, { method: 'DELETE' }),
  providerModels: (id: string) => request<{ models: ProviderModel[] }>(`/api/providers/${id}/models`),
  addCloudModel: (id: string, model: string) => request<Model>(`/api/providers/${id}/models`, json({ model })),
  modelLog: (id: string, tail = 300) => request<{ lines: string[]; path: string | null }>(`/api/models/${id}/log?tail=${tail}`),

  // agent sessions
  sessions: (workspace?: string, kind?: 'chat' | 'code') => {
    const query = new URLSearchParams()
    if (workspace) query.set('workspace', workspace)
    if (kind) query.set('kind', kind)
    const qs = query.toString()
    return request<Session[]>(`/api/agent/sessions${qs ? `?${qs}` : ''}`)
  },
  createSession: (workspace: string, model_id: string | null, mode: Mode) =>
    request<Session>('/api/agent/sessions', json({ workspace, model_id, mode })),
  session: (id: string) => request<Session>(`/api/agent/sessions/${id}`),
  updateSession: (id: string, fields: Partial<Pick<Session, 'title' | 'model_id' | 'mode'>> & { max_steps?: number }) =>
    request<Session>(`/api/agent/sessions/${id}`, { method: 'PATCH', body: JSON.stringify(fields) }),
  deleteSession: (id: string) => request<{ deleted: boolean }>(`/api/agent/sessions/${id}`, { method: 'DELETE' }),
  events: (id: string, after = 0) => request<AgentEvent[]>(`/api/agent/sessions/${id}/events?after=${after}`),
  send: (id: string, text: string) => request<{ started: boolean; turn?: number }>(`/api/agent/sessions/${id}/messages`, json({ text })),
  interrupt: (id: string) => request<{ interrupted: boolean }>(`/api/agent/sessions/${id}/interrupt`, { method: 'POST' }),
  decide: (id: string, call_id: string, approve: boolean, remember = false) =>
    request<{ ok: boolean }>(`/api/agent/sessions/${id}/decision`, json({ call_id, approve, remember })),
  changes: (id: string) => request<Change[]>(`/api/agent/sessions/${id}/changes`),
  decideFile: (id: string, turn: number, path: string, accept: boolean) =>
    request<Change>(`/api/agent/sessions/${id}/changes`, json({ turn, path, accept })),
  undo: (id: string) => request<{ reverted: string[] }>(`/api/agent/sessions/${id}/undo`, { method: 'POST' }),
  systemPrompt: (id: string) => request<{ prompt: string }>(`/api/agent/sessions/${id}/system-prompt`),
  meta: () => request<{ modes: Mode[]; commands: Record<string, string> }>('/api/agent/meta'),

  // prompts
  prompts: () => request<PromptFile[]>('/api/agent/prompts'),
  savePrompt: (name: string, content: string) =>
    request<{ ok: boolean }>(`/api/agent/prompts/${name}`, { method: 'PUT', body: JSON.stringify({ content }) }),
  resetPrompt: (name: string) => request<{ ok: boolean }>(`/api/agent/prompts/${name}`, { method: 'DELETE' }),

  // workspace
  recentWorkspaces: () => request<RecentWorkspace[]>('/api/workspace/recent'),
  openWorkspace: (workspace: string) => request<{ path: string; name: string }>('/api/workspace/open', json({ workspace })),
  browse: (path?: string, ext?: string) => {
    const query = new URLSearchParams()
    if (path) query.set('path', path)
    if (ext) query.set('ext', ext)
    const qs = query.toString()
    return request<BrowseListing>(`/api/workspace/browse${qs ? `?${qs}` : ''}`)
  },
  tree: (path: string) => request<{ path: string; entries: WorkspaceEntry[] }>(`/api/workspace/tree?path=${encodeURIComponent(path)}`),
  file: (path: string) => request<{ path: string; binary: boolean; content: string }>(`/api/workspace/file?path=${encodeURIComponent(path)}`),
}

/** Live session events over SSE; reconnects with the last seen seq. */
export function subscribeSession(id: string, after: number, onEvent: (e: AgentEvent) => void, onState?: (connected: boolean) => void) {
  let last = after
  let source: EventSource | null = null
  let closed = false
  let retry: ReturnType<typeof setTimeout> | undefined

  const open = () => {
    if (closed) return
    source = new EventSource(`${BASE}/api/agent/sessions/${id}/stream?after=${last}`)
    source.onopen = () => onState?.(true)
    source.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as AgentEvent
        if (event.seq <= last) return
        last = event.seq
        onEvent(event)
      } catch {
        /* ignore malformed */
      }
    }
    source.onerror = () => {
      onState?.(false)
      source?.close()
      if (!closed) retry = setTimeout(open, 1500)
    }
  }
  open()
  return () => {
    closed = true
    if (retry) clearTimeout(retry)
    source?.close()
  }
}

declare global {
  interface Window {
    desktop?: { platform: string; electronVersion: string; pickFolder?: () => Promise<string | null> }
  }
}
