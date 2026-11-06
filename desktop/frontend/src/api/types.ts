export type Mode = 'ask' | 'plan' | 'agent' | 'auto' | 'chat'
export type Surface = 'chat' | 'code'
export type GpuMode = 'auto' | 'gpu' | 'split' | 'cpu'

export interface GpuDevice {
  key: string
  id: string
  runtime: string
  name: string
  total_mib: number
  free_mib: number
  physical_key: string
  integrated: boolean
  binary: string
  loaded_models: string[]
}

export interface DevicesResponse {
  runtime: string
  mode: string
  devices: GpuDevice[]
  binaries: { path: string; runtime: string }[]
  errors: { binary: string; error: string }[]
}

export interface PerDevice {
  id: string
  name: string
  layers: number
  model_mib: number
  kv_mib: number
  compute_mib: number
  total_mib: number
}

export interface PlacementCheck {
  level: 'ok' | 'warning' | 'error' | 'unknown'
  message: string
  per_device: PerDevice[]
}

export interface VramUse {
  adapter: string
  luid: string
  integrated: boolean
  dedicated_mib: number
  shared_mib: number
  utilization_pct?: number
}

export interface Timings {
  prompt_n?: number
  prompt_per_second?: number
  predicted_n?: number
  predicted_per_second?: number
}

export interface ModelStatus {
  loaded: boolean
  state?: 'loading' | 'ready' | 'crashed' | 'no_key'
  provider?: string
  provider_label?: string
  remote_model?: string
  warning?: string
  error?: string | null
  runtime?: string
  backend?: string
  placement?: {
    mode: GpuMode
    requested_mode: GpuMode
    runtime: string
    devices: string[]
    device_ids: string[]
    device_names: string[]
    tensor_split: number[] | null
    partial_offload: boolean
    note: string | null
  }
  device_names?: string[]
  placement_note?: string | null
  placement_check?: PlacementCheck
  vram?: VramUse[] | null
  n_ctx_effective?: number | null
  last_timings?: Timings | null
  native_tools?: boolean
  command?: string[]
  pid?: number
  log_path?: string
}

export interface PlacementRequest {
  mode: GpuMode
  devices?: { key?: string; name: string }[]
  tensor_split?: number[] | null
}

export interface Model {
  id: string
  name: string
  display_name: string | null
  path: string
  quant: string | null
  size_bytes: number | null
  loaded: boolean
  status: ModelStatus
  profile: { placement?: PlacementRequest; n_ctx?: number; n_gpu_layers?: number; tensor_split?: number[] }
  metadata: Record<string, unknown> | null
  runtime?: string
}

export interface Provider {
  id: 'openai' | 'anthropic' | 'deepseek' | 'custom'
  label: string
  configured: boolean
  masked_key: string | null
  base_url: string
  default_base_url: string
  key_url: string
  suggested: string[]
}

export interface ProviderModel {
  id: string
  name: string
  context: number | null
}

export interface TelemetryProcess {
  pid: number
  model_id: string
  dedicated_mib: number
  shared_mib: number
  utilization_pct: number
}

export interface TelemetryAdapter {
  luid: string
  name: string
  vendor: string
  integrated: boolean
  dedicated_total_mib: number
  dedicated_used_mib: number | null
  shared_used_mib: number | null
  utilization: Record<string, number>
  busiest_engine_pct: number
  processes: TelemetryProcess[]
  nvidia_smi?: { used_mib: number; total_mib: number; utilization_pct: number }
}

export interface Telemetry {
  supported: boolean
  reason?: string
  adapters: TelemetryAdapter[]
}

export interface Todo {
  content: string
  status: 'pending' | 'in_progress' | 'completed'
}

export interface Session {
  id: string
  workspace: string
  title: string
  model_id: string | null
  mode: Mode
  state: 'idle' | 'running' | 'awaiting_approval'
  todos: Todo[]
  options: { allowed_commands?: string[]; max_steps?: number }
  created_at: string
  updated_at: string
  running?: boolean
  pending_approvals?: string[]
}

export interface AgentEvent {
  seq: number
  turn: number
  type: string
  [key: string]: unknown
}

export interface Change {
  turn: number
  path: string
  abs_path: string
  diff: string
  added: number
  removed: number
  created: boolean
  status: 'applied' | 'accepted' | 'rejected'
}

export interface WorkspaceEntry {
  name: string
  path: string
  dir: boolean
  size: number | null
}

export interface RecentWorkspace {
  path: string
  name: string
  exists: boolean
  last_used: string | null
}

export interface PromptFile {
  name: string
  version: number | null
  overridden: boolean
  content: string
}

export interface AppSettings {
  theme: string
  internet_enabled: boolean
  allow_multi_model: boolean
  runtime: string
  load_timeout_seconds: number
  llama_server_paths: string[]
  model_directories: string[]
  approved_workspaces: string[]
  inference: { n_ctx: number; n_gpu_layers: number; max_tokens: number; temperature: number; [k: string]: unknown }
  [k: string]: unknown
}
