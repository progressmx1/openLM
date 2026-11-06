export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

export function gib(mib: number | null | undefined, digits = 1): string {
  if (mib == null) return '—'
  return `${(mib / 1024).toFixed(digits)} GB`
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const then = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`).getTime()
  const seconds = Math.round((Date.now() - then) / 1000)
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  return days < 30 ? `${days}d ago` : new Date(then).toLocaleDateString()
}

export function compactNumber(n: number | null | undefined): string {
  if (n == null) return '—'
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n)
}

export function shortGpuName(name: string): string {
  return name
    .replace(/NVIDIA GeForce /i, '')
    .replace(/AMD Radeon /i, '')
    .replace(/\s*Laptop GPU/i, ' Laptop')
    .replace(/Intel\(R\) Arc\(TM\) Graphics/i, 'Intel Arc iGPU')
    .trim()
}

export function basename(path: string): string {
  const parts = path.replace(/[\\/]+$/, '').split(/[\\/]/)
  return parts[parts.length - 1] || path
}

const CLOUD_LABELS: Record<string, string> = { openai: 'OpenAI', anthropic: 'Claude', deepseek: 'DeepSeek', custom: 'Custom' }

export const isCloudModel = (m: { path?: string }) => !!m.path?.startsWith('cloud://')

export function modelLabel(m: { display_name: string | null; name: string; quant?: string | null; path?: string }): string {
  const base = m.display_name || m.name
  const cloud = m.path?.match(/^cloud:\/\/([^/]+)\//)
  if (cloud) return `${base} · ${CLOUD_LABELS[cloud[1]] ?? cloud[1]} API`
  return m.quant && !base.includes(m.quant) ? `${base} · ${m.quant}` : base
}
