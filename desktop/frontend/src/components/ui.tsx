import { forwardRef, useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { IconX } from './icons'

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'ok'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent/90 border-transparent',
  secondary: 'bg-raised text-fg border-line hover:bg-sunken',
  ghost: 'bg-transparent text-muted hover:text-fg hover:bg-raised border-transparent',
  danger: 'bg-err/10 text-err border-err/30 hover:bg-err/20',
  ok: 'bg-ok/15 text-ok border-ok/30 hover:bg-ok/25',
}

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' }>(
  function Button({ variant = 'secondary', size = 'md', className, ...props }, ref) {
    return (
      <button
        ref={ref}
        className={cx(
          'inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none select-none',
          size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3 text-[13px]',
          VARIANTS[variant],
          className,
        )}
        {...props}
      />
    )
  },
)

export function IconButton({ label, className, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      aria-label={label}
      title={label}
      className={cx('inline-flex h-7 w-7 items-center justify-center rounded-md text-muted hover:bg-raised hover:text-fg transition-colors disabled:opacity-40', className)}
      {...props}
    >
      {children}
    </button>
  )
}

export function Badge({ tone = 'neutral', children, className }: { tone?: 'neutral' | 'ok' | 'warn' | 'err' | 'accent' | 'info'; children: ReactNode; className?: string }) {
  const tones = {
    neutral: 'bg-raised text-muted border-line',
    ok: 'bg-ok/12 text-ok border-ok/25',
    warn: 'bg-warn/12 text-warn border-warn/30',
    err: 'bg-err/12 text-err border-err/30',
    accent: 'bg-accent/12 text-accent border-accent/30',
    info: 'bg-info/12 text-info border-info/30',
  }
  return <span className={cx('inline-flex items-center gap-1 rounded border px-1.5 py-px text-2xs font-medium whitespace-nowrap', tones[tone], className)}>{children}</span>
}

export function Spinner({ size = 14, className }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={cx('animate-spin', className)} aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" fill="none" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" fill="none" strokeLinecap="round" />
    </svg>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-raised px-1 font-mono text-2xs text-muted">{children}</kbd>
}

export function Segmented<T extends string>({ value, options, onChange, label, size = 'md' }: {
  value: T
  options: { value: T; label: string; title?: string }[]
  onChange: (v: T) => void
  label: string
  size?: 'sm' | 'md'
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md border border-line bg-sunken p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          aria-label={o.label}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded px-2.5 font-medium transition-colors',
            size === 'sm' ? 'h-6 text-2xs' : 'h-7 text-xs',
            o.value === value ? 'bg-raised text-fg shadow-sm' : 'text-muted hover:text-fg',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon && <div className="mb-1 text-faint">{icon}</div>}
      <div className="text-sm font-medium text-fg">{title}</div>
      {children && <div className="max-w-sm text-xs leading-relaxed text-muted">{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

export function Callout({ tone, children, icon }: { tone: 'ok' | 'warn' | 'err' | 'info'; children: ReactNode; icon?: ReactNode }) {
  const tones = {
    ok: 'border-ok/30 bg-ok/8 text-ok',
    warn: 'border-warn/35 bg-warn/8 text-warn',
    err: 'border-err/35 bg-err/8 text-err',
    info: 'border-info/30 bg-info/8 text-info',
  }
  return (
    <div role={tone === 'err' ? 'alert' : 'status'} className={cx('flex items-start gap-2 rounded-md border px-3 py-2 text-xs leading-relaxed', tones[tone])}>
      {icon && <span className="mt-px shrink-0">{icon}</span>}
      <div className="min-w-0 text-fg/90">{children}</div>
    </div>
  )
}

export function Modal({ open, onClose, title, children, width = 'max-w-lg' }: { open: boolean; onClose: () => void; title: string; children: ReactNode; width?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const node = ref.current
    node?.querySelector<HTMLElement>('input, textarea, button, [tabindex]')?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => {
      window.removeEventListener('keydown', onKey, true)
      previous?.focus?.()
    }
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 p-4 pt-[12vh] backdrop-blur-[2px] animate-fade-in" onMouseDown={onClose}>
      <div
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cx('w-full overflow-hidden rounded-xl border border-line bg-panel shadow-2xl', width)}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
          <h2 className="text-sm font-semibold">{title}</h2>
          <IconButton label="Close" onClick={onClose}>
            <IconX />
          </IconButton>
        </div>
        {children}
      </div>
    </div>
  )
}

export function Meter({ value, max, tone = 'accent', label }: { value: number; max: number; tone?: 'accent' | 'ok' | 'warn' | 'err'; label: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0
  const tones = { accent: 'bg-accent', ok: 'bg-ok', warn: 'bg-warn', err: 'bg-err' }
  return (
    <div role="meter" aria-label={label} aria-valuenow={Math.round(value)} aria-valuemin={0} aria-valuemax={Math.round(max)} className="h-1.5 w-full overflow-hidden rounded-full bg-sunken">
      <div className={cx('h-full rounded-full transition-[width] duration-500', tones[tone])} style={{ width: `${pct}%` }} />
    </div>
  )
}
