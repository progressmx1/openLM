import type { Model, TelemetryAdapter } from '../../api/types'
import { IconGpu, IconWarning } from '../../components/icons'
import { cx } from '../../components/ui'
import { compactNumber, gib, modelLabel, shortGpuName } from '../../lib/format'
import type { Usage } from '../../lib/transcript'
import { useApp } from '../../state/app'

export function StatusBar({ model, usage, onOpenModels }: { model: Model | null; usage: Usage | null; onOpenModels: () => void }) {
  const { telemetry, connected } = useApp()
  const status = model?.status
  const check = status?.placement_check
  const adapters = (telemetry?.adapters ?? []).filter((a) => !a.integrated || a.processes.length > 0)
  const ctxUsed = usage?.context_used ?? null
  const ctxMax = usage?.context_max ?? status?.n_ctx_effective ?? null
  const tps = usage?.tokens_per_second ?? status?.last_timings?.predicted_per_second ?? null
  const ctxPct = ctxUsed && ctxMax ? Math.min(100, (ctxUsed / ctxMax) * 100) : 0

  return (
    <footer className="flex h-7 shrink-0 items-center gap-4 border-t border-line bg-panel px-3 text-2xs text-muted" aria-label="Status">
      <span className="flex items-center gap-1.5" title={connected ? 'Backend connected' : 'Backend unreachable'}>
        <span className={cx('h-1.5 w-1.5 rounded-full', connected ? 'bg-ok' : 'bg-err animate-pulse2')} />
        {connected ? 'Local' : 'Offline'}
      </span>

      <button onClick={onOpenModels} className="flex min-w-0 items-center gap-1.5 hover:text-fg" title="Open the model & GPU manager">
        {model ? (
          <>
            <span className={cx('h-1.5 w-1.5 shrink-0 rounded-full', status?.state === 'crashed' ? 'bg-err' : model.loaded ? 'bg-ok' : status?.state === 'loading' ? 'bg-warn animate-pulse2' : 'bg-faint')} />
            <span className="truncate text-fg/90">{modelLabel(model)}</span>
            {model.loaded && status?.device_names?.length ? (
              <span className="truncate">on {status.device_names.map(shortGpuName).join(' + ')}</span>
            ) : model.loaded && status?.placement?.mode === 'cpu' ? <span>on CPU</span> : !model.loaded ? <span>(not loaded)</span> : null}
            {check && check.level !== 'ok' && model.loaded && (
              <span className={cx('flex items-center gap-1', check.level === 'error' ? 'text-err' : 'text-warn')} title={check.message}>
                <IconWarning size={12} /> {check.level === 'unknown' ? 'unverified' : check.level}
              </span>
            )}
            {check?.level === 'ok' && model.loaded && check.per_device.length > 0 && (
              <span className="text-ok" title={check.message}>
                {check.per_device.reduce((n, d) => n + d.layers, 0)} layers on GPU
              </span>
            )}
          </>
        ) : (
          <span>No model selected</span>
        )}
      </button>

      <div className="ml-auto hidden items-center gap-4 md:flex">
        {adapters.map((a) => <AdapterMeter key={a.luid} adapter={a} />)}
        {tps != null && <span title="Generation speed of the last reply">{tps.toFixed(1)} tok/s</span>}
        {ctxMax != null && (
          <span className="flex items-center gap-1.5" title={`Context: ${ctxUsed ?? 0} of ${ctxMax} tokens used by the last request`}>
            ctx
            <span className="h-1 w-14 overflow-hidden rounded-full bg-sunken">
              <span className={cx('block h-full rounded-full', ctxPct > 85 ? 'bg-warn' : 'bg-accent')} style={{ width: `${ctxPct}%` }} />
            </span>
            {compactNumber(ctxUsed)}/{compactNumber(ctxMax)}
          </span>
        )}
      </div>
    </footer>
  )
}

function AdapterMeter({ adapter }: { adapter: TelemetryAdapter }) {
  const ours = adapter.processes.reduce((n, p) => n + p.dedicated_mib, 0)
  const used = adapter.nvidia_smi?.used_mib ?? adapter.dedicated_used_mib ?? 0
  const total = adapter.nvidia_smi?.total_mib ?? adapter.dedicated_total_mib
  const integrated = adapter.integrated
  const shared = adapter.processes.reduce((n, p) => n + p.shared_mib, 0)
  const pct = total ? Math.min(100, (used / total) * 100) : 0
  const label = integrated
    ? `${shortGpuName(adapter.name)}: a model is using ${gib(shared)} of shared memory on the integrated GPU`
    : `${adapter.name}: ${gib(used)} of ${gib(total)} used, ${gib(ours)} by loaded models, busiest engine ${adapter.busiest_engine_pct}%`
  return (
    <span className={cx('flex items-center gap-1.5', integrated && 'text-warn')} title={label}>
      <IconGpu size={12} />
      {shortGpuName(adapter.name)}
      {integrated ? (
        <span>{gib(shared)} shared</span>
      ) : (
        <>
          <span className="h-1 w-12 overflow-hidden rounded-full bg-sunken">
            <span className={cx('block h-full rounded-full', ours > 0 ? 'bg-ok' : 'bg-faint')} style={{ width: `${pct}%` }} />
          </span>
          <span>{gib(used)}/{gib(total, 0)}</span>
        </>
      )}
    </span>
  )
}
