import { useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { GpuDevice, GpuMode, Model, PlacementRequest, TelemetryAdapter } from '../../api/types'
import { IconCheck, IconCpu, IconEject, IconFolder, IconGpu, IconInfo, IconLog, IconPlay, IconRefresh, IconWarning } from '../../components/icons'
import { CloudModelCard, CloudModels } from './CloudModels'
import { ModelPicker } from './ModelPicker'
import { Badge, Button, Callout, EmptyState, Meter, Segmented, Spinner, cx } from '../../components/ui'
import { formatBytes, gib, isCloudModel, modelLabel, shortGpuName } from '../../lib/format'
import { useApp } from '../../state/app'

const norm = (s: string) => s.replace(/\(R\)|\(TM\)/g, '').replace(/\s+/g, ' ').trim().toLowerCase()

interface PhysicalGpu {
  key: string
  name: string
  integrated: boolean
  totalMib: number
  entries: GpuDevice[]
  loaded: string[]
}

function physicalGpus(devices: GpuDevice[]): PhysicalGpu[] {
  const map = new Map<string, PhysicalGpu>()
  for (const d of devices) {
    const g = map.get(d.physical_key) ?? { key: d.physical_key, name: d.name, integrated: d.integrated, totalMib: d.total_mib, entries: [], loaded: [] }
    g.entries.push(d)
    g.totalMib = Math.max(g.totalMib, d.total_mib)
    g.loaded = [...new Set([...g.loaded, ...d.loaded_models])]
    map.set(d.physical_key, g)
  }
  return [...map.values()].sort((a, b) => Number(a.integrated) - Number(b.integrated) || b.totalMib - a.totalMib)
}

function telemetryFor(gpu: PhysicalGpu, adapters: TelemetryAdapter[]): TelemetryAdapter | undefined {
  return adapters.find((a) => norm(a.name) === norm(gpu.name))
}

export function ModelsView() {
  const { models, devices, telemetry, refreshDevices, refreshModels, toast } = useApp()
  const [scanning, setScanning] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)
  const gpus = useMemo(() => physicalGpus(devices?.devices ?? []), [devices])
  const localModels = models.filter((m) => !isCloudModel(m))
  const cloudModels = models.filter(isCloudModel)

  const rescan = async () => {
    setScanning(true)
    await refreshDevices(true)
    setScanning(false)
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-6">
        <header className="flex items-center gap-3">
          <div>
            <h1 className="text-lg font-semibold">Models & GPUs</h1>
            <p className="text-xs text-muted">Choose where each model runs. Placement is verified from the runtime's own log and live per-process GPU memory.</p>
          </div>
          <Button className="ml-auto" onClick={() => void rescan()} disabled={scanning}>
            {scanning ? <Spinner size={13} /> : <IconRefresh size={14} />} Rescan GPUs
          </Button>
        </header>

        <section aria-label="GPUs" className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {gpus.map((g) => <GpuCard key={g.key} gpu={g} adapter={telemetryFor(g, telemetry?.adapters ?? [])} models={models} />)}
          {!gpus.length && (
            <div className="col-span-full rounded-lg border border-line bg-panel">
              <EmptyState icon={<IconGpu size={28} />} title="No GPUs detected">
                {devices?.binaries.length ? 'The llama.cpp runtimes found no GPU. Models will run on the CPU.' : 'No llama.cpp runtime is installed. Run scripts\\install-llama-runtimes.ps1 -WithCuda.'}
              </EmptyState>
            </div>
          )}
        </section>
        {devices?.errors.map((e) => (
          <Callout key={e.binary} tone="warn" icon={<IconWarning size={14} />}>{e.binary}: {e.error}</Callout>
        ))}
        {telemetry && !telemetry.supported && <Callout tone="info" icon={<IconInfo size={14} />}>{telemetry.reason}</Callout>}

        <section aria-label="Models" className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold">Models</h2>
            <span className="text-xs text-muted">{localModels.length} on this PC</span>
            <Button className="ml-auto" variant="primary" size="sm" onClick={() => setPickerOpen(true)}>
              <IconFolder size={13} /> Add models…
            </Button>
          </div>
          {localModels.map((m) => <ModelCard key={m.id} model={m} gpus={gpus} />)}
          {!localModels.length && (
            <div className="rounded-lg border border-line bg-panel">
              <EmptyState title="No models yet" action={<Button variant="primary" onClick={() => setPickerOpen(true)}><IconFolder size={13} /> Add models…</Button>}>
                Browse to a <code>.gguf</code> file or a folder of models (for example this project's <code>models</code> folder).
              </EmptyState>
            </div>
          )}
        </section>

        <CloudModels />
        {cloudModels.length > 0 && (
          <div className="flex flex-col gap-2">
            {cloudModels.map((m) => <CloudModelCard key={m.id} model={m} />)}
          </div>
        )}
      </div>
      <ModelPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onAdded={(message) => {
          toast(message, 'ok')
          void refreshModels()
        }}
      />
    </div>
  )
}

function GpuCard({ gpu, adapter, models }: { gpu: PhysicalGpu; adapter?: TelemetryAdapter; models: Model[] }) {
  const used = adapter?.nvidia_smi?.used_mib ?? adapter?.dedicated_used_mib ?? null
  const total = adapter?.nvidia_smi?.total_mib ?? (adapter?.dedicated_total_mib || gpu.totalMib)
  const names = (id: string) => {
    const m = models.find((x) => x.id === id)
    return m ? modelLabel(m) : id.slice(0, 8)
  }
  return (
    <div className={cx('flex flex-col gap-2 rounded-lg border bg-panel p-3', gpu.integrated ? 'border-line/70' : 'border-line')}>
      <div className="flex items-start gap-2">
        <IconGpu size={16} className={gpu.integrated ? 'text-faint' : 'text-accent'} />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{gpu.name}</div>
          <div className="flex flex-wrap gap-1 pt-1">
            {gpu.entries.map((e) => <Badge key={e.key} tone="neutral">{e.runtime.toUpperCase()} · {e.id}</Badge>)}
            {gpu.integrated && <Badge tone="warn">integrated · never used automatically</Badge>}
          </div>
        </div>
      </div>
      {!gpu.integrated && (
        <>
          <Meter value={used ?? 0} max={total} tone={(adapter?.processes.length ?? 0) > 0 ? 'ok' : 'accent'} label={`${gpu.name} memory used`} />
          <div className="flex justify-between text-2xs text-muted">
            <span>{used != null ? `${gib(used)} used` : 'usage unknown'}</span>
            <span>{gib(total)} VRAM</span>
          </div>
        </>
      )}
      {adapter?.processes.map((p) => (
        <div key={p.pid} className="flex items-center justify-between rounded bg-sunken px-2 py-1 text-2xs">
          <span className="truncate text-fg/90">{names(p.model_id)}</span>
          <span className="font-mono text-muted">{gpu.integrated ? `${gib(p.shared_mib)} shared` : `${gib(p.dedicated_mib)} VRAM`}{p.utilization_pct ? ` · ${p.utilization_pct.toFixed(0)}%` : ''}</span>
        </div>
      ))}
      {!adapter?.processes.length && gpu.loaded.length === 0 && <div className="text-2xs text-faint">No model on this GPU.</div>}
    </div>
  )
}

function initialRequest(model: Model): PlacementRequest {
  const saved = model.profile.placement
  if (saved && typeof saved === 'object' && saved.mode) return { ...saved, mode: saved.mode === ('single' as GpuMode) ? 'gpu' : saved.mode }
  return { mode: 'auto' }
}

function ModelCard({ model, gpus }: { model: Model; gpus: PhysicalGpu[] }) {
  const { refreshModels, refreshDevices, toast } = useApp()
  const [req, setReq] = useState<PlacementRequest>(() => initialRequest(model))
  const [ctx, setCtx] = useState<number>(model.profile.n_ctx ?? 8192)
  const [layerCap, setLayerCap] = useState<string>(model.profile.n_gpu_layers && model.profile.n_gpu_layers > 0 ? String(model.profile.n_gpu_layers) : '')
  const [busy, setBusy] = useState(false)
  const [showLog, setShowLog] = useState(false)
  const [log, setLog] = useState<string[] | null>(null)
  const status = model.status
  const state = status?.state ?? (model.loaded ? 'ready' : undefined)
  const discrete = gpus.filter((g) => !g.integrated)

  // Single GPU default: first discrete card. Multi default: all discrete cards (Vulkan).
  useEffect(() => {
    if (req.mode === 'gpu' && !req.devices?.length && discrete[0]) setReq({ mode: 'gpu', devices: [{ name: discrete[0].name }] })
  }, [req, discrete])

  const multiNames = req.mode === 'split' ? (req.devices?.length ? req.devices.map((d) => d.name) : discrete.map((g) => g.name)) : []
  const multiGpus = gpus.filter((g) => multiNames.some((n) => norm(n) === norm(g.name)))
  const ratio = req.tensor_split && req.tensor_split.length === multiGpus.length ? req.tensor_split : multiGpus.map((g) => g.totalMib)
  const firstShare = ratio.length === 2 ? Math.round((ratio[0] / (ratio[0] + ratio[1])) * 100) : 50

  const load = async () => {
    setBusy(true)
    try {
      const placement: PlacementRequest = req.mode === 'split'
        ? { mode: 'split', devices: multiGpus.map((g) => ({ name: g.entries.find((e) => e.runtime === 'vulkan')?.name ?? g.name })), tensor_split: multiGpus.length === 2 ? ratio : null }
        : req
      const result = await api.loadModel(model.id, { placement, n_ctx: ctx, n_gpu_layers: layerCap ? Number(layerCap) : -1 })
      const check = result.placement_check
      toast(check ? check.message : 'Model loaded.', check?.level === 'ok' ? 'ok' : check?.level === 'error' ? 'err' : 'warn')
    } catch (e) {
      toast((e as Error).message, 'err')
    } finally {
      setBusy(false)
      await Promise.all([refreshModels(), refreshDevices()])
    }
  }
  const unload = async () => {
    setBusy(true)
    try {
      await api.unloadModel(model.id)
    } catch (e) {
      toast((e as Error).message, 'err')
    } finally {
      setBusy(false)
      await Promise.all([refreshModels(), refreshDevices()])
    }
  }
  useEffect(() => {
    if (!showLog) return
    let stop = false
    const tick = () => api.modelLog(model.id, 200).then((r) => !stop && setLog(r.lines)).catch(() => undefined)
    tick()
    const id = setInterval(tick, 3000)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [showLog, model.id])

  const check = status?.placement_check
  return (
    <article className="overflow-hidden rounded-lg border border-line bg-panel" aria-label={modelLabel(model)}>
      <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3">
        <span className={cx('h-2 w-2 rounded-full', state === 'crashed' ? 'bg-err' : state === 'ready' ? 'bg-ok' : state === 'loading' ? 'bg-warn animate-pulse2' : 'bg-faint')} />
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{modelLabel(model)}</div>
          <div className="truncate text-2xs text-muted">{formatBytes(model.size_bytes)} · {model.path}</div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {status?.native_tools && <Badge tone="info" >native tools</Badge>}
          {model.loaded || state === 'loading' || state === 'crashed' ? (
            <>
              <Button size="sm" onClick={() => void load()} disabled={busy}>{busy ? <Spinner size={12} /> : <IconRefresh size={13} />} Reload</Button>
              <Button size="sm" variant="ghost" onClick={() => void unload()} disabled={busy}><IconEject size={13} /> Unload</Button>
            </>
          ) : (
            <Button size="sm" variant="primary" onClick={() => void load()} disabled={busy}>{busy ? <Spinner size={12} /> : <IconPlay size={12} />} Load</Button>
          )}
        </div>
      </div>

      <div className="grid gap-4 px-4 py-3 lg:grid-cols-[1fr_1fr]">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <span className="w-16 text-xs text-muted">Run on</span>
            <Segmented<GpuMode>
              label="GPU mode"
              value={req.mode}
              onChange={(mode) => setReq({ mode })}
              options={[
                { value: 'auto', label: 'Auto', title: 'Best free discrete GPU that fits; split if none does' },
                { value: 'gpu', label: 'Single GPU', title: 'All layers on one GPU you choose' },
                { value: 'split', label: 'Multi-GPU', title: 'One model split across several GPUs (Vulkan)' },
                { value: 'cpu', label: 'CPU', title: 'No GPU' },
              ]}
            />
          </div>
          {req.mode === 'gpu' && (
            <label className="flex items-center gap-3 text-xs">
              <span className="w-16 text-muted">GPU</span>
              <select
                value={req.devices?.[0]?.name ?? ''}
                onChange={(e) => setReq({ mode: 'gpu', devices: [{ name: e.target.value }] })}
                className="h-8 flex-1 rounded-md border border-line bg-raised px-2 text-xs focus:border-accent/60 focus:outline-none"
              >
                {gpus.map((g) => (
                  <option key={g.key} value={g.name}>{g.name} · {gib(g.totalMib, 0)}{g.integrated ? ' (integrated — slow, shares RAM)' : ''}</option>
                ))}
              </select>
            </label>
          )}
          {req.mode === 'split' && (
            <div className="flex flex-col gap-2 text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <span className="w-16 text-muted">GPUs</span>
                {gpus.map((g) => {
                  const on = multiNames.some((n) => norm(n) === norm(g.name))
                  return (
                    <label key={g.key} className="flex items-center gap-1.5">
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() => {
                          const next = on ? multiNames.filter((n) => norm(n) !== norm(g.name)) : [...multiNames, g.name]
                          setReq({ mode: 'split', devices: next.map((name) => ({ name })), tensor_split: null })
                        }}
                      />
                      {shortGpuName(g.name)}{g.integrated ? ' (iGPU)' : ''}
                    </label>
                  )
                })}
              </div>
              {multiGpus.length === 2 && (
                <div className="flex items-center gap-3">
                  <span className="w-16 text-muted">Split</span>
                  <span className="w-28 truncate text-right">{shortGpuName(multiGpus[0].name)} {firstShare}%</span>
                  <input
                    type="range" min={5} max={95} step={1} value={firstShare}
                    aria-label="Share of layers on the first GPU"
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      setReq({ ...req, devices: multiGpus.map((g) => ({ name: g.name })), tensor_split: [v, 100 - v] })
                    }}
                    className="flex-1 accent-[rgb(var(--accent))]"
                  />
                  <span className="w-28 truncate">{100 - firstShare}% {shortGpuName(multiGpus[1].name)}</span>
                </div>
              )}
              {multiGpus.length < 2 && <div className="text-2xs text-warn">Only one GPU selected/available — the model will load on it alone (with a notice).</div>}
              {multiGpus.length === 2 && <div className="text-2xs text-faint">Default is proportional to VRAM. Uses the Vulkan runtime, which sees NVIDIA and AMD cards together.</div>}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-3">
              <span className="w-16 text-muted">Context</span>
              <input
                type="number" min={0} step={1024} value={ctx}
                onChange={(e) => setCtx(Number(e.target.value))}
                className="h-8 w-24 rounded-md border border-line bg-raised px-2 text-xs focus:border-accent/60 focus:outline-none"
                aria-label="Context length (0 = largest that fits)"
              />
            </label>
            <label className="flex items-center gap-2" title="Leave empty to put every layer on the GPU">
              <span className="text-muted">GPU layers</span>
              <input
                type="number" min={1} placeholder="all" value={layerCap}
                onChange={(e) => setLayerCap(e.target.value)}
                className="h-8 w-16 rounded-md border border-line bg-raised px-2 text-xs placeholder:text-faint focus:border-accent/60 focus:outline-none"
                aria-label="GPU layer cap (empty = all)"
              />
            </label>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          {state === 'loading' && <Callout tone="info" icon={<Spinner size={13} />}>Loading…</Callout>}
          {state === 'crashed' && <Callout tone="err" icon={<IconWarning size={14} />}>{status?.error}</Callout>}
          {status?.placement_note && <Callout tone="warn" icon={<IconInfo size={14} />}>{status.placement_note}</Callout>}
          {model.loaded && check && (
            <Callout tone={check.level === 'ok' ? 'ok' : check.level === 'error' ? 'err' : check.level === 'warning' ? 'warn' : 'info'} icon={check.level === 'ok' ? <IconCheck size={14} /> : <IconWarning size={14} />}>
              <div className="font-medium">{check.message}</div>
            </Callout>
          )}
          {model.loaded && (check?.per_device.length ?? 0) > 0 && (
            <table className="w-full text-2xs">
              <thead className="text-faint">
                <tr><th className="py-1 text-left font-medium">Device</th><th className="text-right font-medium">Layers</th><th className="text-right font-medium">Weights</th><th className="text-right font-medium">KV cache</th><th className="text-right font-medium">Live VRAM</th></tr>
              </thead>
              <tbody>
                {check!.per_device.map((d) => {
                  const live = status?.vram?.find((v) => norm(v.adapter) === norm(d.name))
                  return (
                    <tr key={d.id} className="border-t border-line/60">
                      <td className="py-1">{shortGpuName(d.name)} <span className="text-faint">({d.id})</span></td>
                      <td className="text-right font-mono">{d.layers}</td>
                      <td className="text-right font-mono">{gib(d.model_mib, 2)}</td>
                      <td className="text-right font-mono">{gib(d.kv_mib, 2)}</td>
                      <td className="text-right font-mono">{live ? (live.integrated ? `${gib(live.shared_mib, 2)} shared` : gib(live.dedicated_mib, 2)) : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
          {model.loaded && status?.vram && status.vram.some((v) => v.integrated) && req.mode !== 'cpu' && !check?.per_device.some((d) => /intel/i.test(d.name)) && (
            <Callout tone="warn" icon={<IconWarning size={14} />}>This model's process is also using memory on the integrated GPU.</Callout>
          )}
          {model.loaded && status?.placement?.mode === 'cpu' && <Callout tone="info" icon={<IconCpu size={14} />}>Running on the CPU.</Callout>}
          {model.loaded && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs text-muted">
              <span>runtime {status?.backend ?? status?.runtime}</span>
              {status?.n_ctx_effective && <span>ctx {status.n_ctx_effective.toLocaleString()}</span>}
              {status?.last_timings?.predicted_per_second && <span>{status.last_timings.predicted_per_second.toFixed(1)} tok/s</span>}
              {status?.pid && <span>pid {status.pid}</span>}
            </div>
          )}
          {(model.loaded || state === 'crashed') && (
            <button className="flex items-center gap-1.5 self-start text-2xs text-accent hover:underline" onClick={() => setShowLog((v) => !v)}>
              <IconLog size={12} /> {showLog ? 'Hide' : 'Show'} runtime log
            </button>
          )}
        </div>
      </div>
      {showLog && (
        <div className="border-t border-line">
          {status?.command && <div className="truncate bg-sunken px-4 py-1.5 font-mono text-2xs text-muted" title={status.command.join(' ')}>{status.command.join(' ')}</div>}
          <pre className="max-h-72 overflow-auto bg-sunken px-4 py-2 font-mono text-2xs leading-relaxed text-fg/80">{log ? log.join('\n') : 'Loading…'}</pre>
        </div>
      )}
    </article>
  )
}
