import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { Model, Provider, ProviderModel } from '../../api/types'
import { IconCheck, IconGlobe, IconPlay, IconPlus, IconRefresh, IconTrash, IconWarning } from '../../components/icons'
import { Badge, Button, Callout, Spinner, cx } from '../../components/ui'
import { modelLabel } from '../../lib/format'
import { useApp } from '../../state/app'

const inputClass = 'h-8 min-w-0 flex-1 rounded-md border border-line bg-raised px-2 text-xs placeholder:text-faint focus:border-accent/60 focus:outline-none'

/** Paid API models (OpenAI, Claude, DeepSeek, any OpenAI-compatible endpoint). */
export function CloudModels() {
  const [providers, setProviders] = useState<Provider[] | null>(null)
  const { toast } = useApp()

  useEffect(() => {
    api.providers().then(setProviders).catch((e) => toast((e as Error).message, 'err'))
  }, [toast])

  const replace = (p: Provider) => setProviders((list) => list?.map((x) => (x.id === p.id ? p : x)) ?? null)

  return (
    <section aria-label="Cloud models" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h2 className="text-sm font-semibold">Cloud models</h2>
        <span className="text-xs text-muted">Paid API models. Prompts, code and files you share go to the provider; keys are stored encrypted on this PC.</span>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {providers?.map((p) => <ProviderCard key={p.id} provider={p} onChange={replace} />)}
        {!providers && <div className="text-xs text-muted"><Spinner size={12} /> Loading providers…</div>}
      </div>
    </section>
  )
}

function ProviderCard({ provider, onChange }: { provider: Provider; onChange: (p: Provider) => void }) {
  const { toast, refreshModels, models } = useApp()
  const [key, setKey] = useState('')
  const [url, setUrl] = useState(provider.base_url)
  const [model, setModel] = useState(provider.suggested[0] ?? '')
  const [available, setAvailable] = useState<ProviderModel[] | null>(null)
  const [busy, setBusy] = useState<'' | 'key' | 'list' | 'add'>('')
  const custom = provider.id === 'custom'
  const added = models.filter((m) => m.path.startsWith(`cloud://${provider.id}/`))

  const run = async (what: typeof busy, fn: () => Promise<void>) => {
    setBusy(what)
    try {
      await fn()
    } catch (e) {
      toast((e as Error).message, 'err')
    } finally {
      setBusy('')
    }
  }

  const save = () => run('key', async () => {
    const body: { api_key?: string; base_url?: string } = {}
    if (key.trim()) body.api_key = key.trim()
    if (custom || url !== provider.base_url) body.base_url = url
    const p = await api.updateProvider(provider.id, body)
    onChange(p)
    setKey('')
    await refreshModels()
    toast(`${provider.label} saved.`, 'ok')
  })
  const forget = () => run('key', async () => {
    onChange(await api.deleteProviderKey(provider.id))
    await refreshModels()
  })
  const list = () => run('list', async () => {
    const r = await api.providerModels(provider.id)
    setAvailable(r.models)
    toast(`${provider.label}: ${r.models.length} models available with this key.`, 'ok')
  })
  const add = (name = model) => run('add', async () => {
    const m = await api.addCloudModel(provider.id, name.trim())
    await refreshModels()
    toast(`Added ${modelLabel(m)}. Pick it in Chat or Code.`, 'ok')
  })

  const options = [...new Set([...provider.suggested, ...(available?.map((m) => m.id) ?? [])])].slice(0, 40)
  return (
    <article className="flex flex-col gap-2.5 rounded-lg border border-line bg-panel p-3" aria-label={`${provider.label} provider`}>
      <div className="flex items-center gap-2">
        <IconGlobe size={15} className={provider.configured ? 'text-accent' : 'text-faint'} />
        <span className="text-sm font-medium">{provider.label}</span>
        {provider.configured ? <Badge tone="ok">key {provider.masked_key}</Badge> : <Badge tone="neutral">no key</Badge>}
        {provider.key_url && (
          <a href={provider.key_url} target="_blank" rel="noreferrer" className="ml-auto text-2xs text-accent hover:underline">Get an API key</a>
        )}
      </div>
      <div className="flex gap-2">
        <input
          type="password" autoComplete="off" spellCheck={false} value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder={provider.configured ? 'Replace key…' : 'Paste API key'}
          aria-label={`${provider.label} API key`}
          className={inputClass}
          onKeyDown={(e) => e.key === 'Enter' && key.trim() && void save()}
        />
        <Button size="sm" variant={provider.configured ? 'secondary' : 'primary'} onClick={() => void save()} disabled={busy !== '' || (!key.trim() && !(custom && url !== provider.base_url))}>
          {busy === 'key' ? <Spinner size={12} /> : <IconCheck size={13} />} Save
        </Button>
        {provider.configured && (
          <Button size="sm" variant="ghost" onClick={() => void forget()} disabled={busy !== ''} title="Delete the stored key"><IconTrash size={13} /></Button>
        )}
      </div>
      {custom && (
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="Endpoint, e.g. https://openrouter.ai/api/v1" aria-label="OpenAI-compatible endpoint" className={inputClass} />
      )}
      {provider.configured && (
        <div className="flex gap-2">
          <input
            list={`models-${provider.id}`} value={model} onChange={(e) => setModel(e.target.value)}
            placeholder="Model name" aria-label={`${provider.label} model`} className={cx(inputClass, 'font-mono')}
            onKeyDown={(e) => e.key === 'Enter' && model.trim() && void add()}
          />
          <datalist id={`models-${provider.id}`}>{options.map((o) => <option key={o} value={o} />)}</datalist>
          <Button size="sm" variant="ghost" onClick={() => void list()} disabled={busy !== ''} title="Ask the provider which models this key can use">
            {busy === 'list' ? <Spinner size={12} /> : <IconRefresh size={13} />}
          </Button>
          <Button size="sm" variant="primary" onClick={() => void add()} disabled={busy !== '' || !model.trim()} title="Add the model typed on the left">
            {busy === 'add' ? <Spinner size={12} /> : <IconPlus size={13} />} Add
          </Button>
        </div>
      )}
      {provider.configured && options.length > 0 && (
        <div className="flex flex-wrap items-center gap-1" aria-label={`${provider.label} models`}>
          <span className="mr-1 text-2xs text-faint">{available ? 'Your models:' : 'Latest:'}</span>
          {options.map((o) => {
            const isAdded = added.some((m) => m.name === o)
            return (
              <button
                key={o}
                type="button"
                disabled={isAdded || busy !== ''}
                onClick={() => void add(o)}
                title={isAdded ? 'Already added' : `Add ${o}`}
                className={cx(
                  'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-2xs transition-colors',
                  isAdded ? 'border-accent/40 bg-accent/10 text-accent' : 'border-line text-muted hover:border-accent/60 hover:text-fg',
                )}
              >
                {isAdded ? <IconCheck size={10} /> : <IconPlus size={10} />} {o}
              </button>
            )
          })}
        </div>
      )}
    </article>
  )
}

/** A registered API model: no GPU controls, only a connection check. */
export function CloudModelCard({ model }: { model: Model }) {
  const { refreshModels, toast } = useApp()
  const [busy, setBusy] = useState(false)
  const status = model.status
  const ready = status?.state === 'ready'

  const test = async () => {
    setBusy(true)
    try {
      const s = await api.loadModel(model.id, {})
      toast(`${modelLabel(model)}: connected${s.n_ctx_effective ? ` · ${s.n_ctx_effective.toLocaleString()} token context` : ''}.`, 'ok')
    } catch (e) {
      toast((e as Error).message, 'err')
    } finally {
      setBusy(false)
      await refreshModels()
    }
  }
  const remove = async () => {
    try {
      await api.removeModel(model.id)
    } catch (e) {
      toast((e as Error).message, 'err')
    }
    await refreshModels()
  }

  return (
    <article className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-panel px-4 py-3" aria-label={modelLabel(model)}>
      <span className={cx('h-2 w-2 rounded-full', ready ? 'bg-ok' : 'bg-faint')} />
      <div className="min-w-0">
        <div className="truncate text-sm font-medium">{model.display_name || model.name}</div>
        <div className="truncate text-2xs text-muted">
          {model.path}{status?.n_ctx_effective ? ` · ctx ${status.n_ctx_effective.toLocaleString()}` : ''}
          {status?.last_timings?.predicted_per_second ? ` · ${status.last_timings.predicted_per_second.toFixed(0)} tok/s` : ''}
        </div>
      </div>
      <Badge tone="info">cloud · {status?.provider_label ?? 'API'}</Badge>
      <div className="ml-auto flex items-center gap-2">
        <Button size="sm" onClick={() => void test()} disabled={busy || !ready} title="Check the key and model with the provider">
          {busy ? <Spinner size={12} /> : <IconPlay size={12} />} Test
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void remove()} title="Remove from the list (the key stays)"><IconTrash size={13} /></Button>
      </div>
      {!ready && status?.error && (
        <div className="w-full"><Callout tone="warn" icon={<IconWarning size={14} />}>{status.error}</Callout></div>
      )}
    </article>
  )
}
