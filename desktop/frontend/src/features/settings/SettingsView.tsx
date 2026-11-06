import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import type { PromptFile } from '../../api/types'
import { Badge, Button, Segmented, Spinner, cx } from '../../components/ui'
import { useApp, type Theme } from '../../state/app'

type Tab = 'general' | 'prompts' | 'system'

export function SettingsView() {
  const [tab, setTab] = useState<Tab>('general')
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-4xl flex-col gap-5 px-6 py-6">
        <header className="flex items-center gap-4">
          <h1 className="text-lg font-semibold">Settings</h1>
          <Segmented<Tab>
            label="Settings section"
            value={tab}
            onChange={setTab}
            options={[{ value: 'general', label: 'General' }, { value: 'prompts', label: 'Prompts' }, { value: 'system', label: 'System' }]}
          />
        </header>
        {tab === 'general' && <General />}
        {tab === 'prompts' && <Prompts />}
        {tab === 'system' && <SystemInfo />}
      </div>
    </div>
  )
}

function Row({ title, help, children }: { title: string; help?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-6 border-b border-line py-3 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium">{title}</div>
        {help && <div className="mt-0.5 text-xs leading-relaxed text-muted">{help}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cx('relative h-5 w-9 rounded-full transition-colors', checked ? 'bg-accent' : 'bg-line')}
    >
      <span className={cx('absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-[left]', checked ? 'left-[18px]' : 'left-0.5')} />
    </button>
  )
}

function General() {
  const { settings, saveSettings, theme, setTheme } = useApp()
  if (!settings) return <Spinner />
  return (
    <section className="rounded-lg border border-line bg-panel px-4">
      <Row title="Theme">
        <Segmented<Theme> label="Theme" value={theme} onChange={setTheme} options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }, { value: 'system', label: 'System' }]} />
      </Row>
      <Row title="Internet access" help="Lets the agent use web_search and web_fetch. Off by default; everything else always stays on this computer.">
        <Toggle label="Internet access" checked={settings.internet_enabled} onChange={(v) => void saveSettings({ internet_enabled: v })} />
      </Row>
      <Row title="Allow several models on one GPU" help="Models on different GPUs always run side by side. This also allows stacking models on the same GPU (risk of running out of VRAM).">
        <Toggle label="Allow several models on one GPU" checked={settings.allow_multi_model} onChange={(v) => void saveSettings({ allow_multi_model: v })} />
      </Row>
      <Row title="Max reply tokens" help="Upper bound per model call; the agent also limits it to a third of the context.">
        <input
          type="number" min={256} step={256} defaultValue={settings.inference.max_tokens}
          onBlur={(e) => void saveSettings({ inference: { ...settings.inference, max_tokens: Number(e.target.value) } })}
          className="h-8 w-24 rounded-md border border-line bg-raised px-2 text-xs focus:border-accent/60 focus:outline-none"
          aria-label="Max reply tokens"
        />
      </Row>
      <Row title="Model load timeout" help="Seconds to wait for a model to load (large models over Thunderbolt can be slow).">
        <input
          type="number" min={30} step={30} defaultValue={settings.load_timeout_seconds}
          onBlur={(e) => void saveSettings({ load_timeout_seconds: Number(e.target.value) })}
          className="h-8 w-24 rounded-md border border-line bg-raised px-2 text-xs focus:border-accent/60 focus:outline-none"
          aria-label="Model load timeout"
        />
      </Row>
      <Row title="Runtime" help="llama-server runs each model in its own process (CUDA / Vulkan builds). The in-process fallback is CPU-only.">
        <select
          value={settings.runtime}
          onChange={(e) => void saveSettings({ runtime: e.target.value })}
          className="h-8 rounded-md border border-line bg-raised px-2 text-xs focus:outline-none"
          aria-label="Runtime"
        >
          <option value="auto">Auto</option>
          <option value="llama-server">llama-server</option>
          <option value="llama-cpp-python">llama-cpp-python (CPU)</option>
        </select>
      </Row>
      <Row title="Open workspaces" help="Folders you opened. The agent can only touch files inside these.">
        <div className="flex max-w-xs flex-col items-end gap-1">
          {settings.approved_workspaces.map((w) => (
            <div key={w} className="flex items-center gap-2 text-2xs">
              <span className="truncate font-mono text-muted" title={w}>{w}</span>
              <button className="text-err hover:underline" onClick={() => void saveSettings({ approved_workspaces: settings.approved_workspaces.filter((x) => x !== w) })}>remove</button>
            </div>
          ))}
          {!settings.approved_workspaces.length && <span className="text-2xs text-faint">None</span>}
        </div>
      </Row>
    </section>
  )
}

function Prompts() {
  const { toast } = useApp()
  const [prompts, setPrompts] = useState<PromptFile[] | null>(null)
  const [selected, setSelected] = useState('system.md')
  const [draft, setDraft] = useState('')
  const load = async () => {
    const list = await api.prompts()
    setPrompts(list)
    return list
  }
  useEffect(() => {
    load().then((list) => setDraft(list.find((p) => p.name === selected)?.content ?? '')).catch((e) => toast((e as Error).message, 'err'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const current = prompts?.find((p) => p.name === selected)
  const select = (name: string) => {
    setSelected(name)
    setDraft(prompts?.find((p) => p.name === name)?.content ?? '')
  }
  const save = async () => {
    try {
      await api.savePrompt(selected, draft)
      await load()
      toast(`Saved ${selected}. New turns use it.`, 'ok')
    } catch (e) {
      toast((e as Error).message, 'err')
    }
  }
  const reset = async () => {
    await api.resetPrompt(selected)
    const list = await load()
    setDraft(list.find((p) => p.name === selected)?.content ?? '')
    toast(`Restored the default ${selected}.`, 'info')
  }
  if (!prompts) return <Spinner />
  return (
    <section className="flex min-h-[520px] gap-4">
      <ul className="flex w-48 shrink-0 flex-col gap-0.5">
        {prompts.map((p) => (
          <li key={p.name}>
            <button
              onClick={() => select(p.name)}
              className={cx('flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs', p.name === selected ? 'bg-raised text-fg' : 'text-muted hover:bg-raised/60')}
            >
              <span className="flex-1 truncate font-mono">{p.name}</span>
              {p.overridden && <Badge tone="accent">edited</Badge>}
            </button>
          </li>
        ))}
        <li className="mt-3 px-2.5 text-2xs leading-relaxed text-faint">
          Placeholders like <code>{'{{workspace}}'}</code>, <code>{'{{git}}'}</code>, <code>{'{{tree}}'}</code>, <code>{'{{project_instructions}}'}</code> are filled in for every turn. Per-model variants live in <code>variants.json</code>.
        </li>
      </ul>
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className="font-mono text-fg">{selected}</span>
          {current?.version != null && <Badge>v{current.version}</Badge>}
          <div className="ml-auto flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => void reset()} disabled={!current?.overridden}>Restore default</Button>
            <Button size="sm" variant="primary" onClick={() => void save()} disabled={draft === current?.content}>Save</Button>
          </div>
        </div>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          spellCheck={false}
          aria-label={`Edit ${selected}`}
          className="min-h-[480px] flex-1 resize-none rounded-lg border border-line bg-sunken p-3 font-mono text-[12px] leading-relaxed text-fg focus:border-accent/60 focus:outline-none"
        />
      </div>
    </section>
  )
}

function SystemInfo() {
  const [data, setData] = useState<Record<string, unknown> | null>(null)
  useEffect(() => {
    api.diagnostics().then(setData).catch(() => setData({ error: 'Backend unreachable' }))
  }, [])
  if (!data) return <Spinner />
  return (
    <pre className="overflow-auto rounded-lg border border-line bg-sunken p-3 font-mono text-[11.5px] leading-relaxed text-fg/85">
      {JSON.stringify(data, null, 2)}
    </pre>
  )
}
