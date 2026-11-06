import { useApp } from '../state/app'
import { IconGlobe } from './icons'
import { cx } from './ui'

/**
 * Internet access for web_search / web_fetch (both Chat and Code). Off by default — the
 * app is offline-first — so turning it on is an explicit user choice.
 */
export function WebToggle() {
  const { settings, saveSettings, toast } = useApp()
  const on = !!settings?.internet_enabled
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label="Web access"
      title={on ? 'Web access is on: the model may search the web and read pages. Click to turn off.' : 'Web access is off: everything stays on this computer. Click to let the model search the web when it needs to.'}
      disabled={!settings}
      onClick={async () => {
        await saveSettings({ internet_enabled: !on })
        toast(on ? 'Web access off — everything stays local.' : 'Web access on — the model can search the web when it needs to.', 'info')
      }}
      className={cx(
        'flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-colors',
        on ? 'border-accent/40 bg-accent/12 text-accent' : 'border-line text-muted hover:text-fg',
      )}
    >
      <IconGlobe size={13} /> Web
    </button>
  )
}
