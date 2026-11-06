import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function make(paths: React.ReactNode, viewBox = '0 0 24 24') {
  return function Icon({ size = 16, ...props }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox={viewBox}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        {...props}
      >
        {paths}
      </svg>
    )
  }
}

export const IconFolder = make(<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2h9A1.5 1.5 0 0 1 21 9.5v8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z" />)
export const IconFile = make(<><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>)
export const IconPlus = make(<path d="M12 5v14M5 12h14" />)
export const IconSend = make(<path d="M5 12h14M13 6l6 6-6 6" />)
export const IconStop = make(<rect x="6.5" y="6.5" width="11" height="11" rx="2" />)
export const IconChevron = make(<path d="m9 6 6 6-6 6" />)
export const IconChevronDown = make(<path d="m6 9 6 6 6-6" />)
export const IconCheck = make(<path d="m5 12.5 4.5 4.5L19 7.5" />)
export const IconX = make(<path d="M6 6l12 12M18 6 6 18" />)
export const IconTerminal = make(<><rect x="3" y="4.5" width="18" height="15" rx="2" /><path d="m7 9 3 3-3 3M12.5 15H17" /></>)
export const IconEdit = make(<><path d="M4 20h4l10.5-10.5a2.1 2.1 0 0 0-4-4L4 16z" /><path d="m13.5 6.5 4 4" /></>)
export const IconSearch = make(<><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></>)
export const IconGit = make(<><circle cx="6" cy="6" r="2.2" /><circle cx="6" cy="18" r="2.2" /><circle cx="18" cy="9" r="2.2" /><path d="M6 8.2v7.6M18 11.2c0 3-3 3.8-6 4.3-2 .3-3.5.9-4.5 1.5" /></>)
export const IconGlobe = make(<><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></>)
export const IconList = make(<path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01" />)
export const IconCpu = make(<><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9.5 9.5h5v5h-5zM9 2.5v3M15 2.5v3M9 18.5v3M15 18.5v3M2.5 9h3M2.5 15h3M18.5 9h3M18.5 15h3" /></>)
export const IconGpu = make(<><rect x="2.5" y="6.5" width="19" height="11" rx="2" /><circle cx="9" cy="12" r="2.5" /><path d="M14.5 10h4M14.5 14h4M5 17.5v2M19 17.5v2" /></>)
export const IconSettings = make(<><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></>)
export const IconSun = make(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>)
export const IconMoon = make(<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />)
export const IconSidebar = make(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></>)
export const IconPanelRight = make(<><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></>)
export const IconUndo = make(<><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></>)
export const IconBrain = make(<><path d="M9 4.5a3 3 0 0 0-3 3v.2A3 3 0 0 0 4 10.5a3 3 0 0 0 1 2.2A3 3 0 0 0 6 18a3 3 0 0 0 3 2 2.5 2.5 0 0 0 3-2.4V6.5A2.5 2.5 0 0 0 9 4.5z" /><path d="M15 4.5a3 3 0 0 1 3 3v.2a3 3 0 0 1 2 2.8 3 3 0 0 1-1 2.2A3 3 0 0 1 18 18a3 3 0 0 1-3 2 2.5 2.5 0 0 1-3-2.4" /></>)
export const IconWarning = make(<><path d="M12 3.5 2.5 20h19z" /><path d="M12 10v4.5M12 17.5h.01" /></>)
export const IconInfo = make(<><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>)
export const IconRefresh = make(<><path d="M20 11a8 8 0 0 0-14.3-4.9L4 8" /><path d="M4 4v4h4M4 13a8 8 0 0 0 14.3 4.9L20 16" /><path d="M20 20v-4h-4" /></>)
export const IconTrash = make(<><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></>)
export const IconCommand = make(<path d="M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3z" />)
export const IconSpark = make(<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />)
export const IconLog = make(<><path d="M5 4h14v16H5z" /><path d="M8.5 8.5h7M8.5 12h7M8.5 15.5h4" /></>)
export const IconPlay = make(<path d="M7 5v14l11-7z" />)
export const IconEject = make(<><path d="M5 15h14L12 6z" /><path d="M5 19h14" /></>)
