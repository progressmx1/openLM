import { memo, useMemo } from 'react'
import { parseUnifiedDiff } from '../lib/diff'
import { highlightLine, languageFor } from '../lib/highlight'

/** Unified diff with old/new line numbers and per-line syntax highlighting. */
export const DiffView = memo(function DiffView({ diff, path, maxHeight = 420 }: { diff: string; path?: string; maxHeight?: number }) {
  const hunks = useMemo(() => parseUnifiedDiff(diff), [diff])
  const language = languageFor(path)
  if (!hunks.length) return <div className="px-3 py-2 text-xs text-muted">No changes.</div>
  return (
    <div className="overflow-auto font-mono text-[12px] leading-[1.55]" style={{ maxHeight }}>
      <table className="w-full border-collapse">
        <tbody>
          {hunks.map((hunk, h) => (
            <HunkRows key={h} header={hunk.header} lines={hunk.lines} language={language} />
          ))}
        </tbody>
      </table>
    </div>
  )
})

function HunkRows({ header, lines, language }: { header: string; lines: ReturnType<typeof parseUnifiedDiff>[number]['lines']; language?: string }) {
  return (
    <>
      <tr>
        <td colSpan={4} className="select-none bg-info/8 px-3 py-0.5 text-2xs text-info/90">
          {header}
        </td>
      </tr>
      {lines.map((line, i) => {
        if (line.type === 'meta') {
          return (
            <tr key={i}>
              <td colSpan={4} className="px-3 text-2xs italic text-faint">
                {line.text}
              </td>
            </tr>
          )
        }
        const cls = line.type === 'add' ? 'diff-add' : line.type === 'del' ? 'diff-del' : ''
        const sign = line.type === 'add' ? '+' : line.type === 'del' ? '−' : ' '
        return (
          <tr key={i} className={cls}>
            <td className="w-10 select-none border-r border-line/60 pr-2 text-right align-top text-faint">{line.oldNo ?? ''}</td>
            <td className="w-10 select-none border-r border-line/60 pr-2 text-right align-top text-faint">{line.newNo ?? ''}</td>
            <td className="diff-sign w-5 select-none text-center align-top">{sign}</td>
            <td className="whitespace-pre pr-4 align-top" dangerouslySetInnerHTML={{ __html: highlightLine(line.text, language) || '&nbsp;' }} />
          </tr>
        )
      })}
    </>
  )
}
