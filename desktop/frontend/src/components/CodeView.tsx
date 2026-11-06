import { memo, useMemo } from 'react'
import { highlightCode, languageFor, splitHighlightedLines } from '../lib/highlight'

export const CodeView = memo(function CodeView({ code, path, language }: { code: string; path?: string; language?: string }) {
  const lang = language ?? languageFor(path)
  const lines = useMemo(() => splitHighlightedLines(highlightCode(code, lang)), [code, lang])
  return (
    <div className="overflow-auto font-mono text-[12px] leading-[1.55]">
      <table className="border-collapse">
        <tbody>
          {lines.map((html, i) => (
            <tr key={i}>
              <td className="w-10 select-none pr-3 text-right align-top text-faint">{i + 1}</td>
              <td className="whitespace-pre pr-4" dangerouslySetInnerHTML={{ __html: html || '&nbsp;' }} />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
})
