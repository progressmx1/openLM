export interface DiffLine {
  type: 'add' | 'del' | 'ctx' | 'meta'
  text: string
  oldNo?: number
  newNo?: number
}

export interface DiffHunk {
  header: string
  lines: DiffLine[]
}

/** Parse a unified diff into hunks with old/new line numbers. */
export function parseUnifiedDiff(diff: string): DiffHunk[] {
  const hunks: DiffHunk[] = []
  let hunk: DiffHunk | null = null
  let oldNo = 0
  let newNo = 0
  for (const raw of diff.split('\n')) {
    if (raw.startsWith('--- ') || raw.startsWith('+++ ')) continue
    const header = raw.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/)
    if (header) {
      hunk = { header: raw, lines: [] }
      hunks.push(hunk)
      oldNo = Number(header[1])
      newNo = Number(header[2])
      continue
    }
    if (!hunk) continue
    if (raw.startsWith('+')) hunk.lines.push({ type: 'add', text: raw.slice(1), newNo: newNo++ })
    else if (raw.startsWith('-')) hunk.lines.push({ type: 'del', text: raw.slice(1), oldNo: oldNo++ })
    else if (raw.startsWith('\\')) hunk.lines.push({ type: 'meta', text: raw })
    else if (raw.length || hunk.lines.length) hunk.lines.push({ type: 'ctx', text: raw.slice(1), oldNo: oldNo++, newNo: newNo++ })
  }
  // Drop a trailing empty context line produced by the final newline.
  for (const h of hunks) {
    const tail = h.lines[h.lines.length - 1]
    if (tail && tail.type === 'ctx' && tail.text === '') h.lines.pop()
  }
  return hunks
}
