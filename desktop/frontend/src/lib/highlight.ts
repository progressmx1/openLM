import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import cpp from 'highlight.js/lib/languages/cpp'
import css from 'highlight.js/lib/languages/css'
import go from 'highlight.js/lib/languages/go'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import powershell from 'highlight.js/lib/languages/powershell'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'
import ini from 'highlight.js/lib/languages/ini'
import diff from 'highlight.js/lib/languages/diff'

const languages: Record<string, unknown> = {
  bash, cpp, css, go, java, javascript, json, markdown, powershell, python, rust, sql, typescript, xml, yaml, ini, diff,
}
for (const [name, lang] of Object.entries(languages)) hljs.registerLanguage(name, lang as never)

const BY_EXT: Record<string, string> = {
  py: 'python', js: 'javascript', mjs: 'javascript', cjs: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
  json: 'json', md: 'markdown', ps1: 'powershell', psm1: 'powershell', sh: 'bash', bash: 'bash', c: 'cpp', h: 'cpp',
  cpp: 'cpp', hpp: 'cpp', cc: 'cpp', css: 'css', scss: 'css', go: 'go', java: 'java', rs: 'rust', sql: 'sql',
  html: 'xml', xml: 'xml', svg: 'xml', yml: 'yaml', yaml: 'yaml', toml: 'ini', ini: 'ini', cfg: 'ini', diff: 'diff', patch: 'diff',
}

export function languageFor(path: string | undefined): string | undefined {
  if (!path) return undefined
  const ext = path.split('.').pop()?.toLowerCase() ?? ''
  return BY_EXT[ext]
}

const escape = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** Highlight one line (diff lines are highlighted individually; multi-line state is not kept). */
export function highlightLine(line: string, language: string | undefined): string {
  if (!language || !hljs.getLanguage(language)) return escape(line)
  try {
    return hljs.highlight(line, { language, ignoreIllegals: true }).value
  } catch {
    return escape(line)
  }
}

export function highlightCode(code: string, language: string | undefined): string {
  if (!language || !hljs.getLanguage(language)) return escape(code)
  try {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value
  } catch {
    return escape(code)
  }
}

/** Split highlighted HTML into lines, closing and re-opening spans that cross line breaks. */
export function splitHighlightedLines(html: string): string[] {
  const lines: string[] = []
  const open: string[] = []
  let current = ''
  const re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+|<)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(html))) {
    if (m[1]) {
      open.push(m[1])
      current += m[1]
    } else if (m[2]) {
      open.pop()
      current += m[2]
    } else if (m[3]) {
      lines.push(current + '</span>'.repeat(open.length))
      current = open.join('')
    } else {
      current += m[4]
    }
  }
  lines.push(current + '</span>'.repeat(open.length))
  return lines
}

export { hljs }
