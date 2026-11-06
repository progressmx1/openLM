import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeHighlight from 'rehype-highlight'
import 'katex/dist/katex.min.css'

/** DeepSeek-R1 style \[ \] / \( \) math → remark-math's $$ / $. */
function normalizeLatex(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, body: string) => `$$${body.trim()}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, body: string) => `$${body.trim()}$`)
}

export const Markdown = memo(function Markdown({ content, className }: { content: string; className?: string }) {
  return (
    <div className={`md ${className ?? ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, [rehypeHighlight, { detect: false, ignoreMissing: true }]]}
        components={{ a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}
      >
        {normalizeLatex(content)}
      </ReactMarkdown>
    </div>
  )
})
