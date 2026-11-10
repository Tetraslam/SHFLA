import { autocompletion, type CompletionContext } from '@codemirror/autocomplete'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import { HighlightStyle, StreamLanguage, syntaxHighlighting } from '@codemirror/language'
import { type Diagnostic, forceLinting, linter, lintGutter } from '@codemirror/lint'
import { EditorState } from '@codemirror/state'
import { drawSelection, EditorView, highlightActiveLine, keymap } from '@codemirror/view'
import { tags } from '@lezer/highlight'
import { useEffect, useRef } from 'react'
import { FUNCTIONS, OUTPUTS, VARIABLES } from '@/lang/builtins'
import type { Diagnostic as ShflaDiagnostic } from '@/lang/compile'

const MUSIC = new Set(VARIABLES.filter((v) => v.group === 'music').map((v) => v.name))
const BUILTIN = new Set(VARIABLES.filter((v) => v.group !== 'music').map((v) => v.name))
const FNS = new Set(FUNCTIONS.map((f) => f.name))
const OUTS = new Set<string>(OUTPUTS.map((o) => o.name))
const KEYWORDS = new Set(['and', 'or', 'not', 'if', 'then', 'else'])

/** Token classes: music inputs get the accent, because that is where the song touches the program. */
const shfla = StreamLanguage.define({
  token(stream) {
    if (stream.eatSpace()) return null
    if (stream.peek() === '#') {
      stream.skipToEnd()
      return 'comment'
    }
    if (stream.match(/^\d*\.?\d+/)) return 'number'
    const word = stream.match(/^[A-Za-z_][A-Za-z0-9_]*/) as RegExpMatchArray | null
    if (word) {
      const w = word[0]
      if (KEYWORDS.has(w)) return 'keyword'
      if (MUSIC.has(w)) return 'special'
      if (OUTS.has(w)) return 'definition'
      if (BUILTIN.has(w)) return 'atom'
      if (FNS.has(w)) return 'function'
      return 'variableName'
    }
    stream.next()
    return 'operator'
  },
  tokenTable: {
    special: tags.special(tags.variableName),
    definition: tags.definition(tags.variableName),
    function: tags.function(tags.variableName),
  },
})

const highlight = HighlightStyle.define([
  { tag: tags.comment, color: 'rgb(243 236 230 / 0.36)' },
  { tag: tags.number, color: '#f6c6d4' },
  { tag: tags.keyword, color: 'rgb(243 236 230 / 0.55)' },
  { tag: tags.special(tags.variableName), color: 'var(--sodium)' },
  { tag: tags.definition(tags.variableName), color: '#ffffff', fontWeight: '700' },
  { tag: tags.atom, color: '#cfc4ff' },
  { tag: tags.function(tags.variableName), color: '#a9e3e8' },
  { tag: tags.operator, color: 'rgb(243 236 230 / 0.5)' },
  { tag: tags.variableName, color: 'var(--ink)' },
])

const theme = EditorView.theme(
  {
    '&': { backgroundColor: 'transparent', color: 'var(--ink)', fontSize: '13px', height: '100%' },
    '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.65', overflow: 'auto' },
    '.cm-content': { caretColor: 'var(--ink)', padding: '12px 0' },
    '.cm-line': { padding: '0 20px 0 6px' },
    '&.cm-focused': { outline: 'none' },
    '.cm-cursor': { borderLeftColor: 'var(--ink)', borderLeftWidth: '2px' },
    '.cm-activeLine': { backgroundColor: 'rgb(243 236 230 / 0.035)' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': {
      backgroundColor: 'rgb(243 236 230 / 0.16) !important',
    },
    '.cm-gutters': { backgroundColor: 'transparent', border: 'none' },
    '.cm-lint-marker': { width: '0.7em', height: '0.7em' },
    '.cm-diagnostic-error': { borderLeft: '2px solid var(--err)', color: 'var(--ink)' },
    '.cm-lintRange-error': {
      backgroundImage: 'none',
      textDecoration: 'underline wavy var(--err)',
      textUnderlineOffset: '3px',
    },
    '.cm-tooltip': {
      backgroundColor: 'rgb(14 12 18 / 0.96)',
      border: '1px solid rgb(243 236 230 / 0.1)',
      borderRadius: '10px',
      overflow: 'hidden',
    },
    '.cm-tooltip-autocomplete > ul': { fontFamily: 'var(--font-mono)', fontSize: '12px' },
    '.cm-tooltip-autocomplete > ul > li[aria-selected]': {
      backgroundColor: 'rgb(243 236 230 / 0.12)',
      color: 'var(--ink)',
    },
    '.cm-completionDetail': { color: 'rgb(243 236 230 / 0.5)', fontStyle: 'normal', marginLeft: '0.8em' },
  },
  { dark: true },
)

const completions = [
  ...VARIABLES.map((v) => ({ label: v.name, type: v.group === 'music' ? 'constant' : 'variable', detail: v.doc })),
  ...FUNCTIONS.map((f) => ({ label: f.name, type: 'function', detail: f.sig, info: f.doc, apply: `${f.name}(` })),
  ...OUTPUTS.map((o) => ({ label: o.name, type: 'keyword', detail: o.doc })),
]

function complete(ctx: CompletionContext) {
  const word = ctx.matchBefore(/[A-Za-z_][A-Za-z0-9_]*/)
  if (!word || (word.from === word.to && !ctx.explicit)) return null
  const line = ctx.state.doc.lineAt(ctx.pos)
  if (line.text.slice(0, ctx.pos - line.from).includes('#')) return null
  return { from: word.from, options: completions, validFor: /^[A-Za-z0-9_]*$/ }
}

interface Props {
  value: string
  onChange: (v: string) => void
  errors: ShflaDiagnostic[]
  label: string
}

export function Editor({ value, onChange, errors, label }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const errorsRef = useRef(errors)
  errorsRef.current = errors
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  // biome-ignore lint/correctness/useExhaustiveDependencies: the view is created once; value syncs below
  useEffect(() => {
    if (!host.current) return
    const lint = linter(
      (v): Diagnostic[] =>
        errorsRef.current.map((e) => {
          const len = v.state.doc.length
          const from = Math.min(e.from, len)
          return { from, to: Math.min(Math.max(e.to, from), len), severity: 'error', message: e.message }
        }),
      { delay: 120 },
    )
    view.current = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          history(),
          drawSelection(),
          highlightActiveLine(),
          keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
          shfla,
          syntaxHighlighting(highlight),
          theme,
          lint,
          lintGutter(),
          autocompletion({ override: [complete], icons: false }),
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ 'aria-label': label, spellcheck: 'false' }),
          EditorView.updateListener.of((u) => {
            if (u.docChanged) onChangeRef.current(u.state.doc.toString())
          }),
        ],
      }),
    })
    return () => view.current?.destroy()
  }, [])

  // outside changes (a preset, a shared link) replace the document
  useEffect(() => {
    const v = view.current
    if (v && v.state.doc.toString() !== value) {
      v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } })
    }
  }, [value])

  // re-run the linter when diagnostics change
  // biome-ignore lint/correctness/useExhaustiveDependencies: errors is the trigger
  useEffect(() => {
    const v = view.current
    if (v) forceLinting(v)
  }, [errors])

  return <div ref={host} className="h-full min-h-0" />
}
