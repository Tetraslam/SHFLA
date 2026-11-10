import { ArrowCounterClockwiseIcon, CaretDownIcon, LinkIcon, XIcon } from '@phosphor-icons/react'
import { useState } from 'react'
import { FUNCTIONS, OUTPUTS, VARIABLES } from '@/lang/builtins'
import type { Diagnostic } from '@/lang/compile'
import { PRESETS } from '@/lang/presets'
import { Editor } from './editor'

interface Props {
  code: string
  presetId: string | null
  basedOn: string
  errors: Diagnostic[]
  gpuError: string | null
  onCode: (v: string) => void
  onPreset: (id: string) => void
  onReseed: () => void
  onShare: () => void
  onClose: () => void
}

const GROUPS: Array<{ id: (typeof VARIABLES)[number]['group']; label: string }> = [
  { id: 'cell', label: 'the cell' },
  { id: 'scale', label: 'the scales' },
  { id: 'time', label: 'time' },
  { id: 'music', label: 'the song' },
]

function lineOf(src: string, offset: number) {
  return src.slice(0, offset).split('\n').length
}

export function LanguagePanel(p: Props) {
  const [refOpen, setRefOpen] = useState(false)
  const preset = PRESETS.find((x) => x.id === p.presetId)
  const err = p.errors[0]

  return (
    <aside
      className="focus-in scrim-side absolute top-0 right-0 bottom-16 z-20 flex w-full flex-col border-b border-l border-ink/[0.07] sm:w-[480px] sm:rounded-bl-[14px]"
      aria-label="the language"
    >
      <header className="flex items-start justify-between gap-4 px-6 pt-6 pb-2">
        <div className="min-w-0">
          <h2 className="font-display text-[40px] leading-none text-ink">{preset ? preset.name : 'your rule'}</h2>
          <p className="mt-2 truncate text-[13px] text-ink-muted">
            {preset ? preset.line : `rewritten from ${p.basedOn}. it runs as you type`}
          </p>
        </div>
        <button
          type="button"
          onClick={p.onClose}
          className="-mr-2 flex size-9 shrink-0 items-center justify-center rounded-full text-ink/70 hover:bg-ink/10 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/50"
          aria-label="close the language"
        >
          <XIcon className="size-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1">
        <Editor value={p.code} onChange={p.onCode} errors={p.errors} label="shfla program" />
      </div>

      {/* status: one line, fixed height, so the editor never jumps */}
      <div
        className="flex h-10 items-center gap-2 border-t border-ink/[0.07] px-6 font-mono text-[11.5px]"
        aria-live="polite"
      >
        {err ? (
          <>
            <span className="size-1.5 shrink-0 rounded-full bg-err" />
            <span className="truncate text-err">
              line {lineOf(p.code, err.from)}: {err.message}
            </span>
          </>
        ) : p.gpuError ? (
          <>
            <span className="size-1.5 shrink-0 rounded-full bg-err" />
            <span className="truncate text-err">the gpu refused it: {p.gpuError.split('\n')[0]}</span>
          </>
        ) : (
          <>
            <span className="size-1.5 shrink-0 rounded-full bg-ink/70" />
            <span className="truncate text-ink-muted">running on six scales. edits apply as you type</span>
          </>
        )}
      </div>

      <div className="border-t border-ink/[0.07] px-6 pt-4 pb-5">
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {PRESETS.map((x) => (
            <button
              type="button"
              key={x.id}
              onClick={() => p.onPreset(x.id)}
              title={x.line}
              className={`font-display text-[22px] leading-tight transition-colors focus-visible:outline-none focus-visible:underline ${x.id === p.presetId ? 'text-ink underline decoration-ink/60 decoration-1 underline-offset-[6px]' : 'text-ink/50 hover:text-ink'}`}
            >
              {x.name}
            </button>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-1 text-[13px]">
          <button
            type="button"
            onClick={() => setRefOpen((v) => !v)}
            aria-expanded={refOpen}
            className="-ml-2 flex items-center gap-1.5 rounded-full px-2 py-1 text-ink-muted hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/50"
          >
            <CaretDownIcon
              className={`size-3.5 transition-transform duration-300 ease-[var(--ease-focus)] ${refOpen ? '' : '-rotate-90'}`}
            />
            what a rule can say
          </button>
          <span className="flex-1" />
          <button
            type="button"
            onClick={p.onReseed}
            className="flex items-center gap-1.5 rounded-full px-2 py-1 text-ink-muted hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/50"
          >
            <ArrowCounterClockwiseIcon className="size-3.5" /> reseed
            <kbd className="font-mono text-[10px] opacity-60">r</kbd>
          </button>
          <button
            type="button"
            onClick={p.onShare}
            className="flex items-center gap-1.5 rounded-full px-2 py-1 text-ink-muted hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/50"
          >
            <LinkIcon className="size-3.5" /> copy link
          </button>
        </div>

        <div
          className="grid transition-[grid-template-rows,opacity] duration-500 ease-[var(--ease-focus)]"
          style={{ gridTemplateRows: refOpen ? '1fr' : '0fr', opacity: refOpen ? 1 : 0 }}
        >
          <div className="overflow-hidden">
            <div className="mt-3 max-h-[38vh] space-y-4 overflow-y-auto pr-2 text-[12.5px] leading-relaxed">
              <p className="text-ink-muted">
                one statement per line: <code className="font-mono text-ink">name = expression</code>. every cell at
                every scale runs the whole program each step. true is 1, false is 0, anything from 0.5 up counts as
                true, and <code className="font-mono text-ink">==</code> allows 0.02 of slack.
              </p>
              <dl className="space-y-1">
                {OUTPUTS.map((o) => (
                  <div key={o.name} className="grid grid-cols-[4.5rem_1fr] gap-2">
                    <dt className="font-mono font-bold text-white">{o.name}</dt>
                    <dd className="text-ink-muted">{o.doc}</dd>
                  </div>
                ))}
              </dl>
              {GROUPS.map((g) => (
                <div key={g.id}>
                  <h3 className="mb-1 text-[11px] tracking-wide text-ink-faint uppercase">{g.label}</h3>
                  <dl className="space-y-1">
                    {VARIABLES.filter((v) => v.group === g.id).map((v) => (
                      <div key={v.name} className="grid grid-cols-[4.5rem_1fr] gap-2">
                        <dt className={`font-mono ${g.id === 'music' ? 'text-sodium' : 'text-[#cfc4ff]'}`}>{v.name}</dt>
                        <dd className="text-ink-muted">{v.doc}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
              <div>
                <h3 className="mb-1 text-[11px] tracking-wide text-ink-faint uppercase">functions</h3>
                <dl className="space-y-1">
                  {FUNCTIONS.map((f) => (
                    <div key={f.name} className="grid grid-cols-[8.5rem_1fr] gap-2">
                      <dt className="font-mono text-[#a9e3e8]">{f.sig}</dt>
                      <dd className="text-ink-muted">{f.doc}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              <p className="text-ink-muted">
                the language can run conway's life, and life is turing-complete, so a shfla program can in principle
                compute anything. the browser gives it a finite grid, so in practice it is a very large finite machine.
              </p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
