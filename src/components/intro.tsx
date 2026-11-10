import { ArrowRightIcon } from '@phosphor-icons/react'
import { useEffect, useRef, useState } from 'react'
import type { SourceKind } from '@/audio/engine'

interface Props {
  initialLinkOpen?: boolean
  busy: SourceKind | null
  status: string | null
  error: { message: string; youtube?: string } | null
  onDemo: () => void
  onFile: () => void
  onLink: (url: string) => void
  onMic: () => void
  /** only where the browser can share a tab's sound */
  onTab?: () => void
}

const choice =
  'group relative text-left text-ink transition-colors duration-150 ease-[var(--ease-tick)] hover:text-white focus-visible:outline-none focus-visible:text-white disabled:opacity-50'

function Underline({ active }: { active?: boolean }) {
  return (
    <span
      aria-hidden
      className={`absolute -bottom-1 left-0 h-px w-full origin-left bg-ink transition-transform duration-300 ease-[var(--ease-focus)] group-hover:scale-x-100 group-focus-visible:scale-x-100 ${active ? 'scale-x-100' : 'scale-x-0'}`}
    />
  )
}

/** The first screen: the fractal is already running behind this; the title pulls into focus over it. */
export function Intro({ initialLinkOpen = false, busy, status, error, onDemo, onFile, onLink, onMic, onTab }: Props) {
  const [linkOpen, setLinkOpen] = useState(initialLinkOpen)
  const [link, setLink] = useState('')
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (linkOpen) input.current?.focus()
  }, [linkOpen])

  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col justify-end px-6 pb-28 sm:px-12 sm:pb-32 lg:px-20">
      {/* a soft pool of dark under the words, so they read over any part of the fractal */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 60% at 12% 82%, rgb(7 6 10 / 0.72), rgb(7 6 10 / 0.35) 55%, transparent 80%)',
        }}
      />
      <div
        aria-hidden
        className="absolute inset-x-0 bottom-0 h-[75%] bg-gradient-to-t from-ground/85 via-ground/55 to-transparent sm:hidden"
      />
      <div className="pointer-events-auto relative max-w-3xl">
        <h1 className="focus-in text-glow font-display text-[clamp(5.5rem,19vw,15rem)] leading-[0.8] tracking-[-0.03em] text-ink">
          shfla
        </h1>
        <p
          className="focus-in text-glow mt-6 max-w-[34rem] text-[17px] leading-relaxed text-ink/80"
          style={{ animationDelay: '140ms' }}
        >
          a song grows a fractal. every octave of the music moves its own scale of it, the beat carries you deeper, and
          the whole thing is a program you can rewrite while it plays.
        </p>
        <div
          className="focus-in text-glow mt-9 flex flex-wrap items-baseline gap-x-8 gap-y-3 text-[19px]"
          style={{ animationDelay: '280ms' }}
        >
          <button type="button" className={choice} onClick={onDemo} disabled={busy !== null}>
            {busy === 'demo' ? 'starting…' : 'play the demo'}
            <Underline />
          </button>
          <button type="button" className={choice} onClick={onFile} disabled={busy !== null}>
            open a song
            <Underline />
          </button>
          <button
            type="button"
            className={choice}
            onClick={() => setLinkOpen((v) => !v)}
            disabled={busy !== null && busy !== 'link'}
            aria-expanded={linkOpen}
          >
            paste a link
            <Underline active={linkOpen} />
          </button>
          <button type="button" className={choice} onClick={onMic} disabled={busy !== null}>
            {busy === 'mic' ? 'listening…' : 'use the mic'}
            <Underline />
          </button>
          {onTab && (
            <button type="button" className={choice} onClick={onTab} disabled={busy !== null}>
              {busy === 'tab' ? 'pick a tab…' : 'listen to a tab'}
              <Underline />
            </button>
          )}
        </div>

        {/* the link field grows in place; nothing below it moves because nothing is below it */}
        <div
          className="grid transition-[grid-template-rows,opacity] duration-500 ease-[var(--ease-focus)]"
          style={{ gridTemplateRows: linkOpen ? '1fr' : '0fr', opacity: linkOpen ? 1 : 0 }}
        >
          <div className="overflow-hidden" inert={!linkOpen}>
            <form
              className="mt-5 flex max-w-xl items-center gap-2 rounded-full border border-ink/15 bg-ground/80 py-1 pr-1 pl-5 focus-within:border-ink/40"
              onSubmit={(e) => {
                e.preventDefault()
                if (link.trim()) onLink(link)
              }}
            >
              <input
                ref={input}
                value={link}
                onChange={(e) => setLink(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    setLinkOpen(false)
                    e.currentTarget.blur()
                  }
                }}
                placeholder="a youtube link, or a direct link to an audio file"
                className="h-9 min-w-0 flex-1 bg-transparent text-[15px] text-ink placeholder:text-ink-faint focus:outline-none"
                aria-label="audio link"
                inputMode="url"
                spellCheck={false}
              />
              <button
                type="submit"
                disabled={!link.trim() || busy === 'link'}
                className="flex size-9 items-center justify-center rounded-full bg-ink text-ground transition-opacity hover:opacity-90 disabled:opacity-30"
                aria-label="play link"
              >
                <ArrowRightIcon weight="bold" className="size-4" />
              </button>
            </form>
          </div>
        </div>

        <p aria-live="polite" className="text-glow mt-5 min-h-[1.5em] text-[14px]">
          {error ? (
            <span className="text-err">{error.message}</span>
          ) : status ? (
            <span className="text-ink-muted">{status}…</span>
          ) : (
            <span className="text-ink-faint">or drop a song anywhere</span>
          )}
        </p>

        {/* youtube refused the server: play it in a tab, then listen to that tab */}
        {error?.youtube && onTab && (
          <ol className="focus-in text-glow mt-1 flex flex-wrap items-baseline gap-x-6 gap-y-2 text-[16px]">
            <li>
              <span className="mr-2 font-mono text-[11px] text-ink-faint">1</span>
              <a href={error.youtube} target="_blank" rel="noopener noreferrer" className={choice}>
                open the video in a new tab
                <Underline />
              </a>
            </li>
            <li>
              <span className="mr-2 font-mono text-[11px] text-ink-faint">2</span>
              <button type="button" className={choice} onClick={onTab} disabled={busy !== null}>
                listen to that tab
                <Underline />
              </button>
            </li>
          </ol>
        )}
      </div>
    </div>
  )
}
