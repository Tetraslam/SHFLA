import { CodeIcon, CommandIcon, PauseIcon, PlayIcon, SlidersHorizontalIcon, XIcon } from '@phosphor-icons/react'
import type { RefObject } from 'react'
import type { SourceState } from '@/audio/engine'
import { LEVELS } from '@/audio/frame'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { PALETTES } from '@/gpu/palettes'

export const BAND_NAMES = ['sub', 'bass', 'low', 'mid', 'high', 'air']

export interface MeterRefs {
  band: (HTMLElement | null)[]
  act: (HTMLElement | null)[]
  time: HTMLElement | null
  depth: HTMLElement | null
}

interface Props {
  source: SourceState | null
  meters: RefObject<MeterRefs>
  panelOpen: boolean
  paletteId: string
  voice: number
  speed: number
  zoom: number
  onToggle: () => void
  onStop: () => void
  onPanel: () => void
  onCommand: () => void
  onPalette: (id: string) => void
  onVoice: (v: number) => void
  onSpeed: (v: number) => void
  onZoom: (v: number) => void
}

const iconButton =
  'flex size-9 items-center justify-center rounded-full text-ink/75 transition-colors duration-150 hover:bg-ink/10 hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink/50 aria-pressed:bg-ink/10 aria-pressed:text-ink'

function Tip({ label, keys, children }: { label: string; keys?: string; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent side="top" className="flex items-center gap-2">
        {label}
        {keys && <kbd className="font-mono text-[11px] opacity-60">{keys}</kbd>}
      </TooltipContent>
    </Tooltip>
  )
}

function Row({ label, value, children }: { label: string; value: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-2">
      <div className="flex items-baseline justify-between text-[13px]">
        <span className="text-ink">{label}</span>
        <span className="font-mono text-[11px] text-ink-muted tabular-nums">{value}</span>
      </div>
      {children}
    </div>
  )
}

export function Transport(p: Props) {
  const s = p.source
  return (
    <div className="scrim-bottom pointer-events-none absolute inset-x-0 bottom-0 h-28">
      <div className="pointer-events-auto absolute inset-x-0 bottom-0 flex h-16 items-center gap-4 px-4 sm:gap-5 sm:px-6">
        {/* one column per level: the octave of the song (fill) and how busy that scale is computing (dot) */}
        <Tip label="six scales, six octaves: sub on the left, air on the right">
          <div className="flex h-8 items-end gap-[5px]" role="img" aria-label="octave meters">
            {Array.from({ length: LEVELS }, (_, k) => (
              <div key={BAND_NAMES[k]} className="relative flex h-full w-[6px] flex-col items-center justify-end">
                <div className="absolute inset-0 rounded-full bg-ink/10" />
                <div
                  ref={(el) => {
                    p.meters.current.band[k] = el
                  }}
                  className="relative w-full origin-bottom rounded-full bg-sodium"
                  style={{ height: '100%', transform: 'scaleY(0)' }}
                />
                <div
                  ref={(el) => {
                    p.meters.current.act[k] = el
                  }}
                  className="absolute -top-2 size-[4px] rounded-full bg-ink"
                  style={{ opacity: 0 }}
                />
              </div>
            ))}
          </div>
        </Tip>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[14px] leading-tight text-ink">
            {s?.playing && <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-sodium" />}
            <span className="truncate">
              {s ? s.title : <span className="text-ink-muted">no song yet. it's breathing on its own</span>}
            </span>
          </div>
          <div className="mt-0.5 flex items-center gap-2 font-mono text-[11px] leading-tight text-ink-muted tabular-nums">
            {s && <span>{s.kind === 'mic' ? 'live' : s.kind}</span>}
            {s && s.kind !== 'mic' && s.kind !== 'tab' && (
              <span
                ref={(el) => {
                  p.meters.current.time = el
                }}
              />
            )}
            <span
              className="hidden sm:inline"
              ref={(el) => {
                p.meters.current.depth = el
              }}
            />
          </div>
        </div>

        <div className="flex items-center gap-1">
          {s && (
            <>
              <Tip label={s.playing ? 'pause' : 'play'} keys="space">
                <button
                  type="button"
                  className={iconButton}
                  onClick={p.onToggle}
                  aria-label={s.playing ? 'pause' : 'play'}
                >
                  {s.playing ? (
                    <PauseIcon weight="fill" className="size-4" />
                  ) : (
                    <PlayIcon weight="fill" className="size-4" />
                  )}
                </button>
              </Tip>
              <Tip label="let go of the song">
                <button
                  type="button"
                  className={iconButton}
                  onClick={p.onStop}
                  aria-label="stop and choose another song"
                >
                  <XIcon className="size-4" />
                </button>
              </Tip>
            </>
          )}

          <Popover>
            <Tip label="tune">
              <PopoverTrigger className={iconButton} aria-label="tune">
                <SlidersHorizontalIcon className="size-[18px]" />
              </PopoverTrigger>
            </Tip>
            <PopoverContent side="top" align="end" sideOffset={10} className="w-72 gap-5 p-5">
              <div className="grid gap-2">
                <span className="text-[13px] text-ink">palette</span>
                <div className="flex gap-2">
                  {PALETTES.map((pal) => {
                    const [a, b, c] = pal.stops.map((rgb) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(' ')})`)
                    const active = pal.id === p.paletteId
                    return (
                      <button
                        type="button"
                        key={pal.id}
                        onClick={() => p.onPalette(pal.id)}
                        aria-pressed={active}
                        className="group grid flex-1 gap-1.5 text-left focus-visible:outline-none"
                      >
                        <span
                          className={`h-8 rounded-md ring-offset-2 ring-offset-popover transition-shadow group-focus-visible:ring-2 group-focus-visible:ring-ink/50 ${active ? 'ring-1 ring-ink/70' : ''}`}
                          style={{ background: `linear-gradient(135deg, ${a}, ${b} 55%, ${c})` }}
                        />
                        <span className={`text-[11px] ${active ? 'text-ink' : 'text-ink-muted'}`}>{pal.name}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
              <Row label="the fractal's voice" value={p.voice === 0 ? 'quiet' : `${Math.round(p.voice * 100)}%`}>
                <Slider value={[p.voice]} min={0} max={1} step={0.01} onValueChange={(v) => p.onVoice(first(v))} />
              </Row>
              <Row label="step clock" value={`${p.speed.toFixed(2)}×`}>
                <Slider value={[p.speed]} min={0.25} max={3} step={0.05} onValueChange={(v) => p.onSpeed(first(v))} />
              </Row>
              <Row label="zoom" value={p.zoom === 0 ? 'still' : `${p.zoom.toFixed(2)}×`}>
                <Slider value={[p.zoom]} min={0} max={3} step={0.05} onValueChange={(v) => p.onZoom(first(v))} />
              </Row>
            </PopoverContent>
          </Popover>

          <Tip label="the language" keys="e">
            <button
              type="button"
              className={iconButton}
              onClick={p.onPanel}
              aria-pressed={p.panelOpen}
              aria-label="the language"
            >
              <CodeIcon className="size-[18px]" />
            </button>
          </Tip>
          <Tip label="everything" keys="⌘k">
            <button type="button" className={iconButton} onClick={p.onCommand} aria-label="command menu">
              <CommandIcon className="size-[18px]" />
            </button>
          </Tip>
        </div>
      </div>
    </div>
  )
}

function first(v: number | readonly number[]): number {
  return typeof v === 'number' ? v : (v[0] ?? 0)
}
