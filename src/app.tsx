import { parseAsFloat, parseAsString, throttle, useQueryStates } from 'nuqs'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { AudioEngine, SourceError, type SourceKind, type SourceState } from '@/audio/engine'
import { LEVELS } from '@/audio/frame'
import { Intro } from '@/components/intro'
import { type MeterRefs, Transport } from '@/components/transport'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Engine } from '@/gpu/engine'
import { paletteById } from '@/gpu/palettes'
import { compile, type Program } from '@/lang/compile'
import { DEFAULT_PRESET, PRESETS } from '@/lang/presets'
import { decodeProgram, encodeProgram, formatTime } from '@/lib/share'

// the editor and the command menu are the heavy parts; they load after first paint, on idle
const loadPanel = () => import('@/components/language-panel')
const loadMenu = () => import('@/components/command-menu')
const LanguagePanel = lazy(() => loadPanel().then((m) => ({ default: m.LanguagePanel })))
const CommandMenu = lazy(() => loadMenu().then((m) => ({ default: m.CommandMenu })))

const params = {
  rule: parseAsString,
  code: parseAsString,
  palette: parseAsString.withDefault('dusk'),
  voice: parseAsFloat.withDefault(0.35),
}

function initialCode(rule: string | null, code: string | null): { code: string; presetId: string | null } {
  if (code) {
    const decoded = decodeProgram(code)
    if (decoded) return { code: decoded, presetId: null }
  }
  const preset = PRESETS.find((p) => p.id === rule) ?? DEFAULT_PRESET
  return { code: preset.source, presetId: preset.id }
}

const isTyping = (t: EventTarget | null) =>
  t instanceof HTMLElement && (t.isContentEditable || t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')

export function App() {
  const [q, setQ] = useQueryStates(params, { history: 'replace', limitUrlUpdates: throttle(400) })
  const [{ code: startCode, presetId: startPreset }] = useState(() => initialCode(q.rule, q.code))
  const [code, setCode] = useState(startCode)
  const [presetId, setPresetId] = useState<string | null>(startPreset)
  const [basedOn, setBasedOn] = useState(startPreset ?? 'a shared link')
  const [gpuError, setGpuError] = useState<string | null>(null)
  const [fatal, setFatal] = useState<string | null>(null)
  const [panel, setPanel] = useState(false)
  const [chrome, setChrome] = useState(true)
  const [cmdk, setCmdk] = useState(false)
  const [source, setSource] = useState<SourceState | null>(null)
  const [busy, setBusy] = useState<SourceKind | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [sourceError, setSourceError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [linkSignal, setLinkSignal] = useState(0)

  const canvas = useRef<HTMLCanvasElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const engine = useRef<Engine | null>(null)
  const audio = useRef<AudioEngine | null>(null)
  if (!audio.current) audio.current = new AudioEngine()
  const meters = useRef<MeterRefs>({ band: [], act: [], time: null, depth: null })
  const lastGood = useRef<Program | null>(null)

  const compiled = useMemo(() => compile(code), [code])
  const errors = compiled.ok ? [] : compiled.errors

  useEffect(() => {
    const warm = () => {
      void loadPanel()
      void loadMenu()
    }
    if (typeof requestIdleCallback === 'function') {
      const id = requestIdleCallback(warm, { timeout: 3000 })
      return () => cancelIdleCallback(id)
    }
    const id = setTimeout(warm, 1500)
    return () => clearTimeout(id)
  }, [])

  // ---- engine lifecycle and the frame loop
  useEffect(() => {
    if (!canvas.current) return
    let e: Engine
    try {
      const small =
        Math.max(window.screen.width, window.screen.height) < 1000 || matchMedia('(pointer: coarse)').matches
      e = new Engine(canvas.current, { finest: small ? 512 : 1024 })
    } catch (err) {
      setFatal(err instanceof Error ? err.message : String(err))
      return
    }
    engine.current = e
    const a = audio.current as AudioEngine
    a.onChange = () => setSource(a.source ? { ...a.source } : null)
    ;(window as unknown as { shfla: unknown }).shfla = { engine: e, audio: a }

    let raf = 0
    let last = performance.now()
    let textClock = 0
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      const frame = a.frame(dt, e.activity)
      e.frame(dt, frame)
      const m = meters.current
      for (let k = 0; k < LEVELS; k++) {
        const b = m.band[k]
        if (b) b.style.transform = `scaleY(${Math.max(0.04, frame.bands[k]).toFixed(3)})`
        const d = m.act[k]
        if (d) d.style.opacity = Math.min(1, e.activity[k].change * 3).toFixed(2)
      }
      textClock += dt
      if (textClock > 0.25) {
        textClock = 0
        if (m.time) {
          const dur = a.source?.duration
          m.time.textContent = dur ? `${formatTime(a.position)} / ${formatTime(dur)}` : formatTime(a.position)
        }
        if (m.depth) m.depth.textContent = `${e.depth} octaves deep · ${Math.round(frame.bpm)} bpm`
      }
    }
    raf = requestAnimationFrame(loop)

    const onVis = () => {
      if (document.hidden) cancelAnimationFrame(raf)
      else {
        last = performance.now()
        raf = requestAnimationFrame(loop)
      }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVis)
      e.dispose()
      engine.current = null
      lastGood.current = null
    }
  }, [])

  // ---- live compile: the gpu gets the program once typing pauses; the last good one keeps running
  useEffect(() => {
    if (!compiled.ok) return
    const program = compiled.program
    const t = window.setTimeout(
      () => {
        const e = engine.current
        if (!e) return
        const r = e.setProgram(program)
        if (r.ok) {
          lastGood.current = program
          setGpuError(null)
        } else setGpuError(r.error)
      },
      lastGood.current ? 160 : 0,
    )
    return () => window.clearTimeout(t)
  }, [compiled])

  // ---- settings into the engine
  useEffect(() => {
    const rm = matchMedia('(prefers-reduced-motion: reduce)')
    const apply = () => {
      const e = engine.current
      if (e) e.settings = { palette: paletteById(q.palette), speed, zoom, reducedMotion: rm.matches }
    }
    apply()
    rm.addEventListener('change', apply)
    return () => rm.removeEventListener('change', apply)
  }, [q.palette, speed, zoom])

  useEffect(() => {
    audio.current?.setVoice(source?.kind === 'mic' ? 0 : q.voice)
  }, [q.voice, source?.kind])

  // ---- url: a preset by name, an edited program by value
  useEffect(() => {
    if (presetId) setQ({ rule: presetId === DEFAULT_PRESET.id ? null : presetId, code: null })
    else setQ({ rule: null, code: encodeProgram(code) })
  }, [code, presetId, setQ])

  const onCode = useCallback(
    (v: string) => {
      setCode(v)
      if (presetId) {
        const p = PRESETS.find((x) => x.id === presetId)
        if (p && p.source !== v) {
          setBasedOn(p.name)
          setPresetId(null)
        }
      }
    },
    [presetId],
  )

  const choosePreset = useCallback((id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (!p) return
    setCode(p.source)
    setPresetId(p.id)
    setBasedOn(p.name)
    // a new rule starts from its own seed
    window.setTimeout(() => engine.current?.reseed(), 40)
  }, [])

  // ---- sources
  const run = useCallback(async (kind: SourceKind, fn: () => Promise<void>) => {
    setBusy(kind)
    setSourceError(null)
    setStatus(null)
    try {
      await fn()
    } catch (e) {
      const msg = e instanceof SourceError ? e.message : 'something went wrong starting that'
      setSourceError(msg)
      if (!(e instanceof SourceError)) console.error(e)
    } finally {
      setBusy(null)
      setStatus(null)
    }
  }, [])

  const a = audio.current
  const playDemo = useCallback(() => run('demo', () => a.playDemo()), [run, a])
  const playMic = useCallback(() => run('mic', () => a.playMic()), [run, a])
  const playLink = useCallback((url: string) => run('link', () => a.playLink(url, (s) => setStatus(s))), [run, a])
  const playFile = useCallback((file: File) => run('file', () => a.playFile(file)), [run, a])
  const pickFile = useCallback(() => fileInput.current?.click(), [])

  const share = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      toast('link copied. it carries the rule and the palette')
    } catch {
      toast.error("couldn't reach the clipboard")
    }
  }, [])

  const fullscreen = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void document.documentElement.requestFullscreen?.().catch(() => {})
  }, [])

  const toggleChrome = useCallback(() => {
    setChrome((c) => {
      if (c) toast('press h to bring everything back', { duration: 2200 })
      return !c
    })
  }, [])

  const action = useCallback(
    (x: string) => {
      const [verb, arg] = x.split(':')
      if (verb === 'source') {
        if (arg === 'demo') playDemo()
        else if (arg === 'file') pickFile()
        else if (arg === 'mic') playMic()
        else if (arg === 'link') {
          a.stop()
          setChrome(true)
          setLinkSignal((n) => n + 1)
        }
      } else if (verb === 'preset') choosePreset(arg)
      else if (verb === 'palette') setQ({ palette: arg })
      else if (verb === 'toggle') void a.toggle()
      else if (verb === 'panel') {
        setChrome(true)
        setPanel((v) => !v)
      } else if (verb === 'reseed') engine.current?.reseed()
      else if (verb === 'voice') setQ({ voice: q.voice > 0 ? 0 : 0.35 })
      else if (verb === 'chrome') toggleChrome()
      else if (verb === 'fullscreen') fullscreen()
      else if (verb === 'share') void share()
    },
    [a, choosePreset, fullscreen, pickFile, playDemo, playMic, q.voice, setQ, share, toggleChrome],
  )

  // ---- keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCmdk((v) => !v)
        return
      }
      if (e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target) || cmdk) return
      if (e.key === 'Escape' && panel) setPanel(false)
      else if (e.key === ' ' && a.source) {
        e.preventDefault()
        void a.toggle()
      } else if (e.key === 'e') action('panel')
      else if (e.key === 'h') toggleChrome()
      else if (e.key === 'f') fullscreen()
      else if (e.key === 'r') engine.current?.reseed()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [a, action, cmdk, fullscreen, panel, toggleChrome])

  // ---- drop a song anywhere
  useEffect(() => {
    let depth = 0
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files')
    const enter = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth++
      setDragging(true)
    }
    const over = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault()
    }
    const leave = () => {
      depth = Math.max(0, depth - 1)
      if (depth === 0) setDragging(false)
    }
    const drop = (e: DragEvent) => {
      if (!hasFiles(e)) return
      e.preventDefault()
      depth = 0
      setDragging(false)
      const f = e.dataTransfer?.files[0]
      if (f) playFile(f)
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop)
    }
  }, [playFile])

  if (fatal) {
    return (
      <main className="flex h-full items-end bg-ground px-8 pb-24 sm:px-16">
        <div className="max-w-xl">
          <h1 className="focus-in font-display text-[clamp(4rem,12vw,9rem)] leading-[0.85] text-ink">no light here</h1>
          <p className="mt-6 text-[17px] leading-relaxed text-ink-muted">
            shfla draws on the gpu through webgl2, and this browser didn't give it one ({fatal}). a recent chrome,
            firefox, or safari with hardware acceleration turned on will do.
          </p>
        </div>
      </main>
    )
  }

  return (
    <TooltipProvider delay={400}>
      <main className="relative h-full w-full overflow-hidden bg-ground">
        <canvas ref={canvas} className="absolute inset-0 h-full w-full" aria-label="the fractal" role="img" />

        {chrome && !source && (
          <Intro
            key={linkSignal}
            initialLinkOpen={linkSignal > 0}
            busy={busy}
            status={status}
            error={sourceError}
            onDemo={playDemo}
            onFile={pickFile}
            onLink={playLink}
            onMic={playMic}
          />
        )}

        {chrome && (
          <Transport
            source={source}
            meters={meters}
            panelOpen={panel}
            paletteId={q.palette}
            voice={q.voice}
            speed={speed}
            zoom={zoom}
            onToggle={() => void a.toggle()}
            onStop={() => a.stop()}
            onPanel={() => setPanel((v) => !v)}
            onCommand={() => setCmdk(true)}
            onPalette={(id) => setQ({ palette: id })}
            onVoice={(v) => setQ({ voice: v })}
            onSpeed={setSpeed}
            onZoom={setZoom}
          />
        )}

        {chrome && panel && (
          <Suspense fallback={null}>
            <LanguagePanel
              code={code}
              presetId={presetId}
              basedOn={basedOn}
              errors={errors}
              gpuError={gpuError}
              onCode={onCode}
              onPreset={choosePreset}
              onReseed={() => engine.current?.reseed()}
              onShare={() => void share()}
              onClose={() => setPanel(false)}
            />
          </Suspense>
        )}

        {dragging && (
          <div className="focus-in pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-ground/40 backdrop-blur-sm">
            <p className="font-display text-[clamp(3rem,8vw,6rem)] text-ink">let it go</p>
          </div>
        )}

        <input
          ref={fileInput}
          type="file"
          accept="audio/*,video/*,.mp3,.ogg,.wav,.flac,.m4a,.aac,.opus,.webm"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) playFile(f)
            e.target.value = ''
          }}
        />

        <Suspense fallback={null}>
          <CommandMenu
            open={cmdk}
            onOpenChange={setCmdk}
            run={action}
            presetId={presetId}
            paletteId={q.palette}
            playing={!!source?.playing}
            hasSource={!!source}
            voiceOn={q.voice > 0}
          />
        </Suspense>
        <Toaster position="top-center" />
      </main>
    </TooltipProvider>
  )
}
