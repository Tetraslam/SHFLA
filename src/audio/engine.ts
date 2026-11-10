import type { LevelActivity } from '@/gpu/engine'
import { isYouTube } from '../../api/_youtube'
import { createDemoSong, type DemoSong } from './demo'
import { FeatureExtractor, IdleBreath } from './features'
import type { AudioFrame } from './frame'
import { Voice } from './voice'

export type SourceKind = 'demo' | 'file' | 'link' | 'mic' | 'tab'

export interface SourceState {
  kind: SourceKind
  title: string
  playing: boolean
  /** seconds; null for live sources */
  duration: number | null
}

export class SourceError extends Error {
  /** a youtube link the server couldn't fetch; the ui offers to play it in a tab and listen to that */
  youtube?: string
  constructor(message: string, youtube?: string) {
    super(message)
    this.youtube = youtube
  }
}

/** tab audio capture: chrome and edge on desktop */
export function canListenToTab(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getDisplayMedia &&
    !/Android|iPhone|iPad|iPod/i.test(navigator.userAgent) &&
    /Chrome\/|Edg\//.test(navigator.userAgent)
  )
}

const FFT = 4096

/** One place that owns the AudioContext, the current source, the analysis, and the voice. */
export class AudioEngine {
  private ctx: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private music: GainNode | null = null
  private extractor: FeatureExtractor | null = null
  private idle = new IdleBreath()
  private spectrum: Float32Array<ArrayBuffer> | null = null
  private el: HTMLAudioElement | null = null
  private objectUrl: string | null = null
  private elNode: MediaElementAudioSourceNode | null = null
  private mic: MediaStream | null = null
  private micNode: MediaStreamAudioSourceNode | null = null
  private demo: DemoSong | null = null
  private voice: Voice | null = null
  private voiceLevel = 0.35
  private blend = 0
  source: SourceState | null = null
  onChange: (() => void) | null = null

  private ensure(): AudioContext {
    if (this.ctx) return this.ctx
    const ctx = new AudioContext({ latencyHint: 'interactive' })
    const analyser = ctx.createAnalyser()
    analyser.fftSize = FFT
    analyser.smoothingTimeConstant = 0
    analyser.minDecibels = -110
    analyser.maxDecibels = -10
    const music = ctx.createGain()
    music.connect(ctx.destination)
    this.ctx = ctx
    this.analyser = analyser
    this.music = music
    this.spectrum = new Float32Array(analyser.frequencyBinCount)
    this.extractor = new FeatureExtractor(ctx.sampleRate, FFT)
    this.voice = new Voice(ctx, ctx.destination)
    return ctx
  }

  private teardown() {
    this.demo?.stop()
    if (this.el) {
      this.el.pause()
      this.el.removeAttribute('src')
      this.el.load()
    }
    this.elNode?.disconnect()
    this.el = null
    this.elNode = null
    this.micNode?.disconnect()
    for (const t of this.mic?.getTracks() ?? []) t.stop()
    this.mic = null
    this.micNode = null
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl)
    this.objectUrl = null
    this.source = null
  }

  private emit() {
    this.onChange?.()
  }

  async playDemo() {
    const ctx = this.ensure()
    await ctx.resume()
    this.teardown()
    if (!this.demo) {
      const tap = ctx.createGain()
      tap.connect(this.analyser as AnalyserNode)
      tap.connect(this.music as GainNode)
      this.demo = createDemoSong(ctx, tap)
    }
    this.demo.start()
    this.source = { kind: 'demo', title: this.demo.title, playing: true, duration: null }
    this.applyVoice()
    this.emit()
  }

  private async playElement(src: string, title: string, kind: 'file' | 'link', crossOrigin: boolean, owned = false) {
    const ctx = this.ensure()
    await ctx.resume()
    this.teardown()
    if (owned) this.objectUrl = src
    const el = new Audio()
    if (crossOrigin) el.crossOrigin = 'anonymous'
    el.preload = 'auto'
    el.src = src
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new SourceError('the link took too long to answer')), 15000)
      el.addEventListener(
        'canplay',
        () => {
          window.clearTimeout(timer)
          resolve()
        },
        { once: true },
      )
      el.addEventListener(
        'error',
        () => {
          window.clearTimeout(timer)
          reject(new SourceError('that audio would not load'))
        },
        { once: true },
      )
    })
    const node = ctx.createMediaElementSource(el)
    node.connect(this.analyser as AnalyserNode)
    node.connect(this.music as GainNode)
    el.addEventListener('ended', () => {
      if (this.source && this.el === el) {
        this.source = { ...this.source, playing: false }
        this.emit()
      }
    })
    el.addEventListener('durationchange', () => {
      if (this.source && this.el === el) {
        this.source = { ...this.source, duration: Number.isFinite(el.duration) ? el.duration : null }
        this.emit()
      }
    })
    this.el = el
    this.elNode = node
    await el.play()
    this.source = { kind, title, playing: true, duration: Number.isFinite(el.duration) ? el.duration : null }
    this.applyVoice()
    this.emit()
  }

  async playFile(file: File) {
    if (!file.type.startsWith('audio/') && !file.type.startsWith('video/') && file.type !== '') {
      throw new SourceError(`"${file.name}" doesn't look like audio`)
    }
    const url = URL.createObjectURL(file)
    const title = file.name.replace(/\.[a-z0-9]+$/i, '')
    try {
      await this.playElement(url, title, 'file', false, true)
    } catch (e) {
      URL.revokeObjectURL(url)
      throw e instanceof SourceError ? new SourceError(`the browser can't decode "${file.name}"`) : e
    }
  }

  /** Try the link directly (works when it sends CORS headers), then through our proxy. */
  async playLink(raw: string, onStatus?: (s: string) => void) {
    let url: URL
    try {
      url = new URL(raw.trim())
    } catch {
      throw new SourceError("that isn't a link")
    }
    if (url.protocol !== 'https:' && url.protocol !== 'http:')
      throw new SourceError('links need to start with https://')
    const title = decodeURIComponent(url.pathname.split('/').filter(Boolean).pop() ?? url.hostname).replace(
      /\.[a-z0-9]+$/i,
      '',
    )
    if (isYouTube(url.href)) return this.playYouTube(url.href, onStatus)
    onStatus?.('reaching the link')
    try {
      await this.playElement(url.href, title, 'link', true)
      return
    } catch {
      onStatus?.('going around cors')
    }
    try {
      await this.playElement(`/api/audio?url=${encodeURIComponent(url.href)}`, title, 'link', false)
    } catch {
      throw new SourceError(
        "couldn't play that link. it has to be a youtube video or point straight at an audio file (mp3, ogg, wav, m4a)",
      )
    }
  }

  /** yt-dlp runs on our server; the audio arrives as one blob, so seeking is free afterwards */
  private async playYouTube(href: string, onStatus?: (s: string) => void) {
    // resume inside the click, before the long wait, so playback is still allowed afterwards
    await this.ensure().resume()
    onStatus?.('asking youtube for the audio')
    const tabHint = canListenToTab()
      ? 'youtube turned our server away. play it in a tab and let shfla listen to that tab instead'
      : 'youtube turned our server away. try chrome or edge on a computer, which can listen to a tab'
    let res: Response
    const timeout = new AbortController()
    const timer = setTimeout(() => timeout.abort(), 30_000)
    try {
      res = await fetch(`/api/yt?url=${encodeURIComponent(href)}`, { signal: timeout.signal })
    } catch {
      throw new SourceError(tabHint, href)
    } finally {
      clearTimeout(timer)
    }
    if (!res.ok || !res.body) {
      const body = (await res.json().catch(() => null)) as { error?: string } | null
      // refusals that a tab can get around; a private or missing video can't be helped
      if (res.status === 503 || res.status >= 500) throw new SourceError(tabHint, href)
      throw new SourceError(body?.error ?? "couldn't get the audio out of that video")
    }
    const title = decodeURIComponent(res.headers.get('x-shfla-title') ?? 'youtube')
    const total = Number(res.headers.get('content-length') ?? 0)
    const parts: Uint8Array<ArrayBuffer>[] = []
    let got = 0
    const reader = res.body.getReader()
    try {
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        parts.push(value)
        got += value.byteLength
        onStatus?.(
          total
            ? `pulling the audio, ${Math.round((got / total) * 100)}%`
            : `pulling the audio, ${(got / 1e6).toFixed(1)} mb`,
        )
      }
    } catch {
      throw new SourceError(tabHint, href)
    }
    if (!got) throw new SourceError('youtube sent back nothing')
    const blob = new Blob(parts, { type: res.headers.get('content-type') ?? 'audio/mp4' })
    const url = URL.createObjectURL(blob)
    try {
      await this.playElement(url, title, 'link', false, true)
    } catch {
      URL.revokeObjectURL(url)
      throw new SourceError("the browser couldn't decode the audio youtube sent")
    }
  }

  /** listen to another browser tab: it keeps playing there, we only analyse it */
  async playTab() {
    if (!navigator.mediaDevices?.getDisplayMedia) {
      throw new SourceError("this browser can't share a tab's sound. chrome or edge on a computer can")
    }
    const ctx = this.ensure()
    await ctx.resume()
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        preferCurrentTab: false,
        selfBrowserSurface: 'exclude',
        surfaceSwitching: 'include',
        systemAudio: 'include',
        monitorTypeSurfaces: 'exclude',
      } as DisplayMediaStreamOptions)
    } catch {
      throw new SourceError('no tab was shared')
    }
    const video = stream.getVideoTracks()[0]
    const label = video?.label ?? ''
    for (const t of stream.getVideoTracks()) t.stop()
    const audio = stream.getAudioTracks()
    if (!audio.length) {
      throw new SourceError("that share had no sound. pick a tab, and leave 'also share tab audio' switched on")
    }
    this.teardown()
    const live = new MediaStream(audio)
    const node = ctx.createMediaStreamSource(live)
    // the tab plays itself; we only listen, or the song would double
    node.connect(this.analyser as AnalyserNode)
    this.mic = live
    this.micNode = node
    audio[0].addEventListener('ended', () => {
      if (this.mic === live) this.stop()
    })
    const title = label
      .replace(/^web-contents-media-stream:\/\/.*$/, '')
      .replace(/\s*-\s*YouTube$/, '')
      .trim()
    this.source = { kind: 'tab', title: title || 'another tab', playing: true, duration: null }
    this.applyVoice()
    this.emit()
  }

  async playMic() {
    const ctx = this.ensure()
    await ctx.resume()
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
    } catch (e) {
      const name = e instanceof DOMException ? e.name : ''
      throw new SourceError(
        name === 'NotAllowedError'
          ? 'the microphone is blocked. allow it in the address bar and try again'
          : 'no microphone found',
      )
    }
    this.teardown()
    const node = ctx.createMediaStreamSource(stream)
    // listened to, never played back: that would howl
    node.connect(this.analyser as AnalyserNode)
    this.mic = stream
    this.micNode = node
    const label = stream.getAudioTracks()[0]?.label
    this.source = { kind: 'mic', title: label ? label.toLowerCase() : 'microphone', playing: true, duration: null }
    this.applyVoice()
    this.emit()
  }

  async toggle() {
    const s = this.source
    if (!s || !this.ctx) return
    if (s.kind === 'demo') {
      if (s.playing) this.demo?.stop()
      else this.demo?.start()
    } else if (s.kind === 'mic' || s.kind === 'tab') {
      if (s.playing) await this.ctx.suspend()
      else await this.ctx.resume()
    } else if (this.el) {
      if (s.playing) this.el.pause()
      else {
        if (this.el.ended) this.el.currentTime = 0
        await this.el.play()
      }
    }
    this.source = { ...s, playing: !s.playing }
    this.emit()
  }

  stop() {
    this.teardown()
    this.emit()
  }

  get position(): number {
    if (this.el) return this.el.currentTime
    if (this.source?.kind === 'demo') return this.demo?.position ?? 0
    return 0
  }

  seek(seconds: number) {
    if (this.el && Number.isFinite(seconds)) this.el.currentTime = seconds
  }

  setVoice(level: number) {
    this.voiceLevel = level
    this.applyVoice()
  }

  private applyVoice() {
    this.voice?.setLevel(this.source ? this.voiceLevel : 0)
  }

  /** Called every animation frame. Crossfades to the idle breath when nothing is playing. */
  frame(dt: number, activity: LevelActivity[]): AudioFrame {
    const idle = this.idle.process(dt)
    const playing = !!this.source?.playing && !!this.analyser && !!this.extractor && !!this.spectrum
    this.blend += ((playing ? 1 : 0) - this.blend) * (1 - Math.exp(-dt / 0.6))
    if (!playing && this.blend < 0.01) return idle
    if (!this.analyser || !this.extractor || !this.spectrum) return idle
    this.analyser.getFloatFrequencyData(this.spectrum)
    const live = this.extractor.process(this.spectrum, dt)
    this.voice?.update(activity, live, dt)
    if (this.blend > 0.99) return live
    const b = this.blend
    const mix = (x: number, y: number) => x * (1 - b) + y * b
    return {
      ...live,
      bands: live.bands.map((v, i) => mix(idle.bands[i], v)),
      hits: live.hits.map((v, i) => mix(idle.hits[i], v)),
      bass: mix(idle.bass, live.bass),
      mid: mix(idle.mid, live.mid),
      high: mix(idle.high, live.high),
      loud: mix(idle.loud, live.loud),
      beat: mix(idle.beat, live.beat),
      bright: mix(idle.bright, live.bright),
      bpm: b > 0.5 ? live.bpm : idle.bpm,
      phase: b > 0.5 ? live.phase : idle.phase,
      live: b > 0.5,
    }
  }
}
