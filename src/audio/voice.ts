import type { LevelActivity } from '@/gpu/engine'
import { type AudioFrame, LEVELS } from './frame'

/**
 * The fractal sings back. Each level is one voice in an open chord built on the song's own root,
 * from a low drone (coarsest level) up through the octaves. A voice is as loud as its level is busy
 * computing, so the harmony follows what the rule is doing, not the song.
 */

// scale degree (semitones above the root) and octave for each level, coarse to fine
const VOICING: Array<{ degree: 'root' | 'third' | 'fifth'; octave: number }> = [
  { degree: 'root', octave: 1 },
  { degree: 'fifth', octave: 2 },
  { degree: 'root', octave: 3 },
  { degree: 'third', octave: 3 },
  { degree: 'fifth', octave: 4 },
  { degree: 'root', octave: 5 },
]

function impulse(ctx: BaseAudioContext, seconds: number): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds)
  const buf = ctx.createBuffer(2, len, ctx.sampleRate)
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch)
    let lp = 0
    for (let i = 0; i < len; i++) {
      const t = i / len
      lp += (Math.random() * 2 - 1 - lp) * (0.6 - 0.45 * t)
      d[i] = lp * (1 - t) ** 2.4
    }
  }
  return buf
}

function fuzzCurve(amount: number): Float32Array<ArrayBuffer> {
  const n = 1024
  const c = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1
    c[i] = Math.tanh(x * amount) / Math.tanh(amount)
  }
  return c
}

export class Voice {
  readonly out: GainNode
  private gains: GainNode[] = []
  private oscs: OscillatorNode[][] = []
  private filter: BiquadFilterNode
  private amp = new Float32Array(LEVELS)
  private root = 4
  private minor = false
  private candidate = 4
  private candidateFor = 0
  private level = 0

  private ctx: AudioContext

  constructor(ctx: AudioContext, destination: AudioNode) {
    this.ctx = ctx
    this.out = ctx.createGain()
    this.out.gain.value = 0
    this.out.connect(destination)

    const shaper = ctx.createWaveShaper()
    shaper.curve = fuzzCurve(2.2)
    shaper.oversample = '2x'
    this.filter = ctx.createBiquadFilter()
    this.filter.type = 'lowpass'
    this.filter.frequency.value = 1800
    this.filter.Q.value = 0.4
    const reverb = ctx.createConvolver()
    reverb.buffer = impulse(ctx, 5.5)
    const wet = ctx.createGain()
    wet.gain.value = 0.85
    const dry = ctx.createGain()
    dry.gain.value = 0.25
    const bus = ctx.createGain()
    bus.gain.value = 0.5

    bus.connect(shaper)
    shaper.connect(this.filter)
    this.filter.connect(dry)
    this.filter.connect(reverb)
    reverb.connect(wet)
    dry.connect(this.out)
    wet.connect(this.out)

    for (let k = 0; k < LEVELS; k++) {
      const g = ctx.createGain()
      g.gain.value = 0
      g.connect(bus)
      const type: OscillatorType = k === 0 ? 'sine' : k < 5 ? 'sawtooth' : 'triangle'
      const pair = [-7, 7].map((cents) => {
        const o = ctx.createOscillator()
        o.type = type
        o.detune.value = cents
        o.frequency.value = this.freq(k)
        o.connect(g)
        o.start()
        return o
      })
      this.gains.push(g)
      this.oscs.push(pair)
    }
  }

  private freq(k: number) {
    const v = VOICING[k]
    const semis = v.degree === 'root' ? 0 : v.degree === 'fifth' ? 7 : this.minor ? 3 : 4
    // C1 = 32.703 Hz
    return 32.703 * 2 ** (v.octave - 1 + (this.root + semis) / 12)
  }

  /** 0..1 */
  setLevel(v: number) {
    this.level = v
    this.out.gain.setTargetAtTime(v * 0.55, this.ctx.currentTime, 0.25)
  }

  update(activity: LevelActivity[], frame: AudioFrame, dt: number) {
    if (this.level <= 0) return
    const now = this.ctx.currentTime
    // only move the chord once the song has sat on a new root for a while
    if (frame.root !== this.candidate) {
      this.candidate = frame.root
      this.candidateFor = 0
    } else this.candidateFor += dt
    const retune = frame.live && this.candidateFor > 2.5 && (this.candidate !== this.root || frame.minor !== this.minor)
    if (retune) {
      this.root = this.candidate
      this.minor = frame.minor
    }
    for (let k = 0; k < LEVELS; k++) {
      const a = activity[k]
      const target = Math.min(1, a.change * 1.6 + a.state * 0.15)
      this.amp[k] += (target - this.amp[k]) * (1 - Math.exp(-dt / 0.35))
      const weight = k === 0 ? 0.5 : k === 5 ? 0.12 : 0.18
      this.gains[k].gain.setTargetAtTime(this.amp[k] * weight, now, 0.12)
      if (retune) for (const o of this.oscs[k]) o.frequency.setTargetAtTime(this.freq(k), now, 0.6)
    }
    this.filter.frequency.setTargetAtTime(900 + 3200 * frame.bright, now, 0.3)
  }

  dispose() {
    for (const pair of this.oscs) for (const o of pair) o.stop()
    this.out.disconnect()
  }
}
