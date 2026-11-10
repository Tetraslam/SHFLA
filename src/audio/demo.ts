// "the house hum": a generative shoegaze loop synthesized live with plain Web Audio.
// E major, 96 bpm, Eadd9 – Badd9 – C#m7 – A(add9). Form (24 bars, loops forever):
// intro 4 (pad + glimmer) → verse 8 (drums, bass) → chorus 8 (fuzz up, open hats, 8th bass) → breakdown 4 (kick only).

export interface DemoSong {
  /** starts (or resumes) playback; schedules ahead with a lookahead scheduler */
  start(): void
  /** stops scheduling and fades out quickly (~150ms); can be started again later */
  stop(): void
  /** seconds since start(), for display */
  readonly position: number
  readonly bpm: number
  readonly title: string
}

type Section = 'intro' | 'verse' | 'chorus' | 'breakdown'

interface SectionMix {
  pad: number
  drive: number
  cutoff: number
  bass: number
  bassCutoff: number
  glimmer: number
  glimmerDensity: number
  air: number
  reverb: number
}

const BPM = 96
const STEPS_PER_BAR = 8 // eighth notes
const LOOP_BARS = 24
const LOOP_STEPS = LOOP_BARS * STEPS_PER_BAR
const LOOK_AHEAD = 0.12
const TICK_MS = 25
const FADER_LEVEL = 0.5 // about -6 dB

const MIX: Record<Section, SectionMix> = {
  intro: {
    pad: 0.55,
    drive: 1.6,
    cutoff: 1300,
    bass: 0,
    bassCutoff: 300,
    glimmer: 1,
    glimmerDensity: 1,
    air: 0.012,
    reverb: 1,
  },
  verse: {
    pad: 0.6,
    drive: 2.4,
    cutoff: 1800,
    bass: 0.85,
    bassCutoff: 320,
    glimmer: 0.7,
    glimmerDensity: 0.55,
    air: 0.018,
    reverb: 0.85,
  },
  chorus: {
    pad: 0.85,
    drive: 5,
    cutoff: 3200,
    bass: 1,
    bassCutoff: 620,
    glimmer: 0.75,
    glimmerDensity: 0.8,
    air: 0.03,
    reverb: 0.9,
  },
  breakdown: {
    pad: 0.5,
    drive: 1.5,
    cutoff: 1100,
    bass: 0.6,
    bassCutoff: 220,
    glimmer: 1.05,
    glimmerDensity: 1,
    air: 0.012,
    reverb: 1.1,
  },
}

const SECTION_START: Record<Section, number> = { intro: 0, verse: 4, chorus: 12, breakdown: 20 }

// pad voicings (MIDI) and bass roots
const CHORDS: { pad: number[]; root: number }[] = [
  { pad: [52, 59, 66, 68], root: 40 }, // Eadd9: E3 B3 F#4 G#4
  { pad: [47, 54, 61, 63], root: 35 }, // Badd9: B2 F#3 C#4 D#4
  { pad: [49, 56, 59, 64], root: 37 }, // C#m7: C#3 G#3 B3 E4
  { pad: [45, 52, 59, 61], root: 33 }, // A(add9): A2 E3 B3 C#4
]

// E major pentatonic, B5..E7 (~990 Hz to ~2.6 kHz fundamentals)
const GLIMMER_SCALE = [83, 85, 88, 90, 92, 95, 97, 100]

function sectionOf(bar: number): Section {
  if (bar < 4) return 'intro'
  if (bar < 12) return 'verse'
  if (bar < 20) return 'chorus'
  return 'breakdown'
}

function midiToFreq(m: number): number {
  return 440 * 2 ** ((m - 69) / 12)
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function makeImpulse(ctx: BaseAudioContext, seconds: number): AudioBuffer {
  const rate = ctx.sampleRate
  const len = Math.floor(rate * seconds)
  const buf = ctx.createBuffer(2, len, rate)
  const fadeIn = Math.floor(rate * 0.012)
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch)
    let lp = 0
    for (let i = 0; i < len; i++) {
      const p = i / len
      // one-pole lowpass that darkens as the tail decays
      const a = 0.9 - 0.75 * p
      lp += a * (Math.random() * 2 - 1 - lp)
      const env = (1 - p) ** 2 * Math.exp(-p * 2.2) * Math.min(1, i / fadeIn)
      data[i] = lp * env
    }
  }
  return buf
}

function makeNoise(ctx: BaseAudioContext, seconds: number): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds)
  const buf = ctx.createBuffer(1, len, ctx.sampleRate)
  const data = buf.getChannelData(0)
  for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1
  return buf
}

function makeFuzzCurve(amount: number): Float32Array<ArrayBuffer> {
  const n = 2048
  const curve = new Float32Array(n)
  const norm = Math.tanh(amount)
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1
    curve[i] = Math.tanh(amount * x) / norm
  }
  return curve
}

/** disconnect a finished one-shot source and its private nodes so they can be collected */
function release(src: AudioScheduledSourceNode, ...nodes: AudioNode[]): void {
  src.onended = () => {
    src.disconnect()
    for (const n of nodes) n.disconnect()
  }
}

export function createDemoSong(ctx: AudioContext, output: AudioNode): DemoSong {
  const beat = 60 / BPM
  const stepDur = beat / 2
  const barDur = beat * 4

  const gain = (value: number, ...to: AudioNode[]): GainNode => {
    const g = ctx.createGain()
    g.gain.value = value
    for (const n of to) g.connect(n)
    return g
  }
  const filter = (type: BiquadFilterType, freq: number, q: number, ...to: AudioNode[]): BiquadFilterNode => {
    const f = ctx.createBiquadFilter()
    f.type = type
    f.frequency.value = freq
    f.Q.value = q
    for (const n of to) f.connect(n)
    return f
  }

  // master: mix → compressor → fader (-6 dB when playing) → output
  const comp = ctx.createDynamicsCompressor()
  comp.threshold.value = -16
  comp.knee.value = 10
  comp.ratio.value = 3.5
  comp.attack.value = 0.008
  comp.release.value = 0.22
  const fader = gain(0, output)
  comp.connect(fader)
  const mix = gain(1, comp)

  // shared reverb
  const verbOut = gain(0.55, mix)
  const verb = ctx.createConvolver()
  verb.buffer = makeImpulse(ctx, 4.5)
  verb.connect(verbOut)
  const verbIn = gain(1, filter('highpass', 220, 0.7, verb))

  const noise = makeNoise(ctx, 2)
  const noiseSrc = (t: number, dur: number, loop = false): AudioBufferSourceNode => {
    const src = ctx.createBufferSource()
    src.buffer = noise
    src.loop = loop
    src.start(t, Math.random() * 1.5)
    src.stop(t + dur)
    return src
  }

  // pad: L/R chord gains → panners → fuzz → swept lowpass → out (+ heavy reverb send)
  const padOut = gain(0.55, mix)
  padOut.connect(gain(0.9, verbIn))
  const padFilter = filter('lowpass', 1300, 1.2, padOut)
  const fuzzPost = gain(0.35, padFilter)
  const shaper = ctx.createWaveShaper()
  shaper.curve = makeFuzzCurve(2.5)
  shaper.oversample = '2x'
  shaper.connect(fuzzPost)
  const drive = gain(1.6, shaper)
  const padL = ctx.createStereoPanner()
  padL.pan.value = -0.55
  padL.connect(drive)
  const padR = ctx.createStereoPanner()
  padR.pan.value = 0.55
  padR.connect(drive)

  // slow LFOs: cutoff sweep, and a glide-guitar pitch wobble shared by all pad oscillators
  const cutoffLfo = ctx.createOscillator()
  cutoffLfo.frequency.value = 0.07
  const cutoffDepth = gain(450)
  cutoffDepth.connect(padFilter.frequency)
  cutoffLfo.connect(cutoffDepth)
  cutoffLfo.start()
  const wobbleLfo = ctx.createOscillator()
  wobbleLfo.frequency.value = 0.21
  const wobbleDepth = gain(8)
  wobbleLfo.connect(wobbleDepth)
  wobbleLfo.start()

  // drums
  const drumBus = gain(1, mix)
  drumBus.connect(gain(0.12, verbIn))
  const snareBus = gain(1, drumBus)
  snareBus.connect(gain(0.35, verbIn))
  const snareBp = filter('bandpass', 1900, 0.7, snareBus)
  const hatHp = filter('highpass', 7500, 0.7, gain(1, drumBus))
  const crashBus = gain(1, mix)
  crashBus.connect(gain(0.5, verbIn))
  const crashHp = filter('highpass', 3800, 0.5, crashBus)

  // bass
  const bassBus = gain(0, mix)
  const bassFilter = filter('lowpass', 300, 3, bassBus)

  // glimmer: dry + dotted-eighth feedback delay + reverb
  const glimBus = gain(1, mix)
  const glimSend = gain(0.8, verbIn)
  glimBus.connect(glimSend)
  const glimDelay = ctx.createDelay(2)
  glimDelay.delayTime.value = beat * 0.75
  glimBus.connect(glimDelay)
  const glimFbFilter = filter('lowpass', 4000, 0.5)
  glimDelay.connect(glimFbFilter)
  glimFbFilter.connect(gain(0.42, glimDelay))
  glimFbFilter.connect(gain(0.5, mix, glimSend))

  // air: continuous tape-hiss shimmer above ~9.5 kHz
  const airGain = gain(0, mix)
  const airHp = filter('highpass', 9500, 0.7, airGain)
  let airSrc: AudioBufferSourceNode | null = null

  // --- instruments ---

  type PadVoice = { oscs: OscillatorNode[]; end: number }
  let padVoices: PadVoice[] = []

  const playPad = (chord: number[], t: number, dur: number, sec: Section): void => {
    padVoices = padVoices.filter((v) => v.end > ctx.currentTime)
    const level = 0.07
    const attack = sec === 'intro' || sec === 'breakdown' ? 0.6 : 0.12
    const gl = gain(0, padL)
    const gr = gain(0, padR)
    for (const g of [gl, gr]) {
      g.gain.setValueAtTime(0, t)
      g.gain.linearRampToValueAtTime(level, t + attack)
      g.gain.setValueAtTime(level, t + dur)
      g.gain.setTargetAtTime(0, t + dur, 0.14) // overlaps the next chord's attack: crossfade
    }
    const end = t + dur + 1
    const oscs: OscillatorNode[] = []
    chord.forEach((m, vi) => {
      const f = midiToFreq(m)
      const detunes = vi < 2 ? [-11, 0, 12] : [-7, 8]
      detunes.forEach((d, di) => {
        const o = ctx.createOscillator()
        o.type = 'sawtooth'
        o.frequency.value = f
        o.detune.value = d + (Math.random() - 0.5) * 4
        wobbleDepth.connect(o.detune)
        o.connect((vi + di) % 2 === 0 ? gl : gr)
        o.start(t)
        o.stop(end)
        o.onended = () => {
          o.disconnect()
          try {
            wobbleDepth.disconnect(o.detune)
          } catch {
            // already disconnected
          }
        }
        oscs.push(o)
      })
    })
    const last = oscs[oscs.length - 1]
    const prev = last.onended
    last.onended = (e) => {
      prev?.call(last, e)
      gl.disconnect()
      gr.disconnect()
    }
    padVoices.push({ oscs, end })
  }

  const kick = (t: number, vel: number): void => {
    const o = ctx.createOscillator()
    o.frequency.setValueAtTime(150, t)
    o.frequency.exponentialRampToValueAtTime(48, t + 0.11)
    o.frequency.exponentialRampToValueAtTime(40, t + 0.4)
    const g = gain(0, drumBus)
    g.gain.setValueAtTime(vel, t)
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.42)
    o.connect(g)
    o.start(t)
    o.stop(t + 0.45)
    release(o, g)

    const click = noiseSrc(t, 0.02)
    const cg = gain(0, drumBus)
    cg.gain.setValueAtTime(vel * 0.35, t)
    cg.gain.exponentialRampToValueAtTime(0.001, t + 0.012)
    click.connect(cg)
    release(click, cg)
  }

  const snare = (t: number, vel: number): void => {
    const n = noiseSrc(t, 0.25)
    const ng = gain(0, snareBp, hatHp)
    ng.gain.setValueAtTime(vel * 0.7, t)
    ng.gain.exponentialRampToValueAtTime(0.001, t + 0.22)
    n.connect(ng)
    release(n, ng)

    const o = ctx.createOscillator()
    o.type = 'triangle'
    o.frequency.setValueAtTime(200, t)
    o.frequency.exponentialRampToValueAtTime(170, t + 0.08)
    const tg = gain(0, snareBus)
    tg.gain.setValueAtTime(vel * 0.35, t)
    tg.gain.exponentialRampToValueAtTime(0.001, t + 0.12)
    o.connect(tg)
    o.start(t)
    o.stop(t + 0.14)
    release(o, tg)
  }

  const hat = (t: number, vel: number, open: boolean): void => {
    const decay = open ? 0.32 : 0.045
    const n = noiseSrc(t, decay + 0.02)
    const g = gain(0, hatHp)
    g.gain.setValueAtTime(vel, t)
    g.gain.exponentialRampToValueAtTime(0.001, t + decay)
    n.connect(g)
    release(n, g)
  }

  const crash = (t: number, vel: number): void => {
    const n = noiseSrc(t, 3, true)
    const g = gain(0, crashHp)
    g.gain.setValueAtTime(vel, t)
    g.gain.exponentialRampToValueAtTime(0.001, t + 2.8)
    n.connect(g)
    release(n, g)
  }

  const riser = (t: number, dur: number): void => {
    const n = noiseSrc(t, dur + 0.05, true)
    const g = gain(0, mix, verbIn)
    g.gain.setValueAtTime(0, t)
    g.gain.linearRampToValueAtTime(0.18, t + dur)
    g.gain.linearRampToValueAtTime(0, t + dur + 0.03)
    const bp = filter('bandpass', 400, 2, g)
    bp.frequency.setValueAtTime(400, t)
    bp.frequency.exponentialRampToValueAtTime(9000, t + dur)
    n.connect(bp)
    release(n, bp, g)
  }

  const bass = (t: number, dur: number, midi: number, vel: number, sub: boolean): void => {
    const o = ctx.createOscillator()
    o.type = 'sawtooth'
    o.frequency.value = midiToFreq(midi)
    const g = gain(0, bassFilter)
    g.gain.setValueAtTime(0, t)
    g.gain.linearRampToValueAtTime(vel * 0.32, t + 0.01)
    g.gain.setTargetAtTime(vel * 0.22, t + 0.01, 0.15)
    g.gain.setTargetAtTime(0, t + dur - 0.03, 0.03)
    o.connect(g)
    o.start(t)
    o.stop(t + dur + 0.2)
    release(o, g)
    if (!sub) return
    const s = ctx.createOscillator()
    s.frequency.value = midiToFreq(midi - 12)
    const sg = gain(0, bassBus)
    sg.gain.setValueAtTime(0, t)
    sg.gain.linearRampToValueAtTime(vel * 0.3, t + 0.02)
    sg.gain.setTargetAtTime(0, t + dur - 0.05, 0.05)
    s.connect(sg)
    s.start(t)
    s.stop(t + dur + 0.3)
    release(s, sg)
  }

  const glimmer = (t: number, midi: number, vel: number): void => {
    const f = midiToFreq(midi)
    const g = gain(0, glimBus)
    g.gain.setValueAtTime(0, t)
    g.gain.linearRampToValueAtTime(vel, t + 0.008)
    g.gain.exponentialRampToValueAtTime(0.0005, t + 1.6)
    const o = ctx.createOscillator()
    o.type = 'triangle'
    o.frequency.value = f
    o.connect(g)
    const o2 = ctx.createOscillator()
    o2.frequency.value = f * 2
    const g2 = gain(0.25, g)
    o2.connect(g2)
    for (const osc of [o, o2]) {
      osc.start(t)
      osc.stop(t + 1.7)
    }
    release(o, g)
    release(o2, g2)
  }

  // --- song state ---

  let rng = mulberry32(0x5f1a)
  let motif: number[] = []
  const makeMotif = (): number[] => {
    const out: number[] = []
    let idx = 2 + Math.floor(rng() * 4)
    for (let i = 0; i < 16; i++) {
      const strong = i % 4 === 0
      if (!strong && rng() < 0.5) {
        out.push(-1)
        continue
      }
      idx += Math.floor(rng() * 5) - 2
      idx = Math.max(0, Math.min(GLIMMER_SCALE.length - 1, idx))
      out.push(idx)
    }
    return out
  }
  motif = makeMotif()

  const applySection = (sec: Section, t: number, fast: boolean): void => {
    const m = MIX[sec]
    const tc = fast ? 0.05 : 0.6
    const driveVar = 0.9 + rng() * 0.25
    padOut.gain.setTargetAtTime(m.pad, t, tc)
    drive.gain.setTargetAtTime(m.drive * driveVar, t, tc)
    padFilter.frequency.setTargetAtTime(m.cutoff, t, tc * 1.5)
    bassBus.gain.setTargetAtTime(m.bass, t, 0.08)
    bassFilter.frequency.setTargetAtTime(m.bassCutoff, t, tc)
    glimBus.gain.setTargetAtTime(m.glimmer, t, tc)
    airGain.gain.setTargetAtTime(m.air, t, tc)
    verbOut.gain.setTargetAtTime(0.55 * m.reverb, t, tc)
  }

  const scheduleStep = (step: number, t: number, force: boolean): void => {
    const s = step % LOOP_STEPS
    const loop = Math.floor(step / LOOP_STEPS)
    if (s === 0) {
      rng = mulberry32(0x5f1a + loop * 7919)
      motif = makeMotif()
    }
    const bar = Math.floor(s / STEPS_PER_BAR)
    const sib = s % STEPS_PER_BAR
    const off = sib % 2 === 1
    const sec = sectionOf(bar)
    const chord = CHORDS[bar % 4]
    const m = MIX[sec]

    if (force || (sib === 0 && bar === SECTION_START[sec])) applySection(sec, t, sec === 'chorus' && !force)
    if (sib === 0) {
      playPad(chord.pad, t, barDur, sec)
      if (bar === 4 || bar === 16 || bar === 20) crash(t, 0.22)
      if (bar === 12) crash(t, 0.32)
      if (bar === 0 && loop > 0) crash(t, 0.1)
      if (bar === 11) riser(t, barDur)
    } else if (force) {
      playPad(chord.pad, t, barDur - sib * stepDur, sec)
    }

    // drums
    if (sec === 'verse') {
      if (sib === 0) kick(t, 1)
      if (sib === 4) kick(t, 0.95)
      if (sib === 5 && rng() < 0.3) kick(t, 0.5)
      if (bar === 11 && sib >= 4) {
        for (let k = 0; k < 2; k++) {
          const p = (sib - 4) * 2 + k
          snare(t + k * stepDur * 0.5, 0.3 + p * 0.07)
        }
      } else if (sib === 2 || sib === 6) {
        snare(t, 0.8)
      }
      hat(t, off ? 0.16 : 0.09, false)
    } else if (sec === 'chorus') {
      if (!off) kick(t, 1)
      if (sib === 2 || sib === 6) snare(t, 0.95)
      hat(t, off ? 0.14 : 0.08, off)
    } else if (sec === 'breakdown') {
      if (!off) kick(t, 0.75)
    }

    // bass
    if (sec === 'verse') {
      if (sib === 0 || sib === 4) bass(t, beat * 2 - 0.02, chord.root, 1, true)
    } else if (sec === 'chorus') {
      bass(t, stepDur * 0.9, off ? chord.root + 12 : chord.root, off ? 0.8 : 1, false)
      if (sib === 0 || sib === 4) bass(t, beat * 2 - 0.02, chord.root, 0, true)
    } else if (sec === 'breakdown') {
      if (sib === 0) bass(t, barDur - 0.05, chord.root, 0.6, true)
    }

    // glimmer
    const mi = motif[s % 16]
    if (mi >= 0 && rng() < m.glimmerDensity) {
      let idx = mi
      if (rng() < 0.12) idx = Math.max(0, Math.min(GLIMMER_SCALE.length - 1, idx + (rng() < 0.5 ? -1 : 1)))
      glimmer(t + rng() * 0.012, GLIMMER_SCALE[idx], 0.08 + rng() * 0.03)
    }
  }

  // --- scheduler ---

  let running = false
  let timer: ReturnType<typeof setInterval> | null = null
  let step = 0
  let nextTime = 0
  let playStartStep = 0
  let playStartTime = 0
  let frozenPosition = 0
  let force = false

  const currentPosition = (): number => Math.max(0, playStartStep * stepDur + (ctx.currentTime - playStartTime))

  const tick = (): void => {
    if (!running) return
    const now = ctx.currentTime
    if (nextTime < now - 0.2) {
      // fell behind (throttled background tab): realign without rushing through missed notes
      nextTime = now + 0.03
      playStartStep = step
      playStartTime = nextTime
      force = true
    }
    while (nextTime < now + LOOK_AHEAD) {
      scheduleStep(step, nextTime, force)
      force = false
      step++
      nextTime += stepDur
    }
  }

  return {
    start() {
      if (running) return
      running = true
      if (ctx.state === 'suspended') ctx.resume().catch(() => {})
      const now = ctx.currentTime
      step = Math.floor(step / STEPS_PER_BAR) * STEPS_PER_BAR // resume from the start of the bar
      nextTime = now + 0.06
      playStartStep = step
      playStartTime = nextTime
      force = true

      airSrc = ctx.createBufferSource()
      airSrc.buffer = noise
      airSrc.loop = true
      airSrc.connect(airHp)
      airSrc.start(now)

      fader.gain.cancelScheduledValues(now)
      fader.gain.setValueAtTime(fader.gain.value, now)
      fader.gain.linearRampToValueAtTime(FADER_LEVEL, now + 0.05)

      tick()
      timer = setInterval(tick, TICK_MS)
    },

    stop() {
      if (!running) return
      running = false
      if (timer !== null) clearInterval(timer)
      timer = null
      const now = ctx.currentTime
      frozenPosition = currentPosition()
      step = Math.max(playStartStep, playStartStep + Math.floor((now - playStartTime) / stepDur))

      fader.gain.cancelScheduledValues(now)
      fader.gain.setValueAtTime(fader.gain.value, now)
      fader.gain.setTargetAtTime(0, now, 0.04)

      const cut = now + 0.25
      for (const v of padVoices) {
        for (const o of v.oscs) {
          try {
            o.stop(cut)
          } catch {
            // already stopped
          }
        }
      }
      padVoices = []
      if (airSrc) {
        const src = airSrc
        src.stop(cut)
        src.onended = () => src.disconnect()
        airSrc = null
      }
    },

    get position() {
      return running ? currentPosition() : frozenPosition
    },
    bpm: BPM,
    title: 'the house hum',
  }
}
