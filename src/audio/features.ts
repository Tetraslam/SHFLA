import { type AudioFrame, LEVELS, silentFrame } from './frame'

/**
 * Turns raw FFT frames into the signals the language reads.
 * Pure: no Web Audio here, so it is testable with synthetic spectra.
 */

/** octave band edges in Hz, coarse to fine: sub, bass, low-mid, mid, high-mid, air */
export const BAND_EDGES = [28, 90, 250, 700, 2000, 5500, 16000]

export function bandBins(sampleRate: number, fftSize: number): Array<[number, number]> {
  const hzPerBin = sampleRate / fftSize
  const out: Array<[number, number]> = []
  for (let b = 0; b < LEVELS; b++) {
    const lo = Math.max(1, Math.floor(BAND_EDGES[b] / hzPerBin))
    const hi = Math.max(lo + 1, Math.min(fftSize / 2, Math.ceil(BAND_EDGES[b + 1] / hzPerBin)))
    out.push([lo, hi])
  }
  return out
}

/**
 * Tempo from an onset envelope sampled at `rate` Hz: autocorrelation over lags for 70-180 bpm,
 * weighted toward ~110 bpm so it prefers the felt beat over double or half time.
 */
export function estimateTempo(env: ArrayLike<number>, rate: number): { bpm: number; confidence: number } {
  const n = env.length
  // smooth first: sharp onsets on a coarse grid lose their peak at non-integer lags
  const sm = new Float32Array(n)
  const kern = [1, 4, 6, 4, 1]
  for (let i = 0; i < n; i++) {
    let acc = 0
    let w = 0
    for (let j = -2; j <= 2; j++) {
      const v = env[i + j]
      if (v === undefined) continue
      acc += v * kern[j + 2]
      w += kern[j + 2]
    }
    sm[i] = acc / w
  }
  env = sm
  let mean = 0
  for (let i = 0; i < n; i++) mean += env[i]
  mean /= n || 1
  const x = new Float32Array(n)
  let energy = 0
  for (let i = 0; i < n; i++) {
    x[i] = env[i] - mean
    energy += x[i] * x[i]
  }
  if (energy < 1e-9) return { bpm: 0, confidence: 0 }
  // fractional lags (quarter samples, linear interpolation): at a coarse envelope rate the true
  // period rarely lands on an integer lag. each candidate also collects its second harmonic,
  // so the real beat outscores the half-time lag that lines up just as well.
  const ac = (lag: number) => {
    const l0 = Math.floor(lag)
    const f = lag - l0
    if (l0 + 1 >= n) return 0
    let s = 0
    for (let i = l0 + 1; i < n; i++) s += x[i] * (x[i - l0] * (1 - f) + x[i - l0 - 1] * f)
    return s / energy
  }
  let best = 0
  let bestLag = 0
  let bestRaw = 0
  const minLag = (rate * 60) / 180
  const maxLag = (rate * 60) / 70
  for (let lag = minLag; lag <= maxLag && lag < n - 1; lag += 0.25) {
    const raw = ac(lag)
    const bpm = (rate * 60) / lag
    const prior = Math.exp(-0.5 * (Math.log2(bpm / 110) / 0.8) ** 2)
    const score = (raw + 0.5 * Math.max(0, ac(lag * 2))) * (0.6 + 0.4 * prior)
    if (score > best) {
      best = score
      bestLag = lag
      bestRaw = raw
    }
  }
  if (!bestLag) return { bpm: 0, confidence: 0 }
  const lag = bestLag
  return { bpm: (rate * 60) / lag, confidence: Math.max(0, Math.min(1, bestRaw)) }
}

const NOTE_A4 = 440

export class FeatureExtractor {
  private bins: Array<[number, number]>
  private hzPerBin: number
  private prevMag: Float32Array
  private bandDb = new Float32Array(LEVELS).fill(-100)
  private peak = new Float32Array(LEVELS).fill(-60)
  private mean = new Float32Array(LEVELS).fill(-60)
  private fluxMean = new Float32Array(LEVELS).fill(0.01)
  private bandsS = new Float32Array(LEVELS)
  private hitsS = new Float32Array(LEVELS)
  private loudPeak = -40
  private loudS = 0
  private brightS = 0.4
  private chroma = new Float32Array(12)
  private chromaBin: Int16Array
  // tempo
  private envRate = 50
  private env = new Float32Array(50 * 8)
  private envHead = 0
  private envAcc = 0
  private envTime = 0
  private sinceTempo = 0
  private globalFluxMean = 0.01
  private lastOnset = -1
  bpm = 0
  private tempoConf = 0
  private phase = 0
  private beatEnv = 0
  private time = 0

  readonly sampleRate: number
  readonly fftSize: number

  constructor(sampleRate: number, fftSize: number) {
    this.sampleRate = sampleRate
    this.fftSize = fftSize
    this.bins = bandBins(sampleRate, fftSize)
    this.hzPerBin = sampleRate / fftSize
    this.prevMag = new Float32Array(fftSize / 2)
    this.chromaBin = new Int16Array(fftSize / 2).fill(-1)
    for (let i = 1; i < fftSize / 2; i++) {
      const hz = i * this.hzPerBin
      if (hz < 70 || hz > 2200) continue
      const midi = 69 + 12 * Math.log2(hz / NOTE_A4)
      this.chromaBin[i] = ((Math.round(midi) % 12) + 12) % 12
    }
  }

  /** db: analyser float frequency data (dBFS per bin). dt: seconds since the last call. */
  process(db: Float32Array, dt: number): AudioFrame {
    dt = Math.max(1e-3, Math.min(dt, 0.1))
    this.time += dt
    const n = this.fftSize / 2
    const mag = new Float32Array(n)
    let total = 0
    let centroidNum = 0
    for (let i = 0; i < n; i++) {
      const v = db[i]
      const m = Number.isFinite(v) ? 10 ** (v / 20) : 0
      mag[i] = m
      total += m * m
      centroidNum += m * m * i
    }

    const k = (tau: number) => 1 - Math.exp(-dt / tau)
    let globalFlux = 0
    for (let b = 0; b < LEVELS; b++) {
      const [lo, hi] = this.bins[b]
      let p = 0
      let flux = 0
      for (let i = lo; i < hi; i++) {
        p += mag[i] * mag[i]
        // log-compressed positive spectral difference
        const d = Math.log1p(1000 * mag[i]) - Math.log1p(1000 * this.prevMag[i])
        if (d > 0) flux += d
      }
      flux /= hi - lo
      globalFlux += flux
      const levelDb = 10 * Math.log10(p / (hi - lo) + 1e-12)
      this.bandDb[b] = levelDb
      // automatic gain, two parts: presence (is this octave sounding at all, against a slowly
      // falling peak) and motion (where it sits against its own recent average). compressed
      // masters sit near their peak all the time, so motion is what keeps the levels moving.
      this.peak[b] = Math.max(levelDb, this.peak[b] - 4 * dt)
      this.mean[b] += (levelDb - this.mean[b]) * k(4)
      const norm = Math.min(1, Math.max(0, (levelDb - (this.peak[b] - 40)) / 14))
      const motion = Math.min(1, Math.max(0, 0.45 + (levelDb - this.mean[b]) / 10))
      const target = norm * motion
      this.bandsS[b] += (target - this.bandsS[b]) * (target > this.bandsS[b] ? k(0.03) : k(0.22))

      this.fluxMean[b] += (flux - this.fluxMean[b]) * k(1.2)
      const ratio = flux / (this.fluxMean[b] + 1e-4)
      const onset = Math.min(1, Math.max(0, (ratio - 1.5) / 2.5)) * (norm > 0.15 ? 1 : 0)
      this.hitsS[b] = Math.max(this.hitsS[b] * Math.exp(-dt / 0.18), onset)
    }
    this.prevMag = mag

    // loudness
    const rmsDb = 10 * Math.log10(total / n + 1e-12)
    this.loudPeak = Math.max(rmsDb, this.loudPeak - 3 * dt)
    const loud = Math.min(1, Math.max(0, (rmsDb - (this.loudPeak - 36)) / 36)) * (rmsDb > -95 ? 1 : 0)
    this.loudS += (loud - this.loudS) * (loud > this.loudS ? k(0.05) : k(0.35))

    // brightness: log spectral centroid between ~200 Hz and ~6 kHz
    const centroidHz = total > 1e-12 ? (centroidNum / total) * this.hzPerBin : 0
    const bright = centroidHz > 0 ? Math.min(1, Math.max(0, Math.log2(centroidHz / 200) / Math.log2(30))) : 0.4
    this.brightS += (bright - this.brightS) * k(0.8)

    // chroma, slow: the voice should not chase every note
    for (let i = 0; i < n; i++) {
      const c = this.chromaBin[i]
      if (c >= 0) this.chroma[c] += (mag[i] * mag[i] - this.chroma[c]) * k(3) * 0.08
    }

    this.trackBeat(globalFlux, dt)

    const bands = Array.from(this.bandsS)
    const hits = Array.from(this.hitsS)
    let root = 0
    for (let c = 1; c < 12; c++) if (this.chroma[c] > this.chroma[root]) root = c
    const minor = this.chroma[(root + 3) % 12] > this.chroma[(root + 4) % 12]
    return {
      bands,
      hits,
      bass: (bands[0] + bands[1]) / 2,
      mid: (bands[2] + bands[3]) / 2,
      high: (bands[4] + bands[5]) / 2,
      loud: this.loudS,
      beat: this.beatEnv,
      phase: this.phase,
      bright: this.brightS,
      bpm: this.bpm || 96,
      live: true,
      root,
      minor,
    }
  }

  private trackBeat(flux: number, dt: number) {
    // resample the onset envelope to a fixed rate for the autocorrelation
    this.envAcc = Math.max(this.envAcc, flux)
    this.envTime += dt
    const step = 1 / this.envRate
    while (this.envTime >= step) {
      this.envTime -= step
      this.env[this.envHead] = this.envAcc
      this.envHead = (this.envHead + 1) % this.env.length
      this.envAcc = 0
    }
    this.sinceTempo += dt
    if (this.sinceTempo > 0.5 && this.time > 3) {
      this.sinceTempo = 0
      const ordered = new Float32Array(this.env.length)
      for (let i = 0; i < this.env.length; i++) ordered[i] = this.env[(this.envHead + i) % this.env.length]
      const { bpm, confidence } = estimateTempo(ordered, this.envRate)
      if (bpm > 0 && confidence > 0.08) {
        this.bpm = this.bpm ? this.bpm + (bpm - this.bpm) * 0.35 : bpm
        this.tempoConf = confidence
      }
    }

    const bpm = this.bpm || 96
    const period = 60 / bpm
    const prevPhase = this.phase
    this.phase += dt / period
    // phase-locked: onsets near the expected beat pull the phase toward it
    this.globalFluxMean += (flux - this.globalFluxMean) * (1 - Math.exp(-dt / 1.5))
    const isOnset = flux > this.globalFluxMean * 1.8 && this.time - this.lastOnset > period * 0.4
    if (isOnset) {
      this.lastOnset = this.time
      const err = this.phase - Math.round(this.phase)
      if (Math.abs(err) < 0.25) this.phase -= err * (0.25 + 0.3 * this.tempoConf)
    }
    if (Math.floor(this.phase) > Math.floor(prevPhase)) this.beatEnv = 1
    this.phase -= Math.floor(this.phase)
    this.beatEnv *= Math.exp(-dt / (period * 0.35))
  }
}

/**
 * When nothing is playing the fractal still breathes: a slow pulse with soft hits that
 * wander across the levels, so the first paint is already alive.
 */
export class IdleBreath {
  private t = 0
  private hits = new Float32Array(LEVELS)
  private next = 0.4
  process(dt: number): AudioFrame {
    this.t += dt
    const f = silentFrame()
    const bpm = 72
    const phase = (this.t * bpm) / 60
    f.bpm = bpm
    f.phase = phase - Math.floor(phase)
    f.beat = Math.exp(-f.phase * 3) * 0.6
    for (let b = 0; b < LEVELS; b++) {
      f.bands[b] = 0.22 + 0.16 * Math.sin(this.t * 0.31 + b * 1.1)
      this.hits[b] *= Math.exp(-dt / 0.4)
    }
    this.next -= dt
    if (this.next <= 0) {
      this.next = 0.35 + Math.random() * 0.9
      this.hits[Math.floor(Math.random() * LEVELS)] = 0.55 + Math.random() * 0.45
    }
    f.hits = Array.from(this.hits)
    f.loud = 0.25 + 0.08 * Math.sin(this.t * 0.4)
    f.bass = (f.bands[0] + f.bands[1]) / 2
    f.mid = (f.bands[2] + f.bands[3]) / 2
    f.high = (f.bands[4] + f.bands[5]) / 2
    f.bright = 0.45 + 0.1 * Math.sin(this.t * 0.07)
    return f
  }
}
