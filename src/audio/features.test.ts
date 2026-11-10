import { describe, expect, it } from 'vitest'
import { BAND_EDGES, bandBins, estimateTempo, FeatureExtractor } from './features'
import { LEVELS } from './frame'

/** an onset envelope at `rate` Hz: strong pulses on beats, weaker on eighths, a little noise */
function envelope(bpm: number, seconds: number, rate: number, eighths = 0.4) {
  const n = Math.round(seconds * rate)
  const out = new Float32Array(n)
  const beat = (60 / bpm) * rate
  let seed = 7
  const rnd = () => {
    seed = (seed * 16807) % 2147483647
    return seed / 2147483647
  }
  for (let i = 0; i < n; i++) {
    const pos = i / beat
    const fb = pos - Math.floor(pos)
    const fe = pos * 2 - Math.floor(pos * 2)
    out[i] = Math.exp(-fb * 25) + eighths * Math.exp(-fe * 25) + 0.08 * rnd()
  }
  return out
}

describe('tempo', () => {
  for (const bpm of [78, 96, 120, 140]) {
    it(`finds ${bpm} bpm`, () => {
      const { bpm: got, confidence } = estimateTempo(envelope(bpm, 8, 50), 50)
      expect(Math.abs(got - bpm)).toBeLessThan(2.5)
      expect(confidence).toBeGreaterThan(0.1)
    })
  }

  it('hears very fast music at its felt half time, never something unrelated', () => {
    const { bpm } = estimateTempo(envelope(172, 8, 50), 50)
    expect(Math.min(Math.abs(bpm - 172), Math.abs(bpm - 86))).toBeLessThan(2.5)
  })

  it('reports nothing for silence', () => {
    expect(estimateTempo(new Float32Array(400), 50).bpm).toBe(0)
  })
})

describe('bands', () => {
  it('covers every octave band without gaps at 48k', () => {
    const bins = bandBins(48000, 4096)
    expect(bins).toHaveLength(LEVELS)
    for (let b = 1; b < LEVELS; b++) expect(bins[b][0]).toBeLessThanOrEqual(bins[b - 1][1])
    expect(bins[LEVELS - 1][1] * (48000 / 4096)).toBeGreaterThanOrEqual(BAND_EDGES[LEVELS] - 48000 / 4096)
  })

  it('lights the band a tone sits in, and only that one', () => {
    const fft = 4096
    const sr = 48000
    const fx = new FeatureExtractor(sr, fft)
    const tone = 1000 // the high-mid band, index 4 is 2000..5500, 3 is 700..2000
    const db = new Float32Array(fft / 2).fill(-140)
    const bin = Math.round(tone / (sr / fft))
    for (let i = bin - 2; i <= bin + 2; i++) db[i] = -12
    let f = fx.process(db, 1 / 60)
    for (let i = 0; i < 90; i++) f = fx.process(db, 1 / 60)
    const loudest = f.bands.indexOf(Math.max(...f.bands))
    expect(loudest).toBe(3)
    expect(f.bands[0]).toBeLessThan(0.05)
  })
})
