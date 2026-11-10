/** Everything the fractal hears about the song, once per animation frame. */
export const LEVELS = 6

export interface AudioFrame {
  /** loudness per octave band, coarse (sub) to fine (air), 0..1 */
  bands: number[]
  /** onset envelope per band, 0..1 */
  hits: number[]
  bass: number
  mid: number
  high: number
  loud: number
  /** 1 on the beat, decaying toward the next */
  beat: number
  /** position inside the current beat, 0..1 */
  phase: number
  bright: number
  bpm: number
  /** true when this frame comes from a real source rather than the idle breath */
  live: boolean
  /** dominant pitch class 0..11 and whether the song leans minor, for the voice */
  root: number
  minor: boolean
}

export function silentFrame(): AudioFrame {
  return {
    bands: Array(LEVELS).fill(0),
    hits: Array(LEVELS).fill(0),
    bass: 0,
    mid: 0,
    high: 0,
    loud: 0,
    beat: 0,
    phase: 0,
    bright: 0.4,
    bpm: 84,
    live: false,
    root: 4,
    minor: false,
  }
}
