/** Every name the language knows, with the one-line explanation the editor shows. */

export interface VarDoc {
  name: string
  doc: string
  group: 'cell' | 'scale' | 'time' | 'music'
}

export const VARIABLES: VarDoc[] = [
  { name: 'self', group: 'cell', doc: 'this cell, 0 to 1. in split, the parent being divided' },
  { name: 'sum', group: 'cell', doc: 'the 8 neighbors added up' },
  { name: 'rand', group: 'cell', doc: 'a fresh random number, per cell, per step' },
  { name: 'x', group: 'cell', doc: 'column of this cell' },
  { name: 'y', group: 'cell', doc: 'row of this cell' },
  { name: 'qx', group: 'cell', doc: 'x mod 2: which half of its parent this cell sits in' },
  { name: 'qy', group: 'cell', doc: 'y mod 2' },
  { name: 'up', group: 'scale', doc: 'the coarser cell that contains this one' },
  { name: 'down', group: 'scale', doc: 'mean of the four finer cells inside this one' },
  { name: 'level', group: 'scale', doc: '0 is the coarsest level, levels - 1 the finest' },
  { name: 'levels', group: 'scale', doc: 'how many levels are alive at once' },
  { name: 'size', group: 'scale', doc: 'cells across this level' },
  { name: 'depth', group: 'scale', doc: 'octaves zoomed since the start' },
  { name: 'tick', group: 'time', doc: 'steps this level has taken' },
  { name: 't', group: 'time', doc: 'seconds since the start' },
  { name: 'band', group: 'music', doc: "loudness of this level's octave of the song, 0 to 1" },
  { name: 'hit', group: 'music', doc: "an onset in this level's octave: jumps toward 1, then decays" },
  { name: 'bass', group: 'music', doc: 'low end, 0 to 1' },
  { name: 'mid', group: 'music', doc: 'middle, 0 to 1' },
  { name: 'high', group: 'music', doc: 'top end, 0 to 1' },
  { name: 'loud', group: 'music', doc: 'overall loudness, 0 to 1' },
  { name: 'beat', group: 'music', doc: '1 on the beat, falling to 0 before the next' },
  { name: 'phase', group: 'music', doc: 'where we are inside the beat, 0 to 1' },
  { name: 'bright', group: 'music', doc: 'spectral brightness of the song, 0 to 1' },
]

export interface FnDoc {
  name: string
  arity: number
  sig: string
  doc: string
}

export const FUNCTIONS: FnDoc[] = [
  { name: 'nb', arity: 2, sig: 'nb(dx, dy)', doc: 'the neighbor dx across and dy down' },
  { name: 'count', arity: 1, sig: 'count(v)', doc: 'how many of the 8 neighbors equal v' },
  { name: 'hash', arity: 1, sig: 'hash(k)', doc: 'a random number per cell, per step, per k' },
  { name: 'min', arity: 2, sig: 'min(a, b)', doc: 'the smaller' },
  { name: 'max', arity: 2, sig: 'max(a, b)', doc: 'the larger' },
  { name: 'clamp', arity: 3, sig: 'clamp(x, lo, hi)', doc: 'x held between lo and hi' },
  { name: 'mix', arity: 3, sig: 'mix(a, b, k)', doc: 'a when k is 0, b when k is 1' },
  { name: 'smooth', arity: 3, sig: 'smooth(lo, hi, x)', doc: 'a soft 0 to 1 ramp as x crosses lo..hi' },
  { name: 'mod', arity: 2, sig: 'mod(a, b)', doc: 'a % b, always positive' },
  { name: 'abs', arity: 1, sig: 'abs(x)', doc: 'distance from 0' },
  { name: 'floor', arity: 1, sig: 'floor(x)', doc: 'round down' },
  { name: 'ceil', arity: 1, sig: 'ceil(x)', doc: 'round up' },
  { name: 'fract', arity: 1, sig: 'fract(x)', doc: 'the part after the point' },
  { name: 'sin', arity: 1, sig: 'sin(x)', doc: 'sine, radians' },
  { name: 'cos', arity: 1, sig: 'cos(x)', doc: 'cosine, radians' },
  { name: 'wave', arity: 1, sig: 'wave(x)', doc: '0 to 1 and back once per unit of x' },
  { name: 'sqrt', arity: 1, sig: 'sqrt(x)', doc: 'square root' },
  { name: 'exp', arity: 1, sig: 'exp(x)', doc: 'e to the x' },
  { name: 'log', arity: 1, sig: 'log(x)', doc: 'natural log' },
]

export const OUTPUTS = [
  { name: 'step', doc: 'required. the next state of every cell, every step' },
  { name: 'split', doc: 'what a new finer cell becomes when the zoom passes an octave. default: self' },
  { name: 'seed', doc: 'the state everything starts from. default: rand < 0.2' },
] as const

export const VARIABLE_NAMES = new Set(VARIABLES.map((v) => v.name))
export const FUNCTION_MAP = new Map(FUNCTIONS.map((f) => [f.name, f]))
export const OUTPUT_NAMES = new Set<string>(OUTPUTS.map((o) => o.name))
