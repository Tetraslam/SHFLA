export type RGB = [number, number, number]

export interface Palette {
  id: string
  name: string
  /** coarse (deep, cool) to fine (bright, warm) */
  stops: [RGB, RGB, RGB]
  /** the floor the light sits on */
  ground: RGB
  /** the milky lift that rises with loudness */
  haze: RGB
}

const hex = (h: string): RGB => {
  const n = Number.parseInt(h.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export const PALETTES: Palette[] = [
  {
    id: 'dusk',
    name: 'dusk',
    stops: [hex('#3d3c94'), hex('#e0628f'), hex('#ffd3ab')],
    ground: hex('#07060a'),
    haze: hex('#2a1630'),
  },
  {
    id: 'bleach',
    name: 'bleach',
    stops: [hex('#c76b98'), hex('#f6b4a6'), hex('#fff6ec')],
    ground: hex('#2b2026'),
    haze: hex('#8a5f72'),
  },
  {
    id: 'tape',
    name: 'tape',
    stops: [hex('#1e6e69'), hex('#cdb65a'), hex('#ffeccc')],
    ground: hex('#040706'),
    haze: hex('#10271f'),
  },
  {
    id: 'sea',
    name: 'sea',
    stops: [hex('#173a73'), hex('#3ea8b6'), hex('#e8fbff')],
    ground: hex('#03060b'),
    haze: hex('#0c2236'),
  },
]

export function paletteById(id: string): Palette {
  return PALETTES.find((p) => p.id === id) ?? PALETTES[0]
}

/** sample the coarse-to-fine ramp at t in 0..1 */
export function ramp(p: Palette, t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * 2
  const i = Math.min(1, Math.floor(x))
  const f = x - i
  const a = p.stops[i]
  const b = p.stops[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}
