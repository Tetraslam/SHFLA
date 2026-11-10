import { describe, expect, it } from 'vitest'
import { type CellContext, compile, evalCell } from './compile'
import type { Stmt } from './parser'
import { PRESETS } from './presets'

function ok(src: string) {
  const r = compile(src)
  if (!r.ok) throw new Error(r.errors.map((e) => e.message).join('; '))
  return r.program
}

/** Run a program on a torus on the cpu, one level, no audio. */
function run(stmts: Stmt[], grid: number[][], steps: number, extra: (y: number, step: number) => object = () => ({})) {
  let g = grid.map((r) => [...r])
  const h = g.length
  const w = g[0].length
  for (let s = 0; s < steps; s++) {
    const prev = g
    g = prev.map((row, y) =>
      row.map((self, x) => {
        const nb = (dx: number, dy: number) => prev[(((y + dy) % h) + h) % h][(((x + dx) % w) + w) % w]
        let sum = 0
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx || dy) sum += nb(dx, dy)
        const ctx: CellContext = {
          vars: { self, sum, x, y, qx: x % 2, qy: y % 2, size: w, rand: 0.99, hit: 0, tick: s + 1, ...extra(y, s) },
          nb,
          hash: () => 0.99,
        }
        return evalCell(stmts, ctx).step
      }),
    )
  }
  return g
}

const cells = (g: number[][]) => g.flatMap((row, y) => row.flatMap((v, x) => (v > 0.5 ? [`${x},${y}`] : []))).sort()

describe('parser and checker', () => {
  it('compiles every preset', () => {
    for (const p of PRESETS) expect(compile(p.source).ok, p.id).toBe(true)
  })

  it('requires a step line', () => {
    const r = compile('a = 1\n')
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.errors[0].message).toMatch(/step/)
  })

  it('points at undefined names with their position', () => {
    const src = 'step = sefl + 1'
    const r = compile(src)
    expect(r.ok).toBe(false)
    if (!r.ok) {
      expect(r.errors[0].message).toMatch(/sefl/)
      expect(src.slice(r.errors[0].from, r.errors[0].to)).toBe('sefl')
    }
  })

  it('refuses to reassign builtins and checks arity', () => {
    const a = compile('self = 1\nstep = 0')
    expect(a.ok).toBe(false)
    const b = compile('step = nb(1)')
    expect(b.ok).toBe(false)
    if (!b.ok) expect(b.errors[0].message).toMatch(/nb\(dx, dy\)/)
  })

  it('rejects chained comparisons and stray tokens', () => {
    expect(compile('step = 1 < 2 < 3').ok).toBe(false)
    expect(compile('step = 1 2').ok).toBe(false)
    expect(compile('step = @').ok).toBe(false)
  })

  it('continues statements inside parentheses and skips comments', () => {
    const p = ok('# hi\nstep = (1 +\n  2) # trailing\n')
    expect(p.stmts).toHaveLength(1)
  })

  it('tracks which builtins a program reads', () => {
    const p = ok('step = sum + up')
    expect(p.uses.has('sum')).toBe(true)
    expect(p.uses.has('up')).toBe(true)
    expect(p.uses.has('down')).toBe(false)
  })

  it('emits glsl that only references declared names', () => {
    const p = ok('a = self * 2\na = a + 1\nstep = if a > 1 then 1 else 0')
    expect(p.glsl).toContain('float v_a = (b_self * 2.0);')
    expect(p.glsl).toContain('  v_a = (v_a + 1.0);')
    expect(p.glsl).toMatch(/out_split = b_self/)
  })
})

describe('semantics', () => {
  it('has fuzzy equality, truthiness at 0.5, and safe division', () => {
    const p = ok('a = 0.33 == 0.34\nb = 0.5 and 0.49\nc = 1 / 0\nstep = a * 4 + b * 2 + c')
    const r = evalCell(p.stmts, { vars: {}, nb: () => 0, hash: () => 0 })
    expect(r.step).toBe(1) // clamped from 4
    const env = p.stmts.map((s) => s.name)
    expect(env).toEqual(['a', 'b', 'c', 'step'])
  })

  it("runs conway's life: a glider moves one cell diagonally every four steps", () => {
    const life = ok(PRESETS[0].source).stmts
    const n = 10
    const g = Array.from({ length: n }, () => Array(n).fill(0))
    for (const [x, y] of [
      [1, 0],
      [2, 1],
      [0, 2],
      [1, 2],
      [2, 2],
    ])
      g[y][x] = 1
    const after = run(life, g, 4)
    const moved = cells(g).map((c) => {
      const [x, y] = c.split(',').map(Number)
      return `${x + 1},${y + 1}`
    })
    expect(cells(after)).toEqual(moved.sort())
  })

  it('runs rule 110 exactly, row by row', () => {
    const rule = ok(PRESETS.find((p) => p.id === 'rule110')?.source ?? '').stmts
    const w = 16
    const rows = 8
    const g = Array.from({ length: rows }, () => Array(w).fill(0))
    g[0][w - 1] = 1
    // step s writes row s+1 (tick = s + 1), like the shader
    const out = run(rule, g, rows - 1)
    // reference rule 110
    const ref: number[][] = [g[0]]
    for (let r = 1; r < rows; r++) {
      const p = ref[r - 1]
      ref.push(p.map((_, x) => (110 >> ((p[(x - 1 + w) % w] << 2) | (p[x] << 1) | p[(x + 1) % w])) & 1))
    }
    expect(out.map((r) => r.map((v) => (v > 0.5 ? 1 : 0)))).toEqual(ref)
  })
})
