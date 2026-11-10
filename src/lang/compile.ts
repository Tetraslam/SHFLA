import { FUNCTION_MAP, OUTPUT_NAMES, VARIABLE_NAMES } from './builtins'
import { ShflaError } from './lexer'
import { type Expr, parse, type Stmt } from './parser'

export interface Diagnostic {
  message: string
  from: number
  to: number
}

export interface Program {
  source: string
  stmts: Stmt[]
  /** builtin variables and functions the program reads, so the shader only pays for those */
  uses: Set<string>
  has: { split: boolean; seed: boolean }
  glsl: string
}

export type CompileResult = { ok: true; program: Program } | { ok: false; errors: Diagnostic[] }

/** 0.02 tolerance: states live in 8 or 16 bit textures, so 1/3 is never exactly 1/3. */
export const EQ_EPS = 0.02

export function compile(source: string): CompileResult {
  let stmts: Stmt[]
  try {
    stmts = parse(source)
  } catch (e) {
    if (e instanceof ShflaError) return { ok: false, errors: [{ message: e.message, from: e.from, to: e.to }] }
    throw e
  }

  const errors: Diagnostic[] = []
  const defined = new Set<string>()
  const uses = new Set<string>()

  const visit = (e: Expr): void => {
    switch (e.kind) {
      case 'num':
        return
      case 'name':
        if (VARIABLE_NAMES.has(e.name)) uses.add(e.name)
        else if (!defined.has(e.name)) {
          const fn = FUNCTION_MAP.get(e.name)
          errors.push({
            message: fn
              ? `${e.name} is a function: write ${fn.sig}`
              : `"${e.name}" isn't defined yet. define it on a line above, or check the spelling`,
            from: e.from,
            to: e.to,
          })
        }
        return
      case 'unary':
        visit(e.arg)
        return
      case 'binary':
        visit(e.left)
        visit(e.right)
        return
      case 'if':
        visit(e.cond)
        visit(e.yes)
        visit(e.no)
        return
      case 'call': {
        const fn = FUNCTION_MAP.get(e.name)
        if (!fn) {
          errors.push({ message: `there's no function called "${e.name}"`, from: e.from, to: e.nameTo })
        } else if (fn.arity !== e.args.length) {
          errors.push({
            message: `${fn.sig} takes ${fn.arity} ${fn.arity === 1 ? 'value' : 'values'}, got ${e.args.length}`,
            from: e.from,
            to: e.to,
          })
        } else uses.add(e.name)
        for (const a of e.args) visit(a)
        return
      }
    }
  }

  for (const s of stmts) {
    if (VARIABLE_NAMES.has(s.name)) {
      errors.push({ message: `"${s.name}" is built in; pick another name`, from: s.from, to: s.nameTo })
    } else if (FUNCTION_MAP.has(s.name)) {
      errors.push({ message: `"${s.name}" is a function; pick another name`, from: s.from, to: s.nameTo })
    }
    visit(s.expr)
    defined.add(s.name)
  }
  if (!defined.has('step') && errors.length === 0) {
    errors.push({
      message: 'a program needs a "step" line: the next state of every cell',
      from: source.length,
      to: source.length,
    })
  }
  if (errors.length) return { ok: false, errors }

  return {
    ok: true,
    program: {
      source,
      stmts,
      uses,
      has: { split: defined.has('split'), seed: defined.has('seed') },
      glsl: toGLSL(stmts),
    },
  }
}

// ---------------------------------------------------------------------------
// GLSL

const num = (n: number) => {
  const s = String(n)
  return /[.e]/.test(s) ? s : `${s}.0`
}

const GLSL_FN: Record<string, string> = {
  min: 'min',
  max: 'max',
  clamp: 'clamp',
  mix: 'mix',
  smooth: 'smoothstep',
  mod: 'mod',
  abs: 'abs',
  floor: 'floor',
  ceil: 'ceil',
  fract: 'fract',
  sin: 'sin',
  cos: 'cos',
  wave: 'sh_wave',
  sqrt: 'sh_sqrt',
  exp: 'exp',
  log: 'sh_log',
  nb: 'sh_nb',
  count: 'sh_count',
  hash: 'sh_hash',
}

export function exprToGLSL(e: Expr): string {
  switch (e.kind) {
    case 'num':
      return num(e.value)
    case 'name':
      return VARIABLE_NAMES.has(e.name) ? `b_${e.name}` : `v_${e.name}`
    case 'unary':
      return e.op === '-' ? `(-${exprToGLSL(e.arg)})` : `sh_not(${exprToGLSL(e.arg)})`
    case 'if':
      return `(sh_t(${exprToGLSL(e.cond)}) ? ${exprToGLSL(e.yes)} : ${exprToGLSL(e.no)})`
    case 'call':
      return `${GLSL_FN[e.name]}(${e.args.map(exprToGLSL).join(', ')})`
    case 'binary': {
      const a = exprToGLSL(e.left)
      const b = exprToGLSL(e.right)
      switch (e.op) {
        case '+':
        case '-':
        case '*':
          return `(${a} ${e.op} ${b})`
        case '/':
          return `sh_div(${a}, ${b})`
        case '%':
          return `sh_mod(${a}, ${b})`
        case '^':
          return `sh_pow(${a}, ${b})`
        case '==':
          return `sh_eq(${a}, ${b})`
        case '!=':
          return `(1.0 - sh_eq(${a}, ${b}))`
        case '<':
        case '<=':
        case '>':
        case '>=':
          return `((${a} ${e.op} ${b}) ? 1.0 : 0.0)`
        case 'and':
          return `((sh_t(${a}) && sh_t(${b})) ? 1.0 : 0.0)`
        case 'or':
          return `((sh_t(${a}) || sh_t(${b})) ? 1.0 : 0.0)`
      }
      throw new Error(`unknown operator ${e.op}`)
    }
  }
}

function toGLSL(stmts: Stmt[]): string {
  const declared = new Set<string>()
  const lines: string[] = []
  for (const s of stmts) {
    const decl = declared.has(s.name) ? '' : 'float '
    declared.add(s.name)
    lines.push(`  ${decl}v_${s.name} = ${exprToGLSL(s.expr)};`)
  }
  lines.push(`  float out_step = v_step;`)
  lines.push(`  float out_split = ${declared.has('split') ? 'v_split' : 'b_self'};`)
  lines.push(`  float out_seed = ${declared.has('seed') ? 'v_seed' : '(b_rand < 0.2 ? 1.0 : 0.0)'};`)
  return lines.join('\n')
}

/** GLSL helpers the generated body relies on. Shared by every pass. */
export const GLSL_HELPERS = /* glsl */ `
bool sh_t(float x) { return x >= 0.5; }
float sh_not(float x) { return x >= 0.5 ? 0.0 : 1.0; }
float sh_eq(float a, float b) { return abs(a - b) < ${num(EQ_EPS)} ? 1.0 : 0.0; }
float sh_div(float a, float b) { return abs(b) < 1e-6 ? 0.0 : a / b; }
float sh_mod(float a, float b) { return abs(b) < 1e-6 ? 0.0 : a - b * floor(a / b); }
float sh_pow(float a, float b) { return pow(abs(a), b); }
float sh_sqrt(float x) { return sqrt(max(x, 0.0)); }
float sh_log(float x) { return log(max(x, 1e-6)); }
float sh_wave(float x) { return 0.5 - 0.5 * cos(6.28318530718 * x); }
`

// ---------------------------------------------------------------------------
// CPU interpreter: the same semantics as the shader, used by tests and by the
// turing-completeness check.

export interface CellContext {
  vars: Record<string, number>
  nb: (dx: number, dy: number) => number
  hash: (k: number) => number
}

const t = (x: number) => x >= 0.5
const b2n = (b: boolean) => (b ? 1 : 0)
const smoothstep = (a: number, b: number, x: number) => {
  const k = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return k * k * (3 - 2 * k)
}

export function evalExpr(e: Expr, env: Record<string, number>, ctx: CellContext): number {
  switch (e.kind) {
    case 'num':
      return e.value
    case 'name':
      return env[e.name] ?? ctx.vars[e.name] ?? 0
    case 'unary': {
      const v = evalExpr(e.arg, env, ctx)
      return e.op === '-' ? -v : b2n(!t(v))
    }
    case 'if':
      return t(evalExpr(e.cond, env, ctx)) ? evalExpr(e.yes, env, ctx) : evalExpr(e.no, env, ctx)
    case 'call': {
      const a = e.args.map((x) => evalExpr(x, env, ctx))
      switch (e.name) {
        case 'nb':
          return ctx.nb(a[0], a[1])
        case 'count': {
          let c = 0
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              if ((dx || dy) && Math.abs(ctx.nb(dx, dy) - a[0]) < EQ_EPS) c++
            }
          return c
        }
        case 'hash':
          return ctx.hash(a[0])
        case 'min':
          return Math.min(a[0], a[1])
        case 'max':
          return Math.max(a[0], a[1])
        case 'clamp':
          return Math.min(a[2], Math.max(a[1], a[0]))
        case 'mix':
          return a[0] + (a[1] - a[0]) * a[2]
        case 'smooth':
          return smoothstep(a[0], a[1], a[2])
        case 'mod':
          return a[1] === 0 ? 0 : a[0] - a[1] * Math.floor(a[0] / a[1])
        case 'abs':
          return Math.abs(a[0])
        case 'floor':
          return Math.floor(a[0])
        case 'ceil':
          return Math.ceil(a[0])
        case 'fract':
          return a[0] - Math.floor(a[0])
        case 'sin':
          return Math.sin(a[0])
        case 'cos':
          return Math.cos(a[0])
        case 'wave':
          return 0.5 - 0.5 * Math.cos(2 * Math.PI * a[0])
        case 'sqrt':
          return Math.sqrt(Math.max(0, a[0]))
        case 'exp':
          return Math.exp(a[0])
        case 'log':
          return Math.log(Math.max(1e-6, a[0]))
      }
      throw new Error(`unknown function ${e.name}`)
    }
    case 'binary': {
      if (e.op === 'and') return b2n(t(evalExpr(e.left, env, ctx)) && t(evalExpr(e.right, env, ctx)))
      if (e.op === 'or') return b2n(t(evalExpr(e.left, env, ctx)) || t(evalExpr(e.right, env, ctx)))
      const a = evalExpr(e.left, env, ctx)
      const b = evalExpr(e.right, env, ctx)
      switch (e.op) {
        case '+':
          return a + b
        case '-':
          return a - b
        case '*':
          return a * b
        case '/':
          return Math.abs(b) < 1e-6 ? 0 : a / b
        case '%':
          return Math.abs(b) < 1e-6 ? 0 : a - b * Math.floor(a / b)
        case '^':
          return Math.abs(a) ** b
        case '==':
          return b2n(Math.abs(a - b) < EQ_EPS)
        case '!=':
          return b2n(Math.abs(a - b) >= EQ_EPS)
        case '<':
          return b2n(a < b)
        case '<=':
          return b2n(a <= b)
        case '>':
          return b2n(a > b)
        case '>=':
          return b2n(a >= b)
      }
      throw new Error(`unknown operator ${e.op}`)
    }
  }
}

/** Run every statement for one cell and return the outputs, clamped like the shader clamps them. */
export function evalCell(stmts: Stmt[], ctx: CellContext): { step: number; split: number; seed: number } {
  const env: Record<string, number> = {}
  for (const s of stmts) env[s.name] = evalExpr(s.expr, env, ctx)
  const clamp01 = (x: number) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0)
  return {
    step: clamp01(env.step),
    split: clamp01('split' in env ? env.split : (ctx.vars.self ?? 0)),
    seed: clamp01('seed' in env ? env.seed : b2n((ctx.vars.rand ?? 0) < 0.2)),
  }
}

export { OUTPUT_NAMES }
