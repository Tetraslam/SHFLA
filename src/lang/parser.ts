import { lex, ShflaError, type Token } from './lexer'

export type Expr =
  | { kind: 'num'; value: number; from: number; to: number }
  | { kind: 'name'; name: string; from: number; to: number }
  | { kind: 'unary'; op: '-' | 'not'; arg: Expr; from: number; to: number }
  | { kind: 'binary'; op: string; left: Expr; right: Expr; from: number; to: number }
  | { kind: 'if'; cond: Expr; yes: Expr; no: Expr; from: number; to: number }
  | { kind: 'call'; name: string; args: Expr[]; from: number; to: number; nameTo: number }

export interface Stmt {
  name: string
  expr: Expr
  from: number
  nameTo: number
}

export function parse(src: string): Stmt[] {
  const tokens = lex(src)
  let p = 0
  const peek = () => tokens[p]
  const next = () => tokens[p++]
  const isOp = (v: string) => peek().kind === 'op' && peek().value === v
  const expectOp = (v: string) => {
    if (!isOp(v)) fail(`expected "${v}" here`, peek())
    return next()
  }
  function fail(message: string, t: Token): never {
    throw new ShflaError(message, t.from, Math.max(t.to, t.from + 1))
  }

  function expr(): Expr {
    if (isOp('if')) {
      const start = next()
      const cond = expr()
      expectOp('then')
      const a = expr()
      expectOp('else')
      const b = expr()
      return { kind: 'if', cond, yes: a, no: b, from: start.from, to: b.to }
    }
    return or()
  }

  function binaryLevel(ops: string[], sub: () => Expr, chain = true): () => Expr {
    return () => {
      let left = sub()
      while (peek().kind === 'op' && ops.includes(peek().value)) {
        const op = next().value
        const right = sub()
        left = { kind: 'binary', op, left, right, from: left.from, to: right.to }
        if (!chain && peek().kind === 'op' && ops.includes(peek().value)) {
          fail('comparisons do not chain; use "and" to join them', peek())
        }
      }
      return left
    }
  }

  function notExpr(): Expr {
    if (isOp('not')) {
      const t = next()
      const arg = notExpr()
      return { kind: 'unary', op: 'not', arg, from: t.from, to: arg.to }
    }
    return comparison()
  }

  function unary(): Expr {
    if (isOp('-')) {
      const t = next()
      const arg = unary()
      return { kind: 'unary', op: '-', arg, from: t.from, to: arg.to }
    }
    return power()
  }

  function power(): Expr {
    const base = primary()
    if (isOp('^')) {
      next()
      const exp = unary() // right-associative, allows 2^-1
      return { kind: 'binary', op: '^', left: base, right: exp, from: base.from, to: exp.to }
    }
    return base
  }

  const multiplicative = binaryLevel(['*', '/', '%'], unary)
  const additive = binaryLevel(['+', '-'], multiplicative)
  const comparison = binaryLevel(['==', '!=', '<', '<=', '>', '>='], additive, false)
  const and = binaryLevel(['and'], notExpr)
  const or = binaryLevel(['or'], and)

  function primary(): Expr {
    const t = peek()
    if (t.kind === 'num') {
      next()
      return { kind: 'num', value: Number(t.value), from: t.from, to: t.to }
    }
    if (t.kind === 'ident') {
      next()
      if (peek().kind === 'lparen') {
        next()
        const args: Expr[] = []
        if (peek().kind !== 'rparen') {
          for (;;) {
            args.push(expr())
            if (peek().kind === 'comma') {
              next()
              continue
            }
            break
          }
        }
        if (peek().kind !== 'rparen') fail('expected ")" to close the call', peek())
        const close = next()
        return { kind: 'call', name: t.value, args, from: t.from, to: close.to, nameTo: t.to }
      }
      return { kind: 'name', name: t.value, from: t.from, to: t.to }
    }
    if (t.kind === 'lparen') {
      next()
      const e = expr()
      if (peek().kind !== 'rparen') fail('expected ")"', peek())
      next()
      return e
    }
    if (t.kind === 'eof' || t.kind === 'newline') fail('this line ends too early', t)
    fail(`didn't expect "${t.value}" here`, t)
  }

  const stmts: Stmt[] = []
  while (peek().kind !== 'eof') {
    if (peek().kind === 'newline') {
      next()
      continue
    }
    const nameTok = peek()
    if (nameTok.kind !== 'ident') fail('each line should look like: name = expression', nameTok)
    next()
    if (peek().kind !== 'assign') fail(`expected "=" after "${nameTok.value}"`, peek())
    next()
    const e = expr()
    if (peek().kind !== 'newline' && peek().kind !== 'eof') {
      fail(`didn't expect "${peek().value}" here; one statement per line`, peek())
    }
    stmts.push({ name: nameTok.value, expr: e, from: nameTok.from, nameTo: nameTok.to })
  }
  return stmts
}
