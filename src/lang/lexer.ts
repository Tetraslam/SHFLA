export type TokenKind = 'num' | 'ident' | 'op' | 'lparen' | 'rparen' | 'comma' | 'assign' | 'newline' | 'eof'

export interface Token {
  kind: TokenKind
  value: string
  /** offset into the source, in UTF-16 code units */
  from: number
  to: number
}

export class ShflaError extends Error {
  from: number
  to: number
  constructor(message: string, from: number, to: number) {
    super(message)
    this.from = from
    this.to = to
  }
}

const KEYWORD_OPS = new Set(['and', 'or', 'not', 'if', 'then', 'else'])
const TWO_CHAR = ['==', '!=', '<=', '>=']
const ONE_CHAR = '+-*/%<>^'

export function lex(src: string): Token[] {
  const tokens: Token[] = []
  let i = 0
  let depth = 0
  const push = (kind: TokenKind, value: string, from: number) => tokens.push({ kind, value, from, to: i })

  while (i < src.length) {
    const c = src[i]
    if (c === '#') {
      while (i < src.length && src[i] !== '\n') i++
      continue
    }
    if (c === '\n') {
      const from = i++
      // newlines inside parentheses continue the statement
      if (depth === 0) push('newline', '\n', from)
      continue
    }
    if (c === ' ' || c === '\t' || c === '\r') {
      i++
      continue
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      const from = i
      while (i < src.length && /[0-9]/.test(src[i])) i++
      if (src[i] === '.') {
        i++
        while (i < src.length && /[0-9]/.test(src[i])) i++
      }
      push('num', src.slice(from, i), from)
      continue
    }
    if (/[A-Za-z_]/.test(c)) {
      const from = i
      while (i < src.length && /[A-Za-z0-9_]/.test(src[i])) i++
      const word = src.slice(from, i)
      push(KEYWORD_OPS.has(word) ? 'op' : 'ident', word, from)
      continue
    }
    const two = src.slice(i, i + 2)
    if (TWO_CHAR.includes(two)) {
      const from = i
      i += 2
      push('op', two, from)
      continue
    }
    if (ONE_CHAR.includes(c)) {
      const from = i++
      push('op', c, from)
      continue
    }
    if (c === '=') {
      const from = i++
      push('assign', '=', from)
      continue
    }
    if (c === '(') {
      const from = i++
      depth++
      push('lparen', c, from)
      continue
    }
    if (c === ')') {
      const from = i++
      depth = Math.max(0, depth - 1)
      push('rparen', c, from)
      continue
    }
    if (c === ',') {
      const from = i++
      push('comma', c, from)
      continue
    }
    throw new ShflaError(`i don't know the character "${c}"`, i, i + 1)
  }
  tokens.push({ kind: 'eof', value: '', from: src.length, to: src.length })
  return tokens
}
