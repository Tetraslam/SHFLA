/**
 * The audio proxy only fetches public http(s) audio. These checks keep it from being
 * pointed at private networks or used as a general-purpose proxy.
 */

export class GuardError extends Error {
  readonly status: number
  constructor(message: string, status = 400) {
    super(message)
    this.status = status
  }
}

export function checkUrl(raw: string | null): URL {
  if (!raw) throw new GuardError('missing url')
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new GuardError('not a url')
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') throw new GuardError('only http and https')
  if (url.port && url.port !== '80' && url.port !== '443') throw new GuardError('only the default ports')
  if (url.username || url.password) throw new GuardError('no credentials in links')
  const host = url.hostname.toLowerCase()
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) {
    throw new GuardError('that host is private')
  }
  return url
}

function v4Private(ip: string): boolean {
  const p = ip.split('.').map(Number)
  if (p.length !== 4 || p.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return true
  const [a, b] = p
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 192 && b === 0) ||
    (a === 198 && (b === 18 || b === 19)) ||
    a >= 224
  )
}

/** true for loopback, private, link-local, carrier-grade nat, multicast, and reserved addresses */
export function isPrivateAddress(ip: string): boolean {
  const s = ip.toLowerCase()
  if (s.includes('.') && !s.includes(':')) return v4Private(s)
  const mapped = s.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/)
  if (mapped) return v4Private(mapped[1])
  if (s === '::' || s === '::1') return true
  const first = Number.parseInt(s.split(':')[0] || '0', 16)
  return (
    (first & 0xfe00) === 0xfc00 || // unique local
    (first & 0xffc0) === 0xfe80 || // link local
    (first & 0xff00) === 0xff00 || // multicast
    s.startsWith('64:ff9b:') || // nat64 can reach v4 private space
    s.startsWith('2002:') // 6to4 likewise
  )
}

const AUDIO_EXT = /\.(mp3|ogg|oga|opus|wav|flac|m4a|aac|webm|mp4)$/i

export function isAudioResponse(contentType: string | null, url: URL): boolean {
  const ct = (contentType ?? '').split(';')[0].trim().toLowerCase()
  if (ct.startsWith('audio/') || ct.startsWith('video/') || ct === 'application/ogg') return true
  if (
    (ct === 'application/octet-stream' || ct === 'binary/octet-stream' || ct === '') &&
    AUDIO_EXT.test(url.pathname)
  ) {
    return true
  }
  return false
}
