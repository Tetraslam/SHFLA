import { lookup } from 'node:dns/promises'
import { checkUrl, GuardError, isAudioResponse, isPrivateAddress } from './_guard.js'

export const config = { maxDuration: 60 }

const MAX_BYTES = 80 * 1024 * 1024
const MAX_HOPS = 4

async function assertPublic(url: URL) {
  const addrs = await lookup(url.hostname, { all: true, verbatim: true }).catch(() => {
    throw new GuardError("that host doesn't resolve", 502)
  })
  if (addrs.length === 0 || addrs.some((a) => isPrivateAddress(a.address))) {
    throw new GuardError('that host is private', 403)
  }
}

/**
 * Streams a public audio file to the browser from our own origin, so Web Audio can analyse it
 * when the original host doesn't send CORS headers. Range requests pass through for seeking.
 */
export async function GET(request: Request): Promise<Response> {
  try {
    let url = checkUrl(new URL(request.url).searchParams.get('url'))
    const headers = new Headers({ 'user-agent': 'shfla-audio/1.0', accept: 'audio/*,video/*;q=0.8,*/*;q=0.5' })
    const range = request.headers.get('range')
    if (range && /^bytes=\d*-\d*$/.test(range)) headers.set('range', range)

    let res: Response | null = null
    for (let hop = 0; hop <= MAX_HOPS; hop++) {
      await assertPublic(url)
      res = await fetch(url, { headers, redirect: 'manual', signal: request.signal })
      if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
        url = checkUrl(new URL(res.headers.get('location') as string, url).href)
        continue
      }
      break
    }
    if (!res || (res.status >= 300 && res.status < 400)) throw new GuardError('too many redirects', 502)
    if (!res.ok) throw new GuardError(`the host answered ${res.status}`, 502)
    if (!isAudioResponse(res.headers.get('content-type'), url)) {
      await res.body?.cancel()
      throw new GuardError("that link isn't an audio file", 415)
    }
    const length = Number(res.headers.get('content-length') ?? 0)
    if (length > MAX_BYTES) {
      await res.body?.cancel()
      throw new GuardError('that file is too big (80 MB max)', 413)
    }

    let seen = 0
    const cap = new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        seen += chunk.byteLength
        if (seen > MAX_BYTES) controller.error(new Error('too big'))
        else controller.enqueue(chunk)
      },
    })
    const out = new Headers()
    for (const h of ['content-type', 'content-length', 'content-range', 'accept-ranges', 'last-modified', 'etag']) {
      const v = res.headers.get(h)
      if (v) out.set(h, v)
    }
    out.set('cache-control', 'public, max-age=3600')
    out.set('x-content-type-options', 'nosniff')
    out.set('content-security-policy', "default-src 'none'; sandbox")
    return new Response(res.body?.pipeThrough(cap) ?? null, { status: res.status, headers: out })
  } catch (e) {
    const status = e instanceof GuardError ? e.status : 502
    const message = e instanceof GuardError ? e.message : 'could not reach that link'
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
    })
  }
}
