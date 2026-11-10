/** Pure helpers for the youtube endpoint: which links we accept and which audio we take. */

const HOSTS = new Set([
  'youtube.com',
  'www.youtube.com',
  'm.youtube.com',
  'music.youtube.com',
  'youtu.be',
  'www.youtube-nocookie.com',
])

const ID = /^[A-Za-z0-9_-]{11}$/

/**
 * A canonical watch url for a youtube video link, or null for anything else.
 * yt-dlp is only ever handed this canonical form, never the raw input, so its
 * generic extractor can't be pointed at other hosts.
 */
export function youtubeId(raw: string | null | undefined): string | null {
  if (!raw) return null
  let u: URL
  try {
    u = new URL(raw.trim())
  } catch {
    return null
  }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') return null
  const host = u.hostname.toLowerCase()
  if (!HOSTS.has(host)) return null
  let id: string | null = null
  if (host === 'youtu.be') id = u.pathname.split('/')[1] ?? null
  else if (u.pathname === '/watch') id = u.searchParams.get('v')
  else {
    const m = u.pathname.match(/^\/(?:shorts|embed|live|v)\/([^/?#]+)/)
    id = m ? m[1] : null
  }
  return id && ID.test(id) ? id : null
}

export const isYouTube = (raw: string) => youtubeId(raw) !== null

export interface YtFormat {
  format_id: string
  ext?: string
  vcodec?: string | null
  acodec?: string | null
  abr?: number | null
  protocol?: string
  url?: string
  filesize?: number | null
  filesize_approx?: number | null
  http_headers?: Record<string, string>
}

/** audio only over plain https; aac in m4a first (every browser plays it), then opus, best bitrate under ~192k */
export function pickAudio(formats: YtFormat[]): YtFormat | null {
  const ok = formats.filter(
    (f) =>
      f.url && f.protocol === 'https' && (f.vcodec === 'none' || f.vcodec == null) && f.acodec && f.acodec !== 'none',
  )
  const score = (f: YtFormat) => {
    const abr = f.abr ?? 0
    const codec = f.ext === 'm4a' ? 1000 : f.ext === 'webm' ? 500 : 0
    return codec + Math.min(abr, 192)
  }
  return ok.sort((a, b) => score(b) - score(a))[0] ?? null
}

export function mimeFor(f: YtFormat): string {
  if (f.ext === 'm4a' || f.ext === 'mp4') return 'audio/mp4'
  if (f.ext === 'webm') return 'audio/webm'
  return 'application/octet-stream'
}

/** turn yt-dlp's stderr into something a person can act on */
export function explain(stderr: string): { status: number; message: string } {
  const s = stderr.toLowerCase()
  if (s.includes('sign in to confirm') || s.includes('not a bot')) {
    return {
      status: 503,
      message: "youtube won't talk to this server right now (it wants a sign-in). try a file or a direct audio link",
    }
  }
  if (s.includes('private video')) return { status: 404, message: 'that video is private' }
  if (s.includes('age')) return { status: 403, message: 'that video is age-restricted, so it needs a sign-in' }
  if (s.includes('live event') || s.includes('is live') || s.includes('live stream')) {
    return { status: 422, message: 'live streams have no end to hold on to. pick a regular video' }
  }
  if (s.includes('unavailable') || s.includes('not available')) {
    return { status: 404, message: "that video isn't available here" }
  }
  return { status: 502, message: "couldn't get the audio out of that video" }
}
