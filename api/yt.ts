import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { constants } from 'node:fs'
import { access, chmod, copyFile, unlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { explain, mimeFor, pickAudio, type YtFormat, youtubeId } from './_youtube.js'

export const config = { maxDuration: 60 }

const MAX_SECONDS = 30 * 60
const CLIENTS = ['', 'tv_simply']
const MAX_BYTES = 80 * 1024 * 1024
const ENV = { ...process.env, HOME: '/tmp', XDG_CACHE_HOME: '/tmp/yt-dlp-cache', TMPDIR: '/tmp' }
/** yt-dlp needs a javascript runtime for youtube's player challenges; node is already here */
const JS = ['--js-runtimes', `node:${process.execPath}`, '--remote-components', 'ejs:github']

class HttpError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** the bundled binary, copied to /tmp so it's executable on a read-only filesystem; system yt-dlp in dev */
async function binary(): Promise<string> {
  const bundled = path.join(process.cwd(), 'bin', 'yt-dlp')
  try {
    await access(bundled, constants.R_OK)
  } catch {
    return 'yt-dlp'
  }
  try {
    await access(bundled, constants.X_OK)
    return bundled
  } catch {
    const tmp = '/tmp/shfla-yt-dlp'
    try {
      await access(tmp, constants.X_OK)
    } catch {
      await copyFile(bundled, tmp)
      await chmod(tmp, 0o755)
    }
    return tmp
  }
}

function run(bin: string, args: string[], signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      env: ENV,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let out = ''
    let err = ''
    const timer = setTimeout(() => child.kill('SIGKILL'), 45_000)
    const onAbort = () => child.kill('SIGKILL')
    signal.addEventListener('abort', onAbort, { once: true })
    child.stdout.on('data', (d) => {
      out += d
      if (out.length > 30_000_000) child.kill('SIGKILL')
    })
    child.stderr.on('data', (d) => {
      err = (err + d).slice(-8000)
    })
    child.on('error', (e) => {
      clearTimeout(timer)
      reject(new HttpError(500, `yt-dlp didn't start: ${e.message}`))
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
      if (code === 0) resolve(out)
      else {
        console.error('yt-dlp failed', code, err.slice(-1500))
        const { status, message } = explain(err)
        reject(new HttpError(status, message))
      }
    })
  })
}

interface Info {
  title?: string
  duration?: number
  is_live?: boolean
  formats?: YtFormat[]
}

/**
 * Pulls the audio of a youtube video and streams it from our own origin. yt-dlp does the
 * extraction; the audio itself is fetched here, from the same machine yt-dlp ran on, because
 * youtube's media urls are tied to the address that asked for them.
 */
export async function GET(request: Request): Promise<Response> {
  try {
    const id = youtubeId(new URL(request.url).searchParams.get('url'))
    if (!id) throw new HttpError(400, "that isn't a youtube video link")
    const bin = await binary()
    // youtube asks datacenter addresses to sign in; some player clients are asked less often,
    // so when the default is refused, try those before giving up
    let json = ''
    let lastErr: unknown = null
    for (const client of CLIENTS) {
      try {
        json = await run(
          bin,
          [
            '-J',
            '--no-playlist',
            '--no-warnings',
            '--use-extractors',
            'youtube',
            ...JS,
            ...(client ? ['--extractor-args', `youtube:player_client=${client}`] : []),
            '--cache-dir',
            '/tmp/yt-dlp-cache',
            '--',
            `https://www.youtube.com/watch?v=${id}`,
          ],
          request.signal,
        )
        console.log('yt-dlp extracted', id, 'with', client || 'default clients')
        break
      } catch (e) {
        lastErr = e
        if (!(e instanceof HttpError) || e.status !== 503) throw e
      }
    }
    if (!json) throw lastErr
    const info = JSON.parse(json) as Info
    if (info.is_live) throw new HttpError(422, 'live streams have no end to hold on to. pick a regular video')
    if ((info.duration ?? 0) > MAX_SECONDS) throw new HttpError(413, 'that video is longer than 30 minutes')
    const f = pickAudio(info.formats ?? [])
    if (!f?.url) throw new HttpError(502, 'that video has no audio we can use')

    const size = f.filesize ?? null
    if (size && size > MAX_BYTES) throw new HttpError(413, 'that audio is too big (80 MB max)')

    // yt-dlp downloads from its own extraction (it knows youtube's chunking and pacing rules);
    // its stdout streams straight into the response
    const infoPath = path.join(tmpdir(), `shfla-yt-${randomUUID()}.json`)
    await writeFile(infoPath, json)
    const child = spawn(
      bin,
      ['--load-info-json', infoPath, '-f', f.format_id, '-o', '-', '--no-warnings', '--no-progress', '--quiet', ...JS],
      { env: ENV, stdio: ['ignore', 'pipe', 'pipe'] },
    )
    let errTail = ''
    child.stderr.on('data', (d) => {
      errTail = (errTail + d).slice(-4000)
    })
    const cleanup = () => {
      child.kill('SIGKILL')
      void unlink(infoPath).catch(() => {})
    }
    request.signal.addEventListener('abort', cleanup, { once: true })

    // wait for the first bytes, so a failure before the audio starts is still a proper error
    await new Promise<void>((resolve, reject) => {
      const onData = () => {
        clearTimeout(timer)
        child.stdout.pause()
        child.off('close', onClose)
        resolve()
      }
      const onClose = (code: number | null) => {
        clearTimeout(timer)
        child.stdout.off('readable', onData)
        void unlink(infoPath).catch(() => {})
        console.error('yt-dlp download failed early', code, errTail.slice(-1500))
        const { status, message } = explain(errTail)
        reject(new HttpError(status, message))
      }
      const timer = setTimeout(() => {
        child.stdout.off('readable', onData)
        child.off('close', onClose)
        cleanup()
        reject(new HttpError(504, 'youtube was too slow to send the audio'))
      }, 20_000)
      child.stdout.once('readable', onData)
      child.once('close', onClose)
    })
    // nothing gets to outlive the function
    setTimeout(cleanup, 55_000).unref()

    let sent = 0
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        child.stdout.on('data', (chunk: Buffer) => {
          sent += chunk.byteLength
          if (sent > MAX_BYTES) {
            cleanup()
            controller.error(new Error('too big'))
            return
          }
          controller.enqueue(new Uint8Array(chunk))
          if ((controller.desiredSize ?? 1) <= 0) child.stdout.pause()
        })
        child.on('close', (code) => {
          void unlink(infoPath).catch(() => {})
          if (code === 0) controller.close()
          else {
            console.error('yt-dlp download failed', code, errTail.slice(-1500))
            try {
              controller.error(new Error('download failed'))
            } catch {}
          }
        })
      },
      pull() {
        child.stdout.resume()
      },
      cancel: cleanup,
    })

    const out = new Headers({
      'content-type': mimeFor(f),
      'cache-control': 'no-store',
      'x-content-type-options': 'nosniff',
      'x-shfla-title': encodeURIComponent(info.title ?? id),
    })
    if (size) out.set('content-length', String(size))
    if (info.duration) out.set('x-shfla-duration', String(info.duration))
    return new Response(body, { status: 200, headers: out })
  } catch (e) {
    const status = e instanceof HttpError ? e.status : 502
    const message = e instanceof HttpError ? e.message : "couldn't get the audio out of that video"
    if (!(e instanceof HttpError)) console.error(e)
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
    })
  }
}
