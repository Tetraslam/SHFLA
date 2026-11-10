// Downloads a pinned yt-dlp standalone binary into bin/ and verifies its checksum.
// Runs before the build on Vercel; locally, `pnpm yt-dlp` does the same.
import { createHash } from 'node:crypto'
import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'

const VERSION = '2026.10.26'
const SHA256 = '767ba9dce7052aa196f8b12652f433d76de128ba168966fa84b6394d5e6c6795'
const URL = `https://github.com/yt-dlp/yt-dlp/releases/download/${VERSION}/yt-dlp_linux`
const OUT = new globalThis.URL('../bin/yt-dlp', import.meta.url)

const sha = (buf) => createHash('sha256').update(buf).digest('hex')

try {
  if (sha(await readFile(OUT)) === SHA256) {
    console.log(`yt-dlp ${VERSION} already present`)
    process.exit(0)
  }
} catch {}

const res = await fetch(URL)
if (!res.ok) throw new Error(`yt-dlp download failed: ${res.status}`)
const buf = Buffer.from(await res.arrayBuffer())
const got = sha(buf)
if (got !== SHA256) throw new Error(`yt-dlp checksum mismatch: ${got}`)
await mkdir(new globalThis.URL('../bin/', import.meta.url), { recursive: true })
await writeFile(OUT, buf)
await chmod(OUT, 0o755)
console.log(`yt-dlp ${VERSION} verified (${(buf.length / 1e6).toFixed(1)} MB)`)
