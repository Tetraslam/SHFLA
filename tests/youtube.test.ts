import { describe, expect, it } from 'vitest'
import { explain, pickAudio, type YtFormat, youtubeId } from '../api/_youtube.js'

describe('youtube links', () => {
  it('finds the video id in every common shape', () => {
    for (const u of [
      'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
      'https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123',
      'https://m.youtube.com/watch?v=dQw4w9WgXcQ',
      'https://music.youtube.com/watch?v=dQw4w9WgXcQ&feature=share',
      'https://youtu.be/dQw4w9WgXcQ?si=abc',
      'https://www.youtube.com/shorts/dQw4w9WgXcQ',
      'https://www.youtube.com/embed/dQw4w9WgXcQ',
      ' https://youtu.be/dQw4w9WgXcQ ',
    ]) {
      expect(youtubeId(u), u).toBe('dQw4w9WgXcQ')
    }
  })

  it('refuses other hosts, lookalikes, and malformed ids', () => {
    for (const u of [
      null,
      '',
      'https://example.com/watch?v=dQw4w9WgXcQ',
      'https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ',
      'https://www.youtube.com/watch?v=short',
      'https://www.youtube.com/watch?v=dQw4w9WgXcQ%22;rm',
      'https://www.youtube.com/playlist?list=PL123',
      'ftp://youtube.com/watch?v=dQw4w9WgXcQ',
      'https://youtu.be/../../etc',
    ]) {
      expect(youtubeId(u), String(u)).toBeNull()
    }
  })
})

describe('audio format choice', () => {
  const f = (o: Partial<YtFormat> & { format_id: string }): YtFormat => ({
    protocol: 'https',
    url: 'https://x',
    vcodec: 'none',
    ...o,
  })
  it('prefers aac in m4a, then opus, and never video or hls', () => {
    const formats = [
      f({ format_id: '251', ext: 'webm', acodec: 'opus', abr: 130 }),
      f({ format_id: '140', ext: 'm4a', acodec: 'mp4a.40.2', abr: 129 }),
      f({ format_id: '139', ext: 'm4a', acodec: 'mp4a.40.5', abr: 48 }),
      f({ format_id: '18', ext: 'mp4', acodec: 'mp4a', vcodec: 'avc1', abr: 96 }),
      f({ format_id: '233', ext: 'mp4', acodec: 'mp4a', protocol: 'm3u8_native', abr: 300 }),
    ]
    expect(pickAudio(formats)?.format_id).toBe('140')
    expect(pickAudio(formats.filter((x) => x.ext !== 'm4a'))?.format_id).toBe('251')
    expect(pickAudio([formats[3], formats[4]])).toBeNull()
  })

  it('explains yt-dlp failures in words', () => {
    expect(explain("ERROR: Sign in to confirm you're not a bot").status).toBe(503)
    expect(explain('ERROR: Private video').message).toMatch(/private/)
    expect(explain('ERROR: Video unavailable').status).toBe(404)
  })
})
