import { describe, expect, it } from 'vitest'
import { checkUrl, isAudioResponse, isPrivateAddress } from '../api/_guard.js'

describe('audio proxy guard', () => {
  it('accepts public http(s) links only', () => {
    expect(checkUrl('https://example.com/a.mp3').hostname).toBe('example.com')
    for (const bad of [
      null,
      'nope',
      'file:///etc/passwd',
      'ftp://x/a.mp3',
      'http://localhost/a.mp3',
      'http://a.internal/x',
      'https://x.com:8080/a.mp3',
      'https://u:p@x.com/a.mp3',
    ]) {
      expect(() => checkUrl(bad), String(bad)).toThrow()
    }
  })

  it('flags private and reserved addresses', () => {
    for (const ip of [
      '127.0.0.1',
      '10.1.2.3',
      '172.20.0.1',
      '192.168.1.1',
      '169.254.169.254',
      '100.100.1.1',
      '0.0.0.0',
      '::1',
      'fd00::1',
      'fe80::1',
      '::ffff:10.0.0.1',
      '64:ff9b::a00:1',
    ]) {
      expect(isPrivateAddress(ip), ip).toBe(true)
    }
    for (const ip of ['93.184.216.34', '1.1.1.1', '2606:4700::1111']) expect(isPrivateAddress(ip), ip).toBe(false)
  })

  it('only passes audio through', () => {
    const u = new URL('https://x.com/song.mp3')
    expect(isAudioResponse('audio/mpeg', u)).toBe(true)
    expect(isAudioResponse('application/octet-stream', u)).toBe(true)
    expect(isAudioResponse('text/html; charset=utf-8', u)).toBe(false)
    expect(isAudioResponse('application/octet-stream', new URL('https://x.com/file.zip'))).toBe(false)
  })
})
