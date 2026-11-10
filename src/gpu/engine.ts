import { type AudioFrame, LEVELS } from '@/audio/frame'
import { evalSpeed, type Program } from '@/lang/compile'
import { freeTarget, GLError, makePass, makeTarget, type Pass, type Target, type TexFormat } from './gl'
import { PALETTES, type Palette, ramp } from './palettes'
import {
  ACTIVITY_FS,
  BLOOM_FS,
  COMPOSITE_FS,
  CROP_FS,
  FINAL_FS,
  MODE_SEED,
  MODE_SPLIT,
  MODE_STEP,
  simShader,
} from './shaders'

/** steps of the finest level per beat; each coarser level steps half as often */
export const TICKS_PER_BEAT = 8
/** beats it takes the zoom to pass one octave, at full loudness */
const BEATS_PER_OCTAVE = 8
/** how far into the world the camera sits, so rotation never shows an edge */
const MAGNIFY = 1.45

interface Level {
  size: number
  cur: Target
  back: Target
}

interface SimPasses {
  step: Pass
  split: Pass
  seed: Pass
}

export interface LevelActivity {
  /** fraction of the level that is lit */
  state: number
  /** how much of it changed on the last step: how hard it is computing */
  change: number
  trail: number
}

export interface EngineSettings {
  palette: Palette
  /** multiplies the step clock */
  speed: number
  /** multiplies the zoom */
  zoom: number
  reducedMotion: boolean
}

const slew = (cur: number, target: number, maxDelta: number) =>
  cur + Math.max(-maxDelta, Math.min(maxDelta, target - cur))

export class Engine {
  readonly gl: WebGL2RenderingContext
  private fmt!: TexFormat
  private levels: Level[] = []
  private sim: SimPasses | null = null
  private program: Program | null = null
  private crop!: Pass
  private composite!: Pass
  private bloom!: Pass
  private final!: Pass
  private activityPass!: Pass
  private fb: [Target, Target] | null = null
  private bloomTargets: Target[] = []
  private actTarget!: Target
  private actBytes = new Uint8Array(LEVELS * 4)
  private width = 0
  private height = 0

  finest: number
  settings: EngineSettings = { palette: PALETTES[0], speed: 1, zoom: 1, reducedMotion: false }
  activity: LevelActivity[] = Array.from({ length: LEVELS }, () => ({ state: 0, change: 0, trail: 0 }))
  depth = 0
  frac = 0
  lost = false
  /** stops the clocks; used by the in-browser checks */
  frozen = false

  private tickPos = 0
  private tickCount = 0
  private t = 0
  private rot = 0
  private salt = (Math.random() * 1e6) | 0
  private frameNo = 0
  private sub = 0
  private dt = 1 / 60
  private exposure = 1
  private adapt = 1
  private loudS = 0
  private bassS = 0
  private midS = 0
  private highS = 0
  /** the focal plane, in levels: the song racks it between coarse and fine */
  focus = 2.6

  readonly canvas: HTMLCanvasElement

  constructor(canvas: HTMLCanvasElement, opts: { finest: number }) {
    this.canvas = canvas
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    })
    if (!gl) throw new GLError('this browser has no webgl2')
    this.gl = gl
    this.finest = opts.finest
    canvas.addEventListener('webglcontextlost', this.onLost)
    canvas.addEventListener('webglcontextrestored', this.onRestored)
    this.init()
  }

  private onLost = (e: Event) => {
    e.preventDefault()
    this.lost = true
  }

  private onRestored = () => {
    this.lost = false
    this.fb = null
    this.bloomTargets = []
    this.init()
    if (this.program) {
      const p = this.program
      this.program = null
      this.sim = null
      this.setProgram(p)
      this.reseed()
    }
  }

  private init() {
    const gl = this.gl
    const half = gl.getExtension('EXT_color_buffer_float')
    gl.getExtension('OES_texture_float_linear')
    this.fmt = half
      ? { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT }
      : { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE }
    this.levels = []
    for (let k = 0; k < LEVELS; k++) {
      const size = this.finest >> (LEVELS - 1 - k)
      this.levels.push({
        size,
        cur: makeTarget(gl, size, size, this.fmt, { repeat: true }),
        back: makeTarget(gl, size, size, this.fmt, { repeat: true }),
      })
    }
    this.crop = makePass(gl, CROP_FS)
    this.composite = makePass(gl, COMPOSITE_FS)
    this.bloom = makePass(gl, BLOOM_FS)
    this.final = makePass(gl, FINAL_FS)
    this.activityPass = makePass(gl, ACTIVITY_FS)
    this.actTarget = makeTarget(
      gl,
      LEVELS,
      1,
      { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE },
      { linear: false },
    )
    // nothing draws with attributes; a bound VAO keeps some drivers happy
    gl.bindVertexArray(gl.createVertexArray())
    gl.disable(gl.DEPTH_TEST)
    gl.disable(gl.BLEND)
  }

  /** Compile a program into its three passes. On a driver error the old program keeps running. */
  setProgram(program: Program): { ok: true } | { ok: false; error: string } {
    try {
      const next: SimPasses = {
        step: makePass(this.gl, simShader(program, MODE_STEP)),
        split: makePass(this.gl, simShader(program, MODE_SPLIT)),
        seed: makePass(this.gl, simShader(program, MODE_SEED)),
      }
      const first = this.sim === null
      if (this.sim) for (const p of Object.values(this.sim)) this.gl.deleteProgram(p.prog)
      this.sim = next
      this.program = program
      if (first) this.reseed()
      return { ok: true }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
  }

  private bindPass(p: Pass) {
    // biome-ignore lint/correctness/useHookAtTopLevel: WebGL's useProgram, not a React hook
    this.gl.useProgram(p.prog)
  }
  private u1f(p: Pass, n: string, v: number) {
    const l = p.uniforms.get(n)
    if (l) this.gl.uniform1f(l, v)
  }
  private u1i(p: Pass, n: string, v: number) {
    const l = p.uniforms.get(n)
    if (l) this.gl.uniform1i(l, v)
  }
  private u2i(p: Pass, n: string, a: number, b: number) {
    const l = p.uniforms.get(n)
    if (l) this.gl.uniform2i(l, a, b)
  }
  private u2f(p: Pass, n: string, a: number, b: number) {
    const l = p.uniforms.get(n)
    if (l) this.gl.uniform2f(l, a, b)
  }
  private u3f(p: Pass, n: string, v: readonly number[]) {
    const l = p.uniforms.get(n)
    if (l) this.gl.uniform3f(l, v[0], v[1], v[2])
  }
  private tex(p: Pass, n: string, unit: number, t: WebGLTexture) {
    const gl = this.gl
    gl.activeTexture(gl.TEXTURE0 + unit)
    gl.bindTexture(gl.TEXTURE_2D, t)
    this.u1i(p, n, unit)
  }
  private drawTo(t: Target | null) {
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, t ? t.fbo : null)
    gl.viewport(0, 0, t ? t.w : this.width, t ? t.h : this.height)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }
  private swap(k: number) {
    const l = this.levels[k]
    const c = l.cur
    l.cur = l.back
    l.back = c
  }

  private setCellUniforms(p: Pass, k: number, a: AudioFrame) {
    this.u1f(p, 'u_level', k)
    this.u1f(p, 'u_levels', LEVELS)
    this.u1f(p, 'u_depth', this.depth)
    this.u1f(p, 'u_tick', Math.floor(this.tickCount / 2 ** (LEVELS - 1 - k)))
    this.u1f(p, 'u_t', this.t)
    this.u1f(p, 'u_salt', (this.salt + this.depth * 7919 + this.sub * 131) % 16777216)
    this.u1f(p, 'u_decay', 0.86)
    this.u1f(p, 'u_band', a.bands[k])
    this.u1f(p, 'u_hit', a.hits[k])
    this.u1f(p, 'u_bass', a.bass)
    this.u1f(p, 'u_mid', a.mid)
    this.u1f(p, 'u_high', a.high)
    this.u1f(p, 'u_loud', a.loud)
    this.u1f(p, 'u_beat', a.beat)
    this.u1f(p, 'u_phase', a.phase)
    this.u1f(p, 'u_bright', a.bright)
    this.u1i(p, 'u_size', this.levels[k].size)
  }

  private speedVars(k: number, a: AudioFrame): Record<string, number> {
    return {
      level: k,
      levels: LEVELS,
      size: this.levels[k].size,
      depth: this.depth,
      tick: Math.floor(this.tickCount / 2 ** (LEVELS - 1 - k)),
      t: this.t,
      band: a.bands[k],
      hit: a.hits[k],
      bass: a.bass,
      mid: a.mid,
      high: a.high,
      loud: a.loud,
      beat: a.beat,
      phase: a.phase,
      bright: a.bright,
    }
  }

  /** Grow the whole pyramid from scratch: seed the coarsest level, then split down to the finest. */
  reseed(a?: AudioFrame) {
    if (!this.sim) return
    const frame = a ?? quietFrame()
    const { seed, split } = this.sim
    this.salt = (Math.random() * 1e6) | 0
    this.bindPass(seed)
    this.setCellUniforms(seed, 0, frame)
    this.drawTo(this.levels[0].back)
    this.swap(0)
    this.bindPass(split)
    for (let k = 1; k < LEVELS; k++) {
      this.setCellUniforms(split, k, frame)
      this.tex(split, 'u_src', 0, this.levels[k - 1].cur.tex)
      this.u1i(split, 'u_srcSize', this.levels[k - 1].size)
      this.u2i(split, 'u_offset', 0, 0)
      this.drawTo(this.levels[k].back)
      this.swap(k)
    }
  }

  private stepLevel(k: number, a: AudioFrame) {
    if (!this.sim) return
    const p = this.sim.step
    this.bindPass(p)
    this.setCellUniforms(p, k, a)
    this.tex(p, 'u_src', 0, this.levels[k].cur.tex)
    this.tex(p, 'u_up', 1, this.levels[Math.max(0, k - 1)].cur.tex)
    this.tex(p, 'u_down', 2, this.levels[Math.min(LEVELS - 1, k + 1)].cur.tex)
    this.u1f(p, 'u_hasUp', k > 0 ? 1 : 0)
    this.u1f(p, 'u_hasDown', k < LEVELS - 1 ? 1 : 0)
    this.u1i(p, 'u_srcSize', this.levels[k].size)
    this.drawTo(this.levels[k].back)
    this.swap(k)
  }

  /** The zoom passed an octave: every level moves up one slot and the finest is born from split. */
  private shift(a: AudioFrame) {
    if (!this.sim) return
    this.depth++
    this.bindPass(this.crop)
    for (let k = 0; k < LEVELS - 1; k++) {
      const half = this.levels[k].size / 2
      this.tex(this.crop, 'u_src', 0, this.levels[k + 1].cur.tex)
      this.u2i(this.crop, 'u_offset', half, half)
      this.drawTo(this.levels[k].back)
    }
    const f = LEVELS - 1
    const p = this.sim.split
    this.bindPass(p)
    this.setCellUniforms(p, f, a)
    this.tex(p, 'u_src', 0, this.levels[f].cur.tex)
    this.u1i(p, 'u_srcSize', this.levels[f].size)
    this.u2i(p, 'u_offset', this.finest / 4, this.finest / 4)
    this.drawTo(this.levels[f].back)
    for (let k = 0; k < LEVELS; k++) this.swap(k)
  }

  private ensureScreen() {
    const gl = this.gl
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
    let w = Math.max(1, Math.round(this.canvas.clientWidth * dpr))
    let h = Math.max(1, Math.round(this.canvas.clientHeight * dpr))
    const maxPx = 2_600_000
    if (w * h > maxPx) {
      const s = Math.sqrt(maxPx / (w * h))
      w = Math.round(w * s)
      h = Math.round(h * s)
    }
    if (w === this.width && h === this.height && this.fb) return
    this.width = w
    this.height = h
    this.canvas.width = w
    this.canvas.height = h
    if (this.fb) for (const t of this.fb) freeTarget(gl, t)
    for (const t of this.bloomTargets) freeTarget(gl, t)
    this.fb = [makeTarget(gl, w, h, this.fmt), makeTarget(gl, w, h, this.fmt)]
    for (const t of this.fb) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo)
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
    }
    const q = (d: number) => [Math.max(1, Math.round(w / d)), Math.max(1, Math.round(h / d))] as const
    const [qw, qh] = q(4)
    const [ew, eh] = q(10)
    this.bloomTargets = [
      makeTarget(gl, qw, qh, this.fmt),
      makeTarget(gl, qw, qh, this.fmt),
      makeTarget(gl, ew, eh, this.fmt),
      makeTarget(gl, ew, eh, this.fmt),
    ]
  }

  /** Advance the clocks and draw one frame. */
  frame(dt: number, a: AudioFrame) {
    if (this.lost || !this.sim || this.frozen) return
    dt = Math.min(dt, 0.1)
    this.dt = dt
    this.t += dt
    this.frameNo++
    const s = this.settings
    const rm = s.reducedMotion

    const k = (tau: number) => 1 - Math.exp(-dt / tau)
    this.loudS += (a.loud - this.loudS) * k(0.6)
    this.bassS += (a.bass - this.bassS) * k(0.25)
    this.midS += (a.mid - this.midS) * k(0.4)
    this.highS += (a.high - this.highS) * k(0.4)
    // rack focus: low-heavy moments pull toward the coarse levels, bright ones toward the fine
    const tilt = (this.highS + 0.5 * this.midS - this.bassS) / (this.highS + this.midS + this.bassS + 0.15)
    const target = a.live ? 2.5 + 1.4 * tilt : 2.5 + 0.9 * Math.sin(this.t * 0.11)
    this.focus += (target - this.focus) * k(rm ? 4 : 1.6)

    // the step clock: finest level steps TICKS_PER_BEAT times per beat
    const beatsPerSec = a.bpm / 60
    this.tickPos += dt * beatsPerSec * TICKS_PER_BEAT * s.speed
    let budget = 6
    while (this.tickPos >= 1 && budget-- > 0) {
      this.tickPos -= 1
      this.tickCount++
      for (let lv = 0; lv < LEVELS; lv++) {
        if (this.tickCount % 2 ** (LEVELS - 1 - lv) !== 0) continue
        const n = this.program ? evalSpeed(this.program, this.speedVars(lv, a)) : 1
        for (this.sub = 0; this.sub < n; this.sub++) this.stepLevel(lv, a)
        this.sub = 0
      }
    }
    this.tickPos = Math.min(this.tickPos, 1)

    // the zoom: one octave every BEATS_PER_OCTAVE beats at full loudness
    const energy = a.live ? 0.3 + 0.7 * this.loudS : 0.45
    const octPerSec = (beatsPerSec / BEATS_PER_OCTAVE) * energy * s.zoom * (rm ? 0.15 : 1)
    const dFrac = octPerSec * dt
    this.frac += dFrac
    if (this.frac >= 1) {
      this.frac -= 1
      this.shift(a)
    }
    const dRot = dt * (rm ? 0.004 : 0.018)
    this.rot += dRot

    this.ensureScreen()
    this.draw(a, dFrac, dRot)

    if (this.frameNo % 4 === 0) this.readActivity()
  }

  private draw(a: AudioFrame, dFrac: number, dRot: number) {
    const gl = this.gl
    const fb = this.fb
    if (!fb) return
    const s = this.settings
    const pal = s.palette
    const rm = s.reducedMotion

    // composite + feedback
    const c = this.composite
    this.bindPass(c)
    const w: number[] = []
    const coc: number[] = []
    const col: number[] = []
    const sizes: number[] = []
    for (let k = 0; k < LEVELS; k++) {
      const p = k - this.frac
      // fades in at the fine end and out at the coarse end so an octave shift never pops,
      // leaning toward the fine levels so structure, not haze, carries the picture
      const u = (p + 1) / LEVELS
      // a window that is zero at both ends, so octave shifts never pop, times a focal plane
      const window = Math.max(0, Math.sin(Math.PI * u)) ** 0.5
      const off = p - this.focus
      const bump = window * (0.18 + Math.exp(-0.5 * (off / 1.05) ** 2))
      coc.push(Math.min(1, Math.max(0, Math.abs(off) - 0.4) * 0.45))
      const hit = rm ? 0 : a.hits[k]
      w.push(bump * (0.3 + 0.9 * a.bands[k] + 0.3 * hit) * 0.5)
      const rgb = ramp(pal, p / (LEVELS - 1) + (a.bright - 0.45) * 0.3)
      col.push(...rgb)
      sizes.push(this.levels[k].size)
      this.tex(c, `u_l${k}`, k, this.levels[k].cur.tex)
    }
    gl.uniform1fv(c.uniforms.get('u_w') ?? null, w)
    gl.uniform3fv(c.uniforms.get('u_col') ?? null, col)
    gl.uniform1fv(c.uniforms.get('u_size') ?? null, sizes)
    gl.uniform1fv(c.uniforms.get('u_coc') ?? null, coc)
    this.tex(c, 'u_prev', LEVELS, fb[0].tex)
    this.u2f(c, 'u_res', this.width, this.height)
    this.u1f(c, 'u_scale', MAGNIFY * 2 ** this.frac)
    this.u1f(c, 'u_rot', this.rot)
    // decay is specified per 60 Hz frame and converted, so trails last the same time at any frame rate
    const decay60 = rm ? 0.7 : Math.min(0.93, 0.8 + 0.13 * this.loudS)
    const decay = decay60 ** (this.dt * 60)
    this.u1f(c, 'u_fbDecay', decay)
    this.u1f(c, 'u_fbZoom', 2 ** dFrac * (1 + (rm ? 0 : 0.0025 + 0.006 * this.bassS)))
    this.u1f(c, 'u_fbRot', dRot)
    this.u1f(c, 'u_warp', rm ? 0 : 0.0008 + 0.003 * this.midS)
    this.u1f(c, 'u_t', this.t)
    this.u1f(c, 'u_lightMode', this.program?.has.light ? 1 : 0)
    this.drawTo(fb[1])

    // bloom chain: quarter res, then a tenth
    const [qa, qb, ea, eb] = this.bloomTargets
    const b = this.bloom
    this.bindPass(b)
    const blur = (src: Target, dst: Target, dx: number, dy: number, threshold: number) => {
      this.tex(b, 'u_src', 0, src.tex)
      this.u2f(b, 'u_texel', 1 / dst.w, 1 / dst.h)
      this.u2f(b, 'u_dir', dx, dy)
      this.u2f(b, 'u_outRes', dst.w, dst.h)
      this.u1f(b, 'u_threshold', threshold)
      this.drawTo(dst)
    }
    blur(fb[1], qa, 1, 0, 0.35)
    blur(qa, qb, 0, 1, 0)
    blur(qb, ea, 1.6, 0, 0)
    blur(ea, eb, 0, 1.6, 0)

    // final look
    const f = this.final
    this.bindPass(f)
    this.tex(f, 'u_src', 0, fb[1].tex)
    this.tex(f, 'u_bloomA', 1, qb.tex)
    this.tex(f, 'u_bloomB', 2, eb.tex)
    this.u2f(f, 'u_res', this.width, this.height)
    this.u1f(f, 'u_ca', rm ? 0.002 : 0.004 + 0.014 * this.bassS)
    // the lens adapts: estimate how much light the levels put out (from the last readback) and
    // expose for it, so a dense rule doesn't blow out and a sparse one doesn't vanish
    let scene = 0
    const lightMode = !!this.program?.has.light
    for (let k = 0; k < LEVELS; k++) {
      const act = this.activity[k]
      scene += w[k] * (lightMode ? act.trail * 1.3 : act.state * 0.9 + act.trail * 0.42)
    }
    const adapt = Math.min(2, Math.max(0.05, 0.2 / Math.max(scene, 0.05)))
    this.adapt += (adapt - this.adapt) * (1 - Math.exp(-this.dt / 0.8))
    // brightness follows the beat, but slew-limited so it never strobes
    const swell = rm ? 0 : 0.16 * a.beat * this.loudS
    this.exposure = slew(this.exposure, (1 - decay) * 2.4 * this.adapt * (1 + swell), 1.2 * this.dt)
    this.u1f(f, 'u_exposure', this.exposure)
    const hz = 0.18 + 0.5 * this.loudS
    this.u3f(
      f,
      'u_haze',
      pal.haze.map((v) => v * hz),
    )
    this.u3f(f, 'u_ground', pal.ground)
    this.u1f(f, 'u_bloom', 0.55 + 0.6 * this.loudS)
    this.u1f(f, 'u_grain', 0.045)
    this.u1f(f, 'u_t', this.t)
    this.drawTo(null)

    fb.reverse()
  }

  private readActivity() {
    const gl = this.gl
    const p = this.activityPass
    this.bindPass(p)
    for (let k = 0; k < LEVELS; k++) this.tex(p, `u_l${k}`, k, this.levels[k].cur.tex)
    this.drawTo(this.actTarget)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.actTarget.fbo)
    gl.readPixels(0, 0, LEVELS, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.actBytes)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    for (let k = 0; k < LEVELS; k++) {
      const a = this.activity[k]
      a.state = this.actBytes[k * 4] / 255
      a.change = this.actBytes[k * 4 + 1] / 255
      a.trail = this.actBytes[k * 4 + 2] / 255
    }
  }

  /** Read one level back to the cpu. Used by the in-browser checks. */
  readLevel(k: number): { size: number; data: Float32Array } {
    const gl = this.gl
    const l = this.levels[k]
    gl.bindFramebuffer(gl.FRAMEBUFFER, l.cur.fbo)
    const data = new Float32Array(l.size * l.size * 4)
    if (this.fmt.type === gl.UNSIGNED_BYTE) {
      const bytes = new Uint8Array(l.size * l.size * 4)
      gl.readPixels(0, 0, l.size, l.size, gl.RGBA, gl.UNSIGNED_BYTE, bytes)
      for (let i = 0; i < bytes.length; i++) data[i] = bytes[i] / 255
    } else gl.readPixels(0, 0, l.size, l.size, gl.RGBA, gl.FLOAT, data)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    const out = new Float32Array(l.size * l.size)
    for (let i = 0; i < out.length; i++) out[i] = data[i * 4]
    return { size: l.size, data: out }
  }

  /** Overwrite one level's state. Used by the in-browser checks. */
  debugWrite(k: number, states: ArrayLike<number>) {
    const gl = this.gl
    const l = this.levels[k]
    const rgba = new Float32Array(l.size * l.size * 4)
    for (let i = 0; i < l.size * l.size; i++) {
      rgba[i * 4] = states[i]
      rgba[i * 4 + 1] = states[i]
      rgba[i * 4 + 2] = states[i]
      rgba[i * 4 + 3] = 1
    }
    gl.bindTexture(gl.TEXTURE_2D, l.cur.tex)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, l.size, l.size, gl.RGBA, gl.FLOAT, rgba)
  }

  /** Step one level by hand, outside the clock. Used by the in-browser checks. */
  debugStep(k: number) {
    this.tickCount++
    this.stepLevel(k, quietFrame())
  }

  dispose() {
    this.canvas.removeEventListener('webglcontextlost', this.onLost)
    this.canvas.removeEventListener('webglcontextrestored', this.onRestored)
    this.lost = true
  }
}

function quietFrame(): AudioFrame {
  return {
    bands: Array(LEVELS).fill(0),
    hits: Array(LEVELS).fill(0),
    bass: 0,
    mid: 0,
    high: 0,
    loud: 0,
    beat: 0,
    phase: 0,
    bright: 0.4,
    bpm: 90,
    live: false,
    root: 4,
    minor: false,
  }
}
