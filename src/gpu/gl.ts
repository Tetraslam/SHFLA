export const FULLSCREEN_VS = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`

export class GLError extends Error {}

export interface Target {
  tex: WebGLTexture
  fbo: WebGLFramebuffer
  w: number
  h: number
}

export interface Pass {
  prog: WebGLProgram
  uniforms: Map<string, WebGLUniformLocation>
}

export function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)
  if (!sh) throw new GLError('could not create a shader')
  gl.shaderSource(sh, src)
  gl.compileShader(sh)
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS) && !gl.isContextLost()) {
    const log = gl.getShaderInfoLog(sh) ?? 'unknown error'
    gl.deleteShader(sh)
    throw new GLError(log)
  }
  return sh
}

export function makePass(gl: WebGL2RenderingContext, fs: string): Pass {
  const vs = compileShader(gl, gl.VERTEX_SHADER, FULLSCREEN_VS)
  const f = compileShader(gl, gl.FRAGMENT_SHADER, fs)
  const prog = gl.createProgram()
  if (!prog) throw new GLError('could not create a program')
  gl.attachShader(prog, vs)
  gl.attachShader(prog, f)
  gl.linkProgram(prog)
  gl.deleteShader(vs)
  gl.deleteShader(f)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost()) {
    const log = gl.getProgramInfoLog(prog) ?? 'unknown error'
    gl.deleteProgram(prog)
    throw new GLError(log)
  }
  const uniforms = new Map<string, WebGLUniformLocation>()
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS) as number
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i)
    if (!info) continue
    const name = info.name.replace(/\[0\]$/, '')
    const loc = gl.getUniformLocation(prog, info.name)
    if (loc) uniforms.set(name, loc)
  }
  return { prog, uniforms }
}

export interface TexFormat {
  internal: number
  format: number
  type: number
}

export function makeTarget(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  fmt: TexFormat,
  opts: { repeat?: boolean; linear?: boolean } = {},
): Target {
  const tex = gl.createTexture()
  const fbo = gl.createFramebuffer()
  if (!tex || !fbo) throw new GLError('out of gpu memory')
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internal, w, h, 0, fmt.format, fmt.type, null)
  const filter = opts.linear === false ? gl.NEAREST : gl.LINEAR
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter)
  const wrap = opts.repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap)
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  if (status !== gl.FRAMEBUFFER_COMPLETE && !gl.isContextLost()) {
    throw new GLError(`framebuffer incomplete (${status})`)
  }
  return { tex, fbo, w, h }
}

export function freeTarget(gl: WebGL2RenderingContext, t: Target | null | undefined) {
  if (!t) return
  gl.deleteTexture(t.tex)
  gl.deleteFramebuffer(t.fbo)
}
