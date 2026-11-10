import { LEVELS } from '@/audio/frame'
import { GLSL_HELPERS, type Program } from '@/lang/compile'

export const MODE_STEP = 0
export const MODE_SPLIT = 1
export const MODE_SEED = 2

/**
 * The cell shader for one pass of a compiled program.
 * step: next state from this level's previous state.
 * split: a new finer level from its parent (zooming past an octave, or growing the initial pyramid).
 * seed: the first coarse level.
 * Output texel: r = state, g = previous state, b = trail (afterglow).
 */
export function simShader(program: Program, mode: number): string {
  const u = program.uses
  const defs = [
    `#define MODE ${mode}`,
    u.has('sum') ? '#define USE_SUM' : '',
    u.has('up') ? '#define USE_UP' : '',
    u.has('down') ? '#define USE_DOWN' : '',
  ].join('\n')
  return /* glsl */ `#version 300 es
precision highp float;
precision highp int;
${defs}
uniform sampler2D u_src;
uniform sampler2D u_up;
uniform sampler2D u_down;
uniform int u_srcSize;
uniform int u_size;
uniform ivec2 u_offset;
uniform float u_hasUp;
uniform float u_hasDown;
uniform float u_level;
uniform float u_levels;
uniform float u_depth;
uniform float u_tick;
uniform float u_t;
uniform float u_salt;
uniform float u_decay;
uniform float u_band;
uniform float u_hit;
uniform float u_bass;
uniform float u_mid;
uniform float u_high;
uniform float u_loud;
uniform float u_beat;
uniform float u_phase;
uniform float u_bright;
out vec4 o;

ivec2 g_cell;
ivec2 g_src;

${GLSL_HELPERS}

ivec2 sh_wrap(ivec2 c) {
  return ivec2(((c.x % u_srcSize) + u_srcSize) % u_srcSize, ((c.y % u_srcSize) + u_srcSize) % u_srcSize);
}

float sh_nb(float dx, float dy) {
#if MODE == ${MODE_SEED}
  return 0.0;
#else
  return texelFetch(u_src, sh_wrap(g_src + ivec2(int(round(dx)), int(round(dy)))), 0).r;
#endif
}

float sh_count(float v) {
  float c = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
      if (dx != 0 || dy != 0) c += sh_eq(sh_nb(float(dx), float(dy)), v);
  return c;
}

uint sh_pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

float sh_hash(float k) {
  uint h = sh_pcg(floatBitsToUint(k) ^ sh_pcg(uint(u_salt) + uint(u_level) * 977u));
  h = sh_pcg(h ^ uint(u_tick));
  h = sh_pcg(h ^ uint(g_cell.y));
  h = sh_pcg(h ^ uint(g_cell.x) * 2654435761u);
  return float(h >> 8u) / 16777216.0;
}

void main() {
  g_cell = ivec2(gl_FragCoord.xy);
#if MODE == ${MODE_SPLIT}
  g_src = g_cell / 2 + u_offset;
#else
  g_src = g_cell;
#endif
#if MODE == ${MODE_SEED}
  vec4 src = vec4(0.0);
#else
  vec4 src = texelFetch(u_src, sh_wrap(g_src), 0);
#endif
  float b_self = src.r;
  float b_prev = src.g;
  float b_x = float(g_cell.x);
  float b_y = float(g_cell.y);
  float b_qx = float(g_cell.x % 2);
  float b_qy = float(g_cell.y % 2);
  float b_size = float(u_size);
  float b_level = u_level;
  float b_levels = u_levels;
  float b_depth = u_depth;
  float b_tick = u_tick;
  float b_t = u_t;
  float b_band = u_band;
  float b_hit = u_hit;
  float b_bass = u_bass;
  float b_mid = u_mid;
  float b_high = u_high;
  float b_loud = u_loud;
  float b_beat = u_beat;
  float b_phase = u_phase;
  float b_bright = u_bright;
  float b_rand = sh_hash(-1.0);
#ifdef USE_SUM
  float b_sum = 0.0;
  for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
      if (dx != 0 || dy != 0) b_sum += sh_nb(float(dx), float(dy));
#endif
#ifdef USE_UP
  float b_up = b_self;
#if MODE == ${MODE_STEP}
  if (u_hasUp > 0.5) b_up = texelFetch(u_up, g_cell / 2, 0).r;
#endif
#endif
#ifdef USE_DOWN
  float b_down = b_self;
#if MODE == ${MODE_STEP}
  if (u_hasDown > 0.5) {
    ivec2 c = g_cell * 2;
    b_down = 0.25 * (texelFetch(u_down, c, 0).r + texelFetch(u_down, c + ivec2(1, 0), 0).r
      + texelFetch(u_down, c + ivec2(0, 1), 0).r + texelFetch(u_down, c + ivec2(1, 1), 0).r);
  }
#endif
#endif

${program.glsl}

#if MODE == ${MODE_STEP}
  float next = out_step;
#elif MODE == ${MODE_SPLIT}
  float next = out_split;
#else
  float next = out_seed;
#endif
  next = (isnan(next) || isinf(next)) ? 0.0 : clamp(next, 0.0, 1.0);
  // b is what you see: the program's light if it has one, otherwise the state with an afterglow
  bool own = out_light >= 0.0 && !isnan(out_light);
#if MODE == ${MODE_STEP}
  o = vec4(next, b_self, own ? out_light : max(next, src.b * u_decay), 1.0);
#elif MODE == ${MODE_SPLIT}
  o = vec4(next, next, own ? out_light : max(next, src.b * 0.85), 1.0);
#else
  o = vec4(next, next, own ? out_light : next, 1.0);
#endif
}
`
}

/** Copies the center half of a level up into the next coarser slot when the zoom crosses an octave. */
export const CROP_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform ivec2 u_offset;
out vec4 o;
void main() {
  o = texelFetch(u_src, ivec2(gl_FragCoord.xy) + u_offset, 0);
}
`

const levelSamples = Array.from(
  { length: LEVELS },
  (_, k) => `  acc += u_col[${k}] * u_w[${k}] * levelLight(u_l${k}, u_size[${k}], uv, u_coc[${k}]);`,
).join('\n')

/**
 * Composites every level into light and folds it into the feedback buffer.
 * The feedback is the reverb tail: last frame, pushed outward with the zoom, decaying.
 */
export const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
${Array.from({ length: LEVELS }, (_, k) => `uniform sampler2D u_l${k};`).join('\n')}
uniform sampler2D u_prev;
uniform vec2 u_res;
uniform float u_scale;
uniform float u_rot;
uniform float u_w[${LEVELS}];
uniform vec3 u_col[${LEVELS}];
uniform float u_size[${LEVELS}];
uniform float u_coc[${LEVELS}];
uniform float u_fbDecay;
uniform float u_fbZoom;
uniform float u_fbRot;
uniform float u_warp;
uniform float u_t;
uniform float u_lightMode;
out vec4 o;

// depth of field: each level is blurred by its circle of confusion (distance from the focal
// plane), over a small golden-angle disc. the in-focus level stays crisp.
vec3 levelLight(sampler2D tex, float size, vec2 uv, float coc) {
  float r = 0.3 / size + coc * 0.0045;
  vec4 s = texture(tex, uv) * 0.25;
  const float GA = 2.39996323;
  for (int i = 0; i < 6; i++) {
    float fi = float(i) + 0.5;
    float a = fi * GA + u_t * 0.05;
    s += texture(tex, uv + vec2(cos(a), sin(a)) * r * sqrt(fi / 6.0)) * 0.125;
  }
  float born = max(s.r - s.g, 0.0);
  float e = u_lightMode > 0.5 ? s.b * 1.3 : s.r * 0.9 + s.b * 0.42 + born * 0.9;
  return vec3(e);
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  float m = max(u_res.x, u_res.y);
  vec2 q = (frag - 0.5 * u_res) / m;
  q.y = -q.y;
  float cs = cos(u_rot), sn = sin(u_rot);
  vec2 uv = vec2(cs * q.x - sn * q.y, sn * q.x + cs * q.y) / u_scale + 0.5;

  vec3 acc = vec3(0.0);
${levelSamples}

  // feedback: sample last frame slightly toward the center so trails stream outward
  vec2 p = frag / u_res - 0.5;
  float c2 = cos(u_fbRot), s2 = sin(u_fbRot);
  p = vec2(c2 * p.x - s2 * p.y, s2 * p.x + c2 * p.y) / u_fbZoom;
  vec2 w = vec2(sin(p.y * 9.0 + u_t * 0.7), cos(p.x * 7.0 - u_t * 0.5)) * u_warp;
  vec3 prev = texture(u_prev, p + 0.5 + w).rgb;
  o = vec4(acc + prev * u_fbDecay, 1.0);
}
`

/** 9-tap separable gaussian (linear-sampled), with an optional soft bright pass. Used for bloom. */
export const BLOOM_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform vec2 u_dir;
uniform vec2 u_outRes;
uniform float u_threshold;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / u_outRes;
  const float w0 = 0.2270270270, w1 = 0.3162162162, w2 = 0.0702702703;
  vec2 d1 = u_dir * u_texel * 1.3846153846;
  vec2 d2 = u_dir * u_texel * 3.2307692308;
  vec3 c = texture(u_src, uv).rgb * w0
    + (texture(u_src, uv + d1).rgb + texture(u_src, uv - d1).rgb) * w1
    + (texture(u_src, uv + d2).rgb + texture(u_src, uv - d2).rgb) * w2;
  if (u_threshold > 0.0) {
    float l = max(c.r, max(c.g, c.b));
    c *= smoothstep(u_threshold, u_threshold * 2.5, l);
  }
  o = vec4(c, 1.0);
}
`

/** Final look: chromatic smear, bloom, milky haze, tonemap, grain, vignette. */
export const FINAL_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform sampler2D u_bloomA;
uniform sampler2D u_bloomB;
uniform vec2 u_res;
uniform float u_ca;
uniform float u_exposure;
uniform vec3 u_haze;
uniform vec3 u_ground;
uniform float u_bloom;
uniform float u_grain;
uniform float u_t;
out vec4 o;

float h12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_res;
  vec2 d = uv - 0.5;
  float r2 = dot(d, d);
  vec2 off = d * u_ca * (0.4 + r2 * 3.0);
  vec3 c;
  c.r = texture(u_src, uv + off).r;
  c.g = texture(u_src, uv).g;
  c.b = texture(u_src, uv - off).b;
  c += texture(u_bloomA, uv).rgb * u_bloom * 0.7 + texture(u_bloomB, uv).rgb * u_bloom;
  c *= u_exposure;
  c = vec3(1.0) - exp(-c);
  // milky blacks: the haze lifts the floor toward the palette's mid tone
  c = u_ground + c * (1.0 - u_ground) + u_haze * (1.0 - c);
  float vig = smoothstep(0.95, 0.15, r2 * 1.6);
  c *= mix(0.72, 1.0, vig);
  float g = h12(gl_FragCoord.xy + fract(u_t * 13.37) * 400.0) - 0.5;
  c += g * u_grain;
  o = vec4(clamp(c, 0.0, 1.0), 1.0);
}
`

/** Per-level summary for the meters and the voice: mean state, mean change, mean trail. */
export const ACTIVITY_FS = /* glsl */ `#version 300 es
precision highp float;
${Array.from({ length: LEVELS }, (_, k) => `uniform sampler2D u_l${k};`).join('\n')}
out vec4 o;
vec3 summarize(sampler2D t) {
  ivec2 sz = textureSize(t, 0);
  vec3 acc = vec3(0.0);
  for (int j = 0; j < 16; j++)
    for (int i = 0; i < 16; i++) {
      vec4 s = texelFetch(t, ivec2((float(i) + 0.5) / 16.0 * float(sz.x), (float(j) + 0.5) / 16.0 * float(sz.y)), 0);
      acc += vec3(s.r, abs(s.r - s.g), s.b);
    }
  return acc / 256.0;
}
void main() {
  int k = int(gl_FragCoord.x);
  vec3 s = vec3(0.0);
${Array.from({ length: LEVELS }, (_, k) => `  if (k == ${k}) s = summarize(u_l${k});`).join('\n')}
  o = vec4(s.x, min(s.y * 4.0, 1.0), s.z, 1.0);
}
`
