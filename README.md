# SHFLA

shoegaze hierarchical fractal language architecture: music and fractal geometry as one thing. a song drives a living fractal in real time, and the fractal is itself a language, expressive enough to be turing-complete.

this branch is a from-scratch rebuild for the web. nothing from the original implementation carries over except the idea. decide everything else: what the music controls, how the fractal speaks back, what makes it computational, and what it should feel like to use.

runs in the browser on the gpu, with audio from a file, a link, or the microphone. use tetrastack.

**live: [shfla.vercel.app](https://shfla.vercel.app)**

## what it is

the fractal is a stack of six grids, one per spatial octave: 32 cells across up to 1024 (16 to 512 on phones). every cell on every grid runs the same small program, written in the shfla language.

- **the music** is split into six octave bands (sub, bass, low, mid, high, air). each band feeds the grid at the matching scale, so bass moves the big shapes and treble moves the fine dust. a program reads its own octave as `band` (loudness) and `hit` (onsets), plus global `bass`, `loud`, `beat`, and the rest.
- **the hierarchy runs both ways.** the beat is the clock. the finest grid steps eight times a beat, and each coarser grid steps half as often, so the coarsest grid moves once a bar. the zoom passes one octave every couple of bars, faster when the song is loud. when it does, every grid moves up a slot and a new finest grid is born from the `split` rule.
- **the lens follows the song.** levels are drawn through a focal plane. bass-heavy passages pull focus to the coarse scales and bright passages pull it to the fine ones. feedback trails stand in for reverb (louder means a longer tail), and there is bloom, chromatic smear, and grain. the exposure adapts to whatever the rule is doing.
- **the fractal speaks back.** each level drives one voice of a drone chord built on the song's own root (taken from a chroma estimate): a low root, then a fifth, octaves, and a third. a voice is as loud as its level is busy computing, so the harmony follows the rule, not the song. turn it up or down under _tune_. it is off for the mic, to avoid feedback.

## the language

one statement per line, `name = expression`. every cell runs the whole program each step. numbers are floats; true is 1 and false is 0; anything from 0.5 up counts as true; `==` allows 0.02 of slack.

| output  | meaning                                                                                |
| ------- | -------------------------------------------------------------------------------------- |
| `step`  | required. the next state of every cell, 0 to 1                                         |
| `split` | what a new finer cell becomes when the zoom passes an octave. default: `self`          |
| `seed`  | the starting state. the coarsest grid is seeded, then split down to the finest         |
| `light` | how bright the cell looks, if that isn't its state (waves use this)                    |
| `speed` | steps per tick for each level, 0 to 16. it may read only the song, time, and the level |

inputs: `self prev sum rand x y qx qy` (the cell), `up down level levels size depth` (the scales), `tick t` (time), and `band hit bass mid high loud beat phase bright` (the song). functions: `nb(dx, dy) count(v) hash(k) min max clamp mix smooth mod abs floor ceil fract sin cos wave sqrt exp log`. operators: `+ - * / % ^ == != < <= > >= and or not` and `if … then … else …`. the editor highlights music inputs in the accent color, because that is where the song touches the program.

```
# conway's life at every scale, sparked by the song
alive = sum == 3 or (self == 1 and sum == 2)
spark = rand < hit * hit * 0.05
step = alive or spark
split = self and not (qx == 1 and qy == 1)
```

the presets are **haze** (life), **carpet** (a sierpinski triangle: context-free, all split and no step), **rule 110**, **brain** (brian's brain), **echo** (life that copies the coarser scale on the beat), and **tide** (the wave equation). edits run as you type, and the last good program keeps running through errors. the url carries the rule and the palette, so a link shares both.

### turing-completeness

the language can express conway's life (the haze preset) and rule 110, and both are turing-complete (berlekamp, conway, and guy; cook, 2004). both are checked bit-for-bit: against a cpu interpreter in `src/lang/compile.test.ts`, and on the gpu in the browser with `tests/browser/*.js`. any real browser gives the grids a finite size, so in practice a program is a very large finite machine. that is the usual caveat for "turing-complete" hardware.

## audio

- **demo**: a generative shoegaze song synthesized live, so the app works with nothing else.
- **file**: pick one or drop it anywhere.
- **link**: tried directly first. if the host doesn't send cors headers, it goes through `/api/audio`, a vercel function that streams public audio from our origin with range support. the function refuses private and reserved addresses (it checks dns), non-default ports, credentials, non-audio responses, and anything over 80 MB, and it follows redirects by hand so each hop is checked again.
- **youtube links** go to `/api/yt`, which runs a pinned, checksum-verified [yt-dlp](https://github.com/yt-dlp/yt-dlp) (fetched at build by `scripts/fetch-yt-dlp.mjs`), uses the function's own node as yt-dlp's javascript runtime, and streams the audio back. yt-dlp is only ever handed a canonical `youtube.com/watch?v=<id>` built from a validated id. youtube often tells datacenter addresses to sign in, so this works for some videos and not others. when it's refused, the ui offers the tab route instead.
- **tab** (chrome and edge on desktop): play anything in another tab (youtube, spotify, soundcloud) and share that tab. shfla only listens. the tab keeps playing itself, and nothing leaves your machine.
- **mic**: analysed, never played back.

## keys

`space` play/pause · `e` the language · `⌘k` everything · `r` reseed · `h` hide everything · `f` fullscreen

## develop

```sh
pnpm install
pnpm dev          # vite, with /api served by a dev middleware
pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

gpu checks: open the app, then paste `tests/browser/glider.js` into the console on `/`, or `tests/browser/rule110.js` on `/?rule=rule110`. each one prints `ok: true` when the gpu matches the reference.

## stack

[tetrastack](https://github.com/tetraslam/tetrastack), with the look decided in [BRIEF.md](BRIEF.md).

- vite + react 19 + typescript, pnpm, biome, vitest
- tailwind 4, shadcn on base ui, phosphor, cmdk, sonner, nuqs (rule and palette in the url)
- fontsource: inter, instrument serif (display), commit mono (the editor and numbers)
- codemirror 6 for the editor: it's a code editor for a small language, which is not what tiptap is for
- raw webgl2 rather than vgpu: the program compiles to glsl es 3.0 for a multi-pass float pyramid, so the code needs direct control of every pass, and webgl2 runs everywhere
- web audio for analysis, the demo song, and the voice
- vercel for hosting and the audio proxy function. there's no database: programs travel in the url
