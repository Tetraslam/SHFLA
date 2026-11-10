export interface Preset {
  id: string
  name: string
  /** one line, shown in the list */
  line: string
  source: string
}

export const PRESETS: Preset[] = [
  {
    id: 'haze',
    name: 'haze',
    line: "conway's life at every scale, sparked by the song",
    source: `# conway's life, running at six scales at once.
# each scale listens to its own octave of the song:
# bass moves the big shapes, treble the fine dust.

alive = sum == 3 or (self == 1 and sum == 2)

# an onset in this level's octave throws sparks into it
spark = rand < hit * hit * 0.05

step = alive or spark

# zooming past an octave splits every cell into four.
# keep three of the four, and the zoom grows a carpet.
split = self and not (qx == 1 and qy == 1)

seed = rand < 0.3
`,
  },
  {
    id: 'carpet',
    name: 'carpet',
    line: 'the oldest fractal: no neighbors, no time, one rule',
    source: `# a sierpinski triangle. the classic fractal is a program
# that ignores its neighbors and only knows how to split.
# every octave of zoom cuts one corner out of every cell.

split = self and not (qx == 1 and qy == 0)

# it doesn't step on its own. the song only flickers it:
# a loud hit in this level's octave briefly inverts a few cells.
flicker = rand < hit * 0.004
step = if flicker then 1 - self else self

seed = 1
`,
  },
  {
    id: 'rule110',
    name: 'rule 110',
    line: 'a universal computer one row wide, scrolling its own history',
    source: `# rule 110: each new row looks at the row above it, three cells at a time.
# matthew cook proved it can run any program (2004).
# the picture is its history: every step the tape scrolls up one row,
# and the newest row (the bottom one) is rewritten from itself.

l = nb(-1, 0)
c = self
r = nb(1, 0)
n = l * 4 + c * 2 + r
on = n == 1 or n == 2 or n == 3 or n == 5 or n == 6

# the song writes into the tape: hits flip cells on the newest row
write = rand < hit * 0.02
newest = if write then 1 - on else on

step = if y == size - 1 then newest else nb(0, 1)

# one row per step is slow, so every level takes eight steps a tick.
# coarse levels still tick less often: they scroll slower, like parallax.
speed = 8

split = rand < 0.5
seed = rand < 0.5
`,
  },
  {
    id: 'brain',
    name: 'brain',
    line: "brian's brain: three states, always firing",
    source: `# brian's brain: a cell fires, rests, then waits.
# it fires again when exactly two neighbors are firing.
# a restless thing. loud music only makes it worse.

firing = count(1)

step = (if self == 1 then 0.5
  else if self == 0.5 then 0
  else if firing == 2 then 1
  else rand < hit * 0.02)

split = if self == 1 then (qx == qy) else self
seed = rand < 0.08
`,
  },
  {
    id: 'echo',
    name: 'echo',
    line: 'each scale copies the one above it on the beat',
    source: `# life again, but the scales talk.
# on a hit, a level stops thinking for itself and takes
# its shape from the coarser level above it, so big
# structure cascades down into fine detail, octave by octave.

life = sum == 3 or (self == 1 and sum == 2)
listen = hit > 0.55 and rand < 0.6

step = if listen then up else life
split = self and not (qx == 0 and qy == 0)
seed = rand < 0.25
`,
  },
  {
    id: 'tide',
    name: 'tide',
    line: 'the wave equation, with the song dropping stones in',
    source: `# ripples. prev is this cell one step ago, and that is all
# a wave needs: where it is, where it was, and its neighbors.
# still water sits at 0.5. every onset in a level's octave
# drops stones into that level.

edge = (nb(1, 0) + nb(-1, 0) + nb(0, 1) + nb(0, -1)) / 4
height = 2 * self - prev + 0.9 * (edge - self)
stone = rand < hit * hit * 0.002

step = if stone then 1 else 0.5 + (height - 0.5) * 0.994

# what you see is how far the water is from still
light = abs(step - 0.5) * 5

speed = 3
split = self
seed = 0.5
`,
  },
]

export const DEFAULT_PRESET = PRESETS[0]
