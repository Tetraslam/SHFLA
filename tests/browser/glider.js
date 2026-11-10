// open / (the haze preset, which is conway's life). checks a glider on the gpu moves one cell diagonally in four steps.
;(() => {
  const e = window.shfla.engine
  e.frozen = true
  const k = 5
  const { size } = e.readLevel(k)
  const g = new Float32Array(size * size)
  const cells = [
    [1, 0],
    [2, 1],
    [0, 2],
    [1, 2],
    [2, 2],
  ]
  for (const [x, y] of cells) g[(y + 10) * size + (x + 10)] = 1
  e.debugWrite(k, g)
  for (let i = 0; i < 4; i++) e.debugStep(k)
  const { data } = e.readLevel(k)
  const lit = []
  for (let i = 0; i < data.length; i++) if (data[i] > 0.5) lit.push(`${i % size},${Math.floor(i / size)}`)
  const want = cells.map(([x, y]) => `${x + 11},${y + 11}`).sort()
  e.frozen = false
  return JSON.stringify({ ok: JSON.stringify(lit.sort()) === JSON.stringify(want), lit, want })
})()
