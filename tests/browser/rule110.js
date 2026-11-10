// open /?rule=rule110 first. checks the gpu against a reference rule 110 for 60 steps.
;(() => {
  const e = window.shfla.engine
  e.frozen = true
  const k = 5
  const { size } = e.readLevel(k)
  const g = new Float32Array(size * size)
  g[(size - 1) * size + (size - 1)] = 1
  e.debugWrite(k, g)
  const steps = 60
  for (let i = 0; i < steps; i++) e.debugStep(k)
  const { data } = e.readLevel(k)
  const ref = [Array.from({ length: size }, (_, x) => (x === size - 1 ? 1 : 0))]
  for (let r = 1; r <= steps; r++) {
    const p = ref[r - 1]
    ref.push(p.map((_, x) => (110 >> ((p[(x - 1 + size) % size] << 2) | (p[x] << 1) | p[(x + 1) % size])) & 1))
  }
  let mismatches = 0
  for (let j = 0; j <= steps; j++) {
    const row = size - 1 - j
    for (let x = 0; x < size; x++) if ((data[row * size + x] > 0.5 ? 1 : 0) !== ref[steps - j][x]) mismatches++
  }
  e.frozen = false
  return JSON.stringify({ ok: mismatches === 0, mismatches, rows: steps + 1, size })
})()
