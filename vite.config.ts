import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

/** Serves the vercel functions in /api during `vite dev`, so the link proxy works locally. */
function devApi(): Plugin {
  return {
    name: 'shfla-dev-api',
    configureServer(server) {
      server.middlewares.use('/api/audio', async (req, res) => {
        const mod = await server.ssrLoadModule('/api/audio.ts')
        const headers = new Headers()
        for (const [k, v] of Object.entries(req.headers)) if (typeof v === 'string') headers.set(k, v)
        const response: Response = await mod.GET(new Request(`http://localhost${req.originalUrl}`, { headers }))
        res.statusCode = response.status
        response.headers.forEach((v, k) => {
          res.setHeader(k, v)
        })
        if (!response.body) return res.end()
        const reader = response.body.getReader()
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          res.write(value)
        }
        res.end()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), devApi()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, 'src') } },
})
