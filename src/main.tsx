import '@fontsource-variable/inter'
import '@fontsource/instrument-serif/latin-400.css'
import '@fontsource/commit-mono/latin-400.css'
import '@fontsource/commit-mono/latin-700.css'
import './index.css'
import { NuqsAdapter } from 'nuqs/adapters/react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app'

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <NuqsAdapter>
      <App />
    </NuqsAdapter>
  </StrictMode>,
)
