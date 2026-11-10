import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from '@/components/ui/command'
import { PALETTES } from '@/gpu/palettes'
import { PRESETS } from '@/lang/presets'

interface Props {
  open: boolean
  onOpenChange: (v: boolean) => void
  run: (action: string) => void
  presetId: string | null
  paletteId: string
  playing: boolean
  hasSource: boolean
  voiceOn: boolean
}

export function CommandMenu(p: Props) {
  const go = (a: string) => () => {
    p.onOpenChange(false)
    p.run(a)
  }
  return (
    <CommandDialog
      open={p.open}
      onOpenChange={p.onOpenChange}
      title="everything"
      description="search for something to do"
    >
      <CommandInput placeholder="what now?" />
      <CommandList>
        <CommandEmpty>nothing by that name</CommandEmpty>
        <CommandGroup heading="listen to">
          <CommandItem onSelect={go('source:demo')}>the demo hum</CommandItem>
          <CommandItem onSelect={go('source:file')}>a song from this computer</CommandItem>
          <CommandItem onSelect={go('source:link')}>a link</CommandItem>
          <CommandItem onSelect={go('source:mic')}>the microphone</CommandItem>
          {p.hasSource && (
            <CommandItem onSelect={go('toggle')}>
              {p.playing ? 'pause' : 'play'}
              <CommandShortcut>space</CommandShortcut>
            </CommandItem>
          )}
        </CommandGroup>
        <CommandGroup heading="rules">
          {PRESETS.map((x) => (
            <CommandItem key={x.id} value={`rule ${x.name} ${x.line}`} onSelect={go(`preset:${x.id}`)}>
              <span className={x.id === p.presetId ? 'text-white' : ''}>{x.name}</span>
              <span className="truncate text-ink-muted">{x.line}</span>
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandGroup heading="palettes">
          {PALETTES.map((x) => (
            <CommandItem key={x.id} value={`palette ${x.name}`} onSelect={go(`palette:${x.id}`)}>
              <span className={x.id === p.paletteId ? 'text-white' : ''}>{x.name}</span>
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandGroup heading="do">
          <CommandItem onSelect={go('panel')}>
            open the language
            <CommandShortcut>e</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={go('reseed')}>
            reseed every scale
            <CommandShortcut>r</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={go('voice')}>
            {p.voiceOn ? 'quiet the fractal’s voice' : 'let the fractal sing'}
          </CommandItem>
          <CommandItem onSelect={go('chrome')}>
            hide everything but the fractal
            <CommandShortcut>h</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={go('fullscreen')}>
            fullscreen
            <CommandShortcut>f</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={go('share')}>copy a link to this rule</CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
