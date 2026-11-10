# brief

## the product
shfla is an instrument. you give it a song (a file, a link, the microphone, or the built-in demo hum) and it grows a fractal that lives on the song. the fractal is a stack of grids, one per spatial octave, and every grid runs the same small program, written in the shfla language. each octave of the music feeds the grid at the matching scale: bass moves the big shapes and treble moves the fine grain. the beat clocks the zoom, so every bar takes you one level deeper. the fractal sings back as a drone built from how hard each level is computing. people come to watch a song turn into structure, and then to rewrite the rule and hear it change.

## the feeling
blurred, loud, tender. shoegaze is a wall of sound you can stand inside. the screen should feel like that wall: overexposed light, things coming in and out of focus, detail when you look closely.

## direction
**lens** (kyoto animation) blended with **native material** (ghostty).
- lens fits because shoegaze is about focus and haze. the scale hierarchy maps directly onto depth of field: coarse levels are soft and out of focus, fine levels are sharp. transitions are focus pulls, not fades. bloom stands in for shadows.
- native material fits because the fractal is the product, so the fractal is also the pitch. there is no landing page in front of it. the first paint is the live fractal running on its own clock, and the title sits on top of it.
- this is a dark canvas, but it is not the "generic dark devtool" attractor. there are no graphite panels, no indigo accent, and no hairline chrome. the dark is the black behind an overexposed light source, and the chrome is text on a soft scrim.

references:
- **kyoto animation's lens work**: rack focus as the transition. the title arrives out of focus and pulls sharp, and panels blur in from depth instead of sliding.
- **ghostty**: the product renders its own pitch in its own medium. the hero is the running fractal, not a screenshot.
- **mintlify agent score**: mono is allowed because the language is real code. the editor and the numbers use commit mono. everything else does not.

## identity
- palette: surfaces are the fractal itself on `#07060a` (near black, slightly warm). ink is `#f3ece6` (warm paper white), with muted ink at 60%. the accent is **sodium** `#ffb48a`, an overexposed streetlight peach. it means "the music is touching this": live meters, the active level, the playing source. errors are `#ff6b6b` at low saturation, shown in place. the fractal palettes are their own set, all ramps from deep and cool (coarse) to bright and warm (fine): *dusk*, *bleach*, *tape*, *sea*.
- type: inter for the ui. **instrument serif** for display only: the title, the empty state, preset names, set large and lowercase, blurred into focus. commit mono for the language editor and numeric readouts.
- shape: almost no boxes. panels are a scrim (a black-to-transparent gradient plus a 24px backdrop blur on the editor only), with a 14px radius where a container is unavoidable. borders are 1px ink at 8%. density is calm.
- texture and shaders: the fractal canvas is the only shader surface. post-processing runs over it: feedback trails (the reverb tail), bloom, radial chromatic smear, film grain. no other gradients or blobs.
- motion: `focus` = cubic-bezier(0.22, 1, 0.36, 1) at 520ms for blur-ins (blur 12px→0 plus opacity). `tick` = cubic-bezier(0.4, 0, 0.2, 1) at 160ms for small state changes. meters follow audio directly with no css easing. under reduced motion the zoom slows to a crawl, trails stay on but beat flashes turn off, and blur-ins become plain opacity.
- safety: beat-driven brightness is rate-limited, so the screen never strobes.

## hierarchy
- **stage (playing)**: first, the fractal fills the screen. second, a thin bottom strip shows the octave meters: one column per level, so you can see which level the music is touching. it also holds the source name, play/pause, and position. third, on demand: the language panel (`e`), the command palette (`⌘k`), and the palette and voice controls. press `h` to hide all chrome.
- **stage (no source yet)**: the fractal is already running on its own clock. on top of it: "shfla" set huge in instrument serif, one line that says what it is, and four source choices: demo, file, link, mic. dropping a file anywhere works too.
- **language panel**: the program comes first, in mono. then the presets as a short list of names. a one-line reference of the inputs sits at the bottom and expands on demand. compile errors show inline on their line, and the last good program keeps running.

## not parc
parc is the closest project on disk, through its shader field. shfla differs from it in three ways:
1. parc is warm paper with ink overlays. shfla is light on black: the surface emits light, and nothing reads as paper.
2. parc's field is ambient texture behind content. in shfla the shader is the content, and the ui is the thing that recedes.
3. parc's accent (ember) means "yours" or "alive". shfla's accent (sodium) means only "the audio is touching this right now", and it shows up nowhere else.
