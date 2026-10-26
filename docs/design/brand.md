# Brand

The `probly` logo: a disk with two waves, red-pink and blue, whose overlap is a third color. All logo files are
plain vector shapes with flat colors (no fonts, blend modes or transparency effects), so they render the same in
browsers, Figma, Illustrator, Inkscape, LaTeX, slides and favicons.

## Files

All files live in `docs/source/_static/logo/`, so the docs site serves them too. Every logo exists in a `-light`
variant for light backgrounds and a `-dark` variant for dark backgrounds.

| File | What | Used in |
|---|---|---|
| `probly-logo-stacked-*.svg` | mark above the wordmark | docs sidebar, landing page, README |
| `probly-logo-horizontal-*.svg` | mark left of the wordmark | wide spaces, e.g. slides and banners |
| `probly-mark-*.svg` | the disk alone | small square spaces, e.g. avatars |
| `probly-wordmark-*.svg` | the word "probly" alone | next to other logos |
| `favicon.svg`, `favicon.ico`, `apple-touch-icon.png` | browser and home-screen icons; the SVG switches to dark by itself | docs `<head>` |
| `probly-mark-light-512.png` | raster mark | fallback thumbnail of gallery examples without a plot |
| `probly-social-preview-light-1280x640.png` | link preview card | docs `og:image`, GitHub social preview |

The SVGs are the masters: export a PNG from them wherever a format does not take SVG (slides, chat avatars, etc.).
The GitHub social preview is not set from the repository; upload the PNG under Settings -> General -> Social preview.

## Colors

The two ink colors, as designed, are a red leaning red-pink `#ff2763` and a blue with a hint of violet `#285bff`
(on light backgrounds multiplied at 85% with a lighter overlap, on dark backgrounds screened at 90%). The red-pink
is the lead color: the docs use it as their accent (`docs/source/_static/css/brand.css`), with links staying blue.
The wordmark carries the same two inks as accents: the **l** in blue and the **y** in red-pink, as the shapiq
wordmark does with its **i** and **q**.

The flat colors used in the files:

| | light | dark |
|---|---|---|
| disk | `#f1f1f1` | `#22262e` |
| red-pink | `#fd4578` | `#e9265d` |
| blue | `#4671fd` | `#2755ea` |
| overlap | `#63338a` | `#ea6aef` |
| wordmark | `#1a1a1a` | `#e6e6e6` |
| wordmark accents **l** / **y** | `#285bff` / `#ff2763` | `#285bff` / `#ff2763` |
| page background (reference) | `#ffffff` | `#16181d` |

## Usage

- **Clear space:** keep at least 1/4 of the disk diameter free around the logo.
- **Minimum size:** the mark at least 16 px (favicon), the horizontal logo at least 120 px wide, the stacked logo at
  least 80 px wide.
- **Don't** recolor single waves, stretch, rotate, add effects or shadows, or put a `-light` file on a dark
  background (use the `-dark` file).

In a GitHub README, switch with the viewer's theme. Use absolute URLs so the image also renders on PyPI:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/logo/probly-logo-stacked-dark.svg">
  <img alt="probly" src="https://raw.githubusercontent.com/pwhofman/probly/main/docs/source/_static/logo/probly-logo-stacked-light.svg" width="160">
</picture>
```

## Font and license

The wordmark is set in **Space Grotesk Bold** (Florian Karsten, based on Colophon's Space Mono), licensed under the
**SIL Open Font License 1.1**. The letters are converted to vector outlines: using outlined text in a logo is allowed
under the OFL, also commercially, and no font file is shipped or needed. The license text is in
[`OFL-SpaceGrotesk.txt`](OFL-SpaceGrotesk.txt) for reference.

## Geometry

For future edits of the mark: viewBox 200 x 200, disk center (100, 100), r = 92. Each wave has the center line
`100 + amp * sin(2 * pi * 1.25 * x / 200 + phase)` and the half-width `5 + 22 * (x / 200) ** 1.6`. Red-pink: amp 22,
phase 0, right end tucked in (top -6, bottom -12, smoothstep from x = 150 to 195). Blue: amp 16, phase 1.2.
