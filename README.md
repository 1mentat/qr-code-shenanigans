# Color photo QR codes

An extension of [Andrew Taylor's dithered QR codes](https://www.andrewt.net/dithered-qr-codes/wtf/)
from 1-bit halftones to full-color photographs.

## The idea

Taylor's technique rests on one observation: scanners sample only the *center*
of each module, so a module can be subdivided and everything outside the center
used freely for image content, with a two-pass error diffusion hiding the
forced data modules inside the halftone.

This project adds a second degree of freedom: **scanners are colorblind**.
Every decoder reduces the image to grayscale before binarizing, but different
decoders use different conversions (Rec.601 luma, green channel only, channel
averages...). Any such conversion is a convex combination of R, G, B — so if a
dark module center has *every* channel below a threshold, and a light center
has *every* channel above one, the module binarizes correctly no matter which
conversion the scanner uses. Hue and saturation remain completely free, and
darkening by multiplicative scaling / lightening by blending toward white
preserves the photo's hue at every forced pixel.

What `colorqr.py` does:

1. **Soft round center dots, not squares** — only a dot covering ~⅓ of each
   data module's area is hard-forced; a smoothstep ring fades the constraint
   out, and the rest of the module is the untouched photo. Where the photo
   already satisfies the bound, the forcing is a no-op and the dot vanishes.
2. **Channel-bound color forcing** — dark centers: scale RGB so max channel
   ≤ 0.30 (hue-preserving). Light centers: blend toward white so min channel
   ≥ 0.72. Guaranteed binarization under any grayscale conversion.
3. **Continuous-tone two-pass error diffusion** — the luminance error injected
   by the forced dots is spread (Gaussian) into surrounding unconstrained
   pixels and subtracted from their luminance (chroma preserved), so the local
   average brightness still tracks the photo and the code "disappears" at
   viewing distance.
4. **Mask selection** — all 8 QR masks are generated and scored against the
   photo's per-module luminance; the best-matching mask wins (a cheap cousin
   of QArt-style codeword steering).
5. **Tinted function patterns** — finder/timing/alignment patterns are forced
   with tighter bounds (≤ 0.14 / ≥ 0.88) but keep the photo's hue, so even the
   "fixed" parts pick up the image's palette.
6. **Photo-continued quiet zone** — the quiet zone shows the photo blended
   toward white (min channel ≥ 0.78) instead of dead whitespace.

Error correction level H, so on top of the guaranteed-correct centers there's
still the full 30% codeword redundancy in reserve for real-world abuse.

## Usage

```sh
uv run colorqr.py photo.jpg "https://example.com" -o out.png
uv run stress.py out.png "https://example.com"     # robustness check
```

Knobs: `--scale` (px/module), `--dot-hard`/`--dot-soft` (dot radii in module
units), `--dark-max`/`--light-min` (channel bounds), `--diffuse` (error
diffusion strength), `--saturation`, `--mask`.

## Reliability

`stress.py` decodes with both zxing-cpp and OpenCV under 20 degradations:
downscaling to 220 px (≈5 px/module), Gaussian blur to σ=3, JPEG q=20,
brightness 0.6–1.3×, low contrast, rotation, perspective warp, and a
print-and-photograph proxy (blur + gamma + sensor noise). The default
parameters were chosen by sweeping dot radius × channel bounds over three test
photos: **60/60 passes**, with dot radius (not contrast) being the binding
constraint — hard-forced dots need to be ≥ ~0.36 module radius to survive
aggressive downscaling.

Honest caveat, echoing the original article: passing a simulation matrix is
not the same as a stranger's potato phone in bad lighting. For print, bump
`--dot-hard` and widen the channel bounds.
