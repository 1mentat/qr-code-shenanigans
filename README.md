# Color photo QR codes

Full-color photographs that still scan. An extension of
[Andrew Taylor's dithered QR codes](https://www.andrewt.net/dithered-qr-codes/wtf/)
from 1-bit halftones to color.

**Writeup with scannable examples: <https://1mentat.github.io/qr-code-shenanigans/>**

## How it works

Taylor's generator rests on one freedom: a scanner reads only the center of
each module, so everything outside a center dot is available for image
content. His codes spend that freedom on a Floyd–Steinberg halftone, with a
two-pass error diffusion that hides the forced data modules in the dither.

Color comes from a second freedom: scanners are colorblind. Every decoder
flattens the image to grayscale before thresholding, but they flatten
differently. Some use Rec.601 luma, some average the channels, some read only
green. All of these are convex combinations of R, G and B, which gives a
guarantee: if every channel of a dark center sits below the threshold, and
every channel of a light center sits above it, the module reads correctly
under any grayscale conversion. Hue and saturation never enter into it.

The pipeline in `colorqr.py`:

1. **Round dots, not square modules.** Only a dot of radius 0.36
   module-widths is fully forced; a smoothstep ring fades the constraint to
   zero at 0.50. Where the photo already satisfies the bound, forcing does
   nothing and the dot disappears.
2. **Channel-bound forcing.** Dark centers: scale RGB until max(R,G,B)
   ≤ 0.30, which preserves hue and saturation. Light centers: blend toward
   white until min(R,G,B) ≥ 0.72, which preserves hue. Function patterns use
   tighter bounds (0.14 / 0.88) with the same math, so the finder squares
   carry the photo's tint.
3. **Error diffusion for continuous tone.** Taylor pre-diffuses the error
   from forced modules into the surrounding halftone. Here the neighbors are
   continuous-tone and absorb error directly: the luminance the dots inject
   is spread over nearby unconstrained pixels and subtracted from their
   luminance, chroma untouched. Local average brightness tracks the photo,
   and the dot grid fades at viewing distance.
4. **Mask selection.** All 8 QR mask patterns are scored against the photo's
   per-module luminance; the closest match wins.
5. **Photo-continued quiet zone.** The mandatory margin shows the photo
   blended toward white (min channel ≥ 0.78) instead of blank space.

Error correction is level H, and none of it is spent at generation time.
Every center is correct, so the full 30% redundancy remains for glare, folds
and bad lighting.

## Usage

```sh
uv run colorqr.py photo.jpg "https://example.com" -o out.png
uv run stress.py out.png "https://example.com"   # 20-way robustness check
```

Knobs: `--scale` (px/module), `--dot-hard`/`--dot-soft` (dot radii in module
units), `--dark-max`/`--light-min` (channel bounds), `--diffuse`
(error-diffusion strength), `--saturation`, `--mask`.

## Verification

`stress.py` decodes with two independent decoders, zxing-cpp and OpenCV's
QRCodeDetector, under twenty degradations: downscaling to 220 px (about
5 px/module), Gaussian blur to σ=3, JPEG quality 20, brightness 0.6–1.3×,
low contrast, rotation, perspective warp, and a print proxy (blur + gamma +
sensor noise). All three test photos pass every case.

The parameter sweep (`sweep.py`) surfaced one useful fact: dot size, not
contrast, is the binding constraint. Hard dots of radius 0.32 fail
aggressive downscaling at any contrast; at 0.36 every test passes with the
mildest color bounds. That is the right trade, since dot size costs less
visually than crushing the photo's tones.

A simulation matrix is not a stranger's phone in bad lighting. For print,
raise `--dot-hard` and widen the channel bounds.

Photos from Wikimedia Commons and picsum.photos.
