# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "segno>=1.6",
#   "numpy>=1.26",
#   "pillow>=10",
# ]
# ///
"""Color photo QR codes.

Extends the dithered QR codes technique (andrewt.net) from 1-bit halftones
to full-color photographs. Scanners binarize via some grayscale conversion
and sample only module centers, so:

  * every RGB channel at a dark-module center is forced below `dark_max`
    (multiplicative scaling, which preserves hue), and every channel at a
    light-module center above `light_min` (blend toward white); the module
    then binarizes correctly under any channel weighting a decoder uses;
  * only a soft-edged dot at each data-module center is constrained; the
    rest of the module shows the photo untouched;
  * the luminance error injected by forced dots is diffused into nearby
    unconstrained pixels (two-pass error diffusion, generalized to
    continuous tone) so local average brightness still tracks the photo;
  * all 8 QR masks are scored against the photo and the best match is
    used; function patterns are hue-tinted; the quiet zone keeps the
    photo, brightness-constrained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import segno
from PIL import Image, ImageOps
from segno import consts as C

FUNCTION_TYPES = {
    C.TYPE_FINDER_PATTERN_DARK, C.TYPE_FINDER_PATTERN_LIGHT,
    C.TYPE_ALIGNMENT_PATTERN_DARK, C.TYPE_ALIGNMENT_PATTERN_LIGHT,
    C.TYPE_TIMING_DARK, C.TYPE_TIMING_LIGHT,
    C.TYPE_FORMAT_DARK, C.TYPE_FORMAT_LIGHT,
    C.TYPE_VERSION_DARK, C.TYPE_VERSION_LIGHT,
    C.TYPE_DARKMODULE, C.TYPE_SEPARATOR,
}
DARK_TYPES = {
    C.TYPE_FINDER_PATTERN_DARK, C.TYPE_ALIGNMENT_PATTERN_DARK,
    C.TYPE_TIMING_DARK, C.TYPE_FORMAT_DARK, C.TYPE_VERSION_DARK,
    C.TYPE_DARKMODULE, C.TYPE_DATA_DARK,
}


@dataclass
class Params:
    scale: int = 16          # pixels per module
    border: int = 4          # quiet zone, in modules
    dot_hard: float = 0.36   # radius (module units) of fully forced dot
    dot_soft: float = 0.50   # radius where forcing has faded to zero
    dark_max: float = 0.30   # ceiling on every channel at dark centers
    light_min: float = 0.72  # floor on every channel at light centers
    fn_dark_max: float = 0.14   # function patterns: stronger contrast
    fn_light_min: float = 0.88
    quiet_min: float = 0.78  # channel floor in the quiet zone
    diffuse: float = 0.5     # strength of luminance error compensation
    saturation: float = 1.25  # pre-boost, photos wash out under constraints
    mask: int | None = None  # force a QR mask, or None = auto-pick


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma on gamma-encoded values (what cv2/most decoders use)."""
    return rgb @ np.array([0.299, 0.587, 0.114])


def force_dark(rgb: np.ndarray, ceiling: float) -> np.ndarray:
    """Scale colors so max channel <= ceiling. Preserves hue and saturation."""
    peak = rgb.max(axis=-1, keepdims=True)
    scale = np.where(peak > ceiling, ceiling / np.maximum(peak, 1e-6), 1.0)
    return rgb * scale


def force_light(rgb: np.ndarray, floor: float) -> np.ndarray:
    """Blend toward white so min channel >= floor. Preserves hue."""
    low = rgb.min(axis=-1, keepdims=True)
    a = np.clip((floor - low) / np.maximum(1.0 - low, 1e-6), 0.0, 1.0)
    return rgb + a * (1.0 - rgb)


def set_luma(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move each pixel's luma to `target`, preserving hue (scale down / blend up)."""
    y = luma(rgb)[..., None]
    t = target[..., None]
    darker = rgb * (t / np.maximum(y, 1e-6))
    a = np.clip((t - y) / np.maximum(1.0 - y, 1e-6), 0.0, 1.0)
    lighter = rgb + a * (1.0 - rgb)
    out = np.where(t <= y, darker, lighter)
    return np.clip(out, 0.0, 1.0)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur for a 2-D float array (no scipy needed)."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, radius, mode="edge")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp)


def module_maps(qr: segno.QRCode) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_dark, is_function) bool arrays at module resolution."""
    rows = list(qr.matrix_iter(scale=1, border=0, verbose=True))
    types = np.array([[int(v) for v in row] for row in rows])
    is_dark = np.isin(types, list(DARK_TYPES))
    is_function = np.isin(types, list(FUNCTION_TYPES))
    return is_dark, is_function


def prepare_photo(path: str, size: int, saturation: float) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    mean = rgb.mean(axis=-1, keepdims=True)
    return np.clip(mean + (rgb - mean) * saturation, 0.0, 1.0)


def pick_mask(text: str, photo_luma_mod: np.ndarray) -> segno.QRCode:
    """Choose the QR mask whose data modules best match the photo's tones."""
    best, best_score = None, np.inf
    for m in range(8):
        qr = segno.make(text, error="h", mask=m, micro=False, boost_error=False)
        is_dark, is_function = module_maps(qr)
        target = np.where(is_dark, 0.12, 0.88)
        score = np.abs(photo_luma_mod - target)[~is_function].sum()
        if score < best_score:
            best, best_score = qr, score
    return best


def make_color_qr(photo_path: str, text: str, p: Params) -> Image.Image:
    probe = segno.make(text, error="h", micro=False, boost_error=False)
    n = len(probe.matrix)  # modules per side
    full = (n + 2 * p.border) * p.scale

    photo = prepare_photo(photo_path, full, p.saturation)

    if p.mask is not None:
        qr = segno.make(text, error="h", mask=p.mask, micro=False, boost_error=False)
    else:
        # per-module mean luma of the symbol area, for mask scoring
        sym = photo[
            p.border * p.scale : (p.border + n) * p.scale,
            p.border * p.scale : (p.border + n) * p.scale,
        ]
        mod_luma = luma(sym).reshape(n, p.scale, n, p.scale).mean(axis=(1, 3))
        qr = pick_mask(text, mod_luma)

    is_dark, is_function = module_maps(qr)

    # --- per-pixel geometry ---------------------------------------------
    idx = np.arange(full)
    mod = idx // p.scale - p.border          # module index per pixel row/col
    frac = (idx % p.scale + 0.5) / p.scale   # position within module, 0..1
    my, mx = np.meshgrid(mod, mod, indexing="ij")
    fy, fx = np.meshgrid(frac, frac, indexing="ij")
    inside = (my >= 0) & (my < n) & (mx >= 0) & (mx < n)
    myc, mxc = my.clip(0, n - 1), mx.clip(0, n - 1)

    pix_dark = np.where(inside, is_dark[myc, mxc], False)
    pix_fn = np.where(inside, is_function[myc, mxc], False)

    dist = np.hypot(fy - 0.5, fx - 0.5)  # distance from module center, module units
    t = (p.dot_soft - dist) / max(p.dot_soft - p.dot_hard, 1e-6)
    dot_w = np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0))  # smoothstep

    # weight of the constraint at each pixel
    w = np.zeros((full, full))
    w[inside & ~pix_fn] = dot_w[inside & ~pix_fn]   # data modules: center dot
    w[inside & pix_fn] = 1.0                        # function patterns: whole module
    w[~inside] = 1.0                                # quiet zone: whole area

    # --- forced colors ---------------------------------------------------
    dark_ceiling = np.where(pix_fn, p.fn_dark_max, p.dark_max)[..., None]
    light_floor = np.where(
        ~inside, p.quiet_min, np.where(pix_fn, p.fn_light_min, p.light_min)
    )[..., None]

    forced_dark = photo * np.minimum(
        1.0, dark_ceiling / np.maximum(photo.max(axis=-1, keepdims=True), 1e-6)
    )
    low = photo.min(axis=-1, keepdims=True)
    a = np.clip((light_floor - low) / np.maximum(1.0 - low, 1e-6), 0.0, 1.0)
    forced_light = photo + a * (1.0 - photo)
    forced = np.where(pix_dark[..., None], forced_dark, forced_light)

    # --- two-pass error diffusion, continuous-tone version ---------------
    # luminance error the hard-forced regions inject, spread to free pixels
    y_photo = luma(photo)
    err = (luma(forced) - y_photo) * w
    sigma = p.scale * 0.9
    err_spread = gaussian_blur(err, sigma)
    free = 1.0 - w
    free_density = np.maximum(gaussian_blur(free, sigma), 0.05)
    comp = -p.diffuse * err_spread / free_density
    comp = np.clip(comp, -0.30, 0.30) * free
    adjusted = set_luma(photo, np.clip(y_photo + comp, 0.0, 1.0))

    # --- compose ---------------------------------------------------------
    out = adjusted * (1.0 - w[..., None]) + forced * w[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Color photo QR code generator")
    ap.add_argument("photo")
    ap.add_argument("text")
    ap.add_argument("-o", "--output", default="color_qr.png")
    ap.add_argument("--scale", type=int, default=Params.scale)
    ap.add_argument("--border", type=int, default=Params.border)
    ap.add_argument("--dot-hard", type=float, default=Params.dot_hard)
    ap.add_argument("--dot-soft", type=float, default=Params.dot_soft)
    ap.add_argument("--dark-max", type=float, default=Params.dark_max)
    ap.add_argument("--light-min", type=float, default=Params.light_min)
    ap.add_argument("--diffuse", type=float, default=Params.diffuse)
    ap.add_argument("--saturation", type=float, default=Params.saturation)
    ap.add_argument("--mask", type=int, default=None)
    args = ap.parse_args()
    p = Params(
        scale=args.scale, border=args.border,
        dot_hard=args.dot_hard, dot_soft=args.dot_soft,
        dark_max=args.dark_max, light_min=args.light_min,
        diffuse=args.diffuse, saturation=args.saturation, mask=args.mask,
    )
    img = make_color_qr(args.photo, args.text, p)
    img.save(args.output)
    print(f"wrote {args.output} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
