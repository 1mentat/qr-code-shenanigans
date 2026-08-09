# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pillow>=10",
#   "opencv-python-headless>=4.9",
#   "zxing-cpp>=2.2",
# ]
# ///
"""Stress-test a QR image: decode under simulated real-world degradations."""

from __future__ import annotations

import argparse
import io
import sys

import cv2
import numpy as np
import zxingcpp
from PIL import Image, ImageEnhance, ImageFilter


def decode(img: Image.Image) -> str | None:
    arr = np.asarray(img.convert("RGB"))
    res = zxingcpp.read_barcode(
        arr, formats=zxingcpp.BarcodeFormat.QRCode, try_rotate=True
    )
    if res and res.valid:
        return res.text
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    det = cv2.QRCodeDetector()
    text, *_ = det.detectAndDecode(gray)
    return text or None


def variants(img: Image.Image):
    yield "original", img
    for size in (640, 480, 360, 280, 220):
        yield f"downscale-{size}", img.resize((size, size), Image.LANCZOS)
    for sigma in (1.0, 2.0, 3.0):
        small = img.resize((480, 480), Image.LANCZOS)
        yield f"480+blur{sigma}", small.filter(ImageFilter.GaussianBlur(sigma))
    for q in (75, 40, 20):
        buf = io.BytesIO()
        img.resize((480, 480), Image.LANCZOS).save(buf, "JPEG", quality=q)
        yield f"480+jpeg{q}", Image.open(buf)
    for b in (0.6, 0.8, 1.3):
        yield f"brightness{b}", ImageEnhance.Brightness(img).enhance(b)
    yield "contrast0.6", ImageEnhance.Contrast(img).enhance(0.6)
    for angle in (7, 25):
        yield f"rotate{angle}", img.rotate(
            angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC
        )
    # perspective: simulate off-axis phone camera
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.06, h * 0.03], [w * 0.97, 0], [w, h], [0, h * 0.94]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, m, (w, h), borderValue=(255, 255, 255))
    yield "perspective", Image.fromarray(warped)
    # print-and-photograph proxy: blur + noise + gamma
    noisy = np.asarray(
        img.resize((420, 420), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2)),
        dtype=np.float64,
    )
    rng = np.random.default_rng(42)
    noisy = np.clip((noisy / 255.0) ** 1.3 * 255 + rng.normal(0, 8, noisy.shape), 0, 255)
    yield "print-proxy", Image.fromarray(noisy.astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("expected")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    img = Image.open(args.image).convert("RGB")
    passed = total = 0
    fails = []
    for name, variant in variants(img):
        got = decode(variant)
        ok = got == args.expected
        total += 1
        passed += ok
        if not ok:
            fails.append(name)
        if not args.quiet:
            print(f"  {'PASS' if ok else 'FAIL':4}  {name}" + ("" if ok or not got else f" (got {got!r})"))
    print(f"{args.image}: {passed}/{total}" + (f"  failed: {', '.join(fails)}" if fails else ""))
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
