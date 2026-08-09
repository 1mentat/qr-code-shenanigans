# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10", "segno>=1.6",
#   "opencv-python-headless>=4.9", "zxing-cpp>=2.2"]
# ///
"""Parameter sweep: find mildest constraints that pass the full stress matrix."""
import itertools

import colorqr
import stress

URL = "https://www.andrewt.net/dithered-qr-codes/"
PHOTOS = ["photos/bird.jpg", "photos/fruit.jpg", "photos/landscape.jpg"]


def score(p: colorqr.Params) -> tuple[int, list[str]]:
    total, fails = 0, []
    for photo in PHOTOS:
        img = colorqr.make_color_qr(photo, URL, p)
        for name, variant in stress.variants(img):
            if stress.decode(variant) == URL:
                total += 1
            else:
                fails.append(f"{photo.split('/')[-1]}:{name}")
    return total, fails


grid = itertools.product(
    [0.32, 0.36, 0.40],   # dot_hard
    [0.30, 0.26, 0.22],   # dark_max
    [0.72, 0.76, 0.80],   # light_min
)
results = []
for dot_hard, dark_max, light_min in grid:
    p = colorqr.Params(dot_hard=dot_hard, dot_soft=dot_hard + 0.14,
                       dark_max=dark_max, light_min=light_min)
    n, fails = score(p)
    results.append((n, dot_hard, dark_max, light_min, fails))
    print(f"dot={dot_hard} dark<={dark_max} light>={light_min}: {n}/60"
          + (f"  {fails[:4]}" if fails else "  ALL PASS"))

results.sort(key=lambda r: (-r[0], r[1], -r[2], r[3]))
print("\nbest:", results[0])
