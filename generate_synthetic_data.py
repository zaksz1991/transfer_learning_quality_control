"""
Synthetic Manufacturing Part Image Generator
==============================================
Generates procedurally-rendered "metal part" images that simulate a
real QC camera feed: a clean circular/ring-shaped part on a noisy
gray background for "good" parts, and the same base part with a
scratch, dent, crack, or dark blob defect overlaid for "defective" parts.

This lets the notebook run fully end-to-end (train -> evaluate -> infer)
without depending on an external dataset download.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random


def make_background(size=300, base_gray=160, noise_std=8, seed=None):
    rng = np.random.default_rng(seed)
    base = np.full((size, size), base_gray, dtype=np.float32)
    noise = rng.normal(0, noise_std, (size, size))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def draw_good_part(size=300, seed=None):
    rng = random.Random(seed)
    img = make_background(size=size, seed=seed)
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-10, 10)
    r_outer = rng.randint(90, 110)
    r_inner = r_outer - rng.randint(25, 40)

    metal_shade = rng.randint(190, 215)
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        fill=(metal_shade, metal_shade, metal_shade + 5),
        outline=(metal_shade - 40, metal_shade - 40, metal_shade - 35),
        width=3,
    )
    bg_shade = rng.randint(150, 170)
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        fill=(bg_shade, bg_shade, bg_shade),
    )

    # subtle machined ring lines (normal manufacturing texture, not a defect)
    for _ in range(4):
        rr = rng.randint(r_inner + 5, r_outer - 5)
        shade = metal_shade - rng.randint(5, 15)
        draw.ellipse(
            [cx - rr, cy - rr, cx + rr, cy + rr],
            outline=(shade, shade, shade),
            width=1,
        )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def add_defect(img, defect_type, seed=None):
    rng = random.Random(seed)
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    cx, cy = size // 2, size // 2

    if defect_type == "scratch":
        x1 = rng.randint(cx - 80, cx + 20)
        y1 = rng.randint(cy - 80, cy + 20)
        length = rng.randint(40, 90)
        angle = rng.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        draw.line([x1, y1, x2, y2], fill=(40, 40, 40), width=rng.randint(2, 4))

    elif defect_type == "dent":
        dx = cx + rng.randint(-60, 60)
        dy = cy + rng.randint(-60, 60)
        r = rng.randint(8, 18)
        draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=(90, 90, 95))
        draw.ellipse(
            [dx - r, dy - r, dx + r, dy + r], outline=(50, 50, 50), width=2
        )

    elif defect_type == "crack":
        x, y = cx + rng.randint(-40, 40), cy + rng.randint(-40, 40)
        points = [(x, y)]
        for _ in range(rng.randint(3, 6)):
            x += rng.randint(-15, 15)
            y += rng.randint(-15, 15)
            points.append((x, y))
        draw.line(points, fill=(20, 20, 20), width=2)

    elif defect_type == "stain":
        dx = cx + rng.randint(-70, 70)
        dy = cy + rng.randint(-70, 70)
        r = rng.randint(12, 25)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=(30, 25, 20, 120))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
    return img


def generate_dataset(root="dataset", n_train_per_class=160, n_test_per_class=40):
    defect_types = ["scratch", "dent", "crack", "stain"]
    seed_counter = 0

    for split, n in [("train", n_train_per_class), ("test", n_test_per_class)]:
        good_dir = os.path.join(root, split, "good")
        bad_dir = os.path.join(root, split, "defective")
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)

        for i in range(n):
            seed_counter += 1
            img = draw_good_part(seed=seed_counter)
            img.save(os.path.join(good_dir, f"good_{split}_{i:04d}.png"))

        for i in range(n):
            seed_counter += 1
            img = draw_good_part(seed=seed_counter)
            n_defects = random.Random(seed_counter).choice([1, 1, 1, 2])
            for d in range(n_defects):
                dtype = random.Random(seed_counter + d).choice(defect_types)
                img = add_defect(img, dtype, seed=seed_counter + d)
            img.save(os.path.join(bad_dir, f"defective_{split}_{i:04d}.png"))

    print(
        f"Generated {n_train_per_class * 2} train images and "
        f"{n_test_per_class * 2} test images under '{root}/'"
    )


if __name__ == "__main__":
    random.seed(42)
    generate_dataset()
