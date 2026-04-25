#!/usr/bin/env python3
"""
Build a 10-image segmentation test corpus from the Oxford-IIIT Pet dataset.

Expects the dataset to already be extracted at /tmp/oxford_pets/oxford-iiit-pet/
(images in images/, trimaps in annotations/trimaps/).

For each selected image we write:
  - pet_NNNN.jpg          — the source image (quality 90)
  - pet_NNNN_mask.png     — single-channel PNG: 255 = pet foreground, 0 = background
  - corpus.json           — SAM 2 prompts + DETR-panoptic expected labels

License: Oxford-IIIT Pet Dataset, CC-BY-SA 4.0.
Attribution: Parkhi et al., "Cats and Dogs", CVPR 2012.

Usage:
    python3 scripts/download_seg_corpus.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/tmp/oxford_pets/oxford-iiit-pet"
IMG_DIR = os.path.join(ROOT, "images")
TRIMAP_DIR = os.path.join(ROOT, "annotations/trimaps")
TEST_LIST = os.path.join(ROOT, "annotations/test.txt")
OUT = "test/support/images/segmentation"

# Oxford-IIIT Pet: 37 breeds, first 12 are cats (indices 1–12), rest are dogs.
# species column in list files: 1 = Cat, 2 = Dog
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}

os.makedirs(OUT, exist_ok=True)

# Read the test split. Format: <name> <class_id> <species> <breed_id>
# species: 1=Cat, 2=Dog
available = []
with open(TEST_LIST) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        name, _, species = parts[0], parts[1], parts[2]
        img_path = os.path.join(IMG_DIR, name + ".jpg")
        trimap_path = os.path.join(TRIMAP_DIR, name + ".png")
        if os.path.exists(img_path) and os.path.exists(trimap_path):
            coco_class = "cat" if species == "1" else "dog"
            breed = "_".join(name.split("_")[:-1])
            available.append((name, coco_class, breed, img_path, trimap_path))

print(f"Found {len(available)} usable test images on disk.")

# Pick 10 spread evenly across the available set for breed diversity.
step = len(available) // 10
indices = [i * step for i in range(10)]
selected = [available[i] for i in indices]

meta = []
for name, coco_class, breed, img_path, trimap_path in selected:
    image = Image.open(img_path).convert("RGB")
    trimap = Image.open(trimap_path)
    trimap_arr = np.array(trimap)

    # Trimap values: 1 = pet foreground, 2 = background, 3 = boundary
    fg = trimap_arr == 1
    if fg.sum() == 0:
        print(f"  WARNING: {name} has empty foreground — skipping")
        continue

    ys, xs = np.where(fg)
    cx, cy = int(xs.mean()), int(ys.mean())
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    fname = f"pet_{name}.jpg"
    mask_fname = f"pet_{name}_mask.png"

    image.save(os.path.join(OUT, fname), quality=90)
    Image.fromarray((fg * 255).astype(np.uint8)).save(os.path.join(OUT, mask_fname))

    entry = {
        "file": fname,
        "mask_file": mask_fname,
        "width": image.width,
        "height": image.height,
        "breed": breed,
        "coco_class": coco_class,
        "prompt_point": [cx, cy],
        "prompt_box": [x0, y0, x1 - x0, y1 - y0],
        "foreground_pixels": int(fg.sum()),
    }
    meta.append(entry)
    cats = sum(1 for e in meta if e["coco_class"] == "cat")
    dogs = sum(1 for e in meta if e["coco_class"] == "dog")
    print(
        f"  {fname:<40s}  {coco_class:<4s}  breed={breed:<25s}"
        f"  center=({cx:4d},{cy:4d})  fg_px={fg.sum():,}"
    )

corpus_path = os.path.join(OUT, "corpus.json")
with open(corpus_path, "w") as f:
    json.dump(meta, f, indent=2)

cats = sum(1 for e in meta if e["coco_class"] == "cat")
dogs = sum(1 for e in meta if e["coco_class"] == "dog")
total_kb = sum(
    os.path.getsize(os.path.join(OUT, e["file"])) +
    os.path.getsize(os.path.join(OUT, e["mask_file"]))
    for e in meta
) // 1024

print(f"\n{len(meta)} images ({cats} cats, {dogs} dogs), {total_kb} KB total")
print(f"Written to {OUT}/")
