# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only



import os
import cv2
import numpy as np
import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to BPA dataset")
parser.add_argument("--out", default="label_data", help="Output folder")
parser.add_argument("--tile", type=int, default=512)
parser.add_argument("--samples", type=int, default=5000)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

records = []

for root, _, files in os.walk(args.dataset):
    for fname in files:
        if not fname.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
            continue

        img_path = os.path.join(root, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        H, W = img.shape

        for y0 in range(0, H, args.tile):
            for x0 in range(0, W, args.tile):
                tile = img[y0:y0+args.tile, x0:x0+args.tile]
                if tile.size == 0:
                    continue

                # =========================
                # Preprocessing (AYNI)
                # =========================

                bw = cv2.adaptiveThreshold(
                    tile,255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    31,7
                )  

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
              

                contours,_ = cv2.findContours(
                    cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for cnt in contours:
                    if len(cnt) < 5:
                        continue
                    area = cv2.contourArea(cnt)
                    if area < 150:
                        continue

                    try:
                        ellipse = cv2.fitEllipse(cnt)
                    except:
                        continue

                    (xc, yc), (MA, ma), angle = ellipse
                    axis_ratio = min(MA, ma) / max(MA, ma)
                    if axis_ratio > 0.8:
                        continue  

                    records.append({
                        "image_path": img_path,
                        "x0": x0,
                        "y0": y0,
                        "tile_size": args.tile,
                        "contour": cnt.tolist(),
                        "ellipse": ellipse
                    })

print("Total candidates:", len(records))

random.shuffle(records)
records = records[:args.samples]

df = pd.DataFrame(records)
df.to_csv(os.path.join(args.out, "to_label.csv"), index=False)

print("Saved:", os.path.join(args.out, "to_label.csv"))
