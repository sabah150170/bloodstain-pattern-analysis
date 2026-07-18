# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


import cv2
import pandas as pd
import numpy as np
import ast
import os
import subprocess
import signal
import time
import argparse

# =========================
# ARGÜMANLAR
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, required=True, help="Başlangıç index")
parser.add_argument("--end", type=int, required=True, help="Bitiş index (inclusive)")
args = parser.parse_args()

START_IDX = args.start
END_IDX = args.end

# =========================
# AYARLAR
# =========================
INPUT_CSV = "label_data/to_label.csv"
OUT_IMG_DIR = "label_images"
OUT_CSV = f"ellipse_labeled_{START_IDX}_{END_IDX}.csv"

WIN_W, WIN_H = 600, 600

os.makedirs(OUT_IMG_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# güvenlik
END_IDX = min(END_IDX, len(df) - 1)

labels = []
kept_rows = []

print(f"""
ETIKETLEME ARALIĞI:
  {START_IDX}  →  {END_IDX}

Kontroller:
  1 -> iyi elips
  0 -> kötü elips
  p -> pass
  q -> çık
""")

# =========================
# SADECE SEÇİLEN ARALIK
# =========================
for idx in range(START_IDX, END_IDX + 1):
    row = df.iloc[idx]

    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    x0, y0, s = int(row["x0"]), int(row["y0"]), int(row["tile_size"])
    tile = img[y0:y0+s, x0:x0+s]
    if tile.size == 0:
        continue

    # ---- kontur & elips ----
    cnt = np.array(ast.literal_eval(row["contour"]), dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)

    ell = ast.literal_eval(row["ellipse"])
    ellipse = (tuple(ell[0]), tuple(ell[1]), float(ell[2]))

    # ---- görselleri kaydet ----
    raw_path = os.path.join(OUT_IMG_DIR, f"raw_{idx:05d}.png")
    overlay_path = os.path.join(OUT_IMG_DIR, f"overlay_{idx:05d}.png")

    cv2.imwrite(raw_path, tile)

    vis = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
    cv2.ellipse(vis, ellipse, (0, 0, 255), 2)
    cv2.imwrite(overlay_path, vis)

    # =========================
    # İKİ AYRI PENCERE
    # =========================
    proc_raw = subprocess.Popen(
        ["feh", "--auto-zoom", "--geometry", f"{WIN_W}x{WIN_H}+100", raw_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    proc_overlay = subprocess.Popen(
        ["feh", "--auto-zoom", "--geometry", f"{WIN_W}x{WIN_H}+{100 + WIN_W + 20}", overlay_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(0.2)

    lab = input(f"[{idx}] Etiket (1/0/p/q): ").strip().lower()
    print(f"--> {lab}\n")

    proc_raw.send_signal(signal.SIGTERM)
    proc_overlay.send_signal(signal.SIGTERM)
    proc_raw.wait()
    proc_overlay.wait()
    
    if lab in ("1", "0"):
        row_copy = row.copy()
        row_copy["overlay_image"] = os.path.basename(overlay_path)

        labels.append(int(lab))
        kept_rows.append(row_copy)
    elif lab == "p":
        continue
    elif lab == "q":
        print("\n[INFO] Kullanıcı tarafından durduruldu.")
        break

# =========================
# KAYDET
# =========================
if kept_rows:
    out_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    out_df["label"] = labels
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nKaydedildi -> {OUT_CSV} ({len(out_df)} etiket)")
else:
    print("\n[WARN] Bu batch için etiket yok.")
