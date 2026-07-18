# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


import cv2
import pandas as pd
import numpy as np
import ast
import joblib
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ======================
# 1) Load labeled csv
# ======================
csv_files = sorted(glob.glob("ellipse_labeled_*.csv"))

if not csv_files:
    raise FileNotFoundError("ellipse_labeled_*.csv dosyası bulunamadı")

df_list = []
for f in csv_files:
    tmp = pd.read_csv(f)
    tmp["source_file"] = os.path.basename(f)  
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)


# ======================
# 2) Feature functions
# ======================
def ellipse_fit_error(contour, ellipse):
    (xc, yc), (MA, ma), angle = ellipse
    if MA <= 0 or ma <= 0:
        return 1.0
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    errs = []
    for p in contour:
        x, y = p[0]
        xt = (x - xc) * cos_t + (y - yc) * sin_t
        yt = -(x - xc) * sin_t + (y - yc) * cos_t
        val = (xt/(MA/2))**2 + (yt/(ma/2))**2
        errs.append(abs(val - 1))
    return float(np.mean(errs))

def overlap_metrics(contour, ellipse, shape):
    h, w = shape
    mask_cnt = np.zeros((h, w), dtype=np.uint8)
    mask_ell = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(mask_cnt, [contour], -1, 255, -1)
    cv2.ellipse(mask_ell, ellipse, 255, -1)

    inter = np.logical_and(mask_cnt, mask_ell).sum()
    cnt_area = mask_cnt.sum()
    ell_area = mask_ell.sum()

    outside_ratio = 1.0 - inter/cnt_area if cnt_area > 0 else 1.0
    area_ratio = cnt_area/ell_area if ell_area > 0 else 10.0
    overlap_ratio = inter/cnt_area if cnt_area > 0 else 0.0
    return outside_ratio, area_ratio, overlap_ratio

# ======================
# 3) Extract features from each labeled row
# ======================
rows = []
bad_paths = 0

for _, row in df.iterrows():
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        bad_paths += 1
        continue

    x0, y0, s = int(row["x0"]), int(row["y0"]), int(row["tile_size"])
    tile = img[y0:y0+s, x0:x0+s]
    if tile.size == 0:
        continue

    cnt = np.array(ast.literal_eval(row["contour"]), dtype=np.int32)
    if cnt.ndim == 2:
        cnt = cnt.reshape(-1, 1, 2)

    ell = ast.literal_eval(row["ellipse"])
    ellipse = (tuple(ell[0]), tuple(ell[1]), float(ell[2]))

    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    peri = cv2.arcLength(cnt, True)
    circularity = (4*np.pi*area)/(peri*peri) if peri > 0 else 0.0

    MA, ma = float(ellipse[1][0]), float(ellipse[1][1])
    aspect_ratio = (MA / ma) if ma > 0 else 999.0

    fit_err = ellipse_fit_error(cnt, ellipse)
    outside_r, area_r, overlap_r = overlap_metrics(cnt, ellipse, tile.shape)

    rows.append([
        outside_r, area_r, overlap_r,
        aspect_ratio, solidity, circularity,
        fit_err, area,
        int(row["label"])
    ])

print("Feature rows:", len(rows), "| unreadable images:", bad_paths)

feat_cols = [
    "outside_ratio","area_ratio","overlap_ratio",
    "aspect_ratio","solidity","circularity",
    "ellipse_fit_error","contour_area","label"
]
df_feat = pd.DataFrame(rows, columns=feat_cols)

# ======================
# 4) Train/val split + model
# ======================
FEATURES = feat_cols[:-1]
X = df_feat[FEATURES].values
y = df_feat["label"].values


model = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


# ======================
# 5) Feature importance
# ======================
print("\n=== Feature importance ===")

for name, imp in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: -x[1]
):
    print(f"{name:20s}: {imp:.3f}")

# ======================
# 6) Save model + feature order
# ======================
bundle = {
    "model": model,
    "features": FEATURES
}
joblib.dump(bundle, "ellipse_confidence_model_manual.pkl")
print("\nSaved -> ellipse_confidence_model_manual.pkl")
