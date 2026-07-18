# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only



import joblib
import numpy as np
import cv2
import math

# =========================
# MODEL LOAD (ONCE)
# =========================
_BUNDLE = joblib.load("ellipse_confidence_model_manual.pkl")
_MODEL = _BUNDLE["model"]
_FEATURES = _BUNDLE["features"]  # sıralama çok önemli

# =========================
# FEATURE FUNCTIONS
# =========================
def ellipse_fit_error(contour, ellipse):
    (xc, yc), (MA, ma), angle = ellipse
    if MA <= 0 or ma <= 0:
        return 1.0

    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

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


# =========================
# PUBLIC API
# =========================
def compute_ellipse_confidence(contour, ellipse, tile_gray):
    """
    contour : np.ndarray (N,1,2)
    ellipse : ((xc,yc),(MA,ma),angle)
    tile_gray : grayscale image used for segmentation
    """

    #(cx, cy), (MA, ma), angle = ellipse
    #a = MA / 2.0
    #b = ma / 2.0
    #hull_area = math.pi * a * b


    # --- geometric features ---
    area = cv2.contourArea(contour)
   
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0.0

    peri = cv2.arcLength(contour, True)
    circularity = (4*np.pi*area)/(peri*peri) if peri > 0 else 0.0

    MA, ma = float(ellipse[1][0]), float(ellipse[1][1])
    aspect_ratio = (MA/ma) if ma > 0 else 999.0

    fit_err = ellipse_fit_error(contour, ellipse)
    outside_r, area_r, overlap_r = overlap_metrics(
        contour, ellipse, tile_gray.shape
    )

    feature_map = {
        "outside_ratio": outside_r,
        "area_ratio": area_r,
        "overlap_ratio": overlap_r,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "circularity": circularity,
        "ellipse_fit_error": fit_err,
        "contour_area": area
    }

    # --- feature vector (ORDER MATTERS) ---
    X = np.array([[feature_map[f] for f in _FEATURES]])

    # --- predict confidence ---
    conf = float(_MODEL.predict_proba(X)[0, 1])
    return conf
