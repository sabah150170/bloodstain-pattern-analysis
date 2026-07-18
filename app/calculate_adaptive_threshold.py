# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only

import cv2
import numpy as np

from config import DEBUG_MODE

def apply_adaptive_threshold(img, run_dir):

    # ==========
    # 1. Gray
    # ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if DEBUG_MODE: cv2.imwrite(f"{run_dir}/01_gray.png", gray)


    # ============================
    # 5. Adaptive Threshold
    # ============================
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,   # blok boyutu
        3     # daha küçük C = daha agresif ayrım
    )
    
    if DEBUG_MODE: cv2.imwrite(f"{run_dir}/03_threshold.png", thresh)

    # ============================
    # 6. Morphology (hafif)
    # ============================
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if DEBUG_MODE: cv2.imwrite(f"{run_dir}/04_cleaned.png", cleaned)

    return cleaned
