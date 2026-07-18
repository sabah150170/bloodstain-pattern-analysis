# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only

from ellipse_confidence import compute_ellipse_confidence

import cv2
import math
import numpy as np
import os

from config import DEBUG_MODE

def calculate_impact_angle(short_axis, long_axis):
    ratio = min(1.0, max(0.0, short_axis / long_axis))
    return math.asin(ratio)


def ellipse_fit_quality(
    cnt,
    ellipse,
    img_shape,
    outside_thresh,
    area_ratio_thresh,
    overlap_thresh
):
    """
    Quantitatively evaluates the ellipse–contour fit
    """

    (cx, cy), (MA, ma), angle = ellipse
    a = MA / 2.0
    b = ma / 2.0
    theta = np.deg2rad(angle)

    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # =================================================================
    # 1. The proportion of the contour extending beyond the ellipse
    # =================================================================
    outside_count = 0
    for p in cnt:
        x, y = p[0]
        xp =  (x - cx) * cos_t + (y - cy) * sin_t
        yp = -(x - cx) * sin_t + (y - cy) * cos_t

        val = (xp**2)/(a**2) + (yp**2)/(b**2)
        if val > 1.0:
            outside_count += 1

    outside_ratio = outside_count / len(cnt)

    # ===================================================================================
    # 2. Quantifies how much the fitted ellipse overestimates the actual contour area
    # ===================================================================================
    contour_area = cv2.contourArea(cnt)
    ellipse_area = math.pi * a * b
    area_ratio = ellipse_area / contour_area if contour_area > 0 else np.inf

    # =====================================================================================
    #  3. Measures how much of the ellipse interior is actually occupied by the contour
    # =====================================================================================
    mask_cnt = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)

    mask_ell = np.zeros(img_shape, dtype=np.uint8)
    cv2.ellipse(mask_ell, ellipse, 255, -1)

    intersection = cv2.bitwise_and(mask_cnt, mask_ell)
    overlap_ratio = (
        cv2.countNonZero(intersection) /
        max(cv2.countNonZero(mask_ell), 1)
    )

    # =======================
    # 4. Validation stains
    # =======================
    is_valid = (
        outside_ratio < outside_thresh and
        area_ratio < area_ratio_thresh and
        overlap_ratio > overlap_thresh
    )

    return is_valid, outside_ratio, area_ratio, overlap_ratio


def drawContoursAndElips(img, cleaned, run_dir, outside_thresh, area_ratio_thresh, overlap_thresh, step_status, ellipse_method, logger):
    if step_status == "final":
        h, w = img.shape[:2]
        diag_len = int(math.hypot(w, h)) * 2

        # =======================================================================
        # 1. Create large canvas - allow lines to extend beyond the boundaries
        # =======================================================================
        canvas_margin = int(max(w, h) * 1.5)
        canvas_h = h + 2 * canvas_margin
        canvas_w = w + 2 * canvas_margin

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        offset_x = canvas_margin
        offset_y = canvas_margin
        canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img
    else:
        offset_x = 0
        offset_y = 0
        canvas = img.copy()

    # ==============
    # 2. Contours
    # ==============
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    origin_lines = []
    origin_rays_3d = []
    stain_id = 1
    accepted_count = 0
    rejected_count = 0

    for cnt in contours:
        if len(cnt) < 5 or cv2.contourArea(cnt) < 150:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        short_axis = min(MA, ma)
        long_axis = max(MA, ma)


        logger.info(f"Short axis: {short_axis}")
        logger.info(f"Long axis: {long_axis}")

        # =========================
        # 3. ML based validation
        # =========================
        if ellipse_method == "ml":
            conf = compute_ellipse_confidence(
                contour=cnt,
                ellipse=ellipse,
                tile_gray=cleaned   # veya threshold öncesi tile
            )

            logger.info(f"Ellipse confidence: {round(conf, 3)}")

            if conf >= 0.7:
                is_valid = True
            else:
                is_valid = False

        # ================================
        # 3. HEURISTIC based validation
        # ================================
        else:
            is_valid, outside_r, area_r, overlap_r = ellipse_fit_quality(
                cnt,
                ellipse,
                cleaned.shape,
                outside_thresh,
                area_ratio_thresh,
                overlap_thresh
            )

            logger.info(
                f"outside={outside_r:.2f} "
                f"area_ratio={area_r:.2f} "
                f"overlap={overlap_r:.2f}"
            )


            ratio = long_axis / short_axis if short_axis > 0 else 0
            if ratio > 6:
                logger.info("Invalid ratio")
                is_valid = False



        alpha = calculate_impact_angle(short_axis, long_axis)
        if is_valid and (math.degrees(alpha) > 60): #BUSEEE: burası 50 idi.
            logger.info(f"no: {stain_id} | more than 50 degree, this degree: {math.degrees(alpha)}")
            is_valid = False


        # =================
        # 4. Coordinates
        # =================
        cx = int(x + offset_x)
        cy = int(y + offset_y)

        if not is_valid: 
            color = (0, 0, 255)
            rejected_count += 1
            logger.info(f"no: {stain_id} | ❌ REJECT\n--------------------------------------------------------\n")

        else: 
            color = (0, 255, 0)
            accepted_count += 1
            logger.info(f"no: {stain_id} | ✅ ACCEPTED")


        cv2.ellipse(
            canvas,
            ((cx, cy), (MA, ma), angle),
            color,
            2
        )

        cv2.putText(
            canvas,
            str(stain_id),
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255), # white numeration
            2
        )

        # ================
        # 5. Draw lines
        # ================
        if step_status == "final" and is_valid:
            theta = math.radians(angle + 90)

            x1 = int(cx + diag_len * math.cos(theta))
            y1 = int(cy + diag_len * math.sin(theta))
            x2 = int(cx - diag_len * math.cos(theta))
            y2 = int(cy - diag_len * math.sin(theta))

            #cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

            origin_lines.append({
                "p1": (x1, y1), 
                "p2": (x2, y2), 
                "center": (cx, cy),
                "alpha_deg": math.degrees(alpha),
                "theta": theta,
                "stain_id": stain_id,
                "long_axis": long_axis
            })

            logger.info(f"Degree is: {math.degrees(alpha)}\n--------------------------------------------------------\n")

            direction = np.array([
                math.cos(theta),
                math.sin(theta),
                math.tan(alpha)
            ])

            origin = np.array([cx, cy, 0])
            origin_rays_3d.append({
                "origin": origin,
                "direction": direction,
                "stain_id": stain_id
            })

        stain_id += 1

    
    if step_status == "final":
        if DEBUG_MODE: 
            os.makedirs(run_dir, exist_ok=True)
            cv2.imwrite(f"{run_dir}/05_canvas_ellipses_lines.png", canvas)

        if len(origin_rays_3d) < 3:
            logger.warn("WARNING: Not enough valid ellipses for 3D AO")

    return canvas, origin_rays_3d, origin_lines, offset_x, offset_y, accepted_count, rejected_count
    
