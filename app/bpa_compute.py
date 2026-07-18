# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only

import os
import numpy as np
import cv2
import logging
import gc

from calculate_adaptive_threshold import apply_adaptive_threshold
from calculate_contours_and_elips import drawContoursAndElips
from calculate_line_intersections import drawIntersections
from calculate_3d_origin import estimate_aoo_tangent
from color import CLUSTER_COLORS
from pixel_to_cm import pixel_to_cm_from_image
from pdf_generater import generate_bpa_report


def run_bpa_analysis(img, corners_world, outside_thresh, area_ratio_thresh, overlap_thresh, step_status, ellipse_method, run_dir, image_name, logger):
    """
    img           : OpenCV image (BGR)
    corners_world : The real-world corner coordinates (cm) provided by the user
                    [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """

    # ================
    # 1. Preprocess
    # ================
    cleaned = apply_adaptive_threshold(img, run_dir)

    # ==================
    # 2. Elips + Rays
    # ==================
    canvas, rays_pixel, origin_lines, offset_x, offset_y, _, _= drawContoursAndElips(
        img,
        cleaned,
        run_dir,
        outside_thresh,
        area_ratio_thresh,
        overlap_thresh,
        step_status,
        ellipse_method,
        logger
    )

    # =======================================
    # Before continuing, check step status
    # =======================================
    if step_status == "non_final":
        return {
            "image": canvas
        }


    # =======================
    # 3. 2D AO (pixel space)
    # =======================
    _, aoc_2d_list = drawIntersections(
        origin_lines=origin_lines,
        run_dir=run_dir,
        canvas=canvas,
        img=img,
        corners_world=corners_world,
        offset_x=offset_x,
        offset_y=offset_y,
        logger=logger
    )

    # =========================
    # 4. AoC → CM (Multiple)
    # =========================
    clusters_world = []
    for cluster in aoc_2d_list:

        clusters_world.append({
            "id": cluster["id"],
            "color": CLUSTER_COLORS[(cluster["id"] - 1) % len(CLUSTER_COLORS)],
            "aoc_pixel_canvas": cluster["pixel_canvas"],
            "aoc_pixel_image": cluster["pixel_image"],
            "aoc_world_cm": cluster["cm"], 
            "line_ids": cluster["stain_ids"]
        })


    # ===============
    # 5. Rays → CM
    # ===============
    rays_world = []
    for ray in rays_pixel:
        origin_px = ray["origin"]
        direction_px = ray["direction"]
        stain_id = ray["stain_id"]

        ox_img = origin_px[0] - offset_x
        oy_img = origin_px[1] - offset_y

        ox_cm, oy_cm = pixel_to_cm_from_image(
            ox_img,
            oy_img,
            img.shape,
            corners_world
        )

        origin_world = np.array([ox_cm, oy_cm, 0.0])

        # ================================================================================
        # The direction vector is already dimensionless, so normalization is sufficient
        # ================================================================================
        direction = direction_px / np.linalg.norm(direction_px)

        rays_world.append({
            "line_id": stain_id,
            "origin": origin_world.tolist(),
            "direction": direction.tolist()
        })

    # =========================================
    # AoO – Tangent method (CM, Multiple)
    # =========================================
    estimate_aoo_tangent(clusters_world, rays_world, logger)


    # =============
    # PDF Report
    # =============
    canvas, rays_pixel, origin_lines, offset_x, offset_y, accepted_count, rejected_count = drawContoursAndElips(
        img,
        cleaned,
        run_dir,
        outside_thresh,
        area_ratio_thresh,
        overlap_thresh,
        "non_final",
        ellipse_method,
        logging.getLogger("null")
    )


    analysis_image_path = os.path.join(run_dir, "analysis.png")
    cv2.imwrite(analysis_image_path, canvas)

    analysis_image_path_2 = os.path.join(run_dir, "06_AoC_2D_density_peaks.png")

    report_path = generate_bpa_report(
        output_image=analysis_image_path, 
        output_image_2=analysis_image_path_2, 
        output_dir=run_dir,
        clusters=clusters_world,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        image_name=image_name,
        unit="cm"
    )

    del canvas
    del cleaned
    gc.collect()


    return {
        "clusters": clusters_world,
        "rays_3d": rays_world,
        "report_path": report_path
    }

