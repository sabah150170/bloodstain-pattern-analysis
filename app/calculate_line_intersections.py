# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


import numpy as np
import cv2
import math 

from itertools import combinations
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from color import CLUSTER_COLORS
from pixel_to_cm import pixel_to_cm_from_image
#from scipy.spatial.distance import cdist

from config import DEBUG_MODE


AXIS_MULTIPLIER = 10
ALPHA_THRESH = 20  # degree
THETA_THRESH = 20  
DIR_DOT_THRESH = -0.3

# ====================
# Line intersection
# ====================
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) -
          (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) -
          (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)


def angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def direction_vec(line):
    v = np.array(line["p2"]) - np.array(line["p1"])
    return v / np.linalg.norm(v)


def geometric_median(points, eps=1e-5):
    y = np.mean(points, axis=0)

    while True:
        d = np.linalg.norm(points - y, axis=1)
        nonzeros = d > 0

        if not np.any(nonzeros):
            return y

        w = 1 / d[nonzeros]
        t = (points[nonzeros] * w[:, None]).sum(axis=0) / w.sum()

        if np.linalg.norm(t - y) < eps:
            return t
        y = t



def same_line_slope(logger, theta1, theta2):
    v1 = np.array([math.cos(theta1), math.sin(theta1)])
    v2 = np.array([math.cos(theta2), math.sin(theta2)])

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_val = abs(np.dot(v1, v2))
    angle_diff = math.degrees(math.acos(np.clip(cos_val, -1, 1)))

    logger.info(f"Angle Diff: {angle_diff}")

    return angle_diff <= THETA_THRESH



def stains_compatible_2(li, lj, logger):
    min_dist = max(
        AXIS_MULTIPLIER * li["long_axis"],
        AXIS_MULTIPLIER * lj["long_axis"]
    )
    if dist(li["center"], lj["center"]) > min_dist:
        return None

    if not same_line_slope(logger, li["theta"], lj["theta"]):
        logger.info(
            f"Different orientation, stains: {li['stain_id']} - {lj['stain_id']}"
        )
        return False

    return True



def filter_stains_by_compatibility(cluster_stains, logger):
    """
    Her stain için:
    - diğer stainlerle compatibility oranını hesaplar
    - oran < 1.0 ise stain'i eler
    """

    removed = set()

    n = len(cluster_stains)

    for i in range(n):
        si = cluster_stains[i]

        incompatible_count = 0
        compatible_count = 0

        for j in range(n):
            if i == j:
                continue

            sj = cluster_stains[j]

            result = stains_compatible_2(si, sj, logger)
            if result == True:
                compatible_count += 1
            if result == False:
                incompatible_count += 1

            logger.info(
                f"Compare stains: {si['stain_id']} - {sj['stain_id']}  result: {result}"
            )

        # oran
        ratio = compatible_count / incompatible_count if incompatible_count > 0 else 1.0
        logger.info(
            f"Count compatible: {compatible_count}, incompatible: {incompatible_count}"
        )

        # DEBUG istersen
        logger.info(
            f"Stain {si['stain_id']} compatibility ratio: {ratio:.2f}"
        )

        if ratio < 1.0 and (compatible_count + incompatible_count) >= 3:
            removed.add(si['stain_id'])
            logger.info(
                f"Stain {si['stain_id']} REMOVED (ratio={ratio:.2f})"
            )
            

    return removed



def dist(a1, a2):
    return np.linalg.norm(np.array(a1) - np.array(a2))


def long_axis(ellipse):
    return max(ellipse[1])
    

# ========================================
# 2D AoC detection - Density Peak Based
# ========================================
def drawIntersections(origin_lines, run_dir, canvas, img, corners_world, offset_x, offset_y, logger):
    """
    origin_lines :
        [((x1,y1),(x2,y2), alpha_deg), stain no, ...]
    returns:
        intersections : list[(x,y)]
        aoc_with_height_group_list    : list of dicts (one per AoC with height group)
    """

    # ============================
    # 1. Find all intersections
    # ============================
    intersections = []
    for li, lj in combinations(origin_lines, 2):
        stain_i = li["stain_id"]
        stain_j = lj["stain_id"]

        pt = line_intersection(li["p1"], li["p2"], lj["p1"], lj["p2"])
        if pt is None:
            continue

        min_dist_i = AXIS_MULTIPLIER * li["long_axis"]
        min_dist_j = AXIS_MULTIPLIER * lj["long_axis"]
        dist_to_intersection_i = dist(pt, li["center"])
        dist_to_intersection_j = dist(pt, lj["center"])

        # too close → skip
        if (dist_to_intersection_i < min_dist_i or dist_to_intersection_j < min_dist_j):
            logger.warn(f"[WARN] too close intersection, stains: {stain_i} - {stain_j}")
            continue


        intersections.append({
            "point": pt,
            "stain_ids": (stain_i, stain_j)
        })



    if len(intersections) < 3:
        return [], []

    pts = np.array([i["point"] for i in intersections])


    # =========================
    # Safety Check
    # =========================
    min_intersection_size = 10
    n_pts = len(pts)
    if n_pts < min_intersection_size:
        logger.warn(
            f"[WARN] AoC skipped: not enough intersection points "
            f"({n_pts} < {min_intersection_size})"
        )
        return [], [] 

    # =====================================================
    # 2. Find dense areas (KNN density labeling) --> AoC
    # =====================================================
    
    nbrs = NearestNeighbors(n_neighbors=min_intersection_size).fit(pts)
    distances, _ = nbrs.kneighbors(pts)

    # Her noktanın yerel yoğunluğu (komşu mesafesi küçükse yoğun)
    local_density = np.mean(distances, axis=1)

    # Auto threshold (min %30 = dense)
    density_thresh = np.percentile(local_density, 90) #CHANGE:30'du 90 yaptım.

    labels = np.full(len(pts), -1, dtype=int)
    cluster_id = 0

    for i in range(len(pts)):
        if labels[i] != -1:
            continue
        if local_density[i] > density_thresh:
            continue

        # New dense region 
        labels[i] = cluster_id 
        stack = [i]

        while stack:
            idx = stack.pop()
            for j in range(len(pts)):
                if labels[j] != -1:
                    continue
                    
                # spatial closeness
                dist_ok = np.linalg.norm(pts[idx] - pts[j]) < density_thresh * 2
                
                if dist_ok:
                    logger.info(f"Cluster Id: {cluster_id}, Intersections Stains: {intersections[idx]['stain_ids']}, {intersections[j]['stain_ids']}")
                    labels[j] = cluster_id
                    stack.append(j)

        cluster_id += 1
    
    label_counts = Counter(labels)
    for i in range(len(labels)):
        if labels[i] != -1 and label_counts[labels[i]] == 1:
            labels[i] = -1

    unique_labels = set(labels)
    unique_labels.discard(-1)

    logger.info("")
    logger.info(f"labels: \n{labels}\n")


    # ====================================================================================
    # 3. In every densely populated area, get the max intersection point + Height Split
    # ====================================================================================
    aoc_with_height_group_list = []
    impatc_id = 1

    
    for aoc_idx, cid in enumerate(sorted(unique_labels)):
        
        # ==============================
        # 4. Stains belong to this AoC
        # ==============================
        cluster_indices = np.where(labels == cid)[0]
        stain_ids = set()
        for k in cluster_indices:
            stain_ids.update(intersections[k]["stain_ids"])

        logger.info(f"AoC {aoc_idx+1} (cluster {cid}); size: {len(stain_ids)} → Stain IDs: {sorted(stain_ids)}")


        # ---- stain objeleri ----
        cluster_stains = [
            s for s in origin_lines
            if s["stain_id"] in stain_ids
        ]
        removed_stain_ids = filter_stains_by_compatibility(cluster_stains, logger)
        stain_ids = stain_ids.difference(removed_stain_ids)

        logger.info(f"Removed Stains: {removed_stain_ids}")

        # ---- intersection filtresi ----
        cluster_pts = pts[labels == cid] # intersection count, 2'li combinasyon
        logger.info(f"Intersection size before stain oriantation elimination: {len(cluster_pts)}")

        kept_indices = []
        for idx in cluster_indices:
            inter_stains = set(intersections[idx]["stain_ids"])

            # Eğer intersection, removed stain'lerden herhangi birini içeriyorsa -> at
            if inter_stains & removed_stain_ids:
                logger.info(f"Removed intersection idx={idx}, stains={sorted(inter_stains)}")
                continue

            kept_indices.append(idx)

        # cluster_pts'yi bu kept indexlerle yeniden oluştur
        cluster_pts = pts[kept_indices]

        logger.info(f"Intersection size after stain oriantation elimination: {len(cluster_pts)}")


        # ---- AoC representative (KNN center) ----
        m = len(cluster_pts)
        if m < 3:
            continue  # AoC için yetersiz destek

        # ---- AoC representative (local density center) ----
        k = max(5, max(3, m // 3))
        #k = min(10, m//4) ESKi

        k = min(max(5, max(3, m // 3)), m - 1) # ERROR HANDLING


        nbrs = NearestNeighbors(n_neighbors=k+1).fit(cluster_pts)
        dists, _ = nbrs.kneighbors(cluster_pts)
        knn_density = dists[:, 1:].mean(axis=1) # ------> VER1: KNN DENSITY

        best_idx = np.argmin(knn_density)
        best_pt = cluster_pts[best_idx]

        #best_pt = geometric_median(cluster_pts) # ------> VER2: GEOMETRIC MEDIAN


        # D = cdist(cluster_pts, cluster_pts)
        # best_idx = np.argmin(D.sum(axis=1))
        # best_pt = cluster_pts[best_idx] # ------> VER3: CDIST

        
        if best_pt is None:
            continue


        
        # ============================================
        # 5. PIXEL → CM (same for all sub-clusters)
        # ============================================
        cx_canvas, cy_canvas = best_pt

        px_img = cx_canvas - offset_x
        py_img = cy_canvas - offset_y

        x_cm, y_cm = pixel_to_cm_from_image(
            px_img,
            py_img,
            img.shape,
            corners_world
        )

        # ==========
        # 6. Draw
        # ==========
        cx, cy = int(cx_canvas), int(cy_canvas)

        color = CLUSTER_COLORS[aoc_idx % len(CLUSTER_COLORS)]
        color_bgr = color[::-1]

        cv2.circle(canvas, (cx, cy), 14, color_bgr, -1)

        label = f"AoC {aoc_idx+1}"
        coord = f"x={x_cm:.1f} cm, y={y_cm:.1f} cm"

        logger.info("")
        logger.info(f"{label} → {coord}")

        cv2.putText(
            canvas,
            label,
            (cx + 18, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            coord,
            (cx + 18, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

        aoc_with_height_group_list.append({
            "id": impatc_id,
            "cluster_id": int(cid),
            "pixel_canvas": best_pt.tolist(),
            "pixel_image": [float(px_img), float(py_img)],
            "cm": [float(x_cm), float(y_cm)],
            "stain_ids": sorted(stain_ids)
        })
        impatc_id += 1


    # ==============
    # DRAW LINES
    # ==============
    for line in origin_lines:
        if (line["stain_id"] not in removed_stain_ids):
            cv2.line(canvas, line["p1"], line["p2"], (0, 0, 255), 2)
            


    # ==========
    # 7. Save
    # ==========
    if DEBUG_MODE: cv2.imwrite(f"{run_dir}/06_AoC_2D_density_peaks.png", canvas)
    #cv2.imwrite(f"{run_dir}/06_AoC_2D_density_peaks.png", canvas)



    if DEBUG_MODE: 
        import matplotlib.pyplot as plt


        # ------------------------
        # %130 büyüt
        # ------------------------
        scale = 1.8
        h, w = canvas.shape[:2]

        zoomed = cv2.resize(
            canvas,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )



        # ------------------------
        # Ortadan kırp (orijinal boyuta geri dön)
        # ------------------------
        zh, zw = zoomed.shape[:2]

        start_x = (zw - w) // 2
        start_y = (zh - h) // 2

        cropped = zoomed[start_y:start_y+h, start_x:start_x+w]


        cv2.imwrite(f"{run_dir}/06_AoC_2D_density_peaks_zoom.png", cropped)

        # ------------------------
        # PDF olarak yüksek kalite kaydet
        # ------------------------
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis("off")

        plt.savefig(
            f"{run_dir}/06_AoC_2D_density_peaks.pdf",
            bbox_inches="tight",
            pad_inches=0,
            dpi=600   # 🔥 tez kalitesi
        )

        plt.close()


    return intersections, aoc_with_height_group_list