# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


#from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
#import random



def estimate_aoo_tangent(clusters_world, rays_world, logger):
    """
    Classical tangent method AoO estimation
    """
    
    for cluster in clusters_world:
        aoc_xy = np.array(cluster["aoc_world_cm"])[:2]
        heights = []

        for ray in rays_world:
            # ================================
            # Only lines belong to this AoC
            # ================================
            if ray["line_id"] not in cluster["line_ids"]:
                continue
                
            o = np.array(ray["origin"])      # [x,y,0]
            d = np.array(ray["direction"])   # unit vector

            o_xy = o[:2]
            dx, dy, dz = d

            # =================================
            # NEW: distance filter (circle ROI)
            # =================================
            v_xy = np.array([dx, dy], dtype=float)
            vv = float(np.dot(v_xy, v_xy))
            if vv < 1e-12:
                logger.warn("[WARN] ROI skipped: line has near-zero XY direction")
                continue
                
            w = aoc_xy - o_xy
            t = float(np.dot(w, v_xy) / vv)     # closest point on infinite line
            closest_xy = o_xy + t * v_xy
            d_perp = float(np.linalg.norm(aoc_xy - closest_xy))



            xy_norm = math.sqrt(dx*dx + dy*dy)
            if xy_norm < 1e-6:
                logger.warn(f"[WARN] AoO skipped: almost 90 degree")
                continue  # almost 90 degree → skip

            d_stain = float(np.linalg.norm(aoc_xy - o_xy))  # stain position to AoC
            z = d_stain * (dz / xy_norm)# Tangent method

            if z > 0:   # physically meaningful
                heights.append(z)
                logger.debug(f"[DEBUG] stain={ray['line_id']} | Heigh={z}")
            else:
                logger.warn(
                    f"[WARN] AoO skipped: tangent - invalid z value: {z}"
                )

        if len(heights) >= 5:
            #z_est = np.mean(heights)
            z_est = np.median(heights)

            cluster["aoo_tangent"] = [
                float(aoc_xy[0]),
                float(aoc_xy[1]),
                float(z_est)
            ]
        else:
            cluster["aoo_tangent"] = None
            logger.warn(
                f"[WARN] AoO skipped: tangent - not enough rays: {len(heights)}"
            )

