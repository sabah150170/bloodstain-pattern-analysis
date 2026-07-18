# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


def pixel_to_cm_from_image(px_img, py_img, img_shape, corners_world):
    """
    px_img, py_img : image pixel koordinatı
    corners_world  : [[x_tl,y_tl],[x_tr,y_tr],[x_br,y_br],[x_bl,y_bl]]
    """

    h, w = img_shape[:2]

    x_min = corners_world[0][0] 
    x_max = corners_world[1][0] 
    y_max = corners_world[0][1] 
    y_min = corners_world[2][1] 

    scale_x = (x_max - x_min) / w
    scale_y = (y_max - y_min) / h

    x_cm = x_min + px_img * scale_x
    y_cm = y_max - py_img * scale_y 

    return x_cm, y_cm