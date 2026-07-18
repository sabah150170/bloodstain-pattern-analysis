# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only

from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from multipart_response import multipart_response
from logger import setup_logger
from bpa_compute import run_bpa_analysis


import os
from datetime import datetime
import numpy as np
import cv2
import base64
import json
import traceback
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# ======
# APP
# ======
app = FastAPI(debug=True)

# =======
# CORS
# =======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS middleware'den sonra ekle
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


# ====================
# Exception Handler
# ====================
@app.exception_handler(Exception)
async def catch_all_exceptions(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": str(exc),
            "type": exc.__class__.__name__
        }
    )



# ============
# Endpoints
# ============
@app.post("/analyze")
async def analyze(
    image: UploadFile,
    corners: str = Form(...),
    metric_method: str = Form(...),
    ellipse_method: str = Form(...),
    outside_thresh: Optional[float] = Form(None),
    area_ratio_thresh: Optional[float] = Form(None),
    overlap_thresh: Optional[float] = Form(None),
    selection_is_ok: Optional[bool] = Form(None)
):

    # ================
    # 1. Read image
    # ================
    data = await image.read()

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError("Image could not be decoded")

    # ============
    # 2. Inputs
    # ============
    try:
        corners_world = json.loads(corners)
    except Exception as e:
        raise ValueError(f"Invalid corners format: {e}")

    if metric_method == "auto" or selection_is_ok == None:
        outside_thresh=0.55
        area_ratio_thresh=1.8
        overlap_thresh=0.9

    if metric_method == "auto" or selection_is_ok:
        step_status = "final"
    else:
        step_status = "non_final"


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("output", f"run_KNN_{ellipse_method}_{image.filename}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("🚀 BPA analysis started")
    logger.info(f"🖼️ Image file received: {image.filename}")
    logger.info(f"📐 Received world corners (cm): {corners_world}")
    logger.info(f"⚙️ Method: {metric_method}")
    logger.info(f"🧠 Ellipse Elimination: {ellipse_method}")
    logger.info(f"🧭 Step Status: {step_status}")
    logger.info(f"🎚️ Thresholds → outside={outside_thresh}, area_ratio={area_ratio_thresh}, overlap={overlap_thresh}\n")


    # ====================
    # 3. Start analysis
    # ====================
    result = run_bpa_analysis(
        img=img,
        corners_world=corners_world,
        outside_thresh=outside_thresh,
        area_ratio_thresh=area_ratio_thresh,
        overlap_thresh=overlap_thresh,
        step_status=step_status,
        ellipse_method=ellipse_method,
        run_dir=run_dir,
        image_name=image.filename,
        logger=logger
    )

   
   

    # ==================
    # 4. Send outputs
    # ==================
    if step_status == "final":
        json_payload = {
            "clusters": result.get("clusters", []),
            "rays_3d": result.get("rays_3d", []),
            "unit": "cm"
        }
        return multipart_response(json_payload, result.get("report_path"))


    img_np = result.get("image")
    _, buffer = cv2.imencode(".png", img_np)
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "status": "selection",
        "image": image_b64,
        "outside_thresh": outside_thresh,
        "area_ratio_thresh": area_ratio_thresh,
        "overlap_thresh": overlap_thresh
        }
