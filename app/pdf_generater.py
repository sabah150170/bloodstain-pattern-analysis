from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.lib.utils import simpleSplit
from datetime import datetime
from color import CLUSTER_COLORS
import os



# ========================
# Image helper
# ========================
def draw_image_full_page(c, image_path, caption):
    width, height = A4

    img = ImageReader(image_path)
    iw, ih = img.getSize()

    max_width = width - 4 * cm
    max_height = height - 6 * cm

    scale = min(max_width / iw, max_height / ih)
    w = iw * scale
    h = ih * scale

    x = (width - w) / 2
    y = height - 3 * cm

    c.drawImage(
        img,
        x,
        y - h,
        width=w,
        height=h,
        preserveAspectRatio=True,
        mask="auto"
    )

    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(
        width / 2,
        y - h - 0.8 * cm,
        caption
    )


# ========================
# Image helper
# ========================
def draw_image_scaled(c, image_path, x, y, max_width, max_height):
    img = ImageReader(image_path)
    iw, ih = img.getSize()

    scale = min(max_width / iw, max_height / ih)
    w = iw * scale
    h = ih * scale

    c.drawImage(
        img,
        x,
        y - h,
        width=w,
        height=h,
        preserveAspectRatio=True,
        mask="auto"
    )
    return h


# ========================
# Main Report Generator
# ========================
def generate_bpa_report(
    output_image,
    output_image_2,
    output_dir,
    clusters,
    accepted_count,
    rejected_count,
    image_name,
    unit="cm"
):
    report_path = os.path.join(output_dir, "report.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # =========
    # Header
    # =========
    c.setFont("Helvetica-Bold", 16)
    c.drawString(
        2 * cm,
        height - 2 * cm,
        "Bloodstain Pattern Analysis Report"
    )

    c.setFont("Helvetica", 9)
    c.drawString(
        2 * cm,
        height - 2.7 * cm,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {image_name}"
    )

    # ====================
    # PAGE 1 – IMAGE 1
    # ====================
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm,
                 "Bloodstain Pattern Analysis Report")

    c.setFont("Helvetica", 9)
    c.drawString(
        2 * cm,
        height - 2.7 * cm,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {image_name}"
    )

    draw_image_full_page(
        c,
        output_image,
        "Figure 1. Bloodstain reconstruction and estimated area of origin."
    )

    c.showPage()


    # ====================
    # PAGE 2 – IMAGE 2
    # ====================
    try:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 2 * cm,
                    "Bloodstain Pattern Analysis Report")

        draw_image_full_page(
            c,
            output_image_2,
            "Figure 2. Bloodstain reconstruction and estimated area of origin."
        )

        c.showPage()
    except:
        pass



    # ==========
    # Summary
    # ==========
    y = height - 3 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Summary")
    y -= 0.7 * cm

    c.setFont("Helvetica", 10)
    c.drawString(
        2 * cm,
        y,
        f"Detected impact sources: {len(clusters)}"
    )
    y -= 0.5 * cm

    c.drawString(
        2 * cm,
        y,
        f"Accepted stains: {accepted_count}"
    )
    y -= 0.4 * cm

    c.drawString(
        2 * cm,
        y,
        f"Rejected stains: {rejected_count}"
    )

    y -= 1.2 * cm
    col_x = [2 * cm, width / 2 + 0.5 * cm]
    col = 0
    start_y = y

    # ===========
    # Clusters
    # ===========
    for cluster in clusters:
        if y < 3 * cm:
            if col == 0:
                col = 1
                y = start_y
            else:
                c.showPage()
                col = 0
                y = height - 3 * cm

        x0 = col_x[col]

        bullet_color = cluster['color']
        c.setFillColor(bullet_color)
        c.circle(x0, y + 0.15 * cm, 0.12 * cm, fill=1)
        c.setFillColorRGB(0, 0, 0)

        c.setFont("Helvetica-Bold", 11)
        c.drawString(
            x0 + 0.4 * cm,
            y,
            f"Impact Source #{cluster['id']}"
        )
        y -= 0.5 * cm

        c.setFont("Helvetica", 9)

        
        text = c.beginText()
        text.setTextOrigin(x0 + 0.8 * cm, y)
        text.setLeading(12)

        max_width = width - (x0 + 1.2 * cm)   # sağa taşmayı engeller

        stain_text = "" #"Stain Ids: " + ", ".join(map(str, cluster["line_ids"]))

        lines = simpleSplit(stain_text, "Helvetica", 9, max_width)
        for line in lines:
            text.textLine(line)


        c.drawText(text)

        # Satır sayısına göre y'yi aşağı indir
        y -= 0.4 * cm * (len(lines) + 1)



        # AoC
        x2d, y2d = cluster["aoc_world_cm"]
        c.drawString(
            x0 + 0.4 * cm,
            y,
            f"AoC (2D): x={x2d:.2f}, y={y2d:.2f} {unit}"
        )
        y -= 0.4 * cm

        # AoO Tangent
        if cluster.get("aoo_tangent"):
            z = cluster["aoo_tangent"][2]
            c.drawString(
                x0 + 0.4 * cm,
                y,
                f"AoO (Tangent): z={z:.2f} {unit}"
            )
            y -= 0.4 * cm

        # Least Squares
        if cluster.get("aoo_least_square"):
            x3, y3, z3 = cluster["aoo_least_square"]
            c.drawString(
                x0 + 0.4 * cm,
                y,
                f"AO (Least Squares): "
                f"x={x3:.2f}, y={y3:.2f}, z={z3:.2f} {unit}"
            )
            y -= 0.4 * cm

        y -= 0.8 * cm

    c.save()
    return report_path
