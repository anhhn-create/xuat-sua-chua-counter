from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tempfile
import json
import os

app = FastAPI()


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "xuat-sua-chua-row-counter"
    }


@app.post("/count-xuat-sua-chua-rows")
async def count_xuat_sua_chua_rows(file: UploadFile = File(...)):
    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name

    try:
        result = count_rows_from_image(image_path)
        return result
    finally:
        try:
            os.remove(image_path)
        except Exception:
            pass


def count_rows_from_image(image_path: str):
    img = cv2.imread(image_path)

    if img is None:
        return {
            "has_table": False,
            "row_count": 0,
            "counted_rows": [],
            "error": "Cannot read image"
        }

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    horizontal_kernel_width = max(40, w // 35)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1)
    )

    horizontal = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)

        if ww >= w * 0.35 and hh <= max(8, h * 0.01):
            segments.append({
                "x": x,
                "y": y,
                "w": ww,
                "h": hh,
                "yc": y + hh / 2
            })

    segments.sort(key=lambda s: s["yc"])

    merged_y = []
    merge_tol = max(3, int(h * 0.004))

    for s in segments:
        y = s["yc"]

        if not merged_y:
            merged_y.append(y)
            continue

        if abs(y - merged_y[-1]) <= merge_tol:
            merged_y[-1] = (merged_y[-1] + y) / 2
        else:
            merged_y.append(y)

    candidate_y = [
        y for y in merged_y
        if y > h * 0.08 and y < h * 0.90
    ]

    groups = []
    current = []
    max_gap = max(22, int(h * 0.035))

    for y in candidate_y:
        if not current:
            current = [y]
            continue

        if y - current[-1] <= max_gap:
            current.append(y)
        else:
            if len(current) >= 8:
                groups.append(current)
            current = [y]

    if len(current) >= 8:
        groups.append(current)

    if not groups:
        return {
            "has_table": False,
            "row_count": 0,
            "counted_rows": [],
            "error": "No table-like horizontal line group found",
            "debug": {
                "image_width": w,
                "image_height": h,
                "candidate_y": [round(float(v), 1) for v in candidate_y]
            }
        }

    groups.sort(key=lambda g: len(g), reverse=True)
    table_lines = groups[0]

    # Với mẫu bảng II.2 hiện tại:
    # 4 đường đầu thường là phần header, từ đó xuống mới là thân dữ liệu.
    header_boundary_lines = 4

    if len(table_lines) <= header_boundary_lines + 1:
        return {
            "has_table": False,
            "row_count": 0,
            "counted_rows": [],
            "error": "Not enough table lines after header",
            "debug": {
                "table_lines_y": [round(float(v), 1) for v in table_lines]
            }
        }

    body_boundaries = table_lines[header_boundary_lines:]

    intervals = []
    for i in range(len(body_boundaries) - 1):
        y1 = body_boundaries[i]
        y2 = body_boundaries[i + 1]
        intervals.append(y2 - y1)

    median_gap = float(np.median(intervals)) if intervals else 0

    valid_rows = []

    for i, gap in enumerate(intervals):
        y1 = body_boundaries[i]
        y2 = body_boundaries[i + 1]

        # Loại khoảng quá lớn bất thường, thường là vùng chữ ký hoặc chân bảng.
        if median_gap > 0 and gap > median_gap * 2.2:
            continue

        valid_rows.append({
            "row_index": len(valid_rows) + 1,
            "y_top": round(float(y1), 1),
            "y_bottom": round(float(y2), 1),
            "height": round(float(gap), 1)
        })

    return {
        "has_table": True,
        "row_count": len(valid_rows),
        "counted_rows": valid_rows,
        "debug": {
            "image_width": w,
            "image_height": h,
            "horizontal_segments": len(segments),
            "merged_horizontal_lines": len(merged_y),
            "selected_table_lines": len(table_lines),
            "header_boundary_lines": header_boundary_lines,
            "median_row_gap": round(float(median_gap), 2),
            "table_lines_y": [round(float(v), 1) for v in table_lines]
        }
    }
