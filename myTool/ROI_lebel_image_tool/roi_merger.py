#!/usr/bin/env python3
# roi_merger.py
import cv2
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_json(jpath: Path):
    with open(jpath, "r", encoding="utf-8") as f:
        return json.load(f)

def paste(canvas, roi_img, pos):
    x, y = pos
    h, w = roi_img.shape[:2]
    canvas[y:y+h, x:x+w] = roi_img

def main():
    ap = argparse.ArgumentParser(description="ROI 合併器")
    ap.add_argument("--json", type=Path, required=True, help="ROI JSON 檔")
    ap.add_argument("--canvas", nargs=2, type=int, required=True, metavar=("W", "H"), help="輸出畫布尺寸")
    ap.add_argument("--output", type=Path, required=True, help="輸出資料夾")
    ap.add_argument("--root", type=Path, default=".", help="影像根目錄（相對 JSON 路徑）")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data = load_json(args.json)
    W, H = args.canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x_cursor = y_cursor = row_max_h = 0
    file_idx = 1

    def save_canvas(nonlocal_canvas, idx):
        path = args.output / f"merged_{idx:04d}.png"
        cv2.imwrite(str(path), nonlocal_canvas)
        print(f"[INFO] 已輸出 {path}")

    # 逐一貼 ROI
    for img_rel, rois in tqdm(data.items(), desc="Merging"):
        img_path = args.root / img_rel
        src = cv2.imread(str(img_path))
        if src is None:
            print(f"[WARN] 無法讀取 {img_path}，跳過")
            continue
        for r in rois:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            crop = src[y1:y2, x1:x2]
            h, w = crop.shape[:2]

            # 若放不下→換行
            if x_cursor + w > W:
                x_cursor = 0
                y_cursor += row_max_h
                row_max_h = 0
            # 若超出畫布→存檔開新
            if y_cursor + h > H:
                save_canvas(canvas, file_idx)
                file_idx += 1
                canvas[:] = 0
                x_cursor = y_cursor = row_max_h = 0

            paste(canvas, crop, (x_cursor, y_cursor))
            x_cursor += w
            row_max_h = max(row_max_h, h)

    # 儲存最後一張
    save_canvas(canvas, file_idx)

if __name__ == "__main__":
    main()

r""" powershell
C:/Users/Rontgen-W11-NB/.conda/envs/ttgui-env/python.exe `
d:/TrafficTrackerGUI-MOTC/Model/tool/ROI_lebel_image_tool/roi_merger.py `
--json 'D:\_Dataset_\TTGUI\20250606_moto\rois.json' `
--root 'D:\_Dataset_\TTGUI\20250606_moto\Capture' `
--canvas 5472 3076 `
--output 'D:\_Dataset_\TTGUI\20250606_moto\merged'

"""