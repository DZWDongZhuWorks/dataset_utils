#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Generator — Interactive rectangle annotation & crop exporter.

Overview
--------
把任意資料夾內的影像縮放置中到 1920×1080 畫布上，支援拖拉建立長方形 ROI、單/多選移動（群組不出界）、單選縮放、寬高就地編輯、複製/貼上（含錨點貼上與原座標貼上）、即時存 JSON 與批次輸出裁切圖片。
An OpenCV-based GUI tool for creating, editing, and exporting rectangular
ROIs over a folder of images. The viewer scales each image to fit a fixed
1920×1080 canvas (letterboxed), while all ROI coordinates are stored in the
**original image coordinate system**. Supports multi-selection group move
(with boundary clamping), single-selection resize via handles, inline width/
height editing, clipboard copy/paste, JSON autosave, and crop exporting.

Key Features
------------
- Draw / select / edit rectangular ROIs on images in a folder
- Multi-select group move (kept in-bounds); resize available for single select
- Inline label editing: click on the displayed "W" or "H" value to type an exact size
- Copy & paste ROIs
  - **V** : paste relative to the first selected ROI (anchor-based)
  - **v** : paste at original coordinates (no anchor)
- JSON autosave (ON/OFF) and crop auto-export (ON/OFF)
- Export crops for current image or all images
- Robust file I/O for paths with non-ASCII characters

Data Model (JSON)
-----------------
JSON is a dict mapping each **relative image path** (POSIX style) to a list of
ROIs:
{
  "<rel/path/to/image.ext>": [
    {"x1": int, "y1": int, "x2": int, "y2": int}, ...
  ],
  ...
}

All coordinates are clamped to image bounds and enforced with a minimal size
(MIN_W×MIN_H). File keys are derived from the image path relative to the input
folder.

Controls
--------
Mouse:
  - Left-drag on empty area : draw a new ROI
  - Left-drag inside ROI    : move (single or multi-selection)
  - Left-drag on handles    : resize (single selection only)
  - Ctrl + Left-click       : add to selection (multi-select)
  - Right-click             : delete selected ROI(s)

Keyboard:
  - Ctrl + A                : select all ROIs on current image
  - a / ← / ↑               : previous image
  - d / → / ↓               : next image
  - s                       : save JSON immediately
  - J                       : toggle JSON autosave ON/OFF
  - C                       : toggle auto-export crops ON/OFF
  - x                       : export crops for current image
  - X                       : export crops for all images
  - c                       : copy selected ROIs to clipboard
  - V (uppercase)           : paste relative to the first selected ROI (anchor)
  - v (lowercase)           : paste at original coordinates (no anchor)
  - h                       : print help to terminal
  - q / Esc                 : save JSON and exit

CLI
---
python roi_generator.py <folder> [-o rois.json] [-c crops]

Dependencies
------------
- OpenCV (cv2), NumPy, tqdm (optional)

Notes & Limitations
-------------------
- Display is fixed to 1920×1080; images are scaled uniformly and centered.
- ROI edits operate in original image coordinates; visual feedback is rendered
  on the scaled canvas.
- Group move clamps displacement to keep all selected ROIs within bounds.
- Resize handles are available only when exactly one ROI is selected.
- Requires a desktop environment with OpenCV highgui support.

(C) 2025 — See source for details.
"""


import cv2
import json
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
    
WIN = "ROI Generator"
DISP_W, DISP_H = 1920, 1080          # 固定顯示大小
COLOR_BOX = (0, 255, 0)
COLOR_SEL = (0, 0, 255)
COLOR_TXT = (255, 255, 255)
TH = 2                               # 線寬

# 控制點（handle）設定（畫在 canvas 座標）
HANDLE_SIZE = 8      # 小方塊邊長（px）
HANDLE_HALF = HANDLE_SIZE // 2
MIN_W, MIN_H = 2, 2  # ROI 最小尺寸（原圖座標）

HELP = f"""
================= ROI 資料產生器 =================
滑鼠
  左鍵拖拉           ：繪製新 ROI（長方形）
  左鍵拖拉(選取ROI內) ：移動（單選/多選）
  左鍵拖拉(選取ROI控制點) ：縮放（僅單選）
  Ctrl + 左鍵        ：逐步增加被選取的 ROI（多選）
  右鍵點擊           ：刪除『目前被選取』的 ROI（可多個）
  中鍵拖曳           ：平移視圖（Pan）
  滾輪               ：縮放視圖（以滑鼠所在位置為錨點）
  
鍵盤
  Ctrl + A           ：選取所有 ROI
  ↑ / ← / a          ：上一張；  ↓ / → / d ：下一張
  s                  ：立即儲存 JSON
  J                  ：切換「自動儲存 JSON」 ON/OFF
  C                  ：切換「自動輸出 Crop」 ON/OFF
  x                  ：輸出「當前影像」的裁切
  X                  ：輸出「所有影像」的裁切
  c                  ：複製『被選取』的 ROI
  v                  ：貼上剪貼簿 ROI（需當前有選取；以第一個選取 ROI 的左上角作為貼上錨點）
  h                  ：於終端機印出本說明
  q / Esc            ：儲存後離開
================================================
"""

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ROIGenerator:
    """Interactive ROI editor and crop exporter.

    Args:
        folder (Path): Source folder containing images (recursive).
        json_path (Path): Output path for ROI JSON file.
        crop_dir (Path): Output directory for exported crops.

    Behavior:
        - Loads images (jpg/jpeg/png/bmp/tif) recursively and presents them in a
          fixed 1920×1080 window, scaled and centered.
        - Maintains ROI coordinates in original image space; converts to/from
          canvas coordinates for interaction and rendering.
        - Supports multi-select group move (kept in-bounds) and single-select
          resize via 8 handles (tl/tr/br/bl/l/r/t/b).
        - Inline width/height editing by clicking the on-canvas "W×H" label.
        - Clipboard copy/paste of ROIs:
            * 'V' (uppercase): anchor-based paste near the first selected ROI
            * 'v' (lowercase): paste at original coordinates
        - Autosave JSON and optional auto-export of crops per image.

    JSON Schema:
        {
          "<rel/path/to/image.ext>": [
            {"x1": int, "y1": int, "x2": int, "y2": int}, ...
          ],
          ...
        }

    Shortcuts:
        See module docstring `Controls` section.
    """
    def __init__(self, folder: Path, json_path: Path, crop_dir: Path):
        self.folder = folder
        self.json_path = json_path
        self.crop_dir = crop_dir
        if self.crop_dir:
            self.crop_dir.mkdir(parents=True, exist_ok=True)

        self.label_hitboxes = {}   # {roi_index: {'w': (x1,y1,x2,y2), 'h': (x1,y1,x2,y2)}}
        self.edit_mode = None      # None / 'w' / 'h'
        self.edit_roi_idx = None
        self.edit_value_str = ""   # 編輯時的輸入緩衝字串

        self.images = sorted([p for p in folder.rglob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
        if not self.images:
            raise RuntimeError("資料夾內無影像")

        self.idx = 0
        self.data = self._load_json()

        # 互動狀態
        self.scale = 1.0
        self.canvas = None             # 1920×1080 畫布
        self.disp_img = None           # 縮放後貼到 canvas 的視圖
        self.offset = (0, 0)           # 原點偏移量（置中用）
        self.view_rect = (0, 0, 0, 0)  # 影像在 canvas 上的矩形 (x1,y1,x2,y2)
        self.mode = "idle"             # idle / drawing / moving / resizing
        self.start_pt = (0, 0)
        self.orig = None               # 當前原圖（BGR）
        self.orig_hw = (0, 0)          # (h, w)
        
        # === [Zoom/Pan] 狀態 ===
        self.base_scale = 1.0        # 影像縮放到 1920x1080 的「基底縮放」
        self.zoom = 1.0              # 額外縮放倍率（滾輪控制）
        self.pan = [0, 0]            # 額外平移（canvas 像素，MBUTTON 拖曳控制）
        self.panning = False         # 中鍵是否正在拖曳
        self.pan_start = (0, 0)      # 中鍵拖曳起點（canvas）
        self.pan_origin = (0, 0)     # 中鍵拖曳起點時的 pan

        # 縮放參數
        self.ZOOM_MIN = 0.1
        self.ZOOM_MAX = 10.0
        self.ZOOM_STEP = 1.2
        
        # 選取（支援多選）
        self.sel = set()               # 被選取的 ROI 索引集合
        self.active_idx = None         # 互動中的主 ROI（例如縮放/移動的基準）

        # 拖曳暫存
        self.drag_start_canvas = (0, 0)
        self.drag_start_orig = (0, 0)
        self.drag_start_rois = None    # 拖曳開始時，所選各 ROI 的深拷貝列表 [(idx, {x1,y1,x2,y2}), ...]
        self.resize_handle = None      # tl,tr,br,bl,l,r,t,b（僅單選）

        # 功能旗標
        self.auto_save_json = True
        self.auto_export_crops = False

        # 複製/貼上緩衝
        self.copy_buffer = []          # list[dict(x1,y1,x2,y2)]

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self._mouse_cb)


    # ---------- JSON ----------
    def _load_json(self):
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] JSON 已儲存到 {self.json_path}")

    # ---------- 影像與縮放 ----------
    def _prepare_display(self):
        """依目前 self.idx 讀圖，計算基底縮放，初始化 zoom/pan，並重組視圖"""
        img_path = self.images[self.idx]
        self.orig = self._imread(str(img_path))
        if self.orig is None:
            raise RuntimeError(f"讀取失敗: {img_path}")
        h, w = self.orig.shape[:2]
        self.orig_hw = (h, w)

        # 計算基底縮放（完整置入 1920x1080 的比例）
        self.base_scale = min(DISP_W / w, DISP_H / h)

        # 初始化視圖狀態
        self.zoom = 1.0
        self.pan = [0, 0]
        self.scale = self.base_scale  # 目前實際縮放 = base * zoom
        self.offset = (0, 0)
        self.view_rect = (0, 0, 0, 0)

        # 重置互動狀態
        self.mode = "idle"
        self.start_pt = (0, 0)
        self.sel.clear()
        self.active_idx = None
        self.resize_handle = None

        # 依照目前 zoom/pan/scale 組出顯示影像
        self._recompose_view()

    def _recompose_view(self):
        """### [Zoom/Pan] 依 base_scale + zoom + pan 重建 self.disp_img / offset / view_rect"""
        h, w = self.orig_hw
        self.scale = self.base_scale * self.zoom

        # 目前縮放後影像尺寸（可能大於或小於 canvas）
        cur_w = max(1, int(w * self.scale))
        cur_h = max(1, int(h * self.scale))

        # 以置中為基準的 offset（未加 pan）
        base_ox = (DISP_W - cur_w) // 2
        base_oy = (DISP_H - cur_h) // 2

        # 依目前 pan 計算 top-left，並做邊界夾固：
        # 若影像寬 < 畫布寬：允許在 [0, DISP_W-cur_w] 之間；反之允許在 [DISP_W-cur_w, 0]
        def clamp_tl(cur_size, disp_size, base_o, pan_val):
            tl_min = min(0, disp_size - cur_size)
            tl_max = max(0, disp_size - cur_size)
            tl = base_o + int(pan_val)
            tl = clamp(tl, tl_min, tl_max)
            # 回推新的 pan，避免長期累積誤差
            return tl, tl - base_o

        tlx, new_pan_x = clamp_tl(cur_w, DISP_W, base_ox, self.pan[0])
        tly, new_pan_y = clamp_tl(cur_h, DISP_H, base_oy, self.pan[1])
        self.pan = [new_pan_x, new_pan_y]
        self.offset = (tlx, tly)
        self.view_rect = (tlx, tly, tlx + cur_w, tly + cur_h)

        # 建立畫布並把「縮放後影像」可見區塊貼上
        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
        if cur_w > 0 and cur_h > 0:
            zimg = cv2.resize(self.orig, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
            x1 = max(0, tlx); y1 = max(0, tly)
            x2 = min(DISP_W, tlx + cur_w); y2 = min(DISP_H, tly + cur_h)
            if x2 > x1 and y2 > y1:
                sx1, sy1 = x1 - tlx, y1 - tly
                sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
                canvas[y1:y2, x1:x2] = zimg[sy1:sy2, sx1:sx2]

        self.canvas = canvas
        self.disp_img = canvas.copy()


    # ---------- 安全讀寫影像 ----------
    def _imread(self, path):
        """支援含中文/全形符號路徑的安全讀圖"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[WARN] _imread 失敗：{path} ({e})")
            return None

    def _imwrite(self, path, img):
        """支援含中文/全形符號路徑的安全寫圖"""
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        try:
            buf.tofile(str(path))
            return True
        except Exception as e:
            print(f"[WARN] _imwrite 失敗：{path} ({e})")
            return False

    # ---------- ROI 相關 ----------
    def _img_key(self, index=None):
        if index is None: index = self.idx
        return str(self.images[index].relative_to(self.folder)).replace("\\", "/")

    def _rois(self, name):
        return self.data.setdefault(name, [])

    def _mouse2orig(self, x, y):
        """將 canvas 座標轉為原圖座標"""
        ox, oy = self.offset
        x_orig = int((x - ox) / self.scale)
        y_orig = int((y - oy) / self.scale)
        return x_orig, y_orig

    def _orig2canvas(self, x, y):
        """將原圖座標轉為 canvas 座標"""
        ox, oy = self.offset
        return int(x * self.scale) + ox, int(y * self.scale) + oy

    def _hit_test(self, x, y, rois):
        """canvas 座標判斷點擊哪個 ROI，回傳 index 或 None"""
        for i, r in enumerate(rois):
            x1c, y1c = self._orig2canvas(r["x1"], r["y1"])
            x2c, y2c = self._orig2canvas(r["x2"], r["y2"])
            if x1c <= x <= x2c and y1c <= y <= y2c:
                return i
        return None

    def _normalize_roi(self, r):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        # clamp to image boundary
        h, w = self.orig_hw
        x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
        y1 = clamp(y1, 0, h-1); y2 = clamp(y2, 0, h-1)
        # 最小尺寸
        if x2 - x1 < MIN_W: x2 = min(w-1, x1 + MIN_W)
        if y2 - y1 < MIN_H: y2 = min(h-1, y1 + MIN_H)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _in_view(self, x, y):
        """滑鼠是否在可視影像區（canvas 座標）"""
        x1, y1, x2, y2 = self.view_rect
        return (x1 <= x < x2) and (y1 <= y < y2)

    # ---------- 控制點判定與繪製 ----------
    def _handle_boxes_for_roi(self, r):
        """回傳控制點小方塊的 canvas 區域字典"""
        p1 = self._orig2canvas(r["x1"], r["y1"])
        p2 = self._orig2canvas(r["x2"], r["y2"])
        x1c, y1c = p1
        x2c, y2c = p2
        xm = (x1c + x2c) // 2
        ym = (y1c + y2c) // 2

        def box(cx, cy):
            return (cx - HANDLE_HALF, cy - HANDLE_HALF, cx + HANDLE_HALF, cy + HANDLE_HALF)

        return {
            "tl": box(x1c, y1c),
            "tr": box(x2c, y1c),
            "br": box(x2c, y2c),
            "bl": box(x1c, y2c),
            "t":  box(xm,  y1c),
            "r":  box(x2c, ym),
            "b":  box(xm,  y2c),
            "l":  box(x1c, ym),
        }

    def _hit_handle(self, x, y, idx):
        """若命中 idx 的控制點，回傳 handle key；否則 None"""
        key = self._img_key()
        rois = self._rois(key)
        if idx is None or not (0 <= idx < len(rois)):
            return None
        boxes = self._handle_boxes_for_roi(rois[idx])
        for k, (x1, y1, x2, y2) in boxes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return k
        return None

    # ---------- 匯出裁切 ----------
    def _export_crops_for_index(self, index, show_progress: bool = False):
        key = self._img_key(index)
        rois = self._rois(key)
        if not rois:
            return 0

        img_path = self.images[index]
        img = self._imread(str(img_path))
        if img is None:
            print(f"[WARN] 讀圖失敗，略過：{img_path}")
            return 0
        
        out_subdir = (self.crop_dir / Path(key).with_suffix('')).resolve()
        out_subdir.mkdir(parents=True, exist_ok=True)

        iterable = enumerate(rois, 1)
        if show_progress and tqdm is not None:
            iterable = tqdm(iterable,
                            total=len(rois),
                            desc=f"Export crops: {key}",
                            unit="crop",
                            dynamic_ncols=True,
                            leave=False)

        count = 0
        for i, r in iterable:
            rr = self._normalize_roi(r)
            x1, y1, x2, y2 = rr["x1"], rr["y1"], rr["x2"], rr["y2"]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            wh = f"{x2-x1}x{y2-y1}"
            out_path = out_subdir / f"{Path(key).stem}_{i:04d}_{x1}_{y1}_{wh}.png"
            self._imwrite(str(out_path), crop)
            count += 1

        return count

    def _maybe_auto_actions(self):
        if self.auto_save_json:
            self._save_json()
        if self.auto_export_crops:
            n = self._export_crops_for_index(self.idx)
            print(f"[INFO] 自動輸出裁切：{n} 張（{self._img_key()}）")

    # ---------- 群組移動輔助 ----------
    def _group_move_apply(self, rois, selected_pairs, dx_o, dy_o):
        """在原圖座標系統中，將整組 ROI 盡可能以 (dx_o, dy_o) 共同位移，並保持不出界。"""
        h_img, w_img = self.orig_hw

        # 為了不出界，計算每個 ROI 的允許 dx, dy 區間，整組取交集
        dx_min, dx_max = -10**9, 10**9
        dy_min, dy_max = -10**9, 10**9
        for _, r0 in selected_pairs:
            dx_min = max(dx_min, -r0["x1"])
            dx_max = min(dx_max, (w_img - 1) - r0["x2"])
            dy_min = max(dy_min, -r0["y1"])
            dy_max = min(dy_max, (h_img - 1) - r0["y2"])

        dx = clamp(dx_o, dx_min, dx_max)
        dy = clamp(dy_o, dy_min, dy_max)

        # 實際套用
        for idx, r0 in selected_pairs:
            rois[idx] = {
                "x1": r0["x1"] + dx, "y1": r0["y1"] + dy,
                "x2": r0["x2"] + dx, "y2": r0["y2"] + dy,
            }

    # ---------- Mouse Callback ----------
    def _mouse_cb(self, event, x, y, flags, param):
        name = self._img_key()
        rois = self._rois(name)
        ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        # === [Zoom/Pan] 事件：滑鼠滾輪縮放（以滑鼠所在位置為錨點） ===
        if event == cv2.EVENT_MOUSEWHEEL:
            # 取出滑鼠所在點對應「原圖」的浮點座標（避免整數誤差）
            if self.scale > 0:
                oxf = (x - self.offset[0]) / self.scale
                oyf = (y - self.offset[1]) / self.scale
            else:
                oxf, oyf = 0.0, 0.0

            step = self.ZOOM_STEP if flags > 0 else (1.0 / self.ZOOM_STEP)
            new_zoom = clamp(self.zoom * step, self.ZOOM_MIN, self.ZOOM_MAX)
            if abs(new_zoom - self.zoom) > 1e-6:
                self.zoom = new_zoom

                # 依新縮放，計算新的置中 offset，並回推 pan 使游標錨點不漂移
                new_scale = self.base_scale * self.zoom
                new_w = max(1, int(self.orig_hw[1] * new_scale))
                new_h = max(1, int(self.orig_hw[0] * new_scale))
                base_ox = (DISP_W - new_w) // 2
                base_oy = (DISP_H - new_h) // 2

                # 期望的新 top-left：讓 (oxf,oyf) 仍落在 canvas 的 (x,y)
                off_x_prime = x - oxf * new_scale
                off_y_prime = y - oyf * new_scale

                # 轉成 pan 值（top-left = base + pan）
                self.pan = [int(round(off_x_prime - base_ox)),
                            int(round(off_y_prime - base_oy))]

                self._recompose_view()
                self._render()
            return

        # === [Zoom/Pan] 事件：中鍵拖曳平移 ===
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
            self.pan_origin = (self.pan[0], self.pan[1])
            return

        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            self.pan = [self.pan_origin[0] + dx, self.pan_origin[1] + dy]
            self._recompose_view()
            self._render()
            return

        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            return



        # 左鍵按下：可能是 label 編輯 / 控制點縮放 / ROI 內移動 / 新建 / 多選加入
        if event == cv2.EVENT_LBUTTONDOWN:
            # label 編輯優先（僅單選合理；多選時忽略）
            if len(self.sel) <= 1:
                for i, rects in self.label_hitboxes.items():
                    x1, y1, x2, y2 = rects['w']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.edit_mode = 'w'; self.edit_roi_idx = i
                        r = rois[i]; self.edit_value_str = str(max(1, r["x2"] - r["x1"]))
                        self._render(); return
                    x1, y1, x2, y2 = rects['h']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.edit_mode = 'h'; self.edit_roi_idx = i
                        r = rois[i]; self.edit_value_str = str(max(1, r["y2"] - r["y1"]))
                        self._render(); return

            hit = self._hit_test(x, y, rois)

            # Ctrl + 左鍵：逐步增加選取（只加不減）
            if ctrl_down and hit is not None:
                self.sel.add(hit)
                self.active_idx = hit
                self._render()
                return

            # 一般左鍵：若命中 ROI → 單/多選的互動起點
            if hit is not None:
                # 若未按 Ctrl，改為單一選取
                if not ctrl_down:
                    self.sel = {hit}
                self.active_idx = hit

                # 單選 → 先檢查是否命中控制點以縮放
                if len(self.sel) == 1:
                    hkey = self._hit_handle(x, y, self.active_idx)
                    if hkey is not None:
                        self.mode = "resizing"
                        self.resize_handle = hkey
                        self.drag_start_canvas = (x, y)
                        self.drag_start_rois = [(self.active_idx, deepcopy(rois[self.active_idx]))]
                        return

                # 其餘情況 → 進入群組移動
                self.mode = "moving"
                self.drag_start_canvas = (x, y)
                self.drag_start_orig = self._mouse2orig(x, y)
                # 記錄所有被選取 ROI 的原始框
                self.drag_start_rois = [(idx, deepcopy(rois[idx])) for idx in sorted(self.sel)]
                self._render()
                return

            # 未命中既有 ROI：在影像範圍內則新建
            if self._in_view(x, y):
                self.mode = "drawing"
                self.start_pt = (x, y)
                self._render(temp_rect=(self.start_pt, (x, y)))

        # 拖曳中
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "drawing":
                if self._in_view(x, y):
                    x1, y1, x2, y2 = self.view_rect
                    cx = clamp(x, x1, x2-1); cy = clamp(y, y1, y2-1)
                    self._render(temp_rect=(self.start_pt, (cx, cy)))

            elif self.mode == "moving" and self.drag_start_rois:
                sx_o, sy_o = self.drag_start_orig
                cx_o, cy_o = self._mouse2orig(x, y)
                dx_o, dy_o = cx_o - sx_o, cy_o - sy_o
                self._group_move_apply(rois, self.drag_start_rois, dx_o, dy_o)
                self._render()

            elif self.mode == "resizing" and self.drag_start_rois:
                idx, r0 = self.drag_start_rois[0]
                cx_o, cy_o = self._mouse2orig(x, y)
                x1, y1, x2, y2 = r0["x1"], r0["y1"], r0["x2"], r0["y2"]

                def apply(nx1, ny1, nx2, ny2):
                    rr = self._normalize_roi({"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2})
                    rois[idx] = rr

                hdl = self.resize_handle
                if hdl == "tl":
                    apply(cx_o, cy_o, x2, y2)
                elif hdl == "tr":
                    apply(x1, cy_o, cx_o, y2)
                elif hdl == "br":
                    apply(x1, y1, cx_o, cy_o)
                elif hdl == "bl":
                    apply(cx_o, y1, x2, cy_o)
                elif hdl == "t":
                    apply(x1, cy_o, x2, y2)
                elif hdl == "r":
                    apply(x1, y1, cx_o, y2)
                elif hdl == "b":
                    apply(x1, y1, x2, cy_o)
                elif hdl == "l":
                    apply(cx_o, y1, x2, y2)
                self._render()

        # 左鍵放開：完成動作
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "drawing":
                self.mode = "idle"
                (sx, sy) = self.start_pt
                ex, ey = x, y
                vx1, vy1, vx2, vy2 = self.view_rect
                ex = clamp(ex, vx1, vx2-1); ey = clamp(ey, vy1, vy2-1)
                x1o, y1o = self._mouse2orig(sx, sy)
                x2o, y2o = self._mouse2orig(ex, ey)
                rr = self._normalize_roi({"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o})
                if rr["x2"] > rr["x1"] and rr["y2"] > rr["y1"]:
                    rois.append(rr)
                    new_idx = len(rois) - 1
                    self.sel = {new_idx}
                    self.active_idx = new_idx
                    self._maybe_auto_actions()
                self._render()

            elif self.mode in ("moving", "resizing"):
                self.mode = "idle"
                self.drag_start_rois = None
                self.resize_handle = None
                self._maybe_auto_actions()
                self._render()

        # 右鍵：刪除『被選取』的 ROI（若沒有選取則不動作）
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.sel:
                # 由大到小刪，避免索引位移
                for idx in sorted(self.sel, reverse=True):
                    if 0 <= idx < len(rois):
                        del rois[idx]
                print(f"[INFO] 已刪除 {len(self.sel)} 個 ROI")
                self.sel.clear()
                self.active_idx = None
                self.mode = "idle"
                self._maybe_auto_actions()
                self._render()
            else:
                print("[INFO] 尚未選取 ROI，右鍵不執行刪除。")

    # ---------- Render ----------
    def _render(self, temp_rect=None):
        disp = self.disp_img.copy()
        key = self._img_key()
        rois = self._rois(key)
        filename = Path(key).name
        
        self.label_hitboxes = {}  # 重新計算本畫面所有 ROI 的點擊區

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        th = 2

        for i, r in enumerate(rois):
            p1 = self._orig2canvas(r["x1"], r["y1"])
            p2 = self._orig2canvas(r["x2"], r["y2"])
            color = COLOR_SEL if i in self.sel else COLOR_BOX
            cv2.rectangle(disp, p1, p2, color, TH)

            # 原圖 w/h
            w_px = max(0, r["x2"] - r["x1"])
            h_px = max(0, r["y2"] - r["y1"])

            # 將 "w x h" 拆成三段分別畫： [W] [x] [H]
            label_y = p1[1] - 6 if p1[1] - 6 > 12 else p1[1] + 20
            x_cursor = p1[0] + 5

            txt_w = str(w_px)
            size_w, base_w = cv2.getTextSize(txt_w, font, fs, th)
            cv2.putText(disp, txt_w, (x_cursor, label_y), font, fs, color, th, cv2.LINE_AA)
            rect_w = (x_cursor, label_y - size_w[1], x_cursor + size_w[0], label_y + base_w)
            x_cursor += size_w[0] + 4

            txt_x = "x"
            size_x, base_x = cv2.getTextSize(txt_x, font, fs, th)
            cv2.putText(disp, txt_x, (x_cursor, label_y), font, fs, color, th, cv2.LINE_AA)
            x_cursor += size_x[0] + 4

            txt_h = str(h_px)
            size_h, base_h = cv2.getTextSize(txt_h, font, fs, th)
            cv2.putText(disp, txt_h, (x_cursor, label_y), font, fs, color, th, cv2.LINE_AA)
            rect_h = (x_cursor, label_y - size_h[1], x_cursor + size_h[0], label_y + base_h)

            # 只有單選時才提供文字編輯 hitbox
            if len(self.sel) <= 1:
                self.label_hitboxes[i] = {'w': rect_w, 'h': rect_h}

        # 單選：顯示控制點
        if len(self.sel) == 1:
            idx = next(iter(self.sel))
            if 0 <= idx < len(rois):
                boxes = self._handle_boxes_for_roi(rois[idx])
                for (x1, y1, x2, y2) in boxes.values():
                    cv2.rectangle(disp, (x1, y1), (x2, y2), COLOR_SEL, -1)

        # 畫暫時 ROI（拖曳新建時）
        if temp_rect is not None and self.mode == "drawing":
            (sx, sy), (ex, ey) = temp_rect
            cv2.rectangle(disp, (sx, sy), (ex, ey), COLOR_BOX, TH)
            x1o, y1o = self._mouse2orig(sx, sy)
            x2o, y2o = self._mouse2orig(ex, ey)
            w = abs(x2o - x1o); h = abs(y2o - y1o)
            tx, ty = sx + 5, sy - 6 if sy - 6 > 12 else sy + 20
            cv2.putText(disp, f"{w}x{h}", (tx, ty), font, fs, COLOR_BOX, th, cv2.LINE_AA)

        # 標題＋旗標
        status = (f"{self.idx+1}/{len(self.images)}  "
                  f"ZOOM:{int(self.zoom*100)}%  "
                  f"| JSON:{'ON' if self.auto_save_json else 'OFF'}  CROP:{'ON' if self.auto_export_crops else 'OFF'}")

        cv2.rectangle(disp, (0, 0), (DISP_W, 40), (50, 50, 50), -1)
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.setWindowTitle(WIN, f"{filename} - {status}")

        # 若正在編輯，畫出提示條
        if self.edit_mode is not None:
            prompt = f"Edit {self.edit_mode.upper()}:{self.edit_value_str}  (Enter=Confirm / Esc=Cancel / Backspace=Delete)"
            cv2.rectangle(disp, (0, 0), (DISP_W, 40), (0, 0, 0), -1)
            cv2.putText(disp, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(WIN, disp)

    # ---------- Main Loop ----------
    def run(self):
        print(HELP)
        self._prepare_display()
        self._render()
        while True:
            k = cv2.waitKey(10) & 0xFF

            # Ctrl+A 一般會進來為 0x01
            if k in (1,):  # Ctrl+A
                key = self._img_key()
                n = len(self._rois(key))
                self.sel = set(range(n))
                self.active_idx = (0 if n > 0 else None)
                self._render()
                continue



            elif k == ord('h'):
                print(HELP)

            elif k == ord('s'):
                self._save_json()

            # 切換自動儲存/輸出旗標
            elif k == ord('J'):
                self.auto_save_json = not self.auto_save_json
                print(f"[FLAG] Auto Save JSON = {self.auto_save_json}")
                self._render()

            elif k == ord('C'):
                self.auto_export_crops = not self.auto_export_crops
                print(f"[FLAG] Auto Export Crops = {self.auto_export_crops}")
                self._render()

            # 匯出裁切（全部 ROI；與選取無關）
            elif k == ord('x'):
                n = self._export_crops_for_index(self.idx, show_progress=True)
                print(f"[INFO] 匯出當前影像裁切：{n} 張（{self._img_key()}）")

            elif k == ord('X'):
                total = 0
                if tqdm is not None:
                    for i in tqdm(range(len(self.images)),
                                desc="Export all images",
                                unit="img",
                                dynamic_ncols=True):
                        total += self._export_crops_for_index(i, show_progress=False)
                else:
                    for i in range(len(self.images)):
                        total += self._export_crops_for_index(i, show_progress=False)
                print(f"[INFO] 匯出所有影像裁切：{total} 張")

            # 複製：只複製被選取的 ROI
            elif k == ord('c'):
                key = self._img_key()
                src = self._rois(key)
                if not self.sel:
                    print("[WARN] 沒有選取任何 ROI，無法複製。")
                else:
                    self.copy_buffer = [deepcopy(src[i]) for i in sorted(self.sel) if 0 <= i < len(src)]
                    print(f"[INFO] 已複製 {len(self.copy_buffer)} 個 ROI（{key}）")

            # 貼上：需當前有選取；以第一個被選取 ROI 的左上角為貼上錨點
            elif k == ord('V'):
                key = self._img_key()
                target = self._rois(key)
                if not self.copy_buffer:
                    print("[WARN] 剪貼簿是空的（先按 C 複製）。")
                elif not self.sel:
                    print("[WARN] 貼上需先選取至少一個 ROI（作為錨點）。")
                else:
                    anchor_idx = min(self.sel)
                    if not (0 <= anchor_idx < len(target)):
                        print("[WARN] 錨點索引超界。"); continue
                    anchor = target[anchor_idx]
                    # 計算剪貼簿的參考左上角
                    bx1 = min(r["x1"] for r in self.copy_buffer)
                    by1 = min(r["y1"] for r in self.copy_buffer)
                    dx = anchor["x1"] - bx1
                    dy = anchor["y1"] - by1

                    # 執行貼上（clamp + 最小尺寸保護）
                    new_indices = []
                    for r in self.copy_buffer:
                        rr = self._normalize_roi({
                            "x1": r["x1"] + dx, "y1": r["y1"] + dy,
                            "x2": r["x2"] + dx, "y2": r["y2"] + dy,
                        })
                        target.append(rr)
                        new_indices.append(len(target) - 1)
                    self.sel = set(new_indices)   # 讓新貼上的成為選取
                    self.active_idx = (min(new_indices) if new_indices else None)
                    print(f"[INFO] 已貼上 {len(new_indices)} 個 ROI 至 {key}（錨點 idx={anchor_idx}）")
                    self._maybe_auto_actions()
                    self._render()
            # 在 run() 迴圈內，貼上區塊下面，加上「小寫 v = 原座標貼上」
            elif k == ord('v'):  # 原座標貼上（不需錨點）
                key = self._img_key()
                target = self._rois(key)
                if not self.copy_buffer:
                    print("[WARN] 剪貼簿是空的（先按 C 複製）。")
                else:
                    new_indices = []
                    for r in self.copy_buffer:
                        # 直接在『相同座標』產生；_normalize_roi 會自動做邊界保護/最小尺寸
                        rr = self._normalize_roi({
                            "x1": r["x1"], "y1": r["y1"],
                            "x2": r["x2"], "y2": r["y2"],
                        })
                        target.append(rr)
                        new_indices.append(len(target) - 1)
                    # 讓新貼上的成為選取
                    self.sel = set(new_indices)
                    self.active_idx = (min(new_indices) if new_indices else None)
                    print(f"[INFO] 已在原座標貼上 {len(new_indices)} 個 ROI 至 {key}")
                    self._maybe_auto_actions()
                    self._render()

            # 前一張 / 下一張（←=81, ↑=82, →=83, ↓=84）
            elif k in (ord('a'), 81, 82):
                self.idx = (self.idx - 1) % len(self.images)
                self._prepare_display(); self._render()

            elif k in (ord('d'), 83, 84):
                self.idx = (self.idx + 1) % len(self.images)
                self._prepare_display(); self._render()

            # --- 編輯模式下的鍵盤輸入處理（僅單選時有意義） ---
            if self.edit_mode is not None and len(self.sel) <= 1:
                # 數字鍵 0~9
                if ord('0') <= k <= ord('9'):
                    if self.edit_value_str == "0":
                        self.edit_value_str = chr(k)
                    else:
                        self.edit_value_str += chr(k)
                    self._render()
                    continue

                # Backspace
                if k in (8,):  # Windows Backspace = 8
                    self.edit_value_str = self.edit_value_str[:-1] if self.edit_value_str else ""
                    self._render()
                    continue

                # Enter 確認
                if k in (10, 13):  # LF/CR
                    try:
                        val = int(self.edit_value_str) if self.edit_value_str.strip() else None
                    except ValueError:
                        val = None

                    if val is not None and val > 0:
                        r = self._rois(self._img_key())[self.edit_roi_idx]
                        h_img, w_img = self.orig_hw
                        if self.edit_mode == 'w':
                            r["x2"] = clamp(r["x1"] + val, 0, w_img - 1)
                            if r["x2"] <= r["x1"]: r["x2"] = min(w_img - 1, r["x1"] + 1)
                        else:
                            r["y2"] = clamp(r["y1"] + val, 0, h_img - 1)
                            if r["y2"] <= r["y1"]: r["y2"] = min(h_img - 1, r["y1"] + 1)
                        self._maybe_auto_actions()

                    # 離開編輯模式
                    self.edit_mode = None
                    self.edit_roi_idx = None
                    self.edit_value_str = ""
                    self._render()
                    continue

                # Esc 取消
                if k in (27,):
                    self.edit_mode = None
                    self.edit_roi_idx = None
                    self.edit_value_str = ""
                    self._render()
                    continue
                
            if k in (ord('q'), 27):
                self._save_json()
                break
            
        cv2.destroyAllWindows()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="ROI 資料產生器（顯示統一 1920×1080）")
    ap.add_argument("folder", type=Path, help="影像資料夾（來源）")
    ap.add_argument("-o", "--output", type=Path, default="rois.json", help="JSON 輸出檔（ROI 紀錄）")
    ap.add_argument("-c", "--crop-dir", type=Path, default=Path("crops"), help="裁切影像輸出資料夾")
    args = ap.parse_args()

    gen = ROIGenerator(args.folder, args.output, args.crop_dir)
    gen.run()

if __name__ == "__main__":
    main()
