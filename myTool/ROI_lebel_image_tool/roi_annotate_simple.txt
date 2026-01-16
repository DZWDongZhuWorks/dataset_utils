#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import argparse
from pathlib import Path
import numpy as np


def safe_imread(path: str):
    """支援含非 ASCII 路徑的安全讀圖"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PolygonROIAnnotator:
    def __init__(self, image_path: Path, out_json: Path, disp_w: int, disp_h: int):
        self.image_path = image_path
        self.out_json = out_json
        self.disp_w = disp_w
        self.disp_h = disp_h

        self.orig = safe_imread(str(image_path))
        if self.orig is None:
            raise RuntimeError(f"讀取影像失敗：{image_path}")
        self.h, self.w = self.orig.shape[:2]

        # view scale (fit into disp_w x disp_h)
        self.scale = min(self.disp_w / self.w, self.disp_h / self.h)
        self.offset = (0, 0)
        self.view_rect = (0, 0, 0, 0)  # x1,y1,x2,y2 on canvas

        self.canvas = None
        self._recompose_view()

        # ROI data in ORIGINAL coordinates
        self.rois = []
        self.current_poly = []  # list[[x,y]] in ORIGINAL coords
        self.mouse_pos_canvas = None

        # --- close hint / snapping ---
        # 游標靠近「起點」的距離閾值（以 canvas 像素計）
        self.close_snap_dist_px = 14
        self.close_hint_active = False

        self.win = "Polygon ROI Annotator"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._mouse_cb)

    # ---- view compose ----
    def _recompose_view(self):
        cur_w = max(1, int(round(self.w * self.scale)))
        cur_h = max(1, int(round(self.h * self.scale)))
        ox = (self.disp_w - cur_w) // 2
        oy = (self.disp_h - cur_h) // 2
        self.offset = (ox, oy)
        self.view_rect = (ox, oy, ox + cur_w, oy + cur_h)

        canvas = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        resized = cv2.resize(self.orig, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = self.view_rect
        canvas[y1:y2, x1:x2] = resized
        self.canvas = canvas

    # ---- coordinate transforms ----
    def _orig2canvas(self, x, y):
        ox, oy = self.offset
        return int(round(x * self.scale + ox)), int(round(y * self.scale + oy))

    def _canvas2orig(self, x, y):
        ox, oy = self.offset
        xo = int(round((x - ox) / self.scale))
        yo = int(round((y - oy) / self.scale))
        return xo, yo

    def _in_view(self, cx, cy):
        x1, y1, x2, y2 = self.view_rect
        return x1 <= cx < x2 and y1 <= cy < y2

    # ---- polygon utils ----
    def _sanitize_point(self, x, y):
        x = clamp(int(x), 0, self.w - 1)
        y = clamp(int(y), 0, self.h - 1)
        return [x, y]

    def _finalize_current_polygon(self):
        if len(self.current_poly) < 3:
            return False
        rid = len(self.rois) 
        self.rois.append({"id": rid, "polygon": self.current_poly[:]})
        self.current_poly.clear()
        self.close_hint_active = False
        return True

    def _undo_last_polygon(self):
        if self.rois:
            self.rois.pop()

    def _clear_all(self):
        self.rois.clear()
        self.current_poly.clear()
        self.close_hint_active = False

    def _undo_last_point(self):
        if self.current_poly:
            self.current_poly.pop()
        self.close_hint_active = False

    def _update_close_hint(self):
        """
        判斷滑鼠是否接近 current_poly 的「起點」以進入閉合提示狀態。
        規則：
        - 至少已經有 3 個點才允許進入 close 狀態（避免過早提示）
        - 滑鼠在 view 內
        - 滑鼠到起點的距離 <= close_snap_dist_px
        """
        self.close_hint_active = False
        if self.mouse_pos_canvas is None:
            return
        if len(self.current_poly) < 3:
            return

        mx, my = self.mouse_pos_canvas
        if not self._in_view(mx, my):
            return

        sx, sy = self._orig2canvas(self.current_poly[0][0], self.current_poly[0][1])
        dx = mx - sx
        dy = my - sy
        if (dx * dx + dy * dy) <= (self.close_snap_dist_px * self.close_snap_dist_px):
            self.close_hint_active = True

    # ---- mouse ----
    def _mouse_cb(self, event, x, y, flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos_canvas = (x, y)
            self._update_close_hint()
            self._render()
            return

        # 右鍵：undo 最後一點（等效 backspace）
        if event == cv2.EVENT_RBUTTONDOWN:
            self._undo_last_point()
            self._update_close_hint()
            self._render()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self._in_view(x, y):
                return

            # 若接近起點且符合條件：左鍵直接閉合（等效 Enter），不新增點
            self.mouse_pos_canvas = (x, y)
            self._update_close_hint()
            if self.close_hint_active:
                ok = self._finalize_current_polygon()
                if not ok:
                    print("[WARN] Polygon 至少需要 3 個點才可完成。")
                self._render()
                return

            # 一般狀態：新增點
            xo, yo = self._canvas2orig(x, y)
            self.current_poly.append(self._sanitize_point(xo, yo))
            self._update_close_hint()
            self._render()
            return

    # ---- draw ----
    def _draw_polygon(self, img, pts_orig, color, thickness=2, closed=True, fill_alpha=0.15):
        pts_canvas = np.array([self._orig2canvas(x, y) for x, y in pts_orig], dtype=np.int32)
        if len(pts_canvas) >= 2:
            cv2.polylines(img, [pts_canvas], closed, color, thickness, cv2.LINE_AA)

        if closed and len(pts_canvas) >= 3 and fill_alpha > 0:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts_canvas], color)
            cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

    def _polygon_label_pos(self, pts_orig):
        arr = np.array(pts_orig, dtype=np.float32)
        cx, cy = arr.mean(axis=0)
        return self._orig2canvas(int(cx), int(cy))

    def _render(self):
        disp = self.canvas.copy()

        # status bar
        bar_h = 34
        cv2.rectangle(disp, (0, 0), (self.disp_w, bar_h), (40, 40, 40), -1)
        msg = (
            "L-click:add vertex | (near start) L-click:close | Enter:close | Backspace/R-click:del last | "
            "n:discard current | z:undo polygon | c:clear | s:save | q/esc:quit(no save)"
        )
        cv2.putText(disp, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # draw finished rois
        for r in self.rois:
            pts = r["polygon"]
            self._draw_polygon(disp, pts, (0, 255, 0), closed=True, fill_alpha=0.12)
            lx, ly = self._polygon_label_pos(pts)
            cv2.putText(disp, f"ROI#{r['id']}", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # draw current (unfinished) polygon
        if len(self.current_poly) >= 1:
            # open polyline
            self._draw_polygon(disp, self.current_poly, (0, 255, 255), closed=False, fill_alpha=0)

            # draw vertices
            for x, y in self.current_poly:
                cx, cy = self._orig2canvas(x, y)
                cv2.circle(disp, (cx, cy), 4, (0, 255, 255), -1)

            # close hint: 起點顯示更大的空心黃圈
            if len(self.current_poly) >= 3:
                sx, sy = self._orig2canvas(self.current_poly[0][0], self.current_poly[0][1])
                if self.close_hint_active:
                    cv2.circle(disp, (sx, sy), 10, (0, 255, 255), 2, cv2.LINE_AA)  # 大空心黃圈
                else:
                    cv2.circle(disp, (sx, sy), 6, (0, 255, 255), 1, cv2.LINE_AA)   # 小提示圈（可留可不留）

            # preview line to mouse
            if self.mouse_pos_canvas is not None:
                mx, my = self.mouse_pos_canvas
                if self._in_view(mx, my):
                    last = self.current_poly[-1]
                    lx, ly = self._orig2canvas(last[0], last[1])

                    # 若已進入 close 狀態，預覽線連到起點更直觀
                    if self.close_hint_active and len(self.current_poly) >= 3:
                        sx, sy = self._orig2canvas(self.current_poly[0][0], self.current_poly[0][1])
                        cv2.line(disp, (lx, ly), (sx, sy), (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.line(disp, (lx, ly), (mx, my), (0, 255, 255), 1, cv2.LINE_AA)

        cv2.setWindowTitle(
            self.win,
            f"{self.image_path.name} | ROIs: {len(self.rois)} | Current vertices: {len(self.current_poly)}"
        )
        cv2.imshow(self.win, disp)

    # ---- save ----
    def save_json(self):
        payload = {
            "image": str(self.image_path),
            "image_size": {"width": int(self.w), "height": int(self.h)},
            "rois": [{"id": r["id"], "polygon": r["polygon"]} for r in self.rois],
        }
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] JSON 已儲存：{self.out_json}")

    def run(self):
        self._render()
        while True:
            k = cv2.waitKey(20) & 0xFF

            if k == ord("s"):
                self.save_json()

            elif k == ord("z"):
                self._undo_last_polygon()
                self._render()

            elif k == ord("c"):
                self._clear_all()
                self._render()

            elif k == ord("n"):
                # discard current unfinished polygon
                self.current_poly.clear()
                self.close_hint_active = False
                self._render()

            elif k in (8, 127):  # Backspace (8), DEL (127) in some envs
                self._undo_last_point()
                self._update_close_hint()
                self._render()

            elif k in (13, 10):  # Enter (CR/LF)
                ok = self._finalize_current_polygon()
                if not ok:
                    print("[WARN] Polygon 至少需要 3 個點才可完成。")
                self._render()

            elif k in (ord("q"), 27):  # q or ESC
                # 只退出不儲存
                break

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Polygon ROI 標註工具（多個 polygon，輸出 JSON）")
    ap.add_argument("--image", "-i", type=Path, required=True, help="輸入影像路徑")
    ap.add_argument("--output", "-o", type=Path, default=Path("rois_polygon.json"), help="輸出 JSON 路徑")
    ap.add_argument("--disp", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"),
                    help="顯示畫布大小（預設 1280 720）")
    args = ap.parse_args()

    tool = PolygonROIAnnotator(args.image, args.output, args.disp[0], args.disp[1])
    tool.run()


if __name__ == "__main__":
    main()
