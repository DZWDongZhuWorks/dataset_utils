#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. Data Models (共用資料結構)
# ==========================================

@dataclass(frozen=True)
class Point:
    x: int
    y: int

@dataclass
class Detection:
    """標準化後的偵測結果，與 YOLO 格式解耦"""
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]
    center: Point

@dataclass
class PolygonROI:
    id: int
    polygon: List[Point]  # List of Points
    original_points: List[List[int]] # Keep raw list for easy cv2 drawing

@dataclass
class ClassRule:
    class_key: str
    class_name: str
    trigger: int
    roi_all: bool
    roi_set: Optional[Set[int]]
    rule_id: str

@dataclass
class ROIState:
    """單一 ROI 在當前 Frame 的狀態"""
    roi_id: int
    triggered: bool
    trigger_details: List[str] # e.g. ["person:rule_1(3/2)"]


# ==========================================
# 2. Module: Detect (基礎偵測)
# ==========================================

class ObjectDetector:
    def __init__(self, model_path: Path, device: str = "", conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], Any]:
        """
        回傳:
        1. List[Detection]: 供 ROI 邏輯使用的標準化數據
        2. raw_result: 供 Render 使用的原始 Ultralytics 結果物件
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device if self.device else None,
            verbose=False,
            half=False
        )[0]

        detections = []
        if results.boxes is not None:
            xyxy_list = results.boxes.xyxy.cpu().numpy()
            cls_list = results.boxes.cls.cpu().numpy()
            conf_list = results.boxes.conf.cpu().numpy()

            for xyxy, cid, conf in zip(xyxy_list, cls_list, conf_list):
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                c_int = int(cid)
                c_name = self.names.get(c_int, str(c_int))

                detections.append(Detection(
                    class_id=c_int,
                    class_name=c_name,
                    conf=float(conf),
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    center=Point(cx, cy)
                ))
        
        # 關鍵修改：同時回傳原始 results 物件
        return detections, results

# ==========================================
# 3. Module: ROI Logic (業務判斷)
# ==========================================

class ROIEngine:
    """
    職責：管理多邊形區域與規則，計算是否觸發警報。
    特性：純數學運算，不依賴影像繪製，不依賴 YOLO 模型本身。
    """
    def __init__(self, roi_json_path: Path, rules_path: Path, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        # 載入原始資料
        self.rois_raw, self.roi_img_size = self._load_rois(roi_json_path)
        self.rules = self._load_rules(rules_path)
        
        # 建立索引
        self.rois_active: Dict[int, PolygonROI] = {r.id: r for r in self.rois_raw}
        self.rules_map: Dict[str, List[ClassRule]] = {} # key: class_name_lower
        
        for r in self.rules:
            if r.trigger > 0: # 只索引有效的規則
                k = r.class_key # 已處理過大小寫
                self.rules_map.setdefault(k, []).append(r)
        
        self.initialized_size = False

    def rescale_rois_if_needed(self, frame_w: int, frame_h: int):
        """若 JSON 設定的尺寸與實際輸入不同，進行座標縮放 (只執行一次)"""
        if self.initialized_size:
            return
        
        self.initialized_size = True
        src_w = int(self.roi_img_size.get("width", 0))
        src_h = int(self.roi_img_size.get("height", 0))

        if src_w <= 0 or src_h <= 0 or (src_w == frame_w and src_h == frame_h):
            return

        sx, sy = frame_w / src_w, frame_h / src_h
        print(f"[INFO] Rescaling ROIs from {src_w}x{src_h} to {frame_w}x{frame_h}")

        new_rois = {}
        for rid, roi in self.rois_active.items():
            new_pts_list = [[int(p.x * sx), int(p.y * sy)] for p in roi.polygon]
            # 重新封裝 Point 物件與 list 格式
            pts_objs = [Point(x, y) for x, y in new_pts_list]
            new_rois[rid] = PolygonROI(id=rid, polygon=pts_objs, original_points=new_pts_list)
        
        self.rois_active = new_rois

    def evaluate(self, detections: List[Detection]) -> Dict[int, ROIState]:
        """
        核心邏輯：輸入一幀的所有偵測結果，輸出每個 ROI 的狀態
        """
        # 1. 空間篩選：計算每個 ROI 內各類別的數量
        # counts: roi_id -> class_key -> rule_id -> count
        roi_hits: Dict[int, Dict[str, int]] = {rid: {} for rid in self.rois_active}

        for det in detections:
            key = det.class_name if self.case_sensitive else det.class_name.lower()
            
            # 若此類別無規則，直接跳過計算幾何以節省效能
            relevant_rules = self.rules_map.get(key, [])
            if not relevant_rules:
                continue

            # 檢查每個 ROI
            for rid, roi in self.rois_active.items():
                if self._point_in_polygon(det.center, roi.original_points):
                    # 針對該類別的所有規則進行計數
                    for rule in relevant_rules:
                        if rule.roi_all or (rule.roi_set and rid in rule.roi_set):
                            roi_hits[rid][rule.rule_id] = roi_hits[rid].get(rule.rule_id, 0) + 1

        # 2. 規則判定：檢查數量是否超過門檻
        states = {}
        rule_lookup = {r.rule_id: r for r in self.rules}

        for rid in self.rois_active:
            triggered = False
            details = []
            
            for rule_id, count in roi_hits[rid].items():
                rule = rule_lookup[rule_id]
                if count >= rule.trigger:
                    triggered = True
                    details.append(f"{rule.class_name}({count}/{rule.trigger})")
            
            states[rid] = ROIState(roi_id=rid, triggered=triggered, trigger_details=details)
        
        return states, roi_hits

    # --- Helpers ---
    def _point_in_polygon(self, pt: Point, poly_pts: List[List[int]]) -> bool:
        contour = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, (float(pt.x), float(pt.y)), False) >= 0

    def _load_rois(self, path: Path):
        data = json.loads(path.read_text("utf-8"))
        out = []
        for r in data.get("rois", []):
            poly = r.get("polygon", [])
            if len(poly) < 3: continue
            pts_objs = [Point(int(p[0]), int(p[1])) for p in poly]
            raw_pts = [[int(p[0]), int(p[1])] for p in poly]
            out.append(PolygonROI(int(r.get("id")), pts_objs, raw_pts))
        return out, data.get("image_size", {})

    def _load_rules(self, path: Path) -> List[ClassRule]:
        # (這裡保留原有的解析邏輯，簡化展示)
        rules = []
        lines = path.read_text("utf-8").splitlines()
        counter_map = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            c_name, trig_str, roi_str = parts[0], parts[1], parts[2]
            
            key = c_name if self.case_sensitive else c_name.lower()
            counter_map[key] = counter_map.get(key, 0) + 1
            rule_id = f"{c_name}_{counter_map[key]}"
            
            roi_set = None
            roi_all = (roi_str == "-1")
            if not roi_all:
                roi_set = set(int(x) for x in roi_str.split(",") if x)

            rules.append(ClassRule(key, c_name, int(trig_str), roi_all, roi_set, rule_id))
        return rules


# ==========================================
# 4. Module: Render (視覺化)
# ==========================================

class ResultRenderer:
    def draw(self,
             raw_result: Any,
             detections: List[Detection],
             roi_states: Dict[int, ROIState],
             rois_map: Dict[int, PolygonROI],
             roi_hits: Optional[Dict[int, Dict[str, int]]] = None,
             rules_map: Optional[Dict[str, List[ClassRule]]] = None,
             rule_lookup: Optional[Dict[str, ClassRule]] = None,
             case_sensitive: bool = False,
             draw_centers: bool = True,
             color_center_by_roi: bool = True) -> np.ndarray:
        """
        使用 yolo 原生的 plot() 作為基礎，再疊加 ROI 資訊，並可選擇繪製偵測框中心點
        - 開始 counting：黃色
        - 觸發 trigger：紅色
        - 其他：白色
        """
        annotated_frame = raw_result.plot()

        # 1) 畫 ROI（原本邏輯不動）
        for rid, state in roi_states.items():
            roi = rois_map.get(rid)
            if not roi:
                continue

            color = (0, 0, 255) if state.triggered else (0, 255, 0)
            pts = np.array(roi.original_points, dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(annotated_frame, [pts], True, color, 2, cv2.LINE_AA)

            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame)

            cx = int(np.mean([p[0] for p in roi.original_points]))
            cy = int(np.mean([p[1] for p in roi.original_points]))

            label = f"ROI-{rid}"
            if state.triggered:
                label += " [ALARM]"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (cx - 20, cy - h - 5), (cx - 20 + w, cy + 5), color, -1)
            cv2.putText(annotated_frame, label, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2) 畫 Detection 中心點（改成：白 / 橘 / 紅）
        if draw_centers:
            # ROI contour 預先準備（用排序確保命中 ROI 的選擇是 deterministic）
            contours = {}
            for rid in sorted(rois_map.keys()):
                roi = rois_map[rid]
                contours[rid] = np.array(roi.original_points, dtype=np.int32).reshape((-1, 1, 2))

            for det in detections:
                pt = (det.center.x, det.center.y)

                hit_roi_id = None
                if color_center_by_roi and contours:
                    for rid, contour in contours.items():
                        if cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0:
                            hit_roi_id = rid
                            break

                # 預設：不在 ROI → 白色
                color = (255, 255, 255)

                if color_center_by_roi and hit_roi_id is not None:
                    # 判斷是否「開始 counting」/「已 trigger」
                    key = det.class_name if case_sensitive else det.class_name.lower()

                    relevant_rules = (rules_map.get(key, []) if rules_map else [])
                    is_counting = False
                    is_triggered_center = False

                    # 只有符合規則且該 ROI 會被該規則計數，才算「開始 counting」
                    for rule in relevant_rules:
                        if rule.trigger <= 0:
                            continue
                        if rule.roi_all or (rule.roi_set and hit_roi_id in rule.roi_set):
                            is_counting = True
                            if roi_hits is not None:
                                cnt = roi_hits.get(hit_roi_id, {}).get(rule.rule_id, 0)
                                if cnt >= rule.trigger:
                                    is_triggered_center = True
                                    break

                    # 上色：trigger 紅；counting 黃；其他白
                    if is_triggered_center:
                        color = (0, 0, 255)        # 紅 (BGR)
                    elif is_counting:
                        color = (0, 255, 255)      # 黃 (BGR)

                cv2.circle(annotated_frame, pt, 4, color, -1, lineType=cv2.LINE_AA)

        return annotated_frame

# ==========================================
# 5. Module: IO / Pipeline (流程控制)
# ==========================================

class VideoPipeline:
    def __init__(self, detector: ObjectDetector, roi_engine: ROIEngine, renderer: ResultRenderer):
        self.detector = detector
        self.roi_engine = roi_engine
        self.renderer = renderer

    def run(self, source: str, output_path: Optional[str] = None, show: bool = False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"[INFO] Pipeline started. Source: {w}x{h} @ {fps}fps")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.roi_engine.rescale_rois_if_needed(w, h)

                # Step 1: Detect (現在回傳兩個值)
                detections, raw_result = self.detector.predict(frame)

                # Step 2: ROI Logic (只依賴清洗過的 detections)
                roi_states, roi_hits  = self.roi_engine.evaluate(detections)

                print(f"[DEBUG] ROI States: {roi_states}")

                # Step 3: Render (傳入 raw_result 給 plot() 使用)
                # 注意：這裡回傳的是全新的 annotated_frame，不是原本的 frame
                final_frame = frame # 預設若沒畫圖就是原圖
                
                if writer or show:
                    final_frame = self.renderer.draw(
                        raw_result,
                        detections,
                        roi_states,
                        self.roi_engine.rois_active,
                        roi_hits=roi_hits,
                        rules_map=self.roi_engine.rules_map,
                        rule_lookup={r.rule_id: r for r in self.roi_engine.rules},
                        case_sensitive=self.roi_engine.case_sensitive,
                        draw_centers=True,
                        color_center_by_roi=True
                    )


                # Step 4: IO Output
                if writer:
                    writer.write(final_frame)
                
                if show:
                    cv2.imshow("Smart ROI System", final_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[INFO] Pipeline finished.")

# ==========================================
# Main Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Modular YOLO ROI System")
    parser.add_argument("-m", "--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to video or RTSP url")
    parser.add_argument("--roi", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=str, default="output.mp4")
    parser.add_argument("--show", action="store_true")
    
    args = parser.parse_args()

    # 1. 實例化各模組 (Dependency Injection)
    detector = ObjectDetector(model_path=args.model)
    
    roi_engine = ROIEngine(roi_json_path=args.roi, rules_path=args.classes)
    
    renderer = ResultRenderer()

    # 2. 建立 Pipeline 並執行
    pipeline = VideoPipeline(detector, roi_engine, renderer)
    
    pipeline.run(source=args.video, output_path=args.out, show=args.show)

if __name__ == "__main__":
    main()