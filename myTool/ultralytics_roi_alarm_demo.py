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

VALID_OPS = {"<", "<=", "==", ">", ">="}

def compare_count(count: int, op: str, threshold: int) -> bool:
    if op == ">=":
        return count >= threshold
    if op == ">":
        return count > threshold
    if op == "==":
        return count == threshold
    if op == "<=":
        return count <= threshold
    if op == "<":
        return count < threshold
    raise ValueError(f"Unsupported op: {op}")

def summarize_roi_frame(roi_states, roi_hits, rule_lookup):
    """
    回傳一個 dict，方便後續做統計/記錄
    {
      roi_id: {
        "roi_triggered": bool,
        "rules": {
          rule_id: {"count": int, "threshold": int, "triggered": bool, "class_name": str}
        }
      }
    }
    """
    out = {}
    for rid, st in roi_states.items():
        rules_info = {}
        for rule_id, count in roi_hits.get(rid, {}).items():
            rule = rule_lookup[rule_id]
            rules_info[rule_id] = {
                "class_name": rule.class_name,
                "count": count,
                "threshold": rule.trigger,
                "triggered": compare_count(count, rule.op, rule.trigger),
                "op": rule.op,
            }
        out[rid] = {"roi_triggered": st.triggered, "rules": rules_info}
    return out

def get_summary_trigger_count(summary: Dict[int, Any]) -> int:
    """
    快速回傳 summarize 中 ROI 觸發的數量
    """
    return sum(1 for v in summary.values() if v.get("roi_triggered", False))

@dataclass
class RuleTrackState:
    streak: int = 0       # 連續(或容錯後)命中累積
    miss: int = 0         # streak 期間的連續 miss 計數
    fired: bool = False   # 是否已對本次 streak 觸發過事件（避免每幀都觸發）

@dataclass
class TriggerEvent:
    frame_idx: int
    roi_id: int
    rule_id: str
    class_name: str
    count: int
    threshold: int
    streak: int
    op: str = ">="

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
    op: str     
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

class TriggerTracker:
    def __init__(self, warn_frames: int = 0, max_miss: int = 0):
        self.warn_frames = int(warn_frames)
        self.max_miss = int(max_miss)
        self._state: Dict[Tuple[int, str], RuleTrackState] = {}

    def update(self, frame_idx: int, roi_hits: Dict[int, Dict[str, int]], rule_lookup: Dict[str, Any]) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []

        # 本 frame 哪些 (roi, rule) 是 triggered
        triggered_now: Dict[Tuple[int, str], Tuple[int, int, str]] = {}
        # value: (count, threshold, class_name)

        for rid, rule_counts in roi_hits.items():
            for rule_id, count in rule_counts.items():
                rule = rule_lookup[rule_id]
                threshold = int(rule.trigger)
                op = rule.op

                if compare_count(int(count), op, threshold):
                    triggered_now[(rid, rule_id)] = (int(count), threshold, rule.class_name)


        # 對所有出現過的 key 做狀態更新（包含本 frame 沒命中的也要處理 miss）
        all_keys = set(self._state.keys()) | set(triggered_now.keys())

        for key in all_keys:
            st = self._state.setdefault(key, RuleTrackState())
            rid, rule_id = key

            if key in triggered_now:
                count, threshold, class_name = triggered_now[key]
                st.streak += 1
                st.miss = 0

                # 判斷是否該 emit
                if self.warn_frames <= 0:
                    # 預設行為：本 frame 觸發就算（但仍可用 fired 控制只噴一次）
                    if not st.fired:
                        events.append(TriggerEvent(frame_idx, rid, rule_id, class_name, count, threshold, st.streak))
                        st.fired = True
                else:
                    if st.streak >= self.warn_frames and not st.fired:
                        events.append(TriggerEvent(frame_idx, rid, rule_id, class_name, count, threshold, st.streak))
                        st.fired = True

            else:
                # 沒命中：只有在 streak>0 才計 miss
                if st.streak > 0:
                    st.miss += 1
                    if st.miss > self.max_miss:
                        # streak 斷掉，重置
                        st.streak = 0
                        st.miss = 0
                        st.fired = False

        return events

    def is_rule_active(self, roi_id: int, rule_id: str) -> bool:
        """檢查特定規則是否滿足 warn-frame 條件（Active）"""
        key = (roi_id, rule_id)
        if key not in self._state:
            return False
        st = self._state[key]
        if self.warn_frames <= 0:
            return st.streak > 0
        else:
            return st.streak >= self.warn_frames


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
        self.rule_lookup: Dict[str, ClassRule] = {r.rule_id: r for r in self.rules}

        for r in self.rules:
            k = r.class_key
            self.rules_map.setdefault(k, []).append(r)

        # ==========================================
        # ZERO-sensitive rules index (for <, <=, ==)
        # 目的：支援「count=0 也可能成立」的規則，例如 <1、==0、<=0
        # 只為這些規則補零，避免 roi_hits 膨脹
        # ==========================================
        self._zero_sensitive_ops = {"<", "<=", "=="}
        self._zero_rule_ids_by_roi: Dict[int, List[str]] = {rid: [] for rid in self.rois_active.keys()}

        for rule in self.rules:
            if rule.op not in self._zero_sensitive_ops:
                continue

            if rule.roi_all:
                for rid in self._zero_rule_ids_by_roi.keys():
                    self._zero_rule_ids_by_roi[rid].append(rule.rule_id)
            else:
                for rid in (rule.roi_set or []):
                    if rid in self._zero_rule_ids_by_roi:
                        self._zero_rule_ids_by_roi[rid].append(rule.rule_id)

        # 去重 + 固定排序（讓輸出 deterministic）
        for rid in self._zero_rule_ids_by_roi:
            self._zero_rule_ids_by_roi[rid] = sorted(set(self._zero_rule_ids_by_roi[rid]))

        
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
                            
        # 1.5) 補齊 zero-sensitive 規則的 count=0
        # 讓 <1 / ==0 / <=0 在「完全沒有目標」時也能被比較、進入 tracker
        for rid, rule_ids in self._zero_rule_ids_by_roi.items():
            m = roi_hits[rid]
            for rule_id in rule_ids:
                m.setdefault(rule_id, 0)

        # 2. 規則判定：檢查數量是否超過門檻
        states = {}
        rule_lookup = self.rule_lookup

        for rid in self.rois_active:
            triggered = False
            details = []
            
            for rule_id, count in roi_hits[rid].items():
                rule = rule_lookup[rule_id]
                if compare_count(count, rule.op, rule.trigger):
                    triggered = True
                    details.append(f"{rule.class_name}({count} {rule.op} {rule.trigger})")

            
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
        rules: List[ClassRule] = []
        lines = path.read_text("utf-8").splitlines()
        counter_map: Dict[str, int] = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid rule line (need >=3 tokens): {line}")

            c_name = parts[0]

            # NEW: 支援兩種格式：
            # 1) class trigger roi               -> op default ">="
            # 2) class op threshold roi
            if parts[1] in VALID_OPS:
                if len(parts) < 4:
                    raise ValueError(f"Invalid rule line (op format need 4 tokens): {line}")
                op = parts[1]
                trig_str = parts[2]
                roi_str = parts[3]
            else:
                op = ">="
                trig_str = parts[1]
                roi_str = parts[2]

            threshold = int(trig_str)

            key = c_name if self.case_sensitive else c_name.lower()
            counter_map[key] = counter_map.get(key, 0) + 1
            rule_id = f"{c_name}_{counter_map[key]}"

            roi_set = None
            roi_all = (roi_str == "-1")
            if not roi_all:
                roi_set = set(int(x) for x in roi_str.split(",") if x)

            rules.append(ClassRule(
                class_key=key,
                class_name=c_name,
                op=op,
                trigger=threshold,
                roi_all=roi_all,
                roi_set=roi_set,
                rule_id=rule_id
            ))

        return rules


# ==========================================
# 4. Module: Render (視覺化)
# ==========================================

class ResultRenderer:
    # ---------------------------
    # Debug overlay helpers
    # ---------------------------
    def _build_watermark_lines(self,
                               frame_idx: int,
                               roi_states: Dict[int, ROIState],
                               roi_hits: Optional[Dict[int, Dict[str, int]]],
                               rule_lookup: Optional[Dict[str, ClassRule]],
                               tracker: Optional["TriggerTracker"],
                               warn_frames: int) -> List[str]:
        """
        產生右上角 overlay 的文字行
        格式示例：
        Frame: 1234
        ROI-1 [ACTIVE] streak_max=7 warn=5
          person: cnt=2  op=>= thr=3  streak=4/5  active=0
          car:    cnt=1  op=>= thr=1  streak=8/5  active=1
        ROI-2 [----] ...
        """
        lines: List[str] = []
        lines.append(f"Frame: {frame_idx}")

        if roi_hits is None or rule_lookup is None:
            lines.append("roi_hits/rule_lookup: (none)")
            return lines

        # 依 ROI 排序，方便比對
        for rid in sorted(roi_states.keys()):
            st = roi_states[rid]
            hits = roi_hits.get(rid, {}) if roi_hits else {}

            # 該 ROI 下所有 rule_id（用 hits 為主；若想顯示全部規則也可改）
            rule_ids = sorted(hits.keys())

            counting_rules = len(rule_ids)  # 本幀有計數到的規則（hits 出現）
            pending_rules = 0
            alarm_rules = 0
            streak_max = 0

            if tracker is not None:
                for rule_id in rule_ids:
                    ts = tracker._state.get((rid, rule_id))
                    if ts is not None:
                        streak_max = max(streak_max, ts.streak)
                        if tracker.is_rule_active(rid, rule_id):
                            alarm_rules += 1
                        elif ts.streak > 0:
                            pending_rules += 1

            roi_tag = "ALARM" if alarm_rules > 0 else "----"
            lines.append(
                f"ROI-{rid} [{roi_tag}] C-rl={counting_rules} P-rl={pending_rules} alarms={alarm_rules} "
                f"{streak_max}/{warn_frames}"
            )

            if not rule_ids:
                lines.append("  (no hits)")
                continue

            for rule_id in rule_ids:
                rule = rule_lookup.get(rule_id)
                if rule is None:
                    lines.append(f"  {rule_id}: (missing rule)")
                    continue

                cnt = int(hits.get(rule_id, 0))
                op = getattr(rule, "op", ">=")   # 向下相容
                thr = int(rule.trigger)

                streak = "-"
                active = 0
                if tracker is not None:
                    ts = tracker._state.get((rid, rule_id))
                    if ts is not None:
                        streak = f"{ts.streak}/{warn_frames if warn_frames > 0 else 1}"
                    active = 1 if tracker.is_rule_active(rid, rule_id) else 0

                # 顯示：class name : count / thres + streak/active
                # 你也可加上 miss / fired 等欄位
                lines.append(
                    f"  {rule.class_name}: {cnt}{op}{thr} | {streak} | {active}"
                )

        return lines

    def _draw_text_overlay_top_right(self,
                                    frame: np.ndarray,
                                    lines: List[str],
                                    max_lines: int = 40,
                                    font_scale: float = 0.5,
                                    thickness: int = 1,
                                    pad: int = 8,
                                    line_gap: int = 4,
                                    alpha: float = 0.45) -> None:
        """
        在右上角畫半透明底 + 多行文字
        """
        if frame is None or frame.size == 0:
            return
        if not lines:
            return

        lines = lines[:max_lines]

        font = cv2.FONT_HERSHEY_SIMPLEX
        # 量測最大寬度與總高度
        sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
        text_w = max((w for w, h in sizes), default=0)
        text_h = sum((h for w, h in sizes), 0) + (len(lines) - 1) * line_gap

        H, W = frame.shape[:2]
        box_w = text_w + pad * 2
        box_h = text_h + pad * 2

        # 右上角定位
        x2 = W - 10
        y1 = 10
        x1 = max(0, x2 - box_w)
        y2 = min(H - 1, y1 + box_h)

        # 半透明底
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 文字
        y = y1 + pad
        for (s, (w, h)) in zip(lines, sizes):
            y_text = y + h
            cv2.putText(frame, s, (x1 + pad, y_text), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y = y_text + line_gap

    # ---------------------------
    # Main draw
    # ---------------------------
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
             color_center_by_roi: bool = True,
             tracker: Optional["TriggerTracker"] = None,
             frame_idx: int = 0,
             watermark: bool = False,
             wm_max_lines: int = 40,
             warn_frames: int = 0) -> np.ndarray:
        """
        ROI + centers + (optional) watermark overlay
        """
        annotated_frame = raw_result.plot()

        # 1) 畫 ROI
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

        # 2) 中心點（白/橘/紅，warn_frames 相容：有 tracker 時用 tracker）
        if draw_centers:
            contours: Dict[int, np.ndarray] = {}
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

                color = (255, 255, 255)  # 白

                if color_center_by_roi and hit_roi_id is not None:
                    key = det.class_name if case_sensitive else det.class_name.lower()
                    relevant_rules = (rules_map.get(key, []) if rules_map else [])

                    is_counting = False
                    is_triggered_center = False

                    for rule in relevant_rules:
                        if rule.roi_all or (rule.roi_set and hit_roi_id in rule.roi_set):
                            is_counting = True

                            if tracker is not None:
                                if tracker.is_rule_active(hit_roi_id, rule.rule_id):
                                    is_triggered_center = True
                                    break
                            else:
                                if roi_hits is not None:
                                    cnt = roi_hits.get(hit_roi_id, {}).get(rule.rule_id, 0)
                                    if compare_count(cnt, rule.op, rule.trigger):
                                        is_triggered_center = True
                                        break

                    if is_triggered_center:
                        color = (0, 0, 255)        # 紅
                    elif is_counting:
                        color = (0, 165, 255)      # 橘（若你想黃：改 (0,255,255)）

                cv2.circle(annotated_frame, pt, 4, color, -1, lineType=cv2.LINE_AA)

        # 3) 浮水印 overlay（右上角）
        if watermark:
            lines = self._build_watermark_lines(
                frame_idx=frame_idx,
                roi_states=roi_states,
                roi_hits=roi_hits,
                rule_lookup=rule_lookup,
                tracker=tracker,
                warn_frames=warn_frames,
            )
            self._draw_text_overlay_top_right(
                annotated_frame,
                lines=lines,
                max_lines=wm_max_lines,
                font_scale=0.5,
                thickness=1,
                pad=8,
                line_gap=4,
                alpha=0.45,
            )

        return annotated_frame

# ==========================================
# 5. Module: IO / Pipeline (流程控制)
# ==========================================

class VideoPipeline:
    def __init__(self, detector, roi_engine, renderer, tracker=None, watermark=False, wm_max_lines=40):
        self.detector = detector
        self.roi_engine = roi_engine
        self.renderer = renderer
        self.tracker = tracker
        self.watermark = watermark
        self.wm_max_lines = wm_max_lines


    def run(self, source: str, output_path: Optional[str] = None, show: bool = False):
        rule_lookup = {r.rule_id: r for r in self.roi_engine.rules}  # 你 renderer 也在用
        frame_idx = 0
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

                # Step 2.5: Tracker Update (引入時間維度 / warn-frame 邏輯)
                if self.tracker:
                    self.tracker.update(frame_idx, roi_hits, rule_lookup)
                    
                    # 覆寫 roi_states 的 triggered 狀態，改由 Tracker 判定
                    # 只有當 Tracker 認定該 ROI 有規則處於 Active 狀態時，才算觸發
                    active_rois = set()
                    # 遍歷 tracker 內部狀態找出 active 的 rule/roi
                    for (rid, rule_id) in self.tracker._state:
                        if self.tracker.is_rule_active(rid, rule_id):
                            active_rois.add(rid)
                    
                    for rid, st in roi_states.items():
                        st.triggered = (rid in active_rois)

                summary = summarize_roi_frame(roi_states, roi_hits, rule_lookup)

                # Step 3: Render (傳入 raw_result 給 plot() 使用)
                # 注意：這裡回傳的是全新的 annotated_frame，不是原本的 frame
                final_frame = frame # 預設若沒畫圖就是原圖
                frame_idx += 1
                print(f"[DEBUG] [{frame_idx}] Summary: {get_summary_trigger_count(summary)}", end='\r')
                
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
                        color_center_by_roi=True,
                        tracker=self.tracker,
                        frame_idx=frame_idx,
                        watermark=self.watermark,
                        wm_max_lines=self.wm_max_lines,
                        warn_frames=(self.tracker.warn_frames if self.tracker else 0)
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
    parser.add_argument("--warn-frames", type=int, default=0, help="連續觸發門檻(幀). 0=當幀觸發即算")
    parser.add_argument("--max-miss", type=int, default=0, help="連續觸發期間允許 miss 幀數")
    parser.add_argument("--watermark", action="store_true", help="右上角顯示 debug overlay")
    parser.add_argument("--wm-max-lines", type=int, default=40, help="overlay 最多顯示幾行")

    args = parser.parse_args()

    # 1. 實例化各模組 (Dependency Injection)
    detector = ObjectDetector(model_path=args.model)
    
    roi_engine = ROIEngine(roi_json_path=args.roi, rules_path=args.classes)
    
    renderer = ResultRenderer()
    tracker = TriggerTracker(warn_frames=args.warn_frames, max_miss=args.max_miss)
    
    # 2. 建立 Pipeline 並執行
    pipeline = VideoPipeline(detector, roi_engine, renderer, tracker=tracker,
                            watermark=args.watermark, wm_max_lines=args.wm_max_lines)

    
    pipeline.run(source=args.video, output_path=args.out, show=args.show)

if __name__ == "__main__":
    main()