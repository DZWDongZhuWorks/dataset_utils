#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依類別抽樣影像並輸出路徑 txt

用法範例：
    python select_by_class.py \
        -r /path/to/dataset \
        --quota 0=500 1=500 \
        --out selected_paths.txt \
        --img-exts .jpg,.jpeg,.png \
        --label-exts .txt \
        --seed 42

避開之前已選取過的路徑：
    python select_by_class.py \
        -r /path/to/dataset \
        --quota 0=500 1=500 \
        --exclude selected_old.txt selected_old2.txt \
        --out selected_new.txt

說明：
- 會掃描 root 下所有 label 檔（支援多個副檔名），依 YOLO 格式解析 class_id
- 利用 images/labels 的對應規則尋找影像檔
- 依照 --quota 指定的「每類別要 N 張」抽樣影像
- 若指定 --exclude，會先讀取這些檔案中出現的路徑並排除，再做抽樣
- 最終輸出一個 txt，裡面是一行一個影像完整路徑
"""

import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Set, Optional


# ==========================================================
# 和你原本程式類似的路徑處理與解析工具
# ==========================================================

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXTS_DEFAULT = [".txt"]


def normalize_exts(exts_in: List[str]) -> List[str]:
    """把副檔名標準化成 .xxx 小寫、不重複"""
    out: List[str] = []
    seen = set()
    for raw in exts_in:
        s = str(raw).strip()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        s = s.lower()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _replace_one_component(path: str, src_name_lc: str, dst_name: str) -> Optional[str]:
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == src_name_lc:
            q = parts[:]
            q[i] = dst_name
            return os.sep.join(q)
    return None


def find_image_for_label(label_path: str, img_exts: List[str]) -> Optional[str]:
    """
    由標註檔推測影像檔路徑：
    - 同目錄
    - labels -> images
    - label  -> images
    """
    stem = os.path.splitext(os.path.basename(label_path))[0]
    label_dir = os.path.dirname(label_path)

    # 同目錄
    for ext in img_exts:
        cand = os.path.join(label_dir, stem + ext)
        if os.path.isfile(cand):
            return cand

    # labels -> images
    cand_dir = _replace_one_component(label_dir, "labels", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    # label -> images
    cand_dir = _replace_one_component(label_dir, "label", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    return None


def parse_yolo_label_file(path: str) -> Set[int]:
    """
    讀 YOLO txt 標註，回傳「本檔出現過的 class_id 集合」。
    只關心 class_id，不統計 box 數量。
    """
    classes_in_file: Set[int] = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                    classes_in_file.add(cls)
                except Exception:
                    # 忽略壞行
                    continue
    except Exception:
        # 讀不到就當作沒有類別
        return set()
    return classes_in_file


# ==========================================================
# 讀取要排除的 selected_paths.txt
# ==========================================================

def read_excluded_paths(files: List[str], dataset_root: str) -> Set[str]:
    """
    從多個 selected_paths.txt 讀出要排除的影像路徑。
    - 支援絕對路徑與相對路徑：
      - 絕對路徑：直接 normalize
      - 相對路徑：視為相對於 dataset_root
    """
    excluded: Set[str] = set()
    for fpath in files:
        fpath = os.path.abspath(fpath)
        if not os.path.isfile(fpath):
            print(f"[警告] exclude 檔不存在：{fpath}")
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if os.path.isabs(line):
                    full = os.path.abspath(line)
                else:
                    # 假設是相對於 dataset_root
                    full = os.path.abspath(os.path.join(dataset_root, line))
                excluded.add(full)
    print(f"從 exclude 檔共收集到 {len(excluded)} 個需排除路徑。")
    return excluded


# ==========================================================
# 主邏輯：依 quota 抽樣影像
# ==========================================================

def collect_class_to_images(
    dataset_root: str,
    img_exts: List[str],
    label_exts: List[str],
    exclude_paths: Optional[Set[str]] = None,
) -> Dict[int, List[str]]:
    """
    掃描資料集，建立 class_id -> [image_path, ...] 的 mapping
    一張圖只會在每個 class 的 list 出現一次。
    若 exclude_paths 非空，會在建立 mapping 時先排除這些影像。
    """
    dataset_root = os.path.abspath(dataset_root)
    exclude_paths = exclude_paths or set()
    class_to_images: Dict[int, Set[str]] = defaultdict(set)

    # 找出所有 label 檔
    label_files: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            low = fname.lower()
            if any(low.endswith(ext) for ext in label_exts):
                label_files.append(os.path.join(root, fname))

    print(f"找到標註檔 {len(label_files)} 個，開始建立 class -> images 映射...")

    for lab in label_files:
        classes = parse_yolo_label_file(lab)
        if not classes:
            continue
        img_path = find_image_for_label(lab, img_exts)
        if not img_path:
            continue

        img_path = os.path.abspath(img_path)

        # 若在排除清單中，完全略過
        if img_path in exclude_paths:
            continue

        for cid in classes:
            class_to_images[cid].add(img_path)

    # set 轉 list
    return {cid: sorted(list(paths)) for cid, paths in class_to_images.items()}


def parse_quota_arg(quota_args: List[str]) -> Dict[int, int]:
    """
    解析 --quota 參數（nargs='+'）：
        --quota 0=500 1=500 3=200
    或兼容：
        --quota 0=500,1=500 3=200

    回傳 {0:500, 1:500, 3:200}
    """
    result: Dict[int, int] = {}
    if not quota_args:
        return result

    # quota_args 可能像 ["0=500", "1=500"] 或 ["0=500,1=500", "3=200"]
    tokens: List[str] = []
    for item in quota_args:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece:
                tokens.append(piece)

    for p in tokens:
        if "=" not in p:
            raise ValueError(f"無法解析 quota 片段：{p}，正確格式例：0=500")
        k_str, v_str = p.split("=", 1)
        cid = int(k_str.strip())
        n = int(v_str.strip())
        if n <= 0:
            continue
        result[cid] = n

    if not result:
        raise ValueError("解析後 quota 為空，請檢查 --quota 參數。")
    return result


def select_images_by_class_quota(
    class_to_images: Dict[int, List[str]],
    quota: Dict[int, int],
    seed: Optional[int] = None,
) -> List[str]:
    """
    依每個 class 的 quota 抽樣影像。
    - 一張圖若同時包含 class0、class1，被挑到時會同時滿足兩種需求。
    - 整體結果去重之後輸出。
    """
    rng = random.Random(seed)
    selected: Set[str] = set()

    for cid, want_n in quota.items():
        imgs = class_to_images.get(cid, [])
        if not imgs:
            print(f"[警告] 類別 {cid} 沒有可用影像（可能都被排除或原本就不存在）。")
            continue

        # 打亂順序，做隨機抽樣
        imgs_copy = imgs[:]
        rng.shuffle(imgs_copy)

        take_n = min(want_n, len(imgs_copy))
        chosen = imgs_copy[:take_n]
        if take_n < want_n:
            print(f"[提示] 類別 {cid} 只找到 {take_n} 張（少於要求的 {want_n} 張）。")

        for p in chosen:
            selected.add(p)

    print(f"總共選到 {len(selected)} 張影像（去重後）。")
    return sorted(selected)


def save_paths_to_txt(paths: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"已輸出路徑清單到：{out_path}")


# ==========================================================
# CLI
# ==========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="依指定類別張數配額，從 YOLO 資料集中抽樣影像並輸出路徑 txt。"
    )
    p.add_argument(
        "-r",
        "--root",
        required=True,
        help="資料集根目錄（會被遞迴掃描）",
    )
    p.add_argument(
        "--img-exts",
        default=",".join(IMG_EXTS_DEFAULT),
        help=f"影像副檔名清單（逗號分隔），預設：{','.join(IMG_EXTS_DEFAULT)}",
    )
    p.add_argument(
        "--label-exts",
        default=",".join(LABEL_EXTS_DEFAULT),
        help=f"標註副檔名清單（逗號分隔），預設：{','.join(LABEL_EXTS_DEFAULT)}",
    )
    p.add_argument(
        "-q",
        "--quota",
        nargs="+",
        required=True,
        help="各類別張數配額，例如：--quota 0=500 1=500 表示第 0 類 500 張、第 1 類 500 張。",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        help="先前已選取的 selected_paths.txt，可給多個；本次抽樣會避開這些路徑。",
    )
    p.add_argument(
        "--out",
        default="selected_paths.txt",
        help="輸出路徑 txt 檔案路徑，預設：selected_paths.txt",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="隨機種子（給定可重現同樣的抽樣結果）。",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.root)
    img_exts = normalize_exts(args.img_exts.split(","))
    label_exts = normalize_exts(args.label_exts.split(","))

    # 讀取要排除的路徑
    excluded_paths: Set[str] = set()
    if args.exclude:
        excluded_paths = read_excluded_paths(args.exclude, dataset_root)

    print(f"資料集 root      : {dataset_root}")
    print(f"影像副檔名       : {img_exts}")
    print(f"標註副檔名       : {label_exts}")
    print(f"quota            : {args.quota}")
    print(f"輸出 txt         : {args.out}")
    if args.exclude:
        print(f"exclude 檔案     : {args.exclude}")
    if args.seed is not None:
        print(f"隨機種子         : {args.seed}")

    quota = parse_quota_arg(args.quota)

    class_to_images = collect_class_to_images(
        dataset_root=dataset_root,
        img_exts=img_exts,
        label_exts=label_exts,
        exclude_paths=excluded_paths,
    )

    # 簡單列一下現有的類別與張數
    print("\n目前掃描到的各類別影像數（已排除 exclude 路徑後）：")
    for cid in sorted(class_to_images.keys()):
        print(f"  class {cid}: {len(class_to_images[cid])} 張")

    selected = select_images_by_class_quota(
        class_to_images=class_to_images,
        quota=quota,
        seed=args.seed,
    )

    save_paths_to_txt(selected, args.out)


if __name__ == "__main__":
    main()
