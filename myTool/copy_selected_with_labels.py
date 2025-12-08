#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
讀取 selected_paths.txt，複製影像與標註檔到指定目錄，並保持目錄結構。

假設：
- selected_paths.txt 每行為一張影像的完整路徑
- 影像與標註的對應規則與你原本的工具一致：
    - 同資料夾下： image.jpg  <-> image.txt
    - images/xxx.jpg <-> labels/xxx.txt
    - images/xxx.jpg <-> label/xxx.txt

用法範例：

    python copy_selected_with_labels.py \
        -r /path/to/original_dataset_root \
        -s selected_paths.txt \
        -d /path/to/output_root \
        --img-exts .jpg,.jpeg,.png \
        --label-exts .txt \
        --dry-run

參數說明：
- --root/-r      : 原始資料集 root（用來計算相對路徑，保持目錄結構）
- --selected/-s  : selected_paths.txt 檔案路徑
- --dst/-d       : 輸出根目錄
- --img-exts     : 影像副檔名清單，預設 .jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp
- --label-exts   : 標註副檔名清單，預設 .txt
- --dry-run      : 只模擬、不實際複製
"""

import os
import argparse
import shutil
from typing import List, Optional, Tuple, Dict, Set

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXTS_DEFAULT = [".txt"]


# =========================
# 副檔名 & 路徑工具
# =========================

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
    """將路徑中的某一層資料夾名稱換掉，例如 images -> labels"""
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == src_name_lc:
            q = parts[:]
            q[i] = dst_name
            return os.sep.join(q)
    return None


def find_label_for_image(image_path: str, label_exts: List[str]) -> Optional[str]:
    """
    由影像檔推測標註檔路徑：
    - 同目錄
    - images -> labels
    - images -> label
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(image_path)

    # 同目錄
    for ext in label_exts:
        cand = os.path.join(img_dir, stem + ext)
        if os.path.isfile(cand):
            return cand

    # images -> labels
    cand_dir = _replace_one_component(img_dir, "images", "labels")
    if cand_dir:
        for ext in label_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    # images -> label
    cand_dir = _replace_one_component(img_dir, "images", "label")
    if cand_dir:
        for ext in label_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    return None


def ensure_nested_target(dst_root: str, dataset_root: str, src_path: str) -> str:
    """
    在 dst_root 內建立相對於 dataset_root 的巢狀結構並回傳目標路徑。
    若 src_path 不在 dataset_root 底下，則退化成只用檔名。
    """
    try:
        rel = os.path.relpath(src_path, start=dataset_root)
    except ValueError:
        # 不同磁碟之類的情況
        rel = os.path.basename(src_path)

    if rel.startswith(".."):
        # 不在 root 之下，避免把一堆 .. 帶進去
        rel = os.path.basename(src_path)

    return os.path.join(dst_root, rel)


# =========================
# 主流程
# =========================

def read_selected_paths(selected_file: str) -> List[str]:
    """讀取 selected_paths.txt，一行一個路徑，忽略空行與以 # 開頭者"""
    paths: List[str] = []
    with open(selected_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def copy_one_file(src: str, dst: str, dry_run: bool) -> bool:
    """複製單一檔案；回傳 True 表成功（或 dry-run 視為成功）"""
    try:
        if not os.path.exists(src):
            print(f"[警告] 找不到檔案：{src}")
            return False
        dst_dir = os.path.dirname(dst)
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"[錯誤] 無法複製 {src} -> {dst} ({e})")
        return False


def process_copy(
    dataset_root: str,
    selected_paths_file: str,
    dst_root: str,
    img_exts: List[str],
    label_exts: List[str],
    dry_run: bool = False,
) -> None:
    dataset_root = os.path.abspath(dataset_root)
    dst_root = os.path.abspath(dst_root)

    print(f"資料集 root      : {dataset_root}")
    print(f"輸出 root        : {dst_root}")
    print(f"selected 檔案    : {selected_paths_file}")
    print(f"影像副檔名       : {img_exts}")
    print(f"標註副檔名       : {label_exts}")
    print(f"dry-run          : {dry_run}")
    print("--------------------------------------------------")

    img_paths_raw = read_selected_paths(selected_paths_file)
    print(f"從 selected_paths.txt 讀到 {len(img_paths_raw)} 行。")

    # 轉成絕對路徑
    img_paths = [os.path.abspath(p) for p in img_paths_raw]

    copied_images = 0
    missing_images = 0
    copied_labels = 0
    missing_labels = 0

    for idx, img in enumerate(img_paths, start=1):
        # 只處理副檔名符合的影像
        low = img.lower()
        if not any(low.endswith(ext) for ext in img_exts):
            print(f"[略過] 第 {idx} 行看起來不是影像檔：{img}")
            continue

        # 目標影像路徑（保持目錄結構）
        dst_img = ensure_nested_target(dst_root, dataset_root, img)
        ok_img = copy_one_file(img, dst_img, dry_run=dry_run)
        if ok_img:
            copied_images += 1
        else:
            missing_images += 1
            # 找不到影像就沒必要找標註，但可以繼續處理下一筆
            continue

        # 找對應標註檔
        lab = find_label_for_image(img, label_exts)
        if lab:
            dst_lab = ensure_nested_target(dst_root, dataset_root, lab)
            ok_lab = copy_one_file(lab, dst_lab, dry_run=dry_run)
            if ok_lab:
                copied_labels += 1
            else:
                missing_labels += 1
        else:
            missing_labels += 1
            print(f"[提示] 影像無對應標註：{img}")

    print("--------------------------------------------------")
    print(f"影像複製成功數    : {copied_images}")
    print(f"影像缺失/失敗數  : {missing_images}")
    print(f"標註複製成功數    : {copied_labels}")
    print(f"標註缺失/失敗數  : {missing_labels}")
    if dry_run:
        print("（dry-run 模式：沒有實際寫入檔案）")


# =========================
# CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="讀取 selected_paths.txt，複製影像與標註到指定目錄並保持目錄結構。"
    )
    p.add_argument(
        "-r",
        "--root",
        required=True,
        help="原始資料集根目錄（用來計算相對路徑，保持目錄結構）",
    )
    p.add_argument(
        "-s",
        "--selected",
        required=True,
        help="selected_paths.txt 檔案路徑（一行一個影像完整路徑）",
    )
    p.add_argument(
        "-d",
        "--dst",
        required=True,
        help="輸出根目錄（會自動建立）",
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
        "--dry-run",
        action="store_true",
        help="只模擬不實際複製，用來先確認結果。",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    img_exts = normalize_exts(args.img_exts.split(","))
    label_exts = normalize_exts(args.label_exts.split(","))

    process_copy(
        dataset_root=args.root,
        selected_paths_file=args.selected,
        dst_root=args.dst,
        img_exts=img_exts,
        label_exts=label_exts,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
