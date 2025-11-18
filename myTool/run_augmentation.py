#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def move_augmentation_folder(wkpath: Path) -> None:
    """
    3. 將 wkpath/runs/augmentation 移到 wkpath/augmentation
    4. 移除空的 wkpath/runs
    """
    runs_dir = wkpath / "runs"
    src_aug = runs_dir / "augmentation"
    dst_aug = wkpath / "augmentation"

    if not src_aug.exists():
        print(f"[WARN] {src_aug} 不存在，略過移動。")
        return

    # 如果 wkpath/augmentation 已存在，嘗試把內容搬進去（簡單合併）
    if dst_aug.exists() and dst_aug.is_dir():
        print(f"[INFO] {dst_aug} 已存在，將 {src_aug} 底下檔案搬進去。")
        for item in src_aug.iterdir():
            target = dst_aug / item.name
            if target.exists():
                print(f"[WARN] 目標已存在，略過：{target}")
            else:
                shutil.move(str(item), str(target))
        # 如果 src_aug 搬空了就刪掉
        try:
            src_aug.rmdir()
        except OSError:
            pass
    else:
        # 直接把整個資料夾搬上來
        shutil.move(str(src_aug), str(dst_aug))
        print(f"[INFO] 已將 {src_aug} 搬到 {dst_aug}")

    # 如果 runs 變成空的，就刪掉
    if runs_dir.exists():
        try:
            next(runs_dir.iterdir())
        except StopIteration:
            runs_dir.rmdir()
            print(f"[INFO] 已刪除空目錄：{runs_dir}")


def run_single_wkpath(
    wkpath: Path,
    hyp_path: Path,
    data_aug_script: Path,
    new_image: int,
    no_background: bool,
) -> None:
    """
    1. cd wkpath（用 subprocess 的 cwd 模擬）
    2. python data_augmentation.py hyp . --new-image gen_num [--no-background]
    3. / 4. 移動並清理 runs/ 資料夾
    """
    wkpath = wkpath.resolve()
    if not wkpath.is_dir():
        print(f"[ERROR] {wkpath} 不是資料夾，略過。")
        return

    print(f"\n=== 處理資料夾: {wkpath} ===")

    # 組 CLI：dataset 給 "."，並把 cwd 設成 wkpath
    cmd = [
        sys.executable,
        str(data_aug_script),
        str(hyp_path),
        ".",
        "--new-image",
        str(new_image),
    ]

    # 原程式是 action='store_false'，有帶 --no-background 才會變 False
    if no_background is False:
        cmd.append("--no-background")

    print(f"[INFO] 執行指令：{' '.join(cmd)} (cwd={wkpath})")

    # 執行 data_augmentation.py
    subprocess.run(cmd, cwd=str(wkpath), check=True)

    # 移動 runs/augmentation -> augmentation，並刪除空的 runs
    move_augmentation_folder(wkpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批次呼叫 data_augmentation.py，支援多個 wkpath，並自動搬移 runs/augmentation。"
    )

    # 與原本程式相同的必要參數
    parser.add_argument("hyp", help="超參數設定檔路徑 (ex: E:\\dataset_utils\\hyp.yaml)")
    # 差別只有這裡：原本是單一 dataset，現在改成支援多個 wkpath
    parser.add_argument(
        "wkpaths",
        nargs="+",
        help="一個或多個資料集根目錄 (wkpath)",
    )

    # 預設直接找跟本檔案同資料夾的 data_augmentation.py
    parser.add_argument(
        "--script",
        default=None,
        help="data_augmentation.py 的路徑，預設為與本檔同目錄的 data_augmentation.py",
    )

    # 保持與原始程式相同參數名稱與預設值
    parser.add_argument(
        "--new-image",
        type=int,
        default=5,
        help="每張影像要額外生成幾張（對應原程式的 --new-image）",
    )
    parser.add_argument(
        "--no-background",
        action="store_false",
        help="傳遞給 data_augmentation.py 的 --no-background（行為與原程式完全一致）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    hyp_path = Path(args.hyp).resolve()

    # 決定 data_augmentation.py 的實際路徑
    if args.script is not None:
        data_aug_script = Path(args.script).resolve()
    else:
        # 預設假設 batch_data_augmentation.py 與 data_augmentation.py 在同一層
        data_aug_script = Path(__file__).with_name("data_augmentation.py").resolve()

    if not data_aug_script.is_file():
        print(f"[ERROR] 找不到 data_augmentation.py：{data_aug_script}")
        sys.exit(1)

    print(f"[INFO] 使用的 data_augmentation.py：{data_aug_script}")
    print(f"[INFO] 使用的 hyp 檔：{hyp_path}")
    print(f"[INFO] new_image = {args.new_image}, no_background = {args.no_background}")

    for wk in args.wkpaths:
        run_single_wkpath(
            wkpath=Path(wk),
            hyp_path=hyp_path,
            data_aug_script=data_aug_script,
            new_image=args.new_image,
            no_background=args.no_background,
        )

    print("\n[INFO] 所有 wkpath 已處理完成。")


if __name__ == "__main__":
    main()
