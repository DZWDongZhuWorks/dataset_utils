import cv2
import os
import argparse
import numpy as np

def load_image_paths(list_path: str):
    with open(list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    img_paths = []
    for p in lines:
        if os.path.isfile(p):
            img_paths.append(p)
        else:
            print(f"[WARN] 找不到檔案：{p}")
    return img_paths


def shorten_text_to_fit(text: str, width: int, font, font_scale: float, thickness: int):
    """如果字太長超過畫面寬度，就從左邊開始裁掉並加 '...'。"""
    display = text
    (text_w, _), _ = cv2.getTextSize(display, font, font_scale, thickness)

    if text_w <= width - 10:
        return display

    # 太寬就慢慢從左邊砍掉
    while len(display) > 10:
        display = "..." + display[4:]
        (text_w, _), _ = cv2.getTextSize(display, font, font_scale, thickness)
        if text_w <= width - 10:
            break
    return display

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支援中文 / 非 ASCII 路徑的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img

def main(args):
    img_paths = load_image_paths(args.list)

    if not img_paths:
        print("[ERROR] list 檔裡沒有任何可用的圖片路徑")
        return

    # 讀第一張圖決定影像大小
    first = imread_unicode(img_paths[0])
    if first is None:
        print(f"[ERROR] 無法讀取第一張圖：{img_paths[0]}")
        return

    h, w = first.shape[:2]
    size = (w, h)
    print(f"[INFO] 影片大小：{w}x{h}, FPS = {args.fps}, 每張圖片重複 {args.frames_per_image} frame")

    # 建議用 avi + XVID，比較不會有 codec 問題
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.out, fourcc, args.fps, size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_y = 25  # 文字 baseline 的 y
    bar_height = 40

    for idx, path in enumerate(img_paths):
        img = imread_unicode(path)
        if img is None:
            print(f"[WARN] 無法讀取圖片：{path}")
            continue

        # 若大小不一，全部 resize 成跟第一張一樣大
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, size)

        # 畫上方黑底條
        cv2.rectangle(img, (0, 0), (w, bar_height), (0, 0, 0), -1)

        # 要顯示的文字（若太長會縮）
        text = shorten_text_to_fit(path, w, font, font_scale, thickness)

        # 把路徑寫上去
        cv2.putText(
            img,
            text,
            (5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # ★ 新增：重複寫入多次，控制每張圖片出現的 frame 數
        for _ in range(args.frames_per_image):
            out.write(img)

        if (idx + 1) % 100 == 0:
            print(f"[INFO] 已寫入 {idx + 1}/{len(img_paths)} 張", end='\r')

    out.release()
    print(f"[DONE] 影片輸出完成：{args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="將 image_path.txt 中的所有圖片串接成影片，並在上方顯示原始路徑"
    )
    parser.add_argument(
        "-l",
        "--list",
        required=True,
        help="image_path.txt 路徑（每行一個完整圖片路徑）",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="輸出影片路徑，例如 F:\\k00\\k00_concat.avi",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="影片 FPS（預設 5）",
    )
    # ★ 新增參數：每張圖片要佔幾個 frame
    parser.add_argument(
        "--frames-per-image",
        "--fpi",
        dest="frames_per_image",
        type=int,
        default=1,
        help="每張圖片重複寫入的 frame 數（預設 1）",
    )

    args = parser.parse_args()

    if args.frames_per_image <= 0:
        print("[ERROR] --frames-per-image 必須 > 0")
    else:
        main(args)
