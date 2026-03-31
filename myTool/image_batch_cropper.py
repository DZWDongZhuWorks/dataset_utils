import os
import cv2
import argparse
from pathlib import Path
import numpy as np

def crop_image(image_path, output_dir, crop_w, crop_h, out_ext=None):
    """
    將大圖片裁切為指定的 WxH 小圖片並儲存。
    檔名格式: 原始檔名_cropX_cropY.副檔名 (X, Y 代表裁切左上角座標)
    """
    # 使用 np.fromfile 配合 cv2.imdecode，解決讀取中文路徑報錯的問題
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 警告: 無法讀取圖片 {image_path.name}")
        return

    h, w = img.shape[:2]
    filename = image_path.stem
    
    # 決定儲存的副檔名
    ext = out_ext if out_ext else image_path.suffix
    if ext and not ext.startswith('.'):
        ext = f".{ext}"

    # 如果原圖比預期的裁切尺寸還要小，直接原樣儲存或略過
    if h < crop_h or w < crop_w:
        print(f"⚠️ 注意: {image_path.name} 尺寸 ({w}x{h}) 小於指定的裁切尺寸 ({crop_w}x{crop_h})")
        # 這裡選擇直接另存，也可以選擇補邊(padding)
        out_filename = f"{filename}_0_0{ext}"
        out_path = os.path.join(output_dir, out_filename)
        # 使用 cv2.imencode 解決寫入中文路徑報錯的問題
        cv2.imencode(ext, img)[1].tofile(out_path)
        return

    # 使用滑動視窗進行裁切
    for y in range(0, h, crop_h):
        for x in range(0, w, crop_w):
            x1 = x
            y1 = y
            x2 = x + crop_w
            y2 = y + crop_h

            # 若邊界超出原圖，直接以實際邊緣為準，停止延伸，避免產生重疊區域
            # 在此情況下，邊緣切出來的圖片尺寸可能會小於指定的 HxW
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            crop_img = img[y1:y2, x1:x2]
            
            # 組裝輸出檔名: 原圖名稱_X_Y.副檔名
            out_filename = f"{filename}_{x1}_{y1}{ext}"
            out_path = os.path.join(output_dir, out_filename)
            
            # 使用 cv2.imencode 解決寫入中文路徑報錯的問題
            cv2.imencode(ext, crop_img)[1].tofile(out_path)

def batch_process(input_dir, output_dir, crop_w, crop_h, out_ext=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 若輸出資料夾不存在則建立
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    count = 0
    print(f"開始處理資料夾: {input_dir}")
    print(f"設定裁切尺寸: 寬={crop_w}, 高={crop_h}")
    if out_ext:
        print(f"指定輸出格式: {out_ext}")
    print("-" * 40)
    
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # print(f"正在處理: {file_path.name}")
            crop_image(file_path, output_path, crop_w, crop_h, out_ext)
            count += 1
            
    print("-" * 40)
    print(f"✅ 處理完成，共裁切了 {count} 張圖片。")
    print(f"儲存於: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批次將目錄中的大圖片裁切成影像識別所需的小圖片(HxW)")
    parser.add_argument("-i", "--input", type=str, required=True, help="輸入圖片的來源資料夾路徑")
    parser.add_argument("-o", "--output", type=str, required=True, help="處理後圖片的儲存資料夾路徑")
    parser.add_argument("-W", "--width", type=int, default=640, help="裁切後的圖片寬度 (預設: 640)")
    parser.add_argument("-H", "--height", type=int, default=640, help="裁切後的圖片高度 (預設: 640)")
    parser.add_argument("-e", "--ext", type=str, default=None, help="輸出的圖片副檔名 (例如: .png, .jpg)。若未指定則保持與原檔相同")
    
    args = parser.parse_args()
    
    batch_process(args.input, args.output, args.width, args.height, args.ext)
