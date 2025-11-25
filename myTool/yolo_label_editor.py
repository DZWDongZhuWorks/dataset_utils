import os
import argparse

def process_annotation_lines(lines, src_classes: list[str], dst_class: str = None):
    """
    處理一個檔案的所有行，回傳新的行列表：
    - 如果未提供 dst_class：刪除所有以 src_classes 中任一項開頭的行
    - 如果提供 dst_class：把所有目標行的類別改成 dst_class
    """
    new_lines = []
    src_set = set(src_classes)

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        cls = parts[0]

        if cls in src_set:
            if dst_class is None:
                # 刪除
                continue
            else:
                # 改類別
                parts[0] = dst_class
                line = ' '.join(parts) + '\n'

        new_lines.append(line)

    return new_lines


def batch_process_folder(folder_path: str, src_classes: list[str], dst_class: str = None,
                         ext: str = '.txt', output_folder: str = None) -> None:

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(ext):
                continue

            in_path = os.path.join(root, fname)

            # 讀檔
            with open(in_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = process_annotation_lines(lines, src_classes, dst_class)

            # 決定輸出位置
            if output_folder:
                rel_dir = os.path.relpath(root, folder_path)
                out_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
            else:
                out_path = in_path

            # 寫出
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            action = f"{src_classes} → {dst_class}" if dst_class else f"刪除 {src_classes}"
            print(f"[{action}] {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="刪除或轉換 YOLO 標註檔中的類別編號，並可選擇輸出到新資料夾")
    parser.add_argument('-f', '--folder', required=True,
                        help="要處理的資料夾路徑")
    parser.add_argument('-c', '--class', dest='src_classes', nargs='+', required=True,
                        help="原本的類別編號，可一次指定多個，例如: -c 3 5 7")
    parser.add_argument('-t', '--to-class', dest='dst_class',
                        help="要轉換成的類別編號 (例如 '5')，不指定則刪除 src_class")
    parser.add_argument('-e', '--ext', default='.txt',
                        help="要處理的檔案副檔名，預設為 .txt")
    parser.add_argument('-o', '--output-folder', dest='output_folder',
                        help="輸出資料夾路徑，若指定則不覆蓋原始檔案")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"錯誤：找不到資料夾 {args.folder}")
        return

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    batch_process_folder(
        folder_path=args.folder,
        src_class=args.src_class,
        dst_class=args.dst_class,
        ext=args.ext,
        output_folder=args.output_folder
    )
    print("全部處理完成。")

if __name__ == '__main__':
    main()
