import os

def create_blank_txt(txt_path):
    """
    建立一個空白的 txt 檔案
    """
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 建立空白檔案，即檔案內不寫入任何內容
        pass
    print(f"已建立空白 txt：{txt_path}")

def process_folder(folder_path):
    """
    遍歷指定資料夾及其所有子資料夾內的 jpg 檔案，
    若對應的 txt 檔案不存在，則建立一個空白的 txt 檔案。
    """
    # os.walk 會幫你遞迴走遍所有子目錄
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".jpg"):
                file_base = os.path.splitext(file)[0]
                txt_filename = file_base + ".txt"
                txt_path = os.path.join(root, txt_filename)  # 注意用 root，而不是最外層 folder_path
                if not os.path.exists(txt_path):
                    create_blank_txt(txt_path)

if __name__ == "__main__":
    folder = input("請輸入資料夾路徑：")
    process_folder(folder)
