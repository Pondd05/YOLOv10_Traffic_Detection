import os

def delete_invalid_files():
    # Đường dẫn gốc tới thư mục dữ liệu trên ổ G của bạn
    drive_base = r"G:\My Drive\YOLOv10_AI\Dataset\Kaggle\train_1"
    sub_sets = ['train', 'valid', 'test']
    
    # Các tiền tố tên file cần xóa (chuyển về chữ thường để so sánh chính xác)
    prefixes_to_delete = ('ulu', 'highway')

    print("Bắt đầu quét và xóa các file ảnh + nhãn không hợp lệ trên ổ G...")
    print("-" * 65)

    for s in sub_sets:
        img_dir = os.path.join(drive_base, s, 'images')
        lbl_dir = os.path.join(drive_base, s, 'labels')

        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️ Thư mục tập '{s}' không tồn tại, bỏ qua.")
            continue

        deleted_images = 0
        deleted_labels = 0

        # Quét thư mục ảnh
        for filename in os.listdir(img_dir):
            # Kiểm tra xem tên file có bắt đầu bằng ulu, highway, hay Highway không
            if filename.lower().startswith(prefixes_to_delete):
                img_path = os.path.join(img_dir, filename)
                
                # Tìm file nhãn .txt tương ứng
                base_name = os.path.splitext(filename)[0]
                lbl_path = os.path.join(lbl_dir, base_name + ".txt")

                # Tiến hành xóa file ảnh
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted_images += 1
                except Exception as e:
                    print(f"Lỗi khi xóa ảnh {filename}: {e}")

                # Tiến hành xóa file nhãn tương ứng
                try:
                    if os.path.exists(lbl_path):
                        os.remove(lbl_path)
                        deleted_labels += 1
                except Exception as e:
                    print(f"Lỗi khi xóa nhãn {base_name}.txt: {e}")

        print(f"Tập '{s}': Đã xóa {deleted_images} ảnh và {deleted_labels} file nhãn.")

    print("-" * 65)
    print("Hoàn tất quá trình dọn dẹp dữ liệu!")

if __name__ == '__main__':
    delete_invalid_files()