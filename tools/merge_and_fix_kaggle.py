import os

def fix_frame_labels():
    # 1. Đường dẫn gốc tới thư mục dữ liệu trên ổ G
    drive_base = r"G:\My Drive\YOLOv10_AI\Dataset\Kaggle\train_1"
    sub_sets = ['train', 'valid', 'test']
    
    print("Bắt đầu quét và chuyển đổi Class ID cho các file 'frame_'...")
    print("-" * 65)

    for s in sub_sets:
        lbl_dir = os.path.join(drive_base, s, 'labels')

        # Kiểm tra thư mục labels có tồn tại không
        if not os.path.exists(lbl_dir):
            continue

        fixed_files = 0

        # Quét qua toàn bộ file trong thư mục labels
        for filename in os.listdir(lbl_dir):
            # Chỉ xử lý các file nhãn có tên bắt đầu bằng frame_
            if filename.lower().startswith('frame_') and filename.endswith('.txt'):
                file_path = os.path.join(lbl_dir, filename)

                # Đọc nội dung file nhãn hiện tại
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                has_change = False

                for line in lines:
                    parts = line.split()
                    if len(parts) > 0:
                        old_class = int(parts[0])
                        new_class = old_class  # Mặc định giữ nguyên nếu không khớp

                        # Ánh xạ lại Class ID theo đúng yêu cầu của bạn:
                        if old_class == 2:    # Xe máy cũ -> chuyển về 0
                            new_class = 0
                            has_change = True
                        elif old_class == 1:  # Ô tô cũ -> giữ nguyên là 1
                            new_class = 1
                        elif old_class == 0:  # Xe bus cũ -> chuyển về 2
                            new_class = 2
                            has_change = True
                        elif old_class in (3, 4):  # Xe tải cũ -> chuyển hết về 3
                            new_class = 3
                            has_change = True

                        # Cập nhật lại dòng với Class ID mới
                        parts[0] = str(new_class)
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)

                # Chỉ ghi đè file nếu có sự thay đổi về Class ID
                if has_change:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    fixed_files += 1

        print(f"Tập '{s}': Đã sửa thành công {fixed_files} file nhãn 'frame_'.")

    print("-" * 65)
    print("Hoàn tất quá trình đồng bộ Class ID!")

if __name__ == '__main__':
    fix_frame_labels()