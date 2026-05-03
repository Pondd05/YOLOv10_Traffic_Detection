import os

def count_dataset_images():
    # Path gốc dữ liệu ổ G 
    base_path = "G:/My Drive/YOLOv10_AI/Dataset/Kaggle/train_1"
    
    # Danh sách các tập dữ liệu cần kiểm tra
    sub_folders = ['train', 'valid', 'test']
    
    print("="*45)
    print("THỐNG KÊ SỐ LƯỢNG ẢNH TRONG DATASET:")
    print("="*45)
    
    total_images = 0
    
    for folder in sub_folders:
        img_dir = os.path.join(base_path, folder, 'images')
        
        # Kiểm tra xem thư mục có tồn tại không
        if os.path.exists(img_dir):
            # Lọc các file có định dạng ảnh phổ biến
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            num_images = len(images)
            total_images += num_images
            print(f"Thư mục '{folder}/images': {num_images} ảnh")
        else:
            print(f"Thư mục '{folder}/images' KHÔNG tồn tại!")
            
    print("-"*45)
    print(f"Tổng cộng tất cả các tập: {total_images} ảnh")
    print("="*45)

if __name__ == '__main__':
    count_dataset_images()