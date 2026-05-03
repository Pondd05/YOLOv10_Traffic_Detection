import os
from ultralytics import YOLOv10
import torch

def train_full_with_wandb():
    # 1. Đường dẫn file trọng số tuyệt đối trong thư mục của bạn
    weights_path = os.path.join(os.getcwd(), "yolov10m.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file {weights_path}!")

    # 2. Kiểm tra phần cứng (RTX 3050)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Đang Huấn luyện Chính thức trên thiết bị: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 3. Kích hoạt chế độ online để đồng bộ dữ liệu lên W&B
    os.environ["WANDB_MODE"] = "online"

    # 4. Khởi tạo mô hình YOLOv10m
    model = YOLOv10(weights_path)

    # 5. Cấu hình tham số huấn luyện chính thức
    model.train(
        data="G:/My Drive/YOLOv10_AI/Dataset/Kaggle/train_1/data.yaml", 
        epochs=100,               # 100 Epochs để mô hình học sâu và hội tụ tốt nhất
        imgsz=640,                # Kích thước ảnh tiêu chuẩn
        batch=4,                  # Giữ nguyên batch=4 để an toàn cho 4GB VRAM của bạn
        fraction=1.0,             # Sử dụng 100% dữ liệu mới để train
        
        # --- Cấu hình Project trên W&B ---
        project="YOLOv10_Traffic_Full", 
        name="Kaggle_Full_Train_V2",  # Tên phiên chạy chính thức
        device=device,            

        # --- Chiến lược huấn luyện ---
        amp=True,                 # Tiết kiệm VRAM và tăng tốc
        patience=50,              # Dừng sớm nếu sau 50 epoch không cải thiện
        save=True,                # Lưu file weights (.pt)
        exist_ok=True,            
        pretrained=True,          
        workers=2,                # 2 luồng đọc dữ liệu giúp CPU chạy êm ái
    )

    print("Quá trình huấn luyện chính thức hoàn tất!")

if __name__ == '__main__':
    train_full_with_wandb()