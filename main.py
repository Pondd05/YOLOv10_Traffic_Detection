import cv2
import torch
from ultralytics import YOLOv10

# 1. CẤU HÌNH CÁC THÔNG SỐ VÀ ĐƯỜNG DẪN
# Đường dẫn tệp trọng số tốt nhất 
MODEL_PATH = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"

# Đường dẫn input(có thể thay đổi)
VIDEO_INPUT = r"G:\My Drive\YOLOv10_AI\Dataset\Kaggle\train_1\test\images\Sample_Video_HighQuality.mp4"
VIDEO_OUTPUT = r"G:\My Drive\YOLOv10_AI\Dataset\Kaggle\train_1\test\images\output_detection_HQ_1280.mp4"

CLASS_NAMES = {0: "Car", 1: "Motorbike", 2: "Bus", 3: "Truck"}

# 2. LUỒNG XỬ LÝ CHÍNH (MAIN PIPELINE)
def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Đang khởi tạo mô hình trên thiết bị: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLOv10(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở tệp video tại {VIDEO_INPUT}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

    print(f"Đang xử lý video... Kích thước: {width}x{height}, FPS: {fps}")
    print("Bấm phím 'q' tại cửa sổ hiển thị để dừng chương trình.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sử dụng YOLOv10 và ByteTrack với imgsz=1280 và conf=0.2
        results = model.track(
            source=frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            device=device,
            imgsz=1280,  # Tăng kích thước ảnh đầu vào để giữ chi tiết vật thể nhỏ
            conf=0.2,    # Giảm ngưỡng tin cậy để bắt xe ở xa
            iou=0.6,
            verbose=False
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls_id in zip(boxes, ids, clss):
                if cls_id not in CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{CLASS_NAMES[cls_id]} #{obj_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Nhan dien phuong tien - YOLOv10 + ByteTrack", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Đã xử lý xong! Video kết quả lưu tại: {VIDEO_OUTPUT}")

if __name__ == '__main__':
    main()