import os
import sys
from ultralytics import YOLOv10

# CẤU HÌNH HỆ THỐNG: Khử triệt để lỗi ma trận NMS '.shape' của YOLOv10
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != current_dir]
miniconda_packages = [p for p in sys.path if "site-packages" in p and "miniconda3" in p.lower()]
if miniconda_packages:
    sys.path.insert(0, miniconda_packages[0])

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# BƯỚC 2 & 3: HÀM TIỀN XỬ LÝ THÍCH ỨNG TỰ ĐỘNG (LÀM SẠCH DỮ LIỆU)
def auto_adaptive_preprocessing(frame):
    """
    Tiếp nhận ma trận ảnh thô, tự động phân tích thống kê toán học Histogram
    để đưa ra quyết định làm sạch dữ liệu (Tăng sáng/Khử mờ) thích ứng.
    """
    # 2a. Chuyển đổi hệ màu sang GRAY (Ảnh xám) để tính toán toán học tốc độ cao
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2b. Tính toán giá trị trung bình (Mean) và độ lệch chuẩn (StdDev) của các pixel
    mean_brightness, std_dev = cv2.meanStdDev(gray)
    mean_brightness = mean_brightness[0][0]
    std_dev = std_dev[0][0]
    
    # Tạo một bản sao ma trận để xử lý, bảo toàn khung hình gốc
    cleaned_frame = frame.copy()
    
    # TRƯỜNG HỢP 1: Hệ thống phát hiện ảnh bị THIẾU SÁNG / BAN ĐÊM (Mean < 85)
    if mean_brightness < 85:
        # Chuyển sang hệ màu LAB để bóc tách riêng kênh độ sáng (L) ra khỏi màu sắc (A, B)
        lab = cv2.cvtColor(cleaned_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Áp dụng thuật toán CLAHE để cân bằng độ tương phản cục bộ, làm rõ vật thể trong bóng tối
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Gộp các kênh ma trận lại và chuyển trả về hệ màu BGR tiêu chuẩn
        cleaned_frame = cv2.merge((cl, a, b))
        cleaned_frame = cv2.cvtColor(cleaned_frame, cv2.COLOR_LAB2BGR)
        
    # TRƯỜNG HỢP 2: Hệ thống phát hiện ảnh bị MỜ NHÒE / TƯƠNG PHẢN THẤP (StdDev < 45)
    if std_dev < 45:
        # Áp dụng phép toán tích chập ma trận (Matrix Convolution) với nhân Laplacian Kernel
        # Kỹ thuật này giúp làm nét sắc cạnh biên cấu trúc vật lý (khung xe, bánh xe)
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
        cleaned_frame = cv2.filter2D(cleaned_frame, -1, kernel)
        
    return cleaned_frame


def main():
    # 1. KHỞI TẠO ĐƯỜNG DẪN TUYỆT ĐỐI VÀ PHẦN CỨNG (Đồng bộ ổ D của bạn)
    MODEL_PATH = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"
    TRACKER_CONFIG = "custom_bytetrack.yaml"
    VIDEO_PATH = "data/traffic_vietnam.mp4" # <--- Sửa lại tên file video thực tế của bạn tại đây

    if not os.path.exists(TRACKER_CONFIG):
        TRACKER_CONFIG = "bytetrack.yaml"

    # Ép buộc luồng tính toán chạy trên nhân CUDA của GPU NVIDIA RTX 3050 Laptop
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Đang kích hoạt Pipeline trên thiết bị: {device.upper()}")

    # Nạp mô hình YOLOv10m chuẩn hóa tác vụ detect
    model = YOLO(MODEL_PATH, task="detect")
    
    # Khởi tạo bộ giải mã video đầu vào qua OpenCV
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở luồng đọc file video tại: {VIDEO_PATH}")
        return

    # Khống chế bộ đệm khung hình đầu vào để triệt tiêu hoàn toàn hiện tượng trễ hình (Latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # ĐỊNH NGHĨA VÙNG ĐA GIÁC ẢO ROI PHÂN LÀN ĐƯỜNG CẤM
    forbidden_lane_roi = np.array([
        [220, 720],   # Điểm 1: Dưới cùng bên trái
        [520, 420],   # Điểm 2: Trên cùng bên trái (Điểm thắt phối cảnh)
        [820, 420],   # Điểm 3: Trên cùng bên phải (Điểm thắt phối cảnh)
        [1120, 720]   # Điểm 4: Dưới cùng bên phải
    ], np.int32)

    # Bộ đệm thời gian lưu vết khung hình vi phạm tĩnh để lọc nhiễu giật tọa độ
    violation_counter = {}

    print("[SYSTEM] Luồng xử lý thời gian thực bắt đầu vận hành. Nhấn 'q' để thoát...")

    # 4. VÒNG LẶP PIPELINE INFERENCE THỜI GIAN THỰC
    while cap.isOpened():
        # BƯỚC 1: Tiếp nhận ma trận điểm ảnh thô (Raw Matrix uint8 0-255) từ Camera
        success, raw_frame = cap.read()
        if not success:
            print("[SYSTEM] Đã xử lý hết luồng video thực nghiệm.")
            break

        # BƯỚC 2: Thực hiện chuỗi giải thuật LÀM SẠCH DỮ LIỆU thích ứng chủ động
        frame_processed = auto_adaptive_preprocessing(raw_frame)
        
        # Vẽ ranh giới không gian đa giác ảo màu vàng rực lên khung hình
        cv2.polylines(frame_processed, [forbidden_lane_roi], isClosed=True, color=(0, 255, 255), thickness=3)

        # BƯỚC 3: NẠP MA TRẬN ĐÃ LÀM SẠCH VÀO MÔ HÌNH AI (YOLOv10 + Custom ByteTrack)
        # Lưu ý: Tại đây, hàm .track() sẽ tự động chạy ngầm dưới C++/CUDA các bước:
        # Letterbox Resize về 640x640 -> Type Casting sang Float32 -> Min-Max Scaling chia cho 255.0 về khoảng [0.0, 1.0]
        # Tham số half=True ép nén tiếp xuống dữ liệu FP16 để tối ưu băng thông cho GPU 4GB của bạn.
        results = model.track(
            source=frame_processed, persist=True, tracker=TRACKER_CONFIG,
            imgsz=640, half=True, device=device, verbose=False
        )

        # Tính toán tốc độ xử lý thực tế của Pipeline (FPS)
        speed = results[0].speed
        latency_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        # BƯỚC 4: TRÍCH XUẤT MA TRẬN PHÂN ĐỊNH VÀ LOGIC HÌNH HỌC ROI
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()            # Ma trận tọa độ phẳng Bounding Box
            ids = results[0].boxes.id.cpu().numpy().astype(int)    # Mảng danh sách mã định danh tĩnh ID từ ByteTrack
            clss = results[0].boxes.cls.cpu().numpy().astype(int)  # Mảng Class ID phương tiện
            
            current_frame_ids = set(ids)

            for box, obj_id, cls_id in zip(boxes, ids, clss):
                x_min, y_min, x_max, y_max = map(int, box)
                class_name = model.names[cls_id]

                # TOÁN HỌC KHÔNG GIAN: Xác định tọa độ "Điểm chạm đất" (Vị trí bánh xe tiếp xúc mặt đường)
                x_foot = int((x_min + x_max) / 2)
                y_foot = int(y_max)

                # GIẢI THUẬT POINT-IN-POLYGON: Kiểm tra trạng thái quan hệ không gian thực thể
                is_inside = cv2.pointPolygonTest(forbidden_lane_roi, (x_foot, y_foot), False)

                box_color = (0, 255, 0) # Mặc định màu XANH LÁ CÂY = Di chuyển hợp lệ [OK]
                status_text = "OK"

                if is_inside >= 0:
                    # Nếu điểm chạm đất lọt vào đa giác cấm, lũy tiến bộ đếm khung hình lên +1
                    violation_counter[obj_id] = violation_counter.get(obj_id, 0) + 1
                    
                    # Bộ lọc thời gian: Phải lấn làn liên tục 15 frames (~0.5 giây) mới kích hoạt cảnh báo
                    if violation_counter[obj_id] >= 15:
                        box_color = (0, 0, 255) # CHUYỂN TOÀN BỘ KHUNG BAO SANG MÀU ĐỎ RỰC
                        status_text = "VIOLATION"
                else:
                    # Nếu xe thoát khỏi vùng cấm, giảm dần bộ đếm về lại 0
                    if obj_id in violation_counter:
                        violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)

                # Render các thành phần đồ họa trực quan lên màn hình Desktop
                cv2.circle(frame_processed, (x_foot, y_foot), 6, (0, 0, 255), -1) # Chấm đỏ gầm xe
                cv2.rectangle(frame_processed, (x_min, y_min), (x_max, y_max), box_color, 2) # Khung bao xe
                
                label = f"{class_name} #{obj_id} [{status_text}]"
                cv2.putText(frame_processed, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # Giải phóng RAM rác: Xóa vết ID của các phương tiện đã di chuyển ra khỏi góc quay camera
            for cached_id in list(violation_counter.keys()):
                if cached_id not in current_frame_ids:
                    del violation_counter[cached_id]

        # In thông số hiệu năng FPS lên góc màn hình đồ họa
        cv2.putText(frame_processed, f"Pipeline Performance: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Đẩy ma trận ảnh đồ họa hoàn chỉnh hiển thị ra ngoài cửa sổ Desktop của hệ điều hành Windows
        cv2.imshow("UIT Traffic Enforcement System - Main.py High-Speed Pipeline", frame_processed)

        # Lắng nghe sự kiện bàn phím: Bấm phím 'q' để dừng tiến trình chủ động
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SYSTEM] Luồng xử lý được ngắt chủ động từ bàn phím.")
            break

    # Giải phóng hoàn toàn bộ giải mã video và đóng tài nguyên phần cứng đồ họa
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Đã giải phóng hoàn toàn GPU và kết thúc chương trình mượt mà.")

if __name__ == "__main__":
    main()