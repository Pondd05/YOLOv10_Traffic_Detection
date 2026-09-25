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
    # 1. KHỞI TẠO ĐƯỜNG DẪN VÀ PHẦN CỨNG
    current_folder = os.path.dirname(os.path.abspath(__file__))
    local_model = os.path.join(current_folder, "best.pt")
    custom_model = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"
    MODEL_PATH = custom_model if os.path.exists(custom_model) else local_model

    tracker_file = os.path.join(current_folder, "custom_bytetrack.yaml")
    TRACKER_CONFIG = tracker_file if os.path.exists(tracker_file) else "bytetrack.yaml"

    # Cho phép truyền đường dẫn video qua tham số: python main.py [duong_dan_video]
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    else:
        # Tự động tìm video trong thư mục 'video test' nếu có
        video_test_dir = os.path.join(current_folder, "video test")
        test_videos = []
        if os.path.exists(video_test_dir):
            test_videos = [os.path.join(video_test_dir, f) for f in os.listdir(video_test_dir)
                           if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        if test_videos:
            VIDEO_PATH = test_videos[0]
            print(f"[SYSTEM] Tự động chọn video kiểm nghiệm: {os.path.basename(VIDEO_PATH)}")
        else:
            VIDEO_PATH = os.path.join(current_folder, "data", "traffic_vietnam.mp4")

    # Ép buộc luồng tính toán chạy trên nhân CUDA của GPU NVIDIA RTX 3050 Laptop
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Đang kích hoạt Pipeline trên thiết bị: {device.upper()}")
    print(f"[SYSTEM] Sử dụng trọng số mô hình: {MODEL_PATH}")

    # Nạp mô hình YOLOv10m chuẩn hóa tác vụ detect (Hỗ trợ cấu trúc NMS-Free)
    model = YOLOv10(MODEL_PATH)
    
    # Khởi tạo bộ giải mã video đầu vào qua OpenCV
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở luồng đọc file video tại: {VIDEO_PATH}")
        print("[HƯỚNG DẪN] Bạn có thể chạy kèm đường dẫn video, ví dụ:")
        print("            python main.py path/to/video.mp4")
        return

    # Khống chế bộ đệm khung hình đầu vào để triệt tiêu hoàn toàn hiện tượng trễ hình (Latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # CẤU HÌNH BẢNG MÀU VÀ ĐẶC TẢ ĐƯỜNG KẺ
    LANE_PALETTE = [
        (0, 200, 0),      # Xanh lá
        (255, 140, 0),    # Cam
        (0, 180, 255),    # Vàng kim
        (200, 0, 200),    # Tím
        (0, 255, 255),    # Vàng chanh
        (255, 100, 100),  # Xanh dương nhạt
    ]
    BOUNDARY_TYPES = {
        "solid_line": {"color": (0, 0, 255), "thickness": 3},
        "double_solid": {"color": (0, 0, 200), "thickness": 4},
        "dashed_line": {"color": (0, 255, 255), "thickness": 2},
        "curb": {"color": (200, 200, 200), "thickness": 3}
    }

    def load_lane_config(config_path, frame_w, frame_h):
        if not os.path.exists(config_path):
            return None, [], []
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lanes = []
        for lane in cfg.get("lanes", []):
            poly = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in lane["polygon"]], np.int32)
            lanes.append({**lane, "polygon_px": poly})
        boundaries = []
        for b in cfg.get("boundaries", []):
            pts = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in b["points"]], np.int32)
            boundaries.append({**b, "points_px": pts})
        return cfg, lanes, boundaries

    def check_vehicle_lane_violation(point_xy, class_name, lanes):
        px, py = float(point_xy[0]), float(point_xy[1])
        for lane in lanes:
            if cv2.pointPolygonTest(lane["polygon_px"], (px, py), False) >= 0:
                allowed = lane.get("allowed_classes", [])
                if not allowed or class_name in allowed:
                    return False, lane.get("name", "Lane"), "Hop le"
                else:
                    return True, lane.get("name", "Lane"), f"Xe {class_name} sai lan"
        return False, "Ngoai vung", "Khong xac dinh"

    # TỰ ĐỘNG NẠP CẤU HÌNH LÀN THỦ CÔNG TỪ lane_config.json NẾU CÓ
    config_file = os.path.join(current_folder, "lane_config.json")
    lane_cfg, lanes_data, boundaries_data = None, [], []

    if os.path.exists(config_file):
        lane_cfg, lanes_data, boundaries_data = load_lane_config(config_file, width, height)
        print(f"[SYSTEM] Đã nạp thành công {len(lanes_data)} làn đường và {len(boundaries_data)} vạch kẻ từ '{config_file}'")
    else:
        print("[SYSTEM] Không tìm thấy 'lane_config.json'. Sử dụng vùng ROI đa giác cấm mặc định.")

    # ĐỊNH NGHĨA VÙNG ĐA GIÁC ẢO ROI DỰ PHÒNG (NẾU CHƯA CÓ lane_config.json)
    forbidden_lane_roi = np.array([
        [int(width * 0.15), height],
        [int(width * 0.40), int(height * 0.55)],
        [int(width * 0.65), int(height * 0.55)],
        [int(width * 0.90), height]
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
        
        # Vẽ các làn đường từ lane_config.json hoặc ROI mặc định
        if lanes_data:
            overlay = frame_processed.copy()
            for i, lane in enumerate(lanes_data):
                poly = lane["polygon_px"]
                color = LANE_PALETTE[i % len(LANE_PALETTE)]
                cv2.fillPoly(overlay, [poly], color)
                cv2.polylines(frame_processed, [poly], True, color, 2)
                cx, cy = poly.mean(axis=0).astype(int)
                cv2.putText(frame_processed, lane.get("name", ""), (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            frame_processed = cv2.addWeighted(overlay, 0.2, frame_processed, 0.8, 0)

            for b in boundaries_data:
                pts = b["points_px"]
                b_info = BOUNDARY_TYPES.get(b.get("type", "solid_line"), BOUNDARY_TYPES["solid_line"])
                cv2.polylines(frame_processed, [pts], False, b_info["color"], b_info["thickness"], cv2.LINE_AA)
        else:
            # Vẽ ranh giới đa giác cấm mặc định màu vàng rực
            cv2.polylines(frame_processed, [forbidden_lane_roi], isClosed=True, color=(0, 255, 255), thickness=3)

        # BƯỚC 3: NẠP MA TRẬN ĐÃ LÀM SẠCH VÀO MÔ HÌNH AI (YOLOv10 + Custom ByteTrack)
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

                box_color = (0, 255, 0) # Mặc định màu XANH LÁ CÂY = Hợp lệ
                status_text = "OK"

                # KIỂM TRA SAI LÀN: Ưu tiên dùng lane_config.json nếu có
                if lanes_data:
                    is_violation, lane_name, reason = check_vehicle_lane_violation((x_foot, y_foot), class_name, lanes_data)
                    if is_violation:
                        violation_counter[obj_id] = violation_counter.get(obj_id, 0) + 1
                        if violation_counter[obj_id] >= 10: # Lấn làn liên tục > 10 frames
                            box_color = (0, 0, 255) # ĐỔI SANG MÀU ĐỎ
                            status_text = f"SAI LAN ({lane_name})"
                    else:
                        if obj_id in violation_counter:
                            violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)
                        status_text = f"{lane_name}"
                else:
                    # Kiểm tra theo ROI đa giác cấm mặc định
                    is_inside = cv2.pointPolygonTest(forbidden_lane_roi, (x_foot, y_foot), False)
                    if is_inside >= 0:
                        violation_counter[obj_id] = violation_counter.get(obj_id, 0) + 1
                        if violation_counter[obj_id] >= 15:
                            box_color = (0, 0, 255)
                            status_text = "VIOLATION"
                    else:
                        if obj_id in violation_counter:
                            violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)

                # Render các thành phần đồ họa trực quan lên màn hình Desktop
                cv2.circle(frame_processed, (x_foot, y_foot), 5, (0, 0, 255), -1) # Chấm đỏ gầm xe
                cv2.rectangle(frame_processed, (x_min, y_min), (x_max, y_max), box_color, 2) # Khung bao xe
                
                label = f"{class_name} #{obj_id} [{status_text}]"
                cv2.putText(frame_processed, label, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # Giải phóng RAM: Xóa vết ID của các phương tiện đã ra khỏi khung hình
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