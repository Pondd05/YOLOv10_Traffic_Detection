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
from collections import defaultdict, deque

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


class VehicleTrajectoryManager:
    """
    Quản lý quỹ đạo chuyển động (Vehicle Trajectory) và độ ổn định (Stability):
    - Lưu chuỗi lịch sử Bottom-Center (điểm chạm đất) của từng xe theo track_id.
    - Áp dụng bộ lọc trượt Moving Average Filter để khử hiện tượng rung lắc (jitter) do Bounding Box co giãn.
    - Lưu vết đồng thời trong không gian Camera và không gian BEV (Bird's-Eye View).
    - Hỗ trợ cơ chế Temporal Voting để ổn định việc gán làn xe (Lane Assignment), tránh chớp nháy ở ranh giới.
    """
    def __init__(self, max_history=30, smoothing_window=5):
        self.max_history = max_history
        self.smoothing_window = smoothing_window
        self.trajectories = defaultdict(lambda: deque(maxlen=self.max_history))
        self.bev_trajectories = defaultdict(lambda: deque(maxlen=self.max_history))
        self.lane_history = defaultdict(lambda: deque(maxlen=10))

    def update(self, obj_id, point_cam, point_bev=None):
        self.trajectories[obj_id].append(point_cam)
        if point_bev is not None:
            self.bev_trajectories[obj_id].append(point_bev)

    def get_smoothed_trajectory(self, obj_id, use_bev=False):
        raw_pts = self.bev_trajectories[obj_id] if use_bev else self.trajectories[obj_id]
        if len(raw_pts) < 3:
            return list(raw_pts)
        
        # Áp dụng giải thuật Moving Average trên chuỗi tọa độ theo thời gian
        pts_array = np.array(raw_pts, dtype=np.float32)
        smoothed = []
        w = self.smoothing_window
        for i in range(len(pts_array)):
            start_idx = max(0, i - w + 1)
            window_pts = pts_array[start_idx : i + 1]
            avg_pt = np.mean(window_pts, axis=0)
            smoothed.append((int(avg_pt[0]), int(avg_pt[1])))
        return smoothed

    def get_latest_smoothed_point(self, obj_id, use_bev=False):
        smoothed = self.get_smoothed_trajectory(obj_id, use_bev)
        return smoothed[-1] if smoothed else None

    def assign_lane(self, obj_id, current_lane_name):
        """ Majority voting trên cửa sổ 10 frame gần nhất để chống nhiễu đổi làn tức thời """
        self.lane_history[obj_id].append(current_lane_name)
        lanes = list(self.lane_history[obj_id])
        return max(set(lanes), key=lanes.count)

    def cleanup(self, active_ids):
        """ Giải phóng bộ nhớ của các phương tiện đã rời khỏi tầm quan sát """
        tracked_ids = list(self.trajectories.keys())
        for tid in tracked_ids:
            if tid not in active_ids:
                del self.trajectories[tid]
                if tid in self.bev_trajectories:
                    del self.bev_trajectories[tid]
                if tid in self.lane_history:
                    del self.lane_history[tid]


class BEVTransformer:
    def __init__(self, lanes_data, frame_w, frame_h, bev_w=550, bev_h=600):
        self.bev_w = bev_w
        self.bev_h = bev_h
        
        # Extract 4 outermost corners from all lanes to form the Source Quad
        all_points = []
        for lane in lanes_data:
            all_points.extend(lane["polygon_px"])
        all_points = np.array(all_points)
        
        if len(all_points) < 4:
            self.M = None
            return

        # Sort points by Y to separate Top and Bottom points
        sorted_y = all_points[all_points[:, 1].argsort()]
        top_points = sorted_y[:len(sorted_y)//2]
        bottom_points = sorted_y[len(sorted_y)//2:]
        
        # Find corners
        tl = top_points[top_points[:, 0].argmin()]
        tr = top_points[top_points[:, 0].argmax()]
        bl = bottom_points[bottom_points[:, 0].argmin()]
        br = bottom_points[bottom_points[:, 0].argmax()]
        
        self.src_pts = np.float32([tl, tr, br, bl])
        self.dst_pts = np.float32([
            [0, 0], 
            [self.bev_w, 0], 
            [self.bev_w, self.bev_h], 
            [0, self.bev_h]
        ])
        
        # DLT Algorithm to compute Perspective Transform Matrix
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        # Transform Lane Polygons to BEV space
        self.bev_lanes = []
        for lane in lanes_data:
            pts = np.float32(lane["polygon_px"]).reshape(-1, 1, 2)
            bev_pts = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2).astype(np.int32)
            self.bev_lanes.append({**lane, "polygon_px": bev_pts})
            
    def transform_point(self, pt):
        if self.M is None: return pt
        p = np.float32([[[pt[0], pt[1]]]])
        p_bev = cv2.perspectiveTransform(p, self.M)
        return int(p_bev[0][0][0]), int(p_bev[0][0][1])

    def draw_bev_map(self, vehicles_bev, traj_manager=None):
        """ Draw a top-down radar map of the lanes, trajectories, and vehicles """
        bev_map = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
        
        # Draw Lanes in BEV
        for i, lane in enumerate(self.bev_lanes):
            poly = lane["polygon_px"]
            color = (45, 45, 45) if i % 2 == 0 else (65, 65, 65)
            cv2.fillPoly(bev_map, [poly], color)
            cv2.polylines(bev_map, [poly], True, (130, 130, 130), 1)
            # Ghi nhãn tên làn ngắn gọn lên bản đồ BEV
            cx, cy = poly.mean(axis=0).astype(int)
            lane_code = lane.get("name", f"L{i+1}").split(" - ")[0]
            cv2.putText(bev_map, lane_code, (max(5, cx - 25), cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
            
        # Draw Trajectories in BEV (Vệt quỹ đạo trên bản đồ chim nhìn)
        if traj_manager is not None:
            for (x, y, color, obj_id, label_text) in vehicles_bev:
                smoothed_bev = traj_manager.get_smoothed_trajectory(obj_id, use_bev=True)
                if len(smoothed_bev) >= 2:
                    pts_arr = np.array(smoothed_bev, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(bev_map, [pts_arr], False, color, 2, cv2.LINE_AA)

        # Draw Vehicles in BEV
        for (x, y, color, obj_id, label_text) in vehicles_bev:
            cv2.circle(bev_map, (x, y), 5, color, -1)
            cv2.circle(bev_map, (x, y), 7, (255, 255, 255), 1)
            cv2.putText(bev_map, f"#{obj_id}", (x + 9, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            
        return bev_map


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
            # Ưu tiên định dạng chuẩn .mp4 trước các file khác
            test_videos.sort(key=lambda x: (not x.lower().endswith('.mp4'), x))
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

    def check_vehicle_lane_violation(point_xy, class_name, lanes, strict_scenario=None):
        px, py = float(point_xy[0]), float(point_xy[1])
        num_lanes = len(lanes)
        scenario = strict_scenario if strict_scenario is not None else (num_lanes if num_lanes in [3, 4] else None)
        
        for idx, lane in enumerate(lanes):
            if cv2.pointPolygonTest(lane["polygon_px"], (px, py), False) >= 0:
                lane_name = lane.get("name", f"Lan {idx+1}")
                
                # Áp dụng bộ quy tắc phân làn nghiêm ngặt của Thầy (Kịch bản 3 làn / 4 làn)
                if scenario == 4:
                    # 4 làn: L1 (trái cùng)=truck, L2=bus, L3=car, L4 (phải cùng)=motorcycle
                    strict_rules = {
                        0: ["truck"],
                        1: ["bus"],
                        2: ["car"],
                        3: ["motorcycle"]
                    }
                    allowed = strict_rules.get(idx, [])
                elif scenario == 3:
                    # 3 làn: L1 (trái)=truck, bus; L2 (giữa)=car; L3 (phải)=motorcycle
                    strict_rules = {
                        0: ["truck", "bus"],
                        1: ["car"],
                        2: ["motorcycle"]
                    }
                    allowed = strict_rules.get(idx, [])
                else:
                    allowed = lane.get("allowed_classes", [])

                if not allowed or class_name in allowed:
                    return False, lane_name, "Hop le", allowed
                else:
                    allowed_str = ", ".join(allowed) if allowed else "None"
                    return True, lane_name, f"Xe {class_name} sai lan (Chi cho {allowed_str})", allowed
        return False, "Ngoai vung", "Khong xac dinh", []

    # TỰ ĐỘNG NẠP CẤU HÌNH LÀN THỦ CÔNG TỪ lane_config.json NẾU CÓ
    config_file = os.path.join(current_folder, "lane_config.json")
    lane_cfg, lanes_data, boundaries_data = None, [], []

    if os.path.exists(config_file):
        lane_cfg, lanes_data, boundaries_data = load_lane_config(config_file, width, height)
        print(f"[SYSTEM] Đã nạp thành công {len(lanes_data)} làn đường và {len(boundaries_data)} vạch kẻ từ '{config_file}'")
    else:
        print("[SYSTEM] Không tìm thấy 'lane_config.json'. Sử dụng vùng ROI đa giác cấm mặc định.")

    bev_transformer = None
    if lanes_data:
        bev_transformer = BEVTransformer(lanes_data, width, height)
        if bev_transformer.M is not None:
            print("[SYSTEM] Đã khởi tạo thành công không gian Bird's-Eye View (BEV).")

    # ĐỊNH NGHĨA VÙNG ĐA GIÁC ẢO ROI DỰ PHÒNG (NẾU CHƯA CÓ lane_config.json)
    forbidden_lane_roi = np.array([
        [int(width * 0.15), height],
        [int(width * 0.40), int(height * 0.55)],
        [int(width * 0.65), int(height * 0.55)],
        [int(width * 0.90), height]
    ], np.int32)

    # Khởi tạo Bộ quản lý quỹ đạo phương tiện (Vehicle Trajectory Manager)
    traj_manager = VehicleTrajectoryManager(max_history=35, smoothing_window=5)

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
        # Tối ưu conf=0.18 và iou=0.45 để phát hiện nhạy bén toàn bộ xe máy, xe đông và vật thể ở xa
        results = model.track(
            source=frame_processed, persist=True, tracker=TRACKER_CONFIG,
            imgsz=640, conf=0.18, iou=0.45, half=True, device=device, verbose=False
        )

        # Tính toán tốc độ xử lý thực tế của Pipeline (FPS)
        speed = results[0].speed
        latency_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        vehicles_bev = []
        
        # BƯỚC 4: TRÍCH XUẤT MA TRẬN PHÂN ĐỊNH VÀ LOGIC HÌNH HỌC ROI
        boxes_obj = results[0].boxes
        if len(boxes_obj) > 0:
            boxes = boxes_obj.xyxy.cpu().numpy()            # Ma trận tọa độ phẳng Bounding Box
            clss = boxes_obj.cls.cpu().numpy().astype(int)  # Mảng Class ID phương tiện
            
            # Đảm bảo gán ID an toàn (kể cả phương tiện mới xuất hiện chưa kịp có ID từ tracker)
            if boxes_obj.id is not None:
                ids = boxes_obj.id.cpu().numpy().astype(int)
            else:
                ids = np.array([-1] * len(boxes))
            
            current_frame_ids = set([i for i in ids if i != -1])

            for box, obj_id, cls_id in zip(boxes, ids, clss):
                x_min, y_min, x_max, y_max = map(int, box)
                class_name = model.names[cls_id]

                # TOÁN HỌC KHÔNG GIAN: Xác định tọa độ "Điểm chạm đất" (Vị trí bánh xe tiếp xúc mặt đường)
                x_foot = int((x_min + x_max) / 2)
                y_foot = int(y_max)

                # Chuyển đổi tọa độ sang không gian BEV nếu có ma trận biến đổi
                x_bev, y_bev = None, None
                if bev_transformer and bev_transformer.M is not None:
                    x_bev, y_bev = bev_transformer.transform_point((x_foot, y_foot))

                if obj_id != -1:
                    traj_manager.update(obj_id, (x_foot, y_foot), (x_bev, y_bev) if x_bev is not None else None)

                box_color = (0, 255, 0) # Mặc định màu XANH LÁ CÂY = Hợp lệ
                status_text = "OK"
                
                # Kiểm tra trực tiếp trên không gian Camera để khớp chính xác 100% với vùng làn vẽ trên màn hình
                check_pt = (x_foot, y_foot)
                check_lanes = lanes_data

                # KIỂM TRA SAI LÀN VÀ GÁN LÀN NHẠY BÉN (PHẢN HỒI TỨC THÌ)
                if check_lanes:
                    is_violation, raw_lane_name, reason, allowed_cls = check_vehicle_lane_violation(check_pt, class_name, check_lanes)
                    assigned_lane = traj_manager.assign_lane(obj_id, raw_lane_name) if obj_id != -1 else raw_lane_name
                    display_lane = raw_lane_name if raw_lane_name != "Ngoai vung" else assigned_lane

                    if is_violation:
                        v_key = obj_id if obj_id != -1 else f"temp_{x_foot}_{y_foot}"
                        violation_counter[v_key] = violation_counter.get(v_key, 0) + 1
                        # Lấn làn từ 3 frame (~0.1s) trở lên lập tức báo ĐỎ RỰC; 1-2 frame báo CAM CẢNH BÁO
                        if violation_counter[v_key] >= 3:
                            box_color = (0, 0, 255) # ĐỎ RỰC
                            status_text = f"SAI LAN ({display_lane})"
                        else:
                            box_color = (0, 165, 255) # MÀU CAM CẢNH BÁO
                            status_text = f"CANH BAO ({display_lane})"
                    else:
                        v_key = obj_id if obj_id != -1 else f"temp_{x_foot}_{y_foot}"
                        if v_key in violation_counter:
                            violation_counter[v_key] = max(0, violation_counter[v_key] - 2)
                        status_text = f"{display_lane}"
                else:
                    # Kiểm tra theo ROI đa giác cấm mặc định
                    is_inside = cv2.pointPolygonTest(forbidden_lane_roi, (x_foot, y_foot), False)
                    v_key = obj_id if obj_id != -1 else f"temp_{x_foot}_{y_foot}"
                    if is_inside >= 0:
                        violation_counter[v_key] = violation_counter.get(v_key, 0) + 1
                        if violation_counter[v_key] >= 5:
                            box_color = (0, 0, 255)
                            status_text = "VIOLATION"
                    else:
                        if v_key in violation_counter:
                            violation_counter[v_key] = max(0, violation_counter[v_key] - 1)

                if x_bev is not None:
                    vehicles_bev.append((x_bev, y_bev, box_color, obj_id, status_text))

                # BIỂU DIỄN VỆT QUỸ ĐẠO TRÊN ẢNH CAMERA (Perspective Trajectory Trail)
                if obj_id != -1:
                    smoothed_cam = traj_manager.get_smoothed_trajectory(obj_id, use_bev=False)
                    if len(smoothed_cam) >= 2:
                        pts_cam = np.array(smoothed_cam, np.int32).reshape((-1, 1, 2))
                        traj_color = (0, 0, 255) if (box_color == (0, 0, 255)) else (0, 255, 255)
                        cv2.polylines(frame_processed, [pts_cam], isClosed=False, color=traj_color, thickness=2, lineType=cv2.LINE_AA)

                # Render các thành phần đồ họa trực quan lên màn hình Desktop
                cv2.circle(frame_processed, (x_foot, y_foot), 5, (0, 0, 255), -1) # Chấm đỏ gầm xe (Bottom-Center)
                cv2.rectangle(frame_processed, (x_min, y_min), (x_max, y_max), box_color, 2) # Khung bao xe
                
                id_label = f"#{obj_id}" if obj_id != -1 else ""
                label = f"{class_name} {id_label} [{status_text}]"
                cv2.putText(frame_processed, label, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

            # Giải phóng RAM & Bộ đệm Quỹ đạo: Xóa vết ID của các phương tiện đã ra khỏi khung hình
            traj_manager.cleanup(current_frame_ids)
            for cached_id in list(violation_counter.keys()):
                if isinstance(cached_id, int) and cached_id not in current_frame_ids:
                    del violation_counter[cached_id]

        # In thông số hiệu năng FPS lên góc màn hình đồ họa
        cv2.putText(frame_processed, f"Pipeline Performance: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Đẩy ma trận ảnh đồ họa hoàn chỉnh hiển thị ra ngoài cửa sổ Desktop của hệ điều hành Windows
        cv2.imshow("UIT Traffic Enforcement System - Main.py High-Speed Pipeline", frame_processed)
        
        # Hỗ trợ hiển thị Không gian BEV Radar Map
        if bev_transformer and bev_transformer.M is not None:
            bev_map = bev_transformer.draw_bev_map(vehicles_bev, traj_manager)
            cv2.imshow("BEV Radar Map - Hybrid Space", bev_map)

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