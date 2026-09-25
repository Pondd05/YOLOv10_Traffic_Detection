"""
=============================================================================
HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH - UIT
Module: lane_annotator.py - TỰ KẺ LÀN THỦ CÔNG (MANUAL ROI)
=============================================================================
Nhiệm vụ:
  1. Giao diện trực quan dùng cv2.setMouseCallback() để vẽ đa giác (Làn đường)
     và đường gấp khúc (Ranh giới / Vạch kẻ) trực tiếp trên video.
  2. Cơ chế lưu/đọc cấu hình chuẩn hóa (0.0 -> 1.0) qua file `lane_config.json`.
  3. Kiểm tra khả năng tái sử dụng cấu hình cho các video có cùng góc camera.
  4. Cung cấp API kiểm tra vi phạm sai làn (point-in-polygon) cho `main.py`.

Cách sử dụng:
  - Kẻ làn trên video:
      python lane_annotator.py --video path/to/video.mp4 --config lane_config.json
  - Kiểm tra tái sử dụng trên video khác (cùng góc camera):
      python lane_annotator.py --video path/to/video2.mp4 --config lane_config.json --test

Phím tắt trong giao diện:
  - Chuột trái       : Thêm điểm vẽ
  - Chuột phải       : Xóa điểm vừa thêm
  - 1                : Hoàn tất -> Lưu thành [LÀN Ô TÔ] (car, bus, truck)
  - 2                : Hoàn tất -> Lưu thành [LÀN XE MÁY] (motorcycle)
  - 3                : Hoàn tất -> Lưu thành [LÀN HỖN HỢP] (tất cả xe)
  - n                : Hoàn tất -> Nhập thông tin tùy chỉnh ở Terminal
  - b / Tab          : Chuyển đổi giữa chế độ [VẼ LÀN] và [VẼ VẠCH KẺ]
  - [ / ]            : Lùi / Tiến khung hình (chọn frame rõ vạch kẻ nhất)
  - u / Ctrl+Z       : Xóa hình vừa vẽ gần nhất (Undo)
  - c                : Xóa tất cả vẽ lại từ đầu
  - s                : Lưu cấu hình vào file JSON
  - p                : Xem trước video chạy thử để kiểm tra độ khớp làn
  - q / ESC          : Thoát
=============================================================================
"""

import argparse
import copy
import json
import os
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as e:
    print("\n" + "=" * 65)
    print(f"[LỖI THIẾU THƯ VIỆN]: {e}")
    print("Nguyên nhân: Bạn đang ở môi trường Conda '(base)' thay vì 'yolov10_env'.")
    print("Khắc phục ngay: Gõ lệnh sau vào Terminal:")
    print("    conda activate yolov10_env")
    print("Sau đó chạy lại lệnh vừa rồi!")
    print("=" * 65 + "\n")
    sys.exit(1)

# =============================================================================
# 1. QUY ƯỚC HỆ THỐNG (SYSTEM CONVENTIONS)
# =============================================================================
CONFIG_VERSION = 1

# Các lớp đối tượng phương tiện được hỗ trợ bởi mô hình YOLOv10
SUPPORTED_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Quy ước hướng lưu thông
DIRECTIONS = ["down", "up", "any"]

# Quy ước loại ranh giới / vạch kẻ đường (TCVN 41:2019/BGTVT)
BOUNDARY_TYPES = {
    "solid_line": {
        "name": "Vach lien (Cam de vach / Chuyen lan)",
        "color": (0, 0, 255),       # Đỏ
        "thickness": 3
    },
    "dashed_line": {
        "name": "Vach net dut (Cho phep chuyen lan)",
        "color": (0, 255, 255),     # Vàng
        "thickness": 2
    },
    "double_solid": {
        "name": "Vach doi lien (Cam lan lan tuyet doi)",
        "color": (0, 0, 200),       # Đỏ đậm
        "thickness": 4
    },
    "curb": {
        "name": "Via he / Dai phan cach",
        "color": (200, 200, 200),   # Xám trắng
        "thickness": 3
    }
}

# Bảng màu cho các làn đường
LANE_PALETTE = [
    (0, 200, 0),      # Xanh lá (Làn 1)
    (255, 140, 0),    # Cam (Làn 2)
    (0, 180, 255),    # Vàng kim (Làn 3)
    (200, 0, 200),    # Tím (Làn 4)
    (0, 255, 255),    # Vàng chanh (Làn 5)
]


# =============================================================================
# 2. CLASS LẬP TRÌNH GIAO DIỆN KẺ LÀN (LANE ANNOTATOR)
# =============================================================================
class LaneAnnotator:
    def __init__(self, video_path, config_path="lane_config.json", start_frame=0):
        self.video_path = video_path
        self.config_path = config_path
        self.video_name = os.path.basename(video_path) if video_path else "unknown"

        # Mở video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở file video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.current_frame_idx = max(0, min(start_frame, self.total_frames - 1))
        self.current_frame = self._read_frame(self.current_frame_idx)

        # Dữ liệu hình vẽ
        self.lanes = []        # Danh sách làn: dict(id, name, code, direction, allowed_classes, polygon)
        self.boundaries = []   # Danh sách ranh giới: dict(type, name, points)
        self.current = []      # Tọa độ pixel các điểm đang click vẽ: [(x, y), ...]
        self.mode = "lane"     # "lane" (Đa giác) hoặc "boundary" (Đường kẻ)
        self.mouse = None      # Tọa độ chuột hiện tại (x, y)
        self.status_msg = "San sang. Click chuot trai de them diem, 1/2/3 de luu lan nhanh."

        # Nạp cấu hình nếu đã tồn tại từ trước
        self.load()

    def _read_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Fallback tạo frame trống nếu không đọc được
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
        return frame

    # ---------- CHUẨN HÓA VÀ GIẢI CHUẨN HÓA TỌA ĐỘ ----------
    def _norm(self, pts):
        """Chuẩn hóa tọa độ pixel sang tỉ lệ tương đối [0.0, 1.0]."""
        return [[round(float(x) / self.w, 5), round(float(y) / self.h, 5)] for x, y in pts]

    def _denorm(self, pts):
        """Chuyển đổi tọa độ tương đối [0.0, 1.0] sang pixel theo kích thước frame hiện tại."""
        return np.array([[int(p[0] * self.w), int(p[1] * self.h)] for p in pts], np.int32)

    # ---------- XỬ LÝ SỰ KIỆN CHUỘT ----------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
            self.status_msg = f"Da them diem #{len(self.current)} tai ({x}, {y})"
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current:
                popped = self.current.pop()
                self.status_msg = f"Da xoa diem cuoi ({popped[0]}, {popped[1]})"
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)

    # ---------- HOÀN TẤT HÌNH VẼ NHANH HOẶC TÙY CHỌN ----------
    def finish_quick_lane(self, preset_type):
        """Lưu nhanh làn đường theo preset 1, 2, 3 không cần gõ bàn phím Terminal."""
        if self.mode != "lane":
            self.status_msg = "Dang o che do Vach ke, khong ap dung preset lan."
            return

        if len(self.current) < 3:
            self.status_msg = "[LOI] Lan duong can it nhat 3 diem de tao thanh da giac!"
            return

        lane_id = len(self.lanes) + 1
        if preset_type == 1:
            name = f"Lan {lane_id} - O to"
            code = "lane_car"
            allowed = ["car", "bus", "truck"]
        elif preset_type == 2:
            name = f"Lan {lane_id} - Xe may"
            code = "lane_motorcycle"
            allowed = ["motorcycle"]
        elif preset_type == 3:
            name = f"Lan {lane_id} - Hon hop"
            code = "lane_mixed"
            allowed = ["car", "motorcycle", "bus", "truck"]
        else:
            return

        self.lanes.append({
            "id": lane_id,
            "name": name,
            "code": code,
            "direction": "down",
            "allowed_classes": allowed,
            "polygon": self._norm(self.current)
        })
        self.status_msg = f"-> Da tao thanh cong [{name}] ({', '.join(allowed)})"
        self.current = []

    def finish_custom_shape(self):
        """Hoàn tất hình vẽ với thông tin nhập chi tiết qua Terminal."""
        if self.mode == "lane":
            if len(self.current) < 3:
                self.status_msg = "[LOI] Lan duong can it nhat 3 diem de tao da giac!"
                return

            lane_id = len(self.lanes) + 1
            print("\n" + "=" * 50)
            print(f"NHẬP THÔNG TIN CHO LÀN ĐƯỜNG MỚI (ID: {lane_id}):")
            print("=" * 50)
            name = input(f"1. Tên làn (mặc định: Lan {lane_id}): ").strip() or f"Lan {lane_id}"
            direction = input(f"2. Hướng lưu thông {DIRECTIONS} [down]: ").strip().lower() or "down"
            if direction not in DIRECTIONS:
                direction = "down"

            print(f"3. Danh sách loại xe được phép (Hỗ trợ: {SUPPORTED_CLASSES})")
            print("   Ví dụ: 'car, bus, truck' hoặc 'motorcycle' (Để trống = Cho phép tất cả)")
            classes_in = input("   Nhập: ").strip()
            if classes_in:
                allowed = [c.strip() for c in classes_in.split(",") if c.strip() in SUPPORTED_CLASSES]
            else:
                allowed = copy.deepcopy(SUPPORTED_CLASSES)

            self.lanes.append({
                "id": lane_id,
                "name": name,
                "code": f"lane_{lane_id}",
                "direction": direction,
                "allowed_classes": allowed,
                "polygon": self._norm(self.current)
            })
            self.status_msg = f"Da luu [{name}]"
        else:
            if len(self.current) < 2:
                self.status_msg = "[LOI] Ranh gioi can it nhat 2 diem!"
                return

            print("\n" + "=" * 50)
            print("CHỌN LOẠI RANH GIỚI / VẠCH KẺ:")
            for k, v in BOUNDARY_TYPES.items():
                print(f" - {k}: {v['name']}")
            print("=" * 50)
            btype = input("Nhập loại [solid_line]: ").strip() or "solid_line"
            if btype not in BOUNDARY_TYPES:
                btype = "solid_line"

            self.boundaries.append({
                "type": btype,
                "name": BOUNDARY_TYPES[btype]["name"],
                "points": self._norm(self.current)
            })
            self.status_msg = f"Da luu vach ke [{btype}]"

        self.current = []

    # ---------- LƯU / ĐỌC FILE JSON ----------
    def save(self):
        """Lưu cấu hình chuẩn hóa vào file lane_config.json."""
        config_data = {
            "version": CONFIG_VERSION,
            "source_video": self.video_name,
            "reference_resolution": [self.w, self.h],
            "total_lanes": len(self.lanes),
            "total_boundaries": len(self.boundaries),
            "supported_classes": SUPPORTED_CLASSES,
            "lanes": self.lanes,
            "boundaries": self.boundaries
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        self.status_msg = f"[THANH CONG] Da ghi {len(self.lanes)} lan, {len(self.boundaries)} vach ke vao {self.config_path}"
        print(f"\n[SAVE] {self.status_msg}")

    def load(self):
        """Đọc file cấu hình JSON nếu đã có."""
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.lanes = cfg.get("lanes", [])
            self.boundaries = cfg.get("boundaries", [])
            ref_size = cfg.get("reference_resolution", [self.w, self.h])
            print(f"[LOAD] Da nap {len(self.lanes)} lan, {len(self.boundaries)} vach ke tu {self.config_path}")
            print(f"[INFO] Do phan giai goc: {ref_size[0]}x{ref_size[1]} | Video hien tai: {self.w}x{self.h}")
            self.status_msg = f"Da nap {len(self.lanes)} lan tu {os.path.basename(self.config_path)}"
        except Exception as e:
            print(f"[CANH BAO] Loi khi doc file {self.config_path}: {e}")

    # ---------- CHUYỂN ĐỔI FRAME ----------
    def change_frame(self, step):
        new_idx = max(0, min(self.current_frame_idx + step, self.total_frames - 1))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.current_frame = self._read_frame(self.current_frame_idx)
            self.status_msg = f"Khung hinh #{self.current_frame_idx}/{self.total_frames}"

    # ---------- RENDER GIAO DIỆN TRỰC QUAN ----------
    def render(self):
        img = self.current_frame.copy()
        overlay = img.copy()

        # 1. Vẽ các Làn đường (Đa giác có độ trong suốt)
        for i, lane in enumerate(self.lanes):
            poly = self._denorm(lane["polygon"])
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(img, [poly], True, color, 2)

            # Đặt nhãn tên làn ở trọng tâm đa giác
            cx, cy = poly.mean(axis=0).astype(int)
            label = lane.get("name", f"Lane {i+1}")
            cv2.putText(img, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Trộn màu overlay để tạo độ trong suốt 30%
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        # 2. Vẽ các Ranh giới / Vạch kẻ đường
        for b in self.boundaries:
            pts = self._denorm(b["points"])
            b_info = BOUNDARY_TYPES.get(b.get("type", "solid_line"), BOUNDARY_TYPES["solid_line"])
            cv2.polylines(img, [pts], False, b_info["color"], b_info["thickness"], cv2.LINE_AA)

        # 3. Vẽ hình đang click dang dở
        for p in self.current:
            cv2.circle(img, p, 5, (0, 255, 255), -1)
        if len(self.current) > 1:
            is_closed = (self.mode == "lane" and len(self.current) >= 3)
            pts = np.array(self.current, np.int32)
            cv2.polylines(img, [pts], is_closed, (0, 255, 255), 2, cv2.LINE_AA)

        # Đường phụ trợ nối điểm cuối với con trỏ chuột
        if self.current and self.mouse:
            cv2.line(img, self.current[-1], self.mouse, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Vẽ thanh trạng thái và hướng dẫn ở trên cùng và dưới cùng
        h, w = img.shape[:2]
        # Thanh header
        header = np.zeros((70, w, 3), dtype=np.uint8)
        mode_str = "LA DA GIAC (LAN DUONG)" if self.mode == "lane" else "VACH KE (RANH GIOI)"
        cv2.putText(header, f"CHE DO: {mode_str} | So lan: {len(self.lanes)} | Vach ke: {len(self.boundaries)} | Frame: {self.current_frame_idx}/{self.total_frames}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, f"Phim tat: [1]=Oto [2]=Xe may [3]=Hon hop | [n]=Tuy chon | [b]=Doi che do | [s]=Luu | [p]=Phat thu | [q]=Thoat",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Ghép header lên frame
        result = np.vstack([header, img])

        # Vẽ thanh status bar ở dưới cùng
        cv2.putText(result, f"Trang thai: {self.status_msg}",
                    (15, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return result

    # ---------- PHÁT THỬ VIDEO ĐỂ KIỂM TRA ĐỘ KHỚP (TEST PREVIEW) ----------
    def preview_video(self):
        """Phát thử video cùng các làn đã vẽ đè lên để quan sát luồng xe chạy thực tế."""
        print("\n[PREVIEW] Bắt đầu phát thử video để kiểm tra độ khớp làn đường...")
        print("[INFO] Nhấn phím 'p' hoặc 'Space' để tạm dừng, 'q' hoặc ESC để quay lại màn hình vẽ.")
        temp_cap = cv2.VideoCapture(self.video_path)
        paused = False

        while temp_cap.isOpened():
            if not paused:
                ok, f = temp_cap.read()
                if not ok:
                    temp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop lại từ đầu
                    continue

                overlay = f.copy()
                for i, lane in enumerate(self.lanes):
                    poly = np.array([[int(p[0] * self.w), int(p[1] * self.h)] for p in lane["polygon"]], np.int32)
                    color = LANE_PALETTE[i % len(LANE_PALETTE)]
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(f, [poly], True, color, 2)
                    cx, cy = poly.mean(axis=0).astype(int)
                    cv2.putText(f, lane.get("name", ""), (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                f = cv2.addWeighted(overlay, 0.25, f, 0.75, 0)
                for b in self.boundaries:
                    pts = np.array([[int(p[0] * self.w), int(p[1] * self.h)] for p in b["points"]], np.int32)
                    cv2.polylines(f, [pts], False, (0, 0, 255), 2)

                cv2.putText(f, "DANG PHAT THU (PREVIEW) - Nhan 'q' de ve tiep", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Lane Annotator - Preview", f)

            key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('p'), ord(' ')):
                paused = not paused

        temp_cap.release()
        cv2.destroyWindow("Lane Annotator - Preview")


# =============================================================================
# 3. HÀM KIỂM TRA TÁI SỬ DỤNG CẤU HÌNH (REUSE CONFIG TEST)
# =============================================================================
def test_config_reuse(video_path, config_path):
    """
    Kiểm tra khả năng tái sử dụng cấu hình lane_config.json cho video có cùng góc camera.
    Tự động chuẩn hóa tỷ lệ khung hình và chạy video để quan sát đối chiếu trực quan.
    """
    print("\n" + "=" * 65)
    print("KIỂM TRA TÁI SỬ DỤNG CẤU HÌNH LÀN ĐƯỜNG TRÊN VIDEO MỚI")
    print("=" * 65)

    if not os.path.exists(config_path):
        print(f"[LỖI] Không tìm thấy file cấu hình: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở file video: {video_path}")
        return False

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ref_w, ref_h = cfg.get("reference_resolution", [vw, vh])

    print(f"- Video gốc trong cấu hình   : {cfg.get('source_video', 'N/A')} ({ref_w}x{ref_h})")
    print(f"- Video kiểm tra tái sử dụng : {os.path.basename(video_path)} ({vw}x{vh})")
    print(f"- Số lượng làn cấu hình      : {len(cfg.get('lanes', []))}")
    print(f"- Số lượng ranh giới         : {len(cfg.get('boundaries', []))}")

    ratio_ref = ref_w / ref_h
    ratio_new = vw / vh
    if abs(ratio_ref - ratio_new) > 0.05:
        print("[CẢNH BÁO] Tỉ lệ khung hình (Aspect Ratio) giữa 2 video lệch nhau > 5%!")
        print("          Tọa độ làn có thể cần được kiểm tra kỹ lưỡng.")
    else:
        print("[ĐẠT] Tỉ lệ khung hình tương đồng -> Tọa độ chuẩn hóa sẽ khớp chính xác.")

    print("\n[HƯỚNG DẪN] Cửa sổ phát video kiểm tra sẽ mở ra.")
    print("           Quan sát xem các vệt đa giác màu có ôm khít làn xe chạy không.")
    print("           Bấm 'q' hoặc ESC để kết thúc kiểm tra.")

    win = "Kiem Tra Tai Su Dung Cau Hinh Lan"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        overlay = frame.copy()
        for i, lane in enumerate(cfg.get("lanes", [])):
            poly = np.array([[int(p[0] * vw), int(p[1] * vh)] for p in lane["polygon"]], np.int32)
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2)

            # Hiển thị tên làn và loại xe hợp lệ
            cx, cy = poly.mean(axis=0).astype(int)
            allowed_txt = ",".join(lane.get("allowed_classes", []))
            cv2.putText(frame, f"{lane.get('name')}", (cx - 60, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, f"{lane.get('name')}", (cx - 60, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"[{allowed_txt}]", (cx - 60, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        for b in cfg.get("boundaries", []):
            pts = np.array([[int(p[0] * vw), int(p[1] * vh)] for p in b["points"]], np.int32)
            cv2.polylines(frame, [pts], False, (0, 0, 255), 3)

        cv2.putText(frame, "TEST REUSE MODE - Nhan 'q' de thoat", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


# =============================================================================
# 4. API DÙNG TRONG PIPELINE NHẬN DIỆN XE SAI LÀN (CHO MAIN.PY VÀ APP.PY)
# =============================================================================
def load_lane_config(config_path, frame_w, frame_h):
    """
    Nạp cấu hình từ lane_config.json và chuyển đổi tọa độ tương đối sang pixel.
    Trả về: (dict_config, list_lanes_with_pixel_coords, list_boundaries_with_pixel_coords)
    """
    if not os.path.exists(config_path):
        return None, [], []

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lanes = []
    for lane in cfg.get("lanes", []):
        poly = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in lane["polygon"]], np.int32)
        lanes.append({
            **lane,
            "polygon_px": poly
        })

    boundaries = []
    for b in cfg.get("boundaries", []):
        pts = np.array([[int(p[0] * frame_w), int(p[1] * frame_h)] for p in b["points"]], np.int32)
        boundaries.append({
            **b,
            "points_px": pts
        })

    return cfg, lanes, boundaries


def check_vehicle_lane_violation(point_xy, class_name, lanes):
    """
    Thuật toán Point-in-Polygon kiểm tra xem xe có đang đi sai làn quy định không:
    - Input:
        + point_xy: Tọa độ điểm tiếp xúc mặt đường (x_bottom, y_bottom)
        + class_name: Loại phương tiện ('car', 'motorcycle', 'bus', 'truck')
        + lanes: Danh sách các làn đường đã nạp từ load_lane_config
    - Output:
        + (is_violation: bool, lane_name: str, reason: str)
    """
    px, py = float(point_xy[0]), float(point_xy[1])
    for lane in lanes:
        # Kiểm tra điểm có nằm trong đa giác của làn đường không
        dist = cv2.pointPolygonTest(lane["polygon_px"], (px, py), False)
        if dist >= 0:  # Nằm trong hoặc nằm ngay trên cạnh làn
            allowed = lane.get("allowed_classes", [])
            # Nếu danh sách allowed rỗng -> Cho phép mọi loại xe
            if not allowed or class_name in allowed:
                return False, lane.get("name", "Lane"), "Hop le"
            else:
                return True, lane.get("name", "Lane"), f"Xe {class_name} khong duoc phep di vao {lane.get('name')}"

    # Xe không nằm trong bất kỳ làn nào đã định nghĩa (ví dụ làn ngoài lề)
    return False, "Ngoai vung kiem soat", "Khong xac dinh"


# =============================================================================
# 5. MAIN ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Tool Tự Kẻ Làn Thủ Công (Manual ROI) - UIT Traffic Project")
    parser.add_argument("--video", default=None, help="Đường dẫn đến file video thực nghiệm (.mp4, .mov, .avi)")
    parser.add_argument("--config", default="lane_config.json", help="Tên file cấu hình JSON lưu trữ (Mặc định: lane_config.json)")
    parser.add_argument("--frame", type=int, default=0, help="Chỉ số frame bắt đầu vẽ (Mặc định: 0)")
    parser.add_argument("--test", action="store_true", help="Kích hoạt chế độ kiểm tra tái sử dụng cấu hình trên video")
    parser.add_argument("--clean", action="store_true", help="Vẽ mới từ đầu, không nạp lại các làn đã vẽ trước đó")
    args = parser.parse_args()

    # Tự động tìm video nếu người dùng không truyền tham số
    target_video = args.video
    if not target_video:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_test_dir = os.path.join(current_dir, "video test")
        if os.path.exists(video_test_dir):
            videos = [os.path.join(video_test_dir, f) for f in os.listdir(video_test_dir)
                      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
            if videos:
                target_video = videos[0]
                print(f"[SYSTEM] Tự động chọn video từ thư mục 'video test': {os.path.basename(target_video)}")

    if not target_video or not os.path.exists(target_video):
        print("[LỖI] Không tìm thấy video! Vui lòng chỉ định đường dẫn: python lane_annotator.py --video <duong_dan_video>")
        return

    if args.test:
        test_config_reuse(target_video, args.config)
        return

    # Khởi tạo giao diện kẻ làn
    annotator = LaneAnnotator(target_video, args.config, args.frame)
    if args.clean:
        annotator.lanes = []
        annotator.boundaries = []
        annotator.current = []
        annotator.status_msg = "Che do ve moi tu dau. Click chuot trai de them diem."

    win = "UIT - Tu Ke Lan Thu Cong (Manual ROI)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, annotator.on_mouse)

    print("\n" + "=" * 65)
    print("ĐÃ KHỞI CHẠY CÔNG CỤ TỰ KẺ LÀN THỦ CÔNG (MANUAL ROI)")
    print("=" * 65)
    print("HƯỚNG DẪN THAO TÁC TRỰC TIẾP TRÊN VIDEO:")
    print("  1. DÙNG CHUỘT:")
    print("     - Chuột TRÁI : Chấm điểm tạo đỉnh đa giác (làn) hoặc đường kẻ (vạch)")
    print("     - Chuột PHẢI : Xóa điểm vừa chấm gần nhất")
    print("\n  2. PHÍM LƯU NHANH KHI ĐANG VẼ LÀN (Polygon Mode):")
    print("     - Phím 1     : Lưu nhanh thành [LÀN Ô TÔ] (car, bus, truck)")
    print("     - Phím 2     : Lưu nhanh thành [LÀN XE MÁY] (motorcycle)")
    print("     - Phím 3     : Lưu nhanh thành [LÀN HỖN HỢP] (tất cả xe)")
    print("\n  3. PHÍM LƯU NHANH KHI ĐANG VẼ VẠCH KẺ (Boundary Mode):")
    print("     - Phím 4     : Lưu thành [DẢI PHÂN CÁCH GIỮA 2 CHIỀU] (Đỏ)")
    print("     - Phím 5     : Lưu thành [VẠCH NÉT ĐỨT PHÂN LÀN] (Vàng)")
    print("     - Phím 6     : Lưu thành [VẠCH LIỀN CẤM ĐÈ] (Đỏ)")
    print("\n  4. TIỆN ÍCH KHÁC:")
    print("     - Phím Tab/b : Đổi qua lại giữa [VẼ LÀN] và [VẼ VẠCH KẺ]")
    print("     - Phím [ / ] : Tua lùi / tiến khung hình để tìm góc rõ vạch nhất")
    print("     - Phím c     : Xóa trắng toàn bộ để vẽ lại từ đầu")
    print("     - Phím u     : Xóa hình vừa vẽ gần nhất (Undo)")
    print("     - Phím s     : LƯU CẤU HÌNH VÀO FILE 'lane_config.json'")
    print("     - Phím p     : PHÁT THỬ VIDEO xem các vệt kẻ vừa vẽ có khớp xe chạy không")
    print("     - Phím q/ESC : Thoát công cụ")
    print("=" * 65 + "\n")

    while True:
        display_frame = annotator.render()
        cv2.imshow(win, display_frame)

        key = cv2.waitKey(25) & 0xFF

        # Xử lý phím tắt
        if key in (ord('q'), 27):  # 'q' hoặc ESC
            break
        elif key == ord('1'):
            annotator.finish_quick_lane(1)
        elif key == ord('2'):
            annotator.finish_quick_lane(2)
        elif key == ord('3'):
            annotator.finish_quick_lane(3)
        elif key == ord('4'):
            if annotator.mode == "boundary":
                annotator.boundaries.append({"type": "double_solid", "name": "Dai phan cach giua", "points": annotator._norm(annotator.current)})
                annotator.status_msg = "Da luu: Dai phan cach giua (double_solid)"
                annotator.current = []
        elif key == ord('5'):
            if annotator.mode == "boundary":
                annotator.boundaries.append({"type": "dashed_line", "name": "Vach net dut", "points": annotator._norm(annotator.current)})
                annotator.status_msg = "Da luu: Vach net dut (dashed_line)"
                annotator.current = []
        elif key == ord('6'):
            if annotator.mode == "boundary":
                annotator.boundaries.append({"type": "solid_line", "name": "Vach lien", "points": annotator._norm(annotator.current)})
                annotator.status_msg = "Da luu: Vach lien (solid_line)"
                annotator.current = []
        elif key == ord('n'):
            annotator.finish_custom_shape()
        elif key in (ord('b'), 9): # 'b' hoặc Tab (key code 9)
            annotator.mode = "boundary" if annotator.mode == "lane" else "lane"
            annotator.current = []
            annotator.status_msg = f"Da doi sang che do: {'VACH KE' if annotator.mode == 'boundary' else 'LAN DUONG'}"
        elif key in (ord('u'), 26): # 'u' hoặc Ctrl+Z
            target_list = annotator.lanes if annotator.mode == "lane" else annotator.boundaries
            if target_list:
                removed = target_list.pop()
                annotator.status_msg = f"Da xoa: {removed.get('name', 'Hinh cuoi')}"
        elif key == ord('c'):
            annotator.lanes = []
            annotator.boundaries = []
            annotator.current = []
            annotator.status_msg = "Da xoa toan bo hinh ve."
        elif key == ord('s'):
            annotator.save()
        elif key == ord('p'):
            annotator.preview_video()
        elif key == ord('['):
            annotator.change_frame(-15)
        elif key == ord(']'):
            annotator.change_frame(15)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
