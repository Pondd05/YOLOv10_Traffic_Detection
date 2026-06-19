import cv2
import numpy as np
import torch
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# CẤU HÌNH GIAO DIỆN WEB STREAMLIT (UI/UX)
st.set_page_config(page_title="UIT Traffic Enforcement System", page_icon="🚘", layout="wide")
st.title("🚘 Hệ Thống Giám Sát Giao Thông Thông Minh - UIT")
st.markdown("---")

st.sidebar.header("⚙️ Cấu Hình")
source_type = st.sidebar.radio("Chọn đầu vào:", ("Hình ảnh", "Video"))

# KHỞI TẠO MÔ HÌNH VÀ THUẬT TOÁN (Sửa lỗi YOLOv10 & ByteTrack)
MODEL_PATH = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"
TRACKER_CONFIG = "custom_bytetrack.yaml" # <--- Đảm bảo file này có 'fuse_score: True'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Phần cứng: **{device.upper()}**")

@st.cache_resource
def load_yolo_model():
    # Ép tác vụ task="detect" để sửa lỗi .shape của YOLOv10
    return YOLO(MODEL_PATH, task="detect")

with st.spinner("Đang tải YOLOv10m..."):
    model = load_yolo_model()

# Định nghĩa lại vùng đa giác ảo ROI (Màu vàng rực)
# Tọa độ này được tối ưu để ôm khít làn đường hỗn hợp trong phối cảnh camera
forbidden_lane_roi = np.array([
    [220, 720],   # Dưới cùng bên trái
    [520, 420],   # Trên cùng bên trái
    [820, 420],   # Trên cùng bên phải
    [1120, 720]   # Dưới cùng bên phải
], np.int32)

# HÀM TOÁN HỌC HÌNH HỌC NÂNG CAO: TÍNH ĐIỂM CHẠM ĐẤT
def get_foot_point(x_min, x_max, y_max):
    """
    Tính toán Điểm chạm đất ( Bottom-Center Centroid).
    Điểm này triệt tiêu sai số do chiều cao phương tiện tạo ra.
    """
    x_foot = int((x_min + x_max) / 2)
    y_foot = int(y_max)
    return (x_foot, y_foot)

# XỬ LÝ VIDEO VÀ ĐỒ HỌA BYTETRACK
if source_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Tải video lên đây:", type=['mp4', 'avi'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        # Bộ đệm lọc nhiễu thời gian
        violation_counter = {}
        
        stop_button = st.sidebar.button("⏹️ Dừng xử lý")
        
        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                break
                
            # Vẽ ROI màu vàng
            cv2.polylines(frame, [forbidden_lane_roi], isClosed=True, color=(0, 255, 255), thickness=3)
            
            # Kích hoạt Custom ByteTrack
            results = model.track(
                source=frame, persist=True, tracker=TRACKER_CONFIG,
                imgsz=640, half=True, device=device, verbose=False
            )
            
            speed = results[0].speed
            fps = 1000 / (speed['preprocess'] + speed['inference'] + speed['postprocess'])
            
            # Cấu trúc gán vết ByteTrack
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Biến để vẽ nhãn gọn gàng
                labels_positions = []
                current_frame_ids = set(ids)
                
                for box, obj_id, cls_id in zip(boxes, ids, clss):
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_name = model.names[cls_id]
                    
                    # LOGIC HÌNH HỌC PHÂN LÀN ROI
                    # Gọi hàm toán học tính điểm chạm đất chuẩn
                    x_foot, y_foot = get_foot_point(x_min, x_max, y_max)
                    
                    # Giải thuật Point-in-Polygon kiểm tra lấn làn
                    is_inside = cv2.pointPolygonTest(forbidden_lane_roi, (x_foot, y_foot), False)
                    
                    box_color = (0, 255, 0) # Xanh lá = OK
                    status_text = "OK"
                    
                    if is_inside >= 0:
                        # Điểm chạm đất lọt vào đa giác cấm
                        violation_counter[obj_id] = violation_counter.get(obj_id, 0) + 1
                        
                        # Bộ lọc thời gian: Phải đi sai làn liên tục 15 frames (~0.5 giây)
                        if violation_counter[obj_id] >= 15:
                            box_color = (0, 0, 255) # Đỏ rực = VIOLATION
                            status_text = "VIOLATION"
                    else:
                        # Phương tiện đi ra khỏi vùng cấm, giảm dần bộ đếm về 0
                        if obj_id in violation_counter:
                            violation_counter[obj_id] = max(0, violation_counter[obj_id] - 1)
                    
                    # VẼ ĐỒ HỌA TRỰC QUAN (Fix lỗi Render Chồng Chéo)
                    # 1. Vẽ điểm chạm đất (Chấm đỏ)
                    cv2.circle(frame, (x_foot, y_foot), 6, (0, 0, 255), -1)
                    
                    # 2. Vẽ khung bao (Bounding Box) màu xanh/đỏ ôm sát phương tiện
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    
                    # 3. Tối ưu hóa vị trí Text nhãn: "Lớp #ID [Trạng thái]"
                    label_txt = f"{class_name} #{obj_id} [{status_text}]"
                    
                    # Kiểm tra và điều chỉnh vị trí nhãn để không bị che khuất
                    label_y_pos = y_min - 10 if y_min > 20 else y_min + 15
                    cv2.putText(frame, label_txt, (x_min, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
                
                # Giải phóng bộ nhớ đệm ID rác
                for cached_id in list(violation_counter.keys()):
                    if cached_id not in current_frame_ids:
                        del violation_counter[cached_id]
            
            # Đóng gói đồ họa FPS thời gian thực
            cv2.putText(frame, f"Performance: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Chuyển hệ màu để Streamlit Web render chính xác
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(rgb_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        st.sidebar.success("Xử lý dữ liệu hoàn tất!")