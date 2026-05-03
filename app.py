import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLOv10
import torch

# 1. CẤU HÌNH BAN ĐẦU & GIAO DIỆN WEB
st.set_page_config(
    page_title="Hệ thống Nhận diện Phương tiện", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ứng dụng Nhận diện & Theo dõi Phương tiện Giao thông")
st.markdown("---")

# Đường dẫn file best.pt
MODEL_PATH = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"

# class id phương tiện
CLASS_NAMES = {0: "Motobike", 1: "Car", 2: "Bus", 3: "Truck"}

# 2. LOAD MÔ HÌNH VÀO CACHE
@st.cache_resource
def load_yolo_model(model_path):
    """
    Nạp mô hình YOLOv10 vào bộ nhớ đệm của Streamlit.
    """
    return YOLOv10(model_path)

try:
    model = load_yolo_model(MODEL_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
except Exception as e:
    st.error(f"Không thể load mô hình từ đường dẫn: {MODEL_PATH}. Hãy kiểm tra lại file của bạn.")
    st.stop()

# 3. SIDEBAR: ĐIỀU CHỈNH SIÊU THAM SỐ SUY LUẬN TRỰC TIẾP
st.sidebar.header("Cấu hình mô hình")

#điều chỉnh ngưỡng tin cậy từ 0.1 đến 1.0
conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (Confidence threshold)", 
    min_value=0.1, max_value=1.0, value=0.20, step=0.05
)

# Kích thước ảnh đầu vào (1280/640)
img_size = st.sidebar.selectbox("Kích thước ảnh đầu vào (imgsz):", [1280, 640], index=0)

st.sidebar.info(f"Đang chạy trên thiết bị: **{torch.cuda.get_device_name(0) if device == 0 else 'CPU'}**")

# 4. KHU VỰC TẢI LÊN DỮ LIỆU
file_type = st.radio("Chọn loại dữ liệu đầu vào:", ("Hình ảnh", "Video"))
uploaded_file = st.file_uploader(f"Tải lên {file_type.lower()} của bạn tại đây:", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # XỬ LÝ HÌNH ẢNH (IMAGE INFERENCE)
    if file_type == "Hình ảnh":
        # Đọc ảnh bằng OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Dự đoán bằng YOLOv10
        with st.spinner("Đang nhận diện hình ảnh..."):
            results = model.predict(
                source=image, 
                device=device, 
                imgsz=img_size, 
                conf=conf_threshold, 
                iou=0.6
            )

        annotated_image = image.copy()
        # Vẽ khung lên ảnh
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                
                if cls_id in CLASS_NAMES:
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Hiển thị 2 cột ảnh (Ảnh gốc vs Ảnh kết quả)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Ảnh gốc tải lên", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Ảnh kết quả dự đoán", use_container_width=True)

    # XỬ LÝ VIDEO (VIDEO INFERENCE)
    elif file_type == "Video":
        # Lưu file video tải lên vào một file tạm thời
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()  # Khung trống để Streamlit cập nhật video liên tục

        st.info("Đang xử lý video và hiển thị trực tiếp bên dưới...")
        
        # Tạo nút bấm dừng xử lý video nếu cần
        stop_button = st.button("Dừng xử lý video")
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sử dụng YOLOv10 + ByteTrack để theo dõi phương tiện
            results = model.track(
                source=frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                device=device, 
                imgsz=img_size, 
                conf=conf_threshold, 
                iou=0.45,
                verbose=False
            )
            
            # Vẽ kết quả tracking lên từng khung hình của video
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, obj_id, cls_id in zip(boxes, ids, clss):
                    if cls_id in CLASS_NAMES:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{CLASS_NAMES[cls_id]} #{obj_id}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Chuyển đổi màu từ BGR sang RGB và hiển thị lên Web
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Tạo khoảng trễ nhỏ (1ms) để giảm tải cho CPU/GPU
            time.sleep(0.001)
            
        # Giải phóng bộ nhớ sau khi hoàn tất video
        cap.release()
        cv2.destroyAllWindows()
        st.success("Xử lý video hoàn tất!")