# TÀI LIỆU CHỐT QUY ƯỚC LÀN ĐƯỜNG, TÊN LÀN VÀ LOẠI RANH GIỚI
**Đồ án Khóa luận Tốt nghiệp - Hệ thống Giám sát & Phát hiện Phương tiện Đi Sai Làn Đường (YOLOv10m)**

---

## 1. MỤC TIÊU VÀ NGUYÊN LÝ HOẠT ĐỘNG
Chức năng **Tự kẻ làn thủ công (Manual ROI)** cho phép người giám sát cấu hình bản đồ làn đường cho camera giao thông một cách trực quan, linh hoạt trên từng góc máy thực tế.
- **Tọa độ chuẩn hóa (Normalized Coordinates):** Toàn bộ tọa độ các đỉnh của làn đường và vạch kẻ được chuẩn hóa về tỉ lệ thập phân `[0.0, 1.0]` tương đối với độ phân giải `[Width, Height]`.
- **Khả năng tái sử dụng (Config Reuse):** Nhờ cơ chế chuẩn hóa, một file `lane_config.json` có thể được tái sử dụng nguyên vẹn cho tất cả các video/luồng camera ghi hình tại cùng một góc quay, bất kể độ phân giải đầu vào là 720p, 1080p hay 4K.

---

## 2. QUY ƯỚC PHƯƠNG TIỆN (CLASSES CONVENTION)
Hệ thống sử dụng mô hình **YOLOv10m** đã huấn luyện trên bộ dữ liệu giao thông Việt Nam với 4 lớp phương tiện chuẩn:

| Class ID | Mã lớp (Code) | Tên phương tiện | Ghi chú |
| :---: | :---: | :---: | :--- |
| `0` | `car` | Ô tô con, taxi, SUV | Phương tiện 4 bánh thông dụng |
| `1` | `motorcycle` | Xe máy, xe mô tô 2-3 bánh | Phương tiện chiếm đa số tại VN |
| `2` | `bus` | Xe buýt | Xe khách, xe buýt công cộng |
| `3` | `truck` | Xe tải | Xe tải nhẹ, tải nặng, container |

---

## 3. QUY ƯỚC TÊN LÀN VÀ DANH SÁCH XE ĐƯỢC PHÉP (LANE CONVENTIONS)
Dựa theo **Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT)**:

| Tên hiển thị (Name) | Mã định danh (Code) | Loại xe được phép (`allowed_classes`) | Phím tắt Preset | Ý nghĩa / Hành vi kiểm tra |
| :--- | :--- | :--- | :---: | :--- |
| **Làn Ô tô** | `lane_car` | `["car", "bus", "truck"]` | Phím `1` | Xe máy (`motorcycle`) đi vào -> **Báo vi phạm đi sai làn** |
| **Làn Xe máy** | `lane_motorcycle` | `["motorcycle"]` | Phím `2` | Ô tô, xe buýt, xe tải đi vào -> **Báo vi phạm đi sai làn** |
| **Làn Hỗn hợp** | `lane_mixed` | `["car", "motorcycle", "bus", "truck"]` | Phím `3` | Cho phép mọi phương tiện lưu thông hợp lệ |
| **Làn Xe Buýt BRT** | `lane_bus` | `["bus"]` | Tùy chọn `n` | Chỉ cho phép xe buýt; mọi xe khác vào -> **Vi phạm** |

---

## 4. QUY ƯỚC LOẠI RANH GIỚI / VẠCH KẺ ĐƯỜNG (BOUNDARY CONVENTIONS)
Ranh giới được vẽ dưới dạng các đoạn thẳng hoặc đường gấp khúc (Polyline):

| Mã loại (`type`) | Tên mô tả | Quy tắc xử lý | Màu hiển thị |
| :--- | :--- | :--- | :--- |
| `solid_line` | **Vạch liền phân làn** | Cấm đè vạch, cấm chuyển làn giữa các luồng xe | Đỏ `(0, 0, 255)` |
| `dashed_line` | **Vạch nét đứt** | Được phép chuyển làn hợp lệ | Vàng `(0, 255, 255)` |
| `double_solid` | **Vạch đôi liền tim đường** | Cấm đè vạch, cấm lấn sang chiều đường đối diện | Đỏ đậm `(0, 0, 200)` |
| `curb` | **Vỉa hè / Dải phân cách** | Biên an toàn vật lý ngoài cùng của đường | Xám `(200, 200, 200)` |

---

## 5. QUY CÁCH CẤU TRÚC FILE `lane_config.json`
```json
{
  "version": 1,
  "source_video": "traffic_vietnam.mp4",
  "reference_resolution": [1920, 1080],
  "total_lanes": 2,
  "total_boundaries": 1,
  "supported_classes": ["car", "motorcycle", "bus", "truck"],
  "lanes": [
    {
      "id": 1,
      "name": "Lan 1 - O to",
      "code": "lane_car",
      "direction": "down",
      "allowed_classes": ["car", "bus", "truck"],
      "polygon": [
        [0.22, 0.45],
        [0.52, 0.45],
        [0.48, 0.95],
        [0.10, 0.95]
      ]
    },
    {
      "id": 2,
      "name": "Lan 2 - Xe may",
      "code": "lane_motorcycle",
      "direction": "down",
      "allowed_classes": ["motorcycle"],
      "polygon": [
        [0.52, 0.45],
        [0.78, 0.45],
        [0.92, 0.95],
        [0.48, 0.95]
      ]
    }
  ],
  "boundaries": [
    {
      "type": "solid_line",
      "name": "Vach lien (Cam de vach / Chuyen lan)",
      "points": [
        [0.52, 0.45],
        [0.48, 0.95]
      ]
    }
  ]
}
```

---

## 6. HƯỚNG DẪN THỰC THI

### 6.1. Khởi chạy công cụ kẻ làn trên video
```bash
python lane_annotator.py --video path/to/video.mp4 --config lane_config.json
```
- Click chuột trái để chấm điểm đỉnh của làn.
- Nhấn phím `1` để lưu nhanh thành **Làn Ô tô**.
- Nhấn phím `2` để lưu nhanh thành **Làn Xe máy**.
- Nhấn phím `s` để lưu file cấu hình `lane_config.json`.
- Nhấn phím `p` để chạy thử xem luồng xe có khớp với làn vừa vẽ không.

### 6.2. Kiểm tra tái sử dụng cấu hình trên video khác (Cùng góc camera)
```bash
python lane_annotator.py --video path/to/video_khac.mp4 --config lane_config.json --test
```
Hệ thống sẽ:
1. So sánh tỷ lệ khung hình (Aspect Ratio) giữa 2 video.
2. Tự động co giãn tỷ lệ tọa độ theo độ phân giải của video mới.
3. Chạy video với các làn đường đã nạp để người dùng kiểm chứng mắt thường.

### 6.3. Chạy Pipeline phát hiện xe đi sai làn (Real-time Inference)
```bash
python main.py path/to/video.mp4
```
Chương trình `main.py` sẽ tự động đọc `lane_config.json` và xử lý logic vi phạm sai làn tự động.
