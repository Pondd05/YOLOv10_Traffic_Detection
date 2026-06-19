import os
import optuna
import yaml
from ultralytics import YOLOv10 

# 1. ĐƯỜNG DẪN HỆ THỐNG CỐ ĐỊNH (Đã đồng bộ ổ D và ổ G)
VIDEO_PATH = r"G:\My Drive\YOLOv10_AI\Dataset\Kaggle\train_1\test\images\SampleVideo_LowQuality.mp4"
MODEL_PATH = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\YOLOv10_Traffic_Full\Kaggle_Full_Train_V2\weights\best.pt"
TRACKER_CONFIG = r"D:\UIT\Do_An1_2_KLTN\test\yolov10\custom_bytetrack.yaml"

def objective(trial):
    # 2. ĐỊNH NGHĨA KHÔNG GIAN TÌM KIẾM SIÊU THAM SỐ
    track_high_thresh = trial.suggest_float('track_high_thresh', 0.40, 0.70)
    track_low_thresh = trial.suggest_float('track_low_thresh', 0.01, 0.20)
    track_buffer = trial.suggest_int('track_buffer', 30, 100)
    match_thresh = trial.suggest_float('match_thresh', 0.50, 0.90)
    
    # 3. GHI ĐỘNG CÁC SIÊU THAM SỐ VÀO FILE YAML ĐỂ BAYETRACK ĐỌC
    config_data = {
        "tracker_type": "bytetrack",
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": track_high_thresh + 0.05,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "fuse_score": True,
        "gating_local": False,
        "gating_thres": 25.5
    }
    
    with open(TRACKER_CONFIG, 'w') as f:
        yaml.dump(config_data, f)
        
    # 4. GỌI LỚP MÔ HÌNH YOLOV10 ĐỂ HỦY BỎ HÀM NMS MẶC ĐỊNH
    model = YOLOv10(MODEL_PATH)
    
    results = model.track(
        source=VIDEO_PATH, 
        tracker=TRACKER_CONFIG, 
        imgsz=640, 
        save=False, 
        verbose=False,
        stream=True,  # Tiết kiệm RAM
        persist=True  # Đảm bảo giữ vết định danh liên tục
    )
    
    # 5. HÀM CHẤM ĐIỂM AN TOÀN
    all_track_ids = set()
    
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            if r.boxes.is_track and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                all_track_ids.update(ids)
            
    total_vehicles_detected = len(all_track_ids)
    return total_vehicles_detected

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") 
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("BỘ SIÊU THAM SỐ DO THUẬT TOÁN BAYES TÌM ĐƯỢC:")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print("="*50)