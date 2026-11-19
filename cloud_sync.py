"""
Cloud Synchronization Module for Traffic Violation Evidence System (UIT ĐAMH 2)
Author: Võ Việt Tiến (23521591) & Diệp Thanh Phong (23521160)
Features:
- Scans local evidence folder (evidence/snapshots, evidence/videos, evidence/violation_log.csv/json).
- Integrates Google Drive API v3 to upload high-resolution violation snapshots and video clips.
- Integrates Google Sheets API to synchronize real-time violation logs into a cloud spreadsheet for traffic authorities.
- Asynchronous batch / queue worker to ensure zero latency impact on main AI inference pipeline.
"""

import os
import json
import csv
import time
import logging
from threading import Thread
from queue import Queue

logging.basicConfig(level=logging.INFO, format="[CLOUD_SYNC %(asctime)s] %(message)s")

class CloudEvidenceSync:
    def __init__(self, base_dir=".", credentials_file="credentials.json", drive_folder_id=None, spreadsheet_id=None):
        self.base_dir = base_dir
        self.evidence_dir = os.path.join(base_dir, "evidence")
        self.snapshots_dir = os.path.join(self.evidence_dir, "snapshots")
        self.videos_dir = os.path.join(self.evidence_dir, "videos")
        self.csv_log = os.path.join(self.evidence_dir, "violation_log.csv")
        self.json_log = os.path.join(self.evidence_dir, "violation_log.json")
        
        self.credentials_file = os.path.join(base_dir, credentials_file)
        self.drive_folder_id = drive_folder_id or "1_UIT_TRAFFIC_EVIDENCE_FOLDER"
        self.spreadsheet_id = spreadsheet_id or "1_UIT_TRAFFIC_LOG_SPREADSHEET"
        
        self.sync_queue = Queue()
        self.is_running = False
        self.has_credentials = os.path.exists(self.credentials_file)
        
        if not self.has_credentials:
            logging.warning(f"File xác thực '{credentials_file}' chưa được cấu hình. Hệ thống kích hoạt chế độ Giả lập / Sẵn sàng đồng bộ (Dry-run / Standby Mode).")
        else:
            logging.info(f"Đã phát hiện file xác thực Google API: {credentials_file}")

    def enqueue_incident(self, record):
        """Đưa hồ sơ vi phạm vào hàng đợi đẩy lên Cloud chạy nền (Non-blocking)"""
        self.sync_queue.put(record)

    def upload_to_drive(self, file_path, file_name, mime_type="image/jpeg"):
        """Đẩy tệp bằng chứng lên thư mục Google Drive mục tiêu"""
        if not os.path.exists(file_path):
            logging.error(f"Tệp không tồn tại: {file_path}")
            return None
        
        if not self.has_credentials:
            mock_url = f"https://drive.google.com/file/d/MOCK_{os.path.basename(file_path)}/view"
            logging.info(f"[SIMULATED DRIVE UPLOAD] {file_name} -> {mock_url}")
            return mock_url

        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
            service = build('drive', 'v3', credentials=creds)

            file_metadata = {
                'name': file_name,
                'parents': [self.drive_folder_id]
            }
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
            uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
            
            link = uploaded_file.get('webViewLink')
            logging.info(f"Đã đồng bộ lên Google Drive: {file_name} -> {link}")
            return link
        except Exception as e:
            logging.error(f"Lỗi đồng bộ Google Drive ({file_name}): {e}")
            return None

    def append_to_sheet(self, record, snapshot_url, video_url):
        """Ghi nhận thông tin vi phạm vào Google Sheets trực tuyến"""
        row_data = [
            record.get("Incident_ID"),
            record.get("Track_ID"),
            record.get("Vehicle_Class"),
            record.get("Timestamp"),
            record.get("Frame_Index"),
            record.get("Violation_Type"),
            record.get("Lane_Name"),
            snapshot_url or record.get("Snapshot_File"),
            video_url or record.get("Video_File"),
            "SYNCED_ONLINE" if self.has_credentials else "SIMULATED_SYNC"
        ]

        if not self.has_credentials:
            logging.info(f"[SIMULATED SHEET LOG] Ghi nhận dòng vào Sheet: {row_data}")
            return True

        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            service = build('sheets', 'v4', credentials=creds)
            body = {'values': [row_data]}
            service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range="Violations!A:J",
                valueInputOption="USER_ENTERED",
                body=body
            ).execute()
            logging.info(f"Đã cập nhật dòng log lên Google Sheets: {record.get('Incident_ID')}")
            return True
        except Exception as e:
            logging.error(f"Lỗi ghi nhận Google Sheets: {e}")
            return False

    def sync_worker(self):
        """Thread chạy ngầm quét hàng đợi và đẩy lên Cloud để tránh nghẽn FPS"""
        logging.info("Khởi động luồng đồng bộ Cloud ngầm (Background Worker)...")
        while self.is_running:
            try:
                if not self.sync_queue.empty():
                    record = self.sync_queue.get()
                    inc_id = record.get("Incident_ID")
                    logging.info(f"Bắt đầu đồng bộ hồ sơ vi phạm: {inc_id}")

                    # 1. Đồng bộ Snapshot
                    snap_path = os.path.join(self.snapshots_dir, record.get("Snapshot_File", ""))
                    snap_url = self.upload_to_drive(snap_path, record.get("Snapshot_File", ""), mime_type="image/jpeg")

                    # 2. Đợi video clip hoàn tất ghi trước khi đồng bộ (nếu tệp đang ghi dở)
                    vid_path = os.path.join(self.videos_dir, record.get("Video_File", ""))
                    time.sleep(1.0) # Đảm bảo file buffer ghi xong
                    vid_url = self.upload_to_drive(vid_path, record.get("Video_File", ""), mime_type="video/mp4")

                    # 3. Đồng bộ dữ liệu bảng tính Google Sheets
                    self.append_to_sheet(record, snap_url, vid_url)

                    self.sync_queue.task_done()
                else:
                    time.sleep(0.5)
            except Exception as e:
                logging.error(f"Lỗi luồng worker: {e}")
                time.sleep(1.0)

    def start(self):
        self.is_running = True
        self.worker_thread = Thread(target=self.sync_worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=2.0)
        logging.info("Đã đóng luồng Cloud Evidence Sync.")


if __name__ == "__main__":
    print("=== KIỂM THỬ MODULE CLOUD EVIDENCE SYNC (UIT ĐAMH 2) ===")
    syncer = CloudEvidenceSync(base_dir=".")
    syncer.start()
    
    test_incident = {
        "Incident_ID": "VIO_20261120_TEST_ID99",
        "Track_ID": 99,
        "Vehicle_Class": "motorcycle",
        "Timestamp": "2026-11-20 08:30:00",
        "Frame_Index": 120,
        "Violation_Type": "Di chuyen vao lan cam (L2 - O to)",
        "Lane_Name": "Lane 2 - O to",
        "Snapshot_File": "test_snap.jpg",
        "Video_File": "test_vid.mp4"
    }
    
    syncer.enqueue_incident(test_incident)
    time.sleep(2.0)
    syncer.stop()
    print("=== KIỂM THỬ HOÀN TẤT ===")
