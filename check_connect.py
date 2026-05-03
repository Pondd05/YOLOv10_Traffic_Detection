import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

# 1. Định nghĩa quyền truy cập
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# 2. Sử dụng file keys.json để xác thực
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name("keys.json", scope)
    client = gspread.authorize(creds)
    
    # 3. Mở file Sheets (Phải trùng tên 100% với tên trên Web)
    sheet = client.open("CAMERA_AI").sheet1
    
    # 4. Ghi thử một dòng để kiểm tra
    now = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    row = [now, "UIT_TEST", "Xe may", "Ket noi thanh cong!"]
    sheet.append_row(row)
    
    print("--- CHÚC MỪNG PHONG VÀ TIẾN ---")
    print(f"Lúc {now}, dữ liệu đã nhảy lên Google Sheets thành công!")

except Exception as e:
    print(f"Lỗi rồi Phong ơi: {e}")