import os
import shutil
import xml.etree.ElementTree as ET

# =====================================================================
# 1. CAU HINH DUONG DAN CHINH XAC GIUA 2 O DIA
# =====================================================================
# Nguon du lieu tho (Anh + XML) o o D cua ban
SRC_IMG_DIR = r'D:/Downloads/archive/DETRAC-Images/DETRAC-Images'          
SRC_XML_DIR = r'D:/Downloads/archive/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML'

# Dich den (Da loc va chia Train/Valid/Test) o o G
DST_ROOT_DIR = r'G:/My Drive/YOLOv10_AI/Dataset'

# Dinh nghia chinh xac 4 lop doi tuong
CLASSES = ['car', 'motorcycle', 'bus', 'truck']

# =====================================================================
# 2. CAC HAM TIEN XU LY VA CHUYEN DOI
# =====================================================================
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    # UA-DETRAC XML luu: left, top, width, height
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

def process_data(step=10, train_ratio=0.7, val_ratio=0.2):
    # Tao san cac thu muc train, valid, test tren o G
    subsets = ['train', 'valid', 'test']
    for sub in subsets:
        os.makedirs(os.path.join(DST_ROOT_DIR, sub, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DST_ROOT_DIR, sub, 'labels'), exist_ok=True)

    # Kiem tra thu muc nguon o o D
    if not os.path.exists(SRC_IMG_DIR):
        print("Loi: Khong tim thay thu muc anh o o D: " + SRC_IMG_DIR)
        return

    if not os.path.exists(SRC_XML_DIR):
        print("Loi: Khong tim thay thu muc XML o o D: " + SRC_XML_DIR)
        return

    # Quet tat ca cac file XML chuoi video
    xml_files = [f for f in os.listdir(SRC_XML_DIR) if f.endswith('.xml')]
    print("Tim thay " + str(len(xml_files)) + " file XML chuoi video tai o D.")

    all_valid_frames = []
    
    # Duyet qua tung file XML de trich xuat frame
    for xml_file in xml_files:
        video_name = os.path.splitext(xml_file)[0]
        video_img_dir = os.path.join(SRC_IMG_DIR, video_name)
        
        if not os.path.exists(video_img_dir):
            continue
            
        try:
            tree = ET.parse(os.path.join(SRC_XML_DIR, xml_file))
            root = tree.getroot()
        except Exception as e:
            continue
            
        for frame in root.iter('frame'):
            frame_num = int(frame.get('num'))
            
            # Loc theo buoc nhay (step)
            if frame_num % step != 0:
                continue
                
            img_name = "img" + str(frame_num).zfill(5) + ".jpg"
            src_img_path = os.path.join(video_img_dir, img_name)
            
            if os.path.exists(src_img_path):
                all_valid_frames.append({
                    'video_name': video_name,
                    'img_name': img_name,
                    'src_img_path': src_img_path,
                    'frame_element': frame
                })

    total_filtered = len(all_valid_frames)
    print("Tong so frame trich xuat duoc sau khi loc: " + str(total_filtered))
    print("Bat dau chuyen doi XML sang TXT va day sang o G...")
    print("-" * 60)

    # Tinh toan so luong cho tung tap Train/Valid/Test
    train_count = int(total_filtered * train_ratio)
    val_count = int(total_filtered * val_ratio)

    count_success = 0
    for idx, frame_data in enumerate(all_valid_frames):
        if idx < train_count:
            subset = 'train'
        elif idx < train_count + val_count:
            subset = 'valid'
        else:
            subset = 'test'

        # Kich thuoc anh mac dinh cua UA-DETRAC
        img_w, img_h = 960, 540
        yolo_lines = []

        for target in frame_data['frame_element'].iter('target'):
            attr = target.find('.//attribute')
            if attr is None:
                continue
            cls_name = attr.get('vehicle_type', 'car').lower().strip()
            
            if cls_name not in CLASSES:
                continue
            cls_id = CLASSES.index(cls_name)
            
            box = target.find('box')
            if box is None:
                continue
            left = float(box.get('left'))
            top = float(box.get('top'))
            width = float(box.get('width'))
            height = float(box.get('height'))
            
            yolo_box = convert_box_coords((img_w, img_h), (left, top, width, height))
            yolo_lines.append(str(cls_id) + " " + " ".join([f"{v:.6f}" for v in yolo_box]))

        if yolo_lines:
            new_basename = frame_data['video_name'] + "_" + frame_data['img_name']
            dst_img = os.path.join(DST_ROOT_DIR, subset, 'images', new_basename)
            dst_txt = os.path.join(DST_ROOT_DIR, subset, 'labels', os.path.splitext(new_basename)[0] + '.txt')
            
            with open(dst_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
                
            shutil.copy(frame_data['src_img_path'], dst_img)
            count_success += 1

    print("-" * 60)
    print("Hoan thanh quy trinh!")
    print("Da chuyen doi thanh cong " + str(count_success) + " cap sang o G.")
    print("Toan bo du lieu da nam tai: " + DST_ROOT_DIR)

def convert_box_coords(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

if __name__ == '__main__':
    # step=10 de giam tai cho RTX 3050, cu 10 anh giu lai 1 anh
    process_data(step=10, train_ratio=0.7, val_ratio=0.2)