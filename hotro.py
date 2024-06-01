from ultralytics import YOLO
import time
import streamlit as st
import cv2
import numpy as np
import math
from tkinter import Tk, filedialog
import caidat
def load_model(model_path):
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    # Resize the image to a standard size
    image = cv2.resize(image, (800, int(800*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

def PredictionYoloV8RiceSeed(img_path):
    model=YOLO(load_model())
    counter = 0
    img = cv2.imread(img_path)
    results = model.predict(img)
    himg, wimg, channels = img.shape
    print(himg)
    thick_line = int(himg * 1 / 800)
    thick_text = int(himg * 2 / 800)

    fsize_text = himg * 0.4 / 800
    fsixe_text_count = himg * 0.5 / 800

    spacing = int(himg * 20 / 800)
    up_spacing = int(himg * 30 / 800)

    output_lines = []  # To store lines for CSV output
    rice_info = []  # To store rice information for CSV

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for mask in result.masks.xy:#vẽ mặt nạ
            cv2.drawContours(img, [mask.astype(int)], -1, (0,255,0), 1)# vẽ Vẽ đường viền quanh mỗi mặt nạ trên ảnh. Đường viền được vẽ với màu xanh lá cây và độ dày là 1 pixel. Tham số -1 chỉ ra rằng tất cả các đường viền trong danh sách sẽ được vẽ.


        for i, (arrays, box) in enumerate(zip(result.masks.xy, boxes)):#Lặp Qua Từng Phát Hiện Để Thu Thập Thông Tin
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            (x, y), (w, h), angle = cv2.minAreaRect(arrays)#Tính toán hộp chứa nhỏ nhất có thể (rotated rectangle) bao quanh mặt nạ, cho phép xác định tọa độ tâm, kích thước và góc quay.
            if (w > h):
              w, h = h, w
            rice_info.append({
                'Bông lúa': i+1,
                'Giống lúa': class_name,
                'Chiều rộng': w,
                'Chiều dài': h
            })
            output_lines.append(f'Bông lúa thứ #{i+1}')
            output_lines.append(f'--Giống lúa:{class_name}')
            output_lines.append(f'--Chiều rộng= {w} (pixel)')
            output_lines.append(f'--Chiều dài= {h} (pixel)')

        for i, arrays in enumerate(result.masks.xy):
          (x, y), (w, h), angle = cv2.minAreaRect(arrays)#tính hộp giới hạn quay nhỏ nhất cho mặt nạ.
          # vẽ mặt nạ cho từng đối tượng
          box = cv2.boxPoints(((x, y), (w, h), angle))#chuyển hộp giới hạn quay thành các điểm đỉnh, mà sau đó được sử dụng để vẽ hình chữ nhật.
          box = np.int0(box)#Làm tròn các giá trị đỉnh để chúng phù hợp với các thao tác pixel trên ảnh.
          #cv2.drawContours(img, [box], 0, (0, 255, 0), 1)  # Draw rotated rectangle
          if (w > h) & (angle==90):
              w, h = h, w
              angle=0
          elif (w > h) & (angle==45):
            w, h = h, w
            angle=135
          elif w>h:
            w, h = h, w
            angle=angle+90 #Điều chỉnh này đảm bảo rằng chiều rộng w luôn nhỏ hơn chiều cao h, và góc quay angle được thiết lập lại để phản ánh điều này.
          if angle != 0 and angle != 90 and angle != 45 and angle != 135:
            angle_rad = math.radians(angle)
            dx = int(math.cos(angle_rad) * (w / 2))
            dy = int(math.sin(angle_rad) * (w / 2))
            pt1 = (int(x - dx), int(y - dy))
            pt2 = (int(x + dx), int(y + dy))
            cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)
          else:
            for angle_offset in range(-90, 91, 10):  # Change the step as per requirement
                rotated_angle = angle + angle_offset
                if rotated_angle == angle:
                  # Calculate endpoints of the line representing the width
                  angle_rad = math.radians(rotated_angle)
                  dx = int(math.cos(angle_rad) * (w / 2))
                  dy = int(math.sin(angle_rad) * (w / 2))
                  pt1 = (int(x - dx), int(y - dy))
                  pt2 = (int(x + dx), int(y + dy))

                  # Draw the line representing the width
                  cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)
        for i, arrays in enumerate(result.masks.xy):
          (x, y), (w, h), angle = cv2.minAreaRect(arrays)
          box = cv2.boxPoints(((x, y), (w, h), angle))
          box = np.int0(box)
          #cv2.drawContours(img, [box], 0, (0, 255, 0), 1)  # Draw rotated rectangle
          #print(angle)
          if (w < h) & (angle==90):
            angle=90
          if (w < h) & (angle==0):
            angle=90
          elif (w < h) & (angle==45):
            angle=315
          elif w>h:
            w, h = h, w
            angle=angle+180
          else:
            angle=angle+90
          if angle != 0 and angle != 90 and angle != 45 and angle != 315 and angle!=0:
            angle_rad = math.radians(angle)
            dx = int(math.cos(angle_rad) * (h / 2))
            dy = int(math.sin(angle_rad) * (h / 2))
            pt1 = (int(x - dx), int(y - dy))
            pt2 = (int(x + dx), int(y + dy))
            cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)
          else:
            for angle_offset in range(-90, 91, 10):
                rotated_angle = angle + angle_offset
                if rotated_angle == angle:
                  angle_rad = math.radians(rotated_angle)
                  dx = int(math.cos(angle_rad) * (h / 2))
                  dy = int(math.sin(angle_rad) * (h / 2))
                  pt1 = (int(x - dx), int(y - dy))
                  pt2 = (int(x + dx), int(y + dy))

                  cv2.line(img, pt1, pt2, (0, 0, 255), thick_line)

        for box in boxes:
              counter += 1  # Increment counter for each bounding box
              points = box.xyxy[0].astype(int)  # Get corner points as int
              class_id = int(box.cls[0])  # Get class ID
              class_name = model.names[class_id]  # Get class name using the class ID

              # Text for label with class name and number
              label = f"#{counter}:{class_name}"

              # Draw polygon
              points = np.array([(points[0], points[1]), (points[2], points[1]), (points[2], points[3]), (points[0], points[3])])
              cv2.polylines(img, [points], isClosed=True, color=(0,0,255), thickness=thick_line)
              # Draw label
              cv2.putText(img, label, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fsize_text, (0,0,255), thick_line)  # Draw label
    w=spacing
    h=spacing
    cv2.putText(img, "Tổng bông lúa = "+str(counter), (w, h), cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0, 0, 0), thick_text)
    names = model.names
    for i in range(0,len(names)):
      h+=up_spacing
      rice_id = list(names)[list(names.values()).index(str(names[i]))]
      cv2.putText(img, names[i]+"= "+str(results[0].boxes.cls.tolist().count(rice_id)), (w, h), cv2.FONT_HERSHEY_SIMPLEX, fsixe_text_count, (0, 0, 0), thick_text)
    return img,rice_info

def select_folder():
    root = Tk()
    root.withdraw()  # Đóng cửa sổ Tkinter để chỉ hiển thị hộp thoại
    folder_selected = filedialog.askdirectory()  # Mở hộp thoại chọn thư mục
    root.destroy()
    return folder_selected

