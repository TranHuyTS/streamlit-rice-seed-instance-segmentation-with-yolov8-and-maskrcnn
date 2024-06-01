from pathlib import Path

import PIL
import streamlit as st
import numpy as np
import pandas as pd
import ultralytics
from ultralytics import YOLO
import cv2 as cv
import os
import uuid
import math
import csv
import torch
from IPython.display import Image
from PIL import Image

import caidat
import hotro

# Setting page layout
st.set_page_config(
    page_title="Phân tích hình ảnh hạt lúa giống",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Phân đoạn ảnh")
# Sidebar
st.sidebar.header("ML Model")
# Model Options
model_type = st.sidebar.radio(
    "Chọn mô hình", ['YoloV8S', 'YoloV8M'])
confidence = float(st.sidebar.slider(
    "Chọn tỉ lệ", 25, 100, 40)) / 100
# Select model
if model_type == 'YoloV8S':
    model_path = Path(caidat.YOLOV8S_SEG_MODEL)
elif model_type == 'YoloV8M':
    model_path = Path(caidat.YOLOV8M_SEG_MODEL)
# elif model_type == 'Mask R-CNN':
#      model_path = Path(caidat.MASKRCNN_MODEL)
# elif model_type == 'Segmentation':
#     model_path = Path(caidat.SEGMENTATION_MODEL)
try:
    model = hotro.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Ảnh")
source_radio = st.sidebar.radio(
    "Chọn nguồn", caidat.SOURCES_LIST)

source_img = None

# Function to handle image processing and display
def process_and_display_image(image_path):
    # Đoạn mã giả định cho việc xử lý ảnh
    processed_image = Image.open(image_path)
    return processed_image

# If image is selected
if source_radio == caidat.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Chọn ảnh...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(caidat.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Ảnh mặc định",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Ảnh đã chọn",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is None:
            default_detected_image_path = str(caidat.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Ảnh đã xử lí',
                     use_column_width=True)
        else:
            if st.sidebar.button('Xử lí'):
                res = model.predict(uploaded_image,conf=confidence, line_width=1)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Ảnh đã xử lí',
                         use_column_width=True)
                try:
                    st.write(f'Số lượng boxes: {len(boxes)}')
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

        # if source_img is not None:
        #     if source_img is not None:
        #         if st.sidebar.button('Phân đoạn đối tượng'):
        #             # Phân đoạn ảnh được tải lên
        #             res = model.predict(uploaded_image, conf=confidence)
        #             boxes = res[0].boxes
        #             res_plotted = res[0].plot()[:, :, ::-1]
        #             st.image(res_plotted, caption='Ảnh đã phân đoạn', use_column_width=True)
        #
        #             if st.download_button('Chọn thư mục và lưu kết quả vào'):
        #                 save_dir = hotro.select_folder()
        #                 if save_dir:
        #
        #                     img_save_path = os.path.join(save_dir, "segmented_image.png")
        #                     res[0].save(img_save_path)
        #                     st.success(f'Image successfully saved to {img_save_path}')
        #                 else:
        #                     st.error('No folder was selected.')
        #             try:
        #                 with st.expander("Detection Results"):
        #                     for box in boxes:
        #                         st.write(box.data)
        #             except Exception as ex:
        #                 st.error("Failed to display detection results.")
        #                 st.error(ex)
            else:
                st.error("Please upload an image to proceed with object detection.")

else:
    st.error("Please select a valid source type!")



