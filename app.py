# app.py
import io
import tempfile
from pathlib import Path
from ultralytics import YOLO

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
from zipfile import ZipFile

st.set_page_config(page_title="YOLOv8 Face Cropper", layout="wide", page_icon="🧠")

# =================================
# Load Model
# =================================
MODEL_PATH = "best1.pt"

@st.cache_resource
def load_model(path):
    if not Path(path).exists():
        st.error(f"Model not found: {path}")
        return None
    return YOLO(path)

model = load_model(MODEL_PATH)

# =================================
# UI Tabs
# =================================
tab1, tab2 = st.tabs(["📸 Image Face Crop", "🎬 Video Face Crop"])

# =================================
# IMAGE CROP
# =================================
with tab1:
    st.header("📸 Image Face Crop")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    conf_threshold = st.slider("Confidence Threshold (Image)", 0.1, 1.0, 0.35, 0.01)

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🚀 Detect Faces in Image"):
            results = model.predict(np.array(image), conf=conf_threshold)
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()

            if len(boxes) == 0:
                st.warning("⚠️ No faces detected.")
                st.stop()

            st.subheader("✅ Cropped Faces")

            cols = st.columns(3)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cropped = image.crop((x1, y1, x2, y2))

                col = cols[i % 3]
                with col:
                    st.image(cropped, caption=f"Face {i+1}", use_column_width=True)

                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    st.download_button(
                        f"⬇ Face {i+1}",
                        data=buf.getvalue(),
                        file_name=f"face_{i+1}.png",
                        mime="image/png"
                    )

# =================================
# VIDEO CROP
# =================================
with tab2:
    st.header("🎬 Video Face Crop")

    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    conf_video = st.slider("Confidence Threshold (Video)", 0.1, 1.0, 0.30, 0.01)
    frame_skip = st.number_input("Process every Nth frame", min_value=1, max_value=20, value=5)

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        cropped_faces = []
        frame_count = 0  

        if st.button("🎯 Detect & Crop Faces From Video"):
            with st.spinner("⏳ Processing video, please wait..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict(frame_rgb, conf=conf_video)
                    result = results[0]
                    boxes = result.boxes.xyxy.cpu().numpy()

                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame_rgb[y1:y2, x1:x2]
                        cropped_faces.append(crop)

                cap.release()

            if len(cropped_faces) == 0:
                st.warning("⚠️ No faces detected in video.")
                st.stop()

            st.success(f"✅ Extracted {len(cropped_faces)} faces from video!")

            zip_path = "video_faces.zip"
            with ZipFile(zip_path, 'w') as zipf:
                for i, face in enumerate(cropped_faces):
                    face_img = Image.fromarray(face)
                    file_name = f"face_{i+1}.png"
                    face_img.save(file_name)
                    zipf.write(file_name)

            st.download_button(
                "⬇ Download All Cropped Faces (ZIP)",
                data=open(zip_path, "rb"),
                file_name="video_faces.zip",
                mime="application/zip"
            )

            st.subheader("🖼 Preview Samples")
            for i in range(min(6, len(cropped_faces))):
                st.image(cropped_faces[i], width=200)

st.markdown("---")
st.caption("Face Detection + Cropping • YOLOv8 • Streamlit 🚀")
