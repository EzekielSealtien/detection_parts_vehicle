import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import shutil
from datetime import datetime

# Load YOLOv8 model
model = YOLO("best.pt")  # Make sure 'best.pt' is in the same folder as this script

st.set_page_config(page_title="Car Part Detector", layout="centered")
st.title(" Car Part Detection with YOLOv8")
st.write("Upload an image ")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    file_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference
    with st.spinner("Running detection..."):
        results = model(file_path, save=True, line_width=1)
        output_dir = results[0].save_dir
        output_file = os.path.join(output_dir, os.path.basename(file_path))

    # Show results
    st.success(" Detection complete!")
    st.image(output_file, caption="Detection Result", use_column_width=True)

    # Cleanup (optional)
    if st.button("Clear temporary files"):
        shutil.rmtree("runs/detect")
        os.remove(file_path)
        st.success("Temporary files removed.")
