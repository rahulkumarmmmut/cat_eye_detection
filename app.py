# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="Cat Left-Eye Locator", layout="centered")
st.title("🐱 Cat Left-Eye Locator")
st.write("Upload a cat photo and I'll draw a box around its left eye and give you the coordinates.")

# 1️⃣ Lazy-load your model (happens once at cold start)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 2️⃣ File uploader
img_file = st.file_uploader("Choose a cat image", type=["jpg","jpeg","png"])
if not img_file:
    st.info("Please upload an image of a cat.")
    st.stop()

# 3️⃣ Run inference
img = Image.open(img_file).convert("RGB")
results = model.predict(source=np.array(img), conf=0.25, save=False)

# Only one image in, so one result
res = results[0]

# 4️⃣ If we found at least one box, draw it
if res.boxes:
    # Grab the first box
    x1,y1,x2,y2 = res.boxes.xyxy[0].tolist()
    cx, cy = (x1+x2)/2, (y1+y2)/2

    # Draw overlay
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    # small dot at center
    r=5
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="blue")

    st.image(overlay, caption="Detected left eye", use_container_width=True)
    st.success(f"Center = ({cx:.1f}, {cy:.1f}) px")
else:
    st.error("No left-eye detected. Try a clearer cat face.")