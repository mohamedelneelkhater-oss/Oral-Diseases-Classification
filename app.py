import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from pathlib import Path
import json
import numpy as np
from PIL import Image
import io
import requests
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

st.set_page_config(
    page_title="Oral Diseases Classification",
    page_icon="🦷",
    layout="centered",
)

BASE = Path(__file__).resolve().parent

DEFAULT_CLASS_NAMES = [
    'Calculus',
    'Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset',
    'Data caries',
    'Gingivitis',
    'Mouth Ulcer',
    'Tooth Discoloration',
    'hypodontia'
]
DEFAULT_IMG_SIZE = (160, 160)


@st.cache_resource
def load_model_and_config():
    cfg_path = BASE / "app_config.json"
    model_path = BASE / "best_model.keras"

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        class_names = cfg["class_names"]
        img_size = tuple(cfg["img_size"])
    else:
        class_names = DEFAULT_CLASS_NAMES
        img_size = DEFAULT_IMG_SIZE

    model = None
    if model_path.exists():
        model = keras.models.load_model(str(model_path), compile=False)

    return model, class_names, img_size


model, CLASS_NAMES, IMG_SIZE = load_model_and_config()

if model is None:
    st.sidebar.warning("Model file not found. Please upload best_model.keras")
    uploaded_model = st.sidebar.file_uploader("Upload model file (.keras)", type=["keras"])
    if uploaded_model is not None:
        tmp_model_path = BASE / "best_model_uploaded.keras"
        tmp_model_path.write_bytes(uploaded_model.read())
        model = keras.models.load_model(str(tmp_model_path), compile=False)
        st.sidebar.success("Model loaded successfully")
    else:
        st.stop()


def predict_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32")[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    top_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top_idx]
    return top3, probs


def show_prediction(pil_img, top3, probs):
    st.image(pil_img, use_container_width=True)
    best_class, best_conf = top3[0]
    if best_conf >= 0.5:
        st.success(f"Top prediction: **{best_class}** — Confidence: {best_conf * 100:.1f}%")
    else:
        st.warning(f"Top prediction: **{best_class}** — Confidence: {best_conf * 100:.1f}% (low confidence)")
    df_chart = pd.DataFrame({"Class": [c for c, _ in top3], "Probability": [p for _, p in top3]})
    st.bar_chart(df_chart.set_index("Class"))
    with st.expander("Show probabilities for all classes"):
        all_probs = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
        for cls, p in all_probs:
            st.write(f"{cls}: {p * 100:.2f}%")


st.title("🦷 Oral Diseases Image Classification")
st.markdown("Select image source from the sidebar.")

st.sidebar.header("Image source")
option = st.sidebar.radio("Choose source:", ["From device 📱", "From data 📊", "From internet 🌐"])

if option == "From device 📱":
    st.subheader("Upload an image from your device")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        img = Image.open(uploaded)
        top3, probs = predict_image(img)
        show_prediction(img, top3, probs)

elif option == "From data 📊":
    st.subheader("Sample images from dataset")
    examples_dir = BASE / "examples"
    if examples_dir.exists():
        image_paths = sorted(examples_dir.glob("*.*"))[:30]
        st.write(f"Number of available images: {len(image_paths)}")
        cols = st.columns(5)
        for idx, img_path in enumerate(image_paths):
            with cols[idx % 5]:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
                if st.button("Classify this image", key=f"select_{idx}"):
                    top3, probs = predict_image(img)
                    show_prediction(img, top3, probs)
    else:
        st.info("examples folder not found. Add images to it to display them here.")

elif option == "From internet 🌐":
    st.subheader("Classify an image from a direct link")
    url = st.text_input("Enter image URL")
    if st.button("Download and classify"):
        if not url:
            st.warning("Please enter a valid link.")
        else:
            try:
                with st.spinner("Downloading image..."):
                    response = requests.get(url, timeout=20)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content))
                top3, probs = predict_image(img)
                show_prediction(img, top3, probs)
            except Exception as e:
                st.error(f"Failed to download image: {e}")

st.divider()
st.caption("⚠️ This is an educational application only and not a medical diagnostic tool.")
