import urllib.request
from pathlib import Path
import streamlit as st

MODEL_URL = "https://github.com/mohamedelneelkhater-oss/Oral-Diseases-Classification/releases/download/v1.0/best_model.keras"

@st.cache_resource(show_spinner="Downloading model...")
def get_model_file() -> Path:
    path = Path("best_model.keras")
    if not path.exists():
        urllib.request.urlretrieve(MODEL_URL, path)
    return path

model_path = get_model_file()
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
    page_title="تصنيف أمراض الفم والأسنان",
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

# ------------------------------------------------------------
# تحميل النموذج: من ملف محلي أو من رفع المستخدم
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_config():
    cfg_path = BASE / "app_config.json"

    # قراءة إعدادات الفئات إذا وجدت
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        class_names = cfg["class_names"]
        img_size = tuple(cfg["img_size"])
    else:
        class_names = DEFAULT_CLASS_NAMES
        img_size = DEFAULT_IMG_SIZE

    # محاولة تحميل النموذج من نفس مجلد التطبيق
    model_path = BASE / "best_model.keras"
    model = None
    if model_path.exists():
        model = keras.models.load_model(str(model_path), compile=False)

    return model, class_names, img_size, model_path


model, CLASS_NAMES, IMG_SIZE, MODEL_PATH = load_model_and_config()

# إذا لم يوجد النموذج، اعرض زر الرفع في الشريط الجانبي
if model is None:
    st.sidebar.warning("يجب تحميل النموذج على الرابط. يرجى رفع ملف best_model.keras")
    uploaded_model = st.sidebar.file_uploader("Upload model file (.keras)", type=["keras"])
    if uploaded_model is not None:
        # حفظ الملف المرفوع مؤقتاً وتحميله
        tmp_model_path = BASE / "best_model_uploaded.keras"
        tmp_model_path.write_bytes(uploaded_model.read())
        model = keras.models.load_model(str(tmp_model_path), compile=False)
        st.sidebar.success("تم تحميل النموذج بنجاح!")
    else:
        st.stop()


# ------------------------------------------------------------
# دالة التنبؤ
# ------------------------------------------------------------
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
        st.success(f"✅ التوقع الأبرز: **{best_class}** — الثقة: {best_conf * 100:.1f}%")
    else:
        st.warning(f"⚠️ التوقع الأبرز: **{best_class}** — الثقة: {best_conf * 100:.1f}% (ثقة منخفضة)")

    df_chart = pd.DataFrame({"الفئة": [c for c, _ in top3], "الاحتمال": [p for _, p in top3]})
    st.bar_chart(df_chart.set_index("الفئة"))

    with st.expander("🔍 عرض احتمالات جميع الفئات"):
        all_probs = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
        for cls, p in all_probs:
            st.write(f"{cls}: {p * 100:.2f}%")


# ------------------------------------------------------------
# واجهة المستخدم
# ------------------------------------------------------------
st.title("🦷 تطبيق تصنيف أمراض الفم والأسنان")
st.markdown("اختر مصدر الصورة من القائمة الجانبية.")

st.sidebar.header("📂 مصدر الصورة")
option = st.sidebar.radio("اختر المصدر:", ["من الجهاز 📱", "من البيانات 📊", "من الإنترنت 🌐"])

# الخيار 1: من الجهاز
if option == "من الجهاز 📱":
    st.subheader("ارفع صورة من جهازك")
    uploaded = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        img = Image.open(uploaded)
        top3, probs = predict_image(img)
        show_prediction(img, top3, probs)

# الخيار 2: من البيانات (مجلد الأمثلة)
elif option == "من البيانات 📊":
    st.subheader("عينة من صور البيانات")
    examples_dir = BASE / "examples"
    if examples_dir.exists():
        image_paths = sorted(examples_dir.glob("*.*"))[:30]
        st.write(f"عدد الصور المتاحة: {len(image_paths)}")
        cols = st.columns(5)
        for idx, img_path in enumerate(image_paths):
            with cols[idx % 5]:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
                if st.button("تصنيف هذه الصورة", key=f"select_{idx}"):
                    top3, probs = predict_image(img)
                    show_prediction(img, top3, probs)
    else:
        st.info("مجلد `examples` غير موجود. أضف صورًا إليه لتظهر هنا.")

# الخيار 3: من الإنترنت
elif option == "من الإنترنت 🌐":
    st.subheader("تصنيف صورة عبر رابط مباشر")
    url = st.text_input("أدخل رابط الصورة")
    if st.button("تحميل وتوقع"):
        if not url:
            st.warning("الرجاء إدخال رابط صحيح.")
        else:
            try:
                with st.spinner("جاري تحميل الصورة..."):
                    response = requests.get(url, timeout=20)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content))
                top3, probs = predict_image(img)
                show_prediction(img, top3, probs)
            except Exception as e:
                st.error(f"فشل تحميل الصورة: {e}")

st.divider()
st.caption("⚠️ هذا تطبيق تعليمي فقط وليس أداة طبية لتشخيص الأمراض.")
