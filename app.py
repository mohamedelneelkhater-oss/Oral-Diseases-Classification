"""تطبيق تصنيف أمراض الفم والأسنان - Streamlit (تعليمي فقط، ليس أداة تشخيص)."""
import io
import ipaddress
import json
import os
import re
import socket
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="تصنيف أمراض الفم والأسنان", page_icon="🦷", layout="wide")

# ----------------------------------------------------------------------------- الإعدادات
BASE = Path(__file__).resolve().parent
MODEL_URL = ("https://github.com/mohamedelneelkhater-oss/Oral-Diseases-Classification/"
             "releases/download/v1.0/best_model.keras")
MODEL_PATH = BASE / "best_model.keras"
CONFIG_PATH = BASE / "app_config.json"
SAMPLES_DIR = BASE / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SAMPLES = 30
LOW_CONFIDENCE = 0.50

DEFAULT_CLASSES = [
    "Calculus",
    "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "hypodontia",
]
ARABIC_NAMES = {
    "Calculus": "الجير السني (Calculus)",
    "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset": "حالات مختلطة (مجموعة مُعلَّمة)",
    "Data caries": "تسوس الأسنان (Caries)",
    "Gingivitis": "التهاب اللثة (Gingivitis)",
    "Mouth Ulcer": "قرحة الفم (Mouth Ulcer)",
    "Tooth Discoloration": "تغيّر لون الأسنان",
    "hypodontia": "نقص عدد الأسنان (Hypodontia)",
}

st.markdown(
    "<style>[data-testid='stMarkdownContainer'] {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)


def label_ar(raw: str) -> str:
    return ARABIC_NAMES.get(raw, raw.replace("_", " "))


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ----------------------------------------------------------------------------- النموذج
def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """يحمّل النموذج من GitHub Releases مرة واحدة، ويحذف الملف التالف إن وُجد."""
    if path.exists() and not zipfile.is_zipfile(path):
        path.unlink()
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            raise RuntimeError("الملف الذي تم تنزيله ليس نموذج .keras صالحًا. تأكد من MODEL_URL.")
        tmp.replace(path)
    return path


@st.cache_resource(show_spinner="جارٍ تحميل النموذج...")
def load_model_and_config():
    cfg = {"class_names": DEFAULT_CLASSES, "img_size": [160, 160]}
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    model = keras.models.load_model(str(ensure_model()), compile=False)
    return model, list(cfg["class_names"]), tuple(cfg["img_size"])


try:
    model, CLASS_NAMES, IMG_SIZE = load_model_and_config()
except Exception as exc:
    st.error(f"تعذّر تحميل النموذج: {type(exc).__name__}. افتح Manage app ← Logs لمعرفة التفاصيل.")
    st.stop()


# ----------------------------------------------------------------------------- الصور
def open_image(src) -> Image.Image:
    try:
        img = Image.open(src)
        img.load()
    except Exception:
        raise ValueError("الملف ليس صورة صالحة")
    return ImageOps.exif_transpose(img).convert("RGB")


def is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def fetch_image(url: str) -> Image.Image:
    """تنزيل صورة من الإنترنت (مواقع عامة فقط، حد أقصى 10MB)."""
    current = url.strip()
    for _ in range(4):
        u = urlparse(current)
        if u.scheme not in ("http", "https") or not u.hostname or not is_public_host(u.hostname):
            raise ValueError("رابط غير صالح")
        r = requests.get(current, headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
                         stream=True, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, r.headers.get("Location", ""))
            r.close()
            continue
        r.raise_for_status()
        data = bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data) > 10 * 1024 * 1024:
                r.close()
                raise ValueError("الصورة أكبر من 10MB")
        return open_image(io.BytesIO(bytes(data)))
    raise ValueError("تحويلات كثيرة")


@st.cache_data
def list_samples(class_names: tuple):
    """صور المعرض: samples/<class>/*.jpg موزّعة بالتناوب بين الأصناف."""
    if not SAMPLES_DIR.is_dir():
        return []
    names = {slug(c): c for c in class_names}
    groups = []
    for sub in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f for f in sub.iterdir() if f.suffix.lower() in IMG_EXTS)
        if files:
            groups.append([(str(f), names.get(slug(sub.name))) for f in files])
    items = []
    for rnd in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if rnd < len(g) and len(items) < MAX_SAMPLES:
                items.append(g[rnd])
    return items


def predict(img: Image.Image) -> np.ndarray:
    h, w = IMG_SIZE
    arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype="float32")[None]
    probs = np.asarray(model.predict(arr, verbose=0))[0]
    if len(probs) != len(CLASS_NAMES):
        raise ValueError("عدد مخرجات النموذج لا يطابق عدد الأصناف في app_config.json")
    return probs


# ----------------------------------------------------------------------------- الواجهة
st.title("🦷 تطبيق تصنيف أمراض الفم والأسنان")
st.write("اختر مصدر الصورة من القائمة الجانبية، ثم سيعرض النموذج أكثر الحالات احتمالًا.")

st.sidebar.header("📂 مصدر الصورة")
source = st.sidebar.radio("اختر المصدر", ["💻 رفع من الجهاز", "🌐 رابط من الإنترنت", "🖼️ معرض الصور (30 صورة)"])
image, true_label = None, None

if source.startswith("💻"):
    st.subheader("ارفع صورة من جهازك")
    f = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if f:
        try:
            image = open_image(f)
        except ValueError as e:
            st.error(str(e))

elif source.startswith("🌐"):
    st.subheader("صورة من الإنترنت")
    url = st.text_input("رابط مباشر للصورة", placeholder="https://example.com/photo.jpg")
    if url:
        try:
            with st.spinner("جارٍ تحميل الصورة..."):
                image = fetch_image(url)
        except Exception as e:
            st.error(f"تعذّر تحميل الصورة: {e}")

else:
    st.subheader("معرض الصور من مجموعة البيانات")
    items = list_samples(tuple(CLASS_NAMES))
    if not items:
        st.info("فولدر samples غير موجود في المستودع. ارفعه بجانب app.py ثم أعد تشغيل التطبيق.")
    else:
        st.caption("اضغط «اختيار» تحت أي صورة لتصنيفها. قد تكون هذه الصور من بيانات التدريب.")
        cols = st.columns(6)
        for k, (path, label) in enumerate(items):
            with cols[k % 6]:
                st.image(path, caption=label_ar(label) if label else "", width=130)
                if st.button("اختيار", key=f"pick_{k}"):
                    st.session_state["sel"] = (path, label)
        if "sel" in st.session_state:
            path, true_label = st.session_state["sel"]
            image = open_image(path)

# ----------------------------------------------------------------------------- النتيجة
if image is not None:
    st.divider()
    c1, c2 = st.columns([1, 1.3])
    c1.image(image, width=320)
    with st.spinner("جارٍ التحليل..."):
        try:
            probs = predict(image)
        except Exception as e:
            st.error(f"فشل التصنيف: {e}")
            st.stop()
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    c2.subheader(label_ar(CLASS_NAMES[top]))
    c2.metric("الثقة", f"{probs[top]:.1%}")
    if probs[top] < LOW_CONFIDENCE:
        c2.warning("الثقة منخفضة: قد لا تكون الصورة واضحة أو النموذج غير متأكد.")
    if true_label:
        ok = true_label == CLASS_NAMES[top]
        c2.caption(f"التصنيف الأصلي: {label_ar(true_label)} - " + ("✅ صحيح" if ok else "❌ مختلف"))
    for i in order[:3]:
        c2.progress(float(probs[i]), text=f"{label_ar(CLASS_NAMES[i])} - {probs[i]:.1%}")

st.divider()
st.warning("⚠️ هذا تطبيق تعليمي فقط وليس أداة طبية لتشخيص الأمراض. استشر طبيب أسنان مختص دائمًا.")
