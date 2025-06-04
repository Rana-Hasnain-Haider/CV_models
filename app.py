import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import tempfile
import os

# ---------------------------
# Sidebar navigation
# ---------------------------
st.set_page_config(page_title="Disease Detection Suite", layout="centered")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a detector", ["Ginger Disease Detector", "OCT Eye Disease Detector"])

# ---------------------------
# Ginger Disease Detector
# ---------------------------
if page == "Ginger Disease Detector":
    st.title("🌿 Ginger Disease Detector – MobileNet")

    # Load Ginger model
    @st.cache_resource
    def load_ginger_model():
        return load_model("ginger_disease_model.keras")

    ginger_model = load_ginger_model()
    ginger_class_labels = ['Bacterial_wilt', 'Healthy']
    ginger_img_size = (224, 224)

    def predict_ginger(image_path):
        img = load_img(image_path, target_size=ginger_img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = ginger_model.predict(img_array)[0][0]
        label = ginger_class_labels[1] if prediction >= 0.5 else ginger_class_labels[0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return label, confidence

    uploaded_file = st.file_uploader("Upload a ginger leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.image(temp_path, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            label, confidence = predict_ginger(temp_path)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}**")

        os.remove(temp_path)

# ---------------------------
# OCT Eye Disease Detector
# ---------------------------
elif page == "OCT Eye Disease Detector":
    st.title("👁️ OCT Eye Disease Detector – VGG16")

    # Load OCT model
    @st.cache_resource
    def load_oct_model():
        return load_model("vgg16_oct_model2.keras")

    oct_model = load_oct_model()
    oct_class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    oct_img_size = (160, 160)

    def preprocess_oct_image(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize(oct_img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.cast(img_array, tf.float32) / 255.0
        return tf.expand_dims(img_array, axis=0)

    def predict_oct(image_path):
        input_img = preprocess_oct_image(image_path)
        prediction = oct_model.predict(input_img)
        label = oct_class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return label, confidence

    uploaded_file = st.file_uploader("Upload an OCT retina image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.image(temp_path, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            label, confidence = predict_oct(temp_path)
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2%}**")

        os.remove(temp_path)
