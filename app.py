
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Custom Photo Classifier",
    page_icon="📷",
    layout="wide"
)

MODEL_PATH = "custom_photo_classifier.keras"
CLASS_PATH = "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource(show_spinner="Loading AI model into memory...")
def load_classifier_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_PATH):
        st.error(f"Class names file '{CLASS_PATH}' not found.")
        return []
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

# Load model and classes
model = load_classifier_model()
class_names = load_class_names()

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Information")
    st.markdown(
        """
        - **Architecture**: MobileNetV2 (Transfer Learning)
        - **Input Size**: 224 × 224 pixels
        - **Classes Count**: 10
        """
    )
    with st.expander("📋 View Supported Classes", expanded=True):
        if class_names:
            for i, name in enumerate(class_names):
                st.markdown(f"**{i+1}.** `{name}`")
        else:
            st.warning("No classes loaded.")

# Main content
st.title("📷 Custom Photo Classifier")
st.write("Upload an image of a supported hardware or tech object to classify it.")

uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None and class_names:
    col_img, col_pred = st.columns([1, 1], gap="medium")

    image = Image.open(uploaded_file).convert("RGB")

    with col_img:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

    # Preprocess and predict
    with st.spinner("Classifying image..."):
        image_resized = image.resize(IMG_SIZE)
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

    with col_pred:
        st.subheader("🎯 Prediction Result")
        st.success(f"**Predicted Class:** {predicted_class}")
        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")

        st.markdown("---")
        st.subheader("📊 Top Predictions Breakdown")
        
        # Sort top-5 classes by confidence
        top_indices = np.argsort(predictions)[::-1][:5]
        for idx in top_indices:
            class_label = class_names[idx]
            prob = float(predictions[idx])
            col_label, col_val = st.columns([3, 1])
            with col_label:
                st.write(f"**{class_label}**")
            with col_val:
                st.write(f"{prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))

