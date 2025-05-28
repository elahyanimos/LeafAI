import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import pathlib
import shutil
import datetime

# Page configuration without favicon
st.set_page_config(
    page_title="🌿 Plant Disease Classifier",
    layout="centered"
)

# --- CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f5fff5;
            color: #000000;
        }
        .main {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h3 {
            text-align: center;
            color: #2e7d32;
        }
        .stImage > img {
            border-radius: 12px;
        }
        /* Make all text visible */
        .markdown-text-container, .stMarkdown, .stText, .stInfo, .stSuccess, .stError, .stWarning, .stSidebar, .st-bb, .st-cq, .st-cv, .st-cw {
            color: #000000 !important;
        }
        /* Sidebar background and text */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .sidebar-content, .stSidebar, .css-1d391kg, .css-1v3fvcr, .css-1cpxqw2 {
            color: #000000 !important;
        }
        /* Sidebar titles and headings */
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: #2e7d32 !important;
        }
        /* Info box text */
        .stAlert {
            color: #000000 !important;
        }
        /* Uploaded image caption */
        .caption {
            color: #666666 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/model/plant_disease_classifier.h5"
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Load Class Indices ---
try:
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
except Exception as e:
    st.error(f"❌ Failed to load class index JSON: {e}")
    st.stop()

# --- Preprocessing Function ---
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# --- Prediction Function ---
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = np.max(predictions)
    return predicted_class_name, confidence

# --- Sidebar Upload ---
st.sidebar.title("Upload Image")
uploaded_image = st.sidebar.file_uploader(
    "📸 Take or upload a leaf image (use your camera on mobile)",
    type=["jpg", "jpeg", "png"]
)

# --- About Section ---
st.sidebar.markdown("---")
st.sidebar.title("About")
st.sidebar.markdown("""
This app uses a deep learning model to **classify plant leaf diseases**.

**Instructions:**  
- Tap **"Upload"** to either **take a photo with your camera** or **choose one from your gallery**.  
- Wait a few seconds for prediction.  
- The app will return the **disease type** and **confidence**.
""")

# --- Main App ---
st.markdown("<h1>🌿 Plant Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect plant diseases with a single image.</p>", unsafe_allow_html=True)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image.resize((250, 250)), caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
        st.success("✅ Analysis Complete!")
        st.markdown(f"### 🏷️ Disease: `{prediction}`")
        st.markdown(f"### 📊 Confidence: `{confidence*100:.2f}%`")

        # --- Feedback Section ---
        st.markdown("####  Was this prediction correct?")
        feedback = st.radio(
            "Let us know to help improve the model!",
            ("Yes", "No"),
            horizontal=True,
            index=0
        )

        if feedback == "No":
            corrected_label = st.text_input("If you know the correct label, please enter it (optional):")

            if st.button("🚩 Submit Feedback"):
                # Create feedback directory if not exists
                feedback_dir = os.path.join(working_dir, "feedback")
                os.makedirs(feedback_dir, exist_ok=True)

                # Save image
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"misclassified_{timestamp}.png"
                image_path = os.path.join(feedback_dir, image_filename)
                image.save(image_path)

                # Save metadata
                metadata = {
                    "timestamp": timestamp,
                    "model_prediction": prediction,
                    "confidence": float(confidence),
                    "user_corrected_label": corrected_label if corrected_label else None
                }

                metadata_filename = f"{image_filename.replace('.png', '.json')}"
                with open(os.path.join(feedback_dir, metadata_filename), "w") as f:
                    json.dump(metadata, f)

                st.success("✅ Feedback submitted. Thank you for helping us improve!")
        else:
            if st.button("✅ Submit Feedback"):
                st.success("🎉 Thank you for your feedback!")

    st.markdown("---")
    st.info("Upload another image using the sidebar.")
else:
    st.markdown(
        "<div style='text-align: center; font-size: 1.1em;'>👈 Use the sidebar to upload a leaf image.</div>",
        unsafe_allow_html=True
    )
