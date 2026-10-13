"""Streamlit interface for PlantCare AI."""
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from config import MODEL_PATH, CLASS_NAMES_PATH
from src.preprocessing import prepare_batch
from src.utils import load_class_names
from predict import predict_image

st.set_page_config(page_title="PlantCare AI", page_icon="🌱", layout="centered")

DISEASE_GUIDANCE = {
    "healthy": ("Healthy class", "The model classified this leaf as healthy. Continue routine crop monitoring and follow locally recommended care practices."),
    "bacterial_spot": ("Bacterial spot", "General guidance: use disease-free seed/transplants, avoid working with wet plants, reduce leaf wetness, remove infected plant debris, and ask a local extension expert about suitable management."),
    "early_blight": ("Early blight", "General guidance: rotate crops, remove infected plant debris, avoid overhead watering where practical, maintain good airflow, and seek local advice for management options."),
    "late_blight": ("Late blight", "General guidance: isolate suspicious plants where practical, avoid prolonged leaf wetness, remove affected debris safely, and contact an agricultural extension service promptly because this disease can spread quickly."),
    "leaf_mold": ("Leaf mold", "General guidance: improve ventilation, avoid excess humidity and leaf wetness, and remove affected plant material. Confirm the diagnosis with a local expert."),
    "septoria_leaf_spot": ("Septoria leaf spot", "General guidance: remove infected debris, rotate crops, mulch to reduce soil splash, and water near the soil rather than wetting leaves."),
    "spider_mites_two_spotted_spider_mite": ("Spider mites", "General guidance: inspect leaf undersides, reduce plant stress, and seek locally appropriate integrated pest-management advice."),
    "target_spot": ("Target spot", "General guidance: improve airflow, avoid prolonged leaf wetness, remove crop debris, and confirm with an agricultural specialist."),
    "tomato_yellow_leaf_curl_virus": ("Tomato yellow leaf curl virus", "General guidance: monitor whiteflies, use healthy transplants, remove severely affected plants according to local guidance, and consult an agricultural extension expert."),
    "tomato_mosaic_virus": ("Tomato mosaic virus", "General guidance: use clean tools and healthy seed/transplants, wash hands after handling affected plants, and follow local advice for removing infected plants."),
    "powdery_mildew": ("Powdery mildew", "General guidance: improve airflow, avoid excessive nitrogen, and ask a local expert about crop-appropriate controls."),
    "potato": ("Potato class", "Follow local potato crop monitoring and care guidance; consult an expert if symptoms persist."),
    "pepper": ("Pepper class", "Follow local pepper crop monitoring and care guidance; consult an expert if symptoms persist."),
}

def normalize_class_name(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("__", "_")

def get_guidance(class_name):
    key = normalize_class_name(class_name)
    for known, info in DISEASE_GUIDANCE.items():
        if known in key or key in known:
            return info
    if "healthy" in key:
        return DISEASE_GUIDANCE["healthy"]
    return ("General guidance", "The app does not have verified class-specific advice for this label. Check the symptoms with a local agricultural extension service or qualified plant-health expert.")

st.title("🌱 PlantCare AI")
st.subheader("Plant Disease Detection Using Deep Learning")
st.write("Upload a clear photo of a tomato, potato, or pepper leaf to see the class predicted by the trained CNN model.")
st.info("Educational prototype only. Predictions can be wrong and depend on image quality, lighting, plant variety, and training data. Do not use this result as a substitute for expert agricultural advice.")

with st.expander("How does it work?"):
    st.write("A Convolutional Neural Network (CNN) learns visual patterns from labeled leaf images. The app resizes an uploaded image, sends it to the saved model, and displays the class with the highest predicted probability. The confidence score is not a guarantee that the diagnosis is correct.")

if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
    st.warning("The trained model is not available yet. Download and arrange the dataset, then run `python train.py` in the project terminal.")
    st.caption(f"Expected model: {MODEL_PATH.name}; class mapping: {CLASS_NAMES_PATH.name}")
    st.stop()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        # Verify the file can be decoded before showing it.
        from src.preprocessing import load_rgb_image
        preview = load_rgb_image(uploaded_file)
        st.image(preview, caption="Uploaded leaf image", use_container_width=True)
        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing leaf image..."):
                result = predict_image(uploaded_file)
            label = result["class_name"]
            confidence = result["confidence"]
            if "healthy" in label.lower():
                st.success(f"🌿 Predicted class: {label}")
            else:
                st.warning(f"🔎 Predicted class: {label}")
            st.metric("Model confidence", f"{confidence:.1%}")
            if confidence < 0.60:
                st.warning("The model confidence is relatively low. Try a clearer image and seek expert confirmation.")
            title, advice = get_guidance(label)
            st.markdown(f"### {title}")
            st.write(advice)
            st.caption("Model confidence is not the same as a clinically or agriculturally validated probability.")
    except (ValueError, OSError) as exc:
        st.error(f"Could not read this image: {exc}")
    except Exception as exc:
        st.error("Prediction failed. Confirm that the model, class mapping, and installed packages are correct.")
        st.exception(exc)

st.divider()
st.markdown("### About this project")
st.write("**Developer:** Harsh Chaudhary  \n**Project:** PlantCare AI  \n**Technology:** Python, TensorFlow/Keras, CNN, NumPy, Matplotlib, Streamlit, Git and GitHub.")
st.write("This student project demonstrates image classification. It is not a certified diagnostic tool.")
