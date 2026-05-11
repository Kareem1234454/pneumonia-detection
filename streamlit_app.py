import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, BEST_MODEL_PATH, APP_TITLE, APP_DESCRIPTION, CLASS_NAMES
from src.preprocessing import preprocess_image
from src.segmentation import segment_lungs
from src.gradcam import gradcam_single
from src.evaluate import load_model

# Must be the first Streamlit command
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dark mode aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        text-align: center;
    }
    .panel {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .result-box-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    .result-box-pneumonia {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    model_path = BEST_MODEL_PATH
    if not os.path.exists(model_path):
        fallback = os.path.join(MODEL_DIR, "vgg16_finetuned.h5")
        if os.path.exists(fallback):
            model_path = fallback
        else:
            return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.markdown(f'<div class="main-header"><h1>🩺 {APP_TITLE}</h1><p style="color: #94a3b8; font-size: 1.1rem;">{APP_DESCRIPTION}</p></div>', unsafe_allow_html=True)

    with st.spinner("Loading AI Model (this may take a few seconds)..."):
        model = get_model()
    
    if model is None:
        st.error("Model could not be found. Please ensure the model weights exist in the `models/` directory.")
        return

    st.markdown("### 📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center; margin-top: 0;'>Original Image</h4>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Save to temp file because our backend expects paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Analyzing image..."):
            try:
                # 1. Run Grad-CAM (This handles preprocessing, segmentation, and prediction internally)
                overlay, heatmap, pred_prob = gradcam_single(model, tmp_path, save=False)
                
                # 2. Get Visualization assets
                enhanced, _ = preprocess_image(tmp_path)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                
                seg_res = segment_lungs(enhanced)
                segmented_gray = seg_res['lung_only']
                segmented_rgb = cv2.cvtColor(segmented_gray, cv2.COLOR_GRAY2RGB)
                
                pred_class = int(pred_prob >= 0.5)
                class_name = CLASS_NAMES[pred_class]
                confidence = pred_prob * 100 if pred_class == 1 else (1 - pred_prob) * 100
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                os.remove(tmp_path)
                return

        os.remove(tmp_path)

        with col2:
            st.markdown("### 📊 Diagnostic Results")
            if class_name == "PNEUMONIA":
                st.markdown(f"""
                <div class="result-box-pneumonia">
                    <h2 style="color: #ef4444; margin: 0;">⚠️ PNEUMONIA DETECTED</h2>
                    <h3 style="color: #f8fafc; margin-top: 10px;">Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-normal">
                    <h2 style="color: #10b981; margin: 0;">✅ NORMAL (Healthy)</h2>
                    <h3 style="color: #f8fafc; margin-top: 10px;">Confidence: {confidence:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔬 Clinical Explainability Pipeline")
        
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #94a3b8; font-weight: bold; margin-bottom: 0.5rem;'>🛠️ Preprocessed</div>", unsafe_allow_html=True)
            if enhanced_rgb is not None:
                st.image(enhanced_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with vcol2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #94a3b8; font-weight: bold; margin-bottom: 0.5rem;'>🫁 Segmentation</div>", unsafe_allow_html=True)
            if segmented_rgb is not None:
                st.image(segmented_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with vcol3:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; color: #94a3b8; font-weight: bold; margin-bottom: 0.5rem;'>🌡️ Grad-CAM</div>", unsafe_allow_html=True)
            if overlay is not None:
                # overlay from create_gradcam_overlay is RGB
                st.image(overlay, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
