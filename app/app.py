import os
import sys
import numpy as np
import cv2
import gradio as gr
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, BEST_MODEL_PATH, APP_TITLE, APP_DESCRIPTION, CLASS_NAMES
from src.preprocessing import preprocess_image
from src.segmentation import segment_lungs
from src.gradcam import gradcam_single
from src.evaluate import load_model

# Global variable to cache the model
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        vgg_ft = os.path.join(MODEL_DIR, 'vgg16_finetuned.h5')
        vgg_b = os.path.join(MODEL_DIR, 'vgg16_best.h5')
        if os.path.exists(vgg_ft):
            _MODEL = load_model(vgg_ft)
        elif os.path.exists(vgg_b):
            _MODEL = load_model(vgg_b)
        else:
            _MODEL = load_model(BEST_MODEL_PATH)
    return _MODEL


def predict(image_path):
    if image_path is None:
        return None, None, "Please upload an image."
        
    try:
        model = get_model()
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"
    
    # Run Grad-CAM pipeline
    try:
        overlay, heatmap, pred_prob = gradcam_single(model, image_path, save=False)
    except Exception as e:
        return None, None, f"Error generating Grad-CAM: {str(e)}"
    
    if overlay is None:
        return None, None, "Error processing image."
        
    pred_class_idx = int(pred_prob >= 0.5)
    class_name = CLASS_NAMES[pred_class_idx]
    
    # Format confidence
    confidence = pred_prob if pred_class_idx == 1 else 1 - pred_prob
    confidence_str = f"{confidence * 100:.2f}%"
    
    # Preprocessing visualization
    try:
        enhanced, normalized = preprocess_image(image_path)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Segmentation visualization
        seg_res = segment_lungs(enhanced)
        segmented_gray = seg_res['lung_only']
        segmented_rgb = cv2.cvtColor(segmented_gray, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        enhanced_rgb = None
        segmented_rgb = None
        print(f"Preprocessing/Segmentation error: {e}")
    
    # Output message with dynamic styling
    if class_name == "PNEUMONIA":
        color = "#ef4444" # red
        icon = "⚠️"
    else:
        color = "#10b981" # green
        icon = "✅"
        
    msg = f"""
    <div style="text-align: center; padding: 1.5rem; background: rgba(15, 23, 42, 0.6); border-radius: 12px; border: 1px solid {color}50;">
        <h2 style="color: {color}; margin: 0; font-size: 2.2rem; text-shadow: 0 0 10px {color}40;">{icon} Diagnosis: {class_name}</h2>
        <h3 style="color: #94a3b8; margin-top: 0.5rem; font-weight: 500;">AI Confidence: {confidence_str}</h3>
    </div>
    """
    
    return enhanced_rgb, segmented_rgb, overlay, msg


# Custom CSS for a premium look
CUSTOM_CSS = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
}
.header {
    text-align: center;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(56,189,248,0.05) 0%, rgba(0,0,0,0) 70%);
    pointer-events: none;
}
.header h1 {
    color: #e0f2fe;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 15px rgba(56, 189, 248, 0.4);
    letter-spacing: -0.5px;
}
.header p {
    color: #94a3b8;
    font-size: 1.15rem;
    max-width: 600px;
    margin: 0 auto;
}
.panel {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.panel:hover {
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
}
"""

def build_interface():
    # Pre-load model on startup to save time during prediction
    try:
        get_model()
        print("Model loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        
    theme = gr.themes.Default(
        primary_hue="sky",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0b0f19",
        body_background_fill_dark="#0b0f19",
        block_background_fill="#1e293b",
        block_border_width="1px",
        block_border_color="#334155",
        block_radius="16px",
    )

    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div class="header">
                    <h1>🩺 {APP_TITLE}</h1>
                    <p>{APP_DESCRIPTION}</p>
                </div>
                """)
        
        with gr.Row():
            # Left Column: Upload
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📥 1. Upload Chest X-Ray")
                    image_input = gr.Image(type="filepath", label="Input Image", elem_classes="panel")
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    
            # Right Column: Results
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 📊 2. AI Analysis Results")
                    result_text = gr.HTML(
                        "<div style='text-align: center; padding: 2rem; color: #64748b;'>Upload an image and click 'Analyze Image' to see the diagnosis.</div>", 
                        elem_classes="panel"
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("<div style='text-align: center; color: #94a3b8; margin-bottom: 0.5rem;'>🛠️ Preprocessed</div>")
                            preprocessed_img = gr.Image(label="Enhanced Image", interactive=False, elem_classes="panel", show_label=False)
                        with gr.Column():
                            gr.Markdown("<div style='text-align: center; color: #94a3b8; margin-bottom: 0.5rem;'>🫁 Segmentation</div>")
                            segmented_img = gr.Image(label="Segmented Lungs", interactive=False, elem_classes="panel", show_label=False)
                        with gr.Column():
                            gr.Markdown("<div style='text-align: center; color: #94a3b8; margin-bottom: 0.5rem;'>🌡️ Grad-CAM</div>")
                            gradcam_img = gr.Image(label="Grad-CAM Overlay", interactive=False, elem_classes="panel", show_label=False)
                            
        analyze_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[preprocessed_img, segmented_img, gradcam_img, result_text]
        )
        
    demo.theme = theme
    demo.css = CUSTOM_CSS
    return demo
