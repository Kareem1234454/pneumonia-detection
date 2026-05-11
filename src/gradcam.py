import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GRADCAM_LAYER, GRADCAM_ALPHA, FIGURES_DIR, VGG_INPUT_SHAPE, CLASS_NAMES
from src.preprocessing import preprocess_image
from src.segmentation import segment_lungs


def get_gradcam_models(model):
    """
    Returns the grad_model and classifier_model for Grad-CAM.
    Handles both flat models and models with one nested base model (like VGG16 base).
    """
    has_nested = False
    nested_model = None
    nested_idx = -1
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            has_nested = True
            nested_model = layer
            nested_idx = i
            break
            
    if has_nested:
        # Find last conv layer in nested model
        last_conv_name = None
        for layer in reversed(nested_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break
                
        if not last_conv_name:
            raise ValueError("No convolutional layer found in the nested base model.")
            
        last_conv_layer = nested_model.get_layer(last_conv_name)
        grad_model = tf.keras.models.Model(
            [nested_model.inputs],
            [last_conv_layer.output, nested_model.output]
        )
        
        # Classifier model is the rest of the main model
        classifier_input = tf.keras.Input(shape=nested_model.output.shape[1:])
        x = classifier_input
        for layer in model.layers[nested_idx+1:]:
            x = layer(x)
        classifier_model = tf.keras.models.Model(classifier_input, x)
        
        return grad_model, classifier_model, True
        
    else:
        # Flat model
        last_conv_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break
        if not last_conv_name:
            last_conv_name = GRADCAM_LAYER # Fallback from config
            
        last_conv_layer = model.get_layer(last_conv_name)
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output]
        )
        
        # Classifier is the rest of the layers
        layer_idx = model.layers.index(last_conv_layer)
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[layer_idx+1:]:
            x = layer(x)
        classifier_model = tf.keras.models.Model(classifier_input, x)
        
        return grad_model, classifier_model, False


def make_gradcam_heatmap(img_array, model):
    """Generates the Grad-CAM heatmap."""
    grad_model, classifier_model, is_nested = get_gradcam_models(model)
    
    with tf.GradientTape() as tape:
        if is_nested:
            last_conv_layer_output, base_output = grad_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(base_output)
        else:
            last_conv_layer_output = grad_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            
        # Binary classification, so we just use the output probability directly
        class_channel = preds[:, 0]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Global average pooling over the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps by the pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and Normalize
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = tf.maximum(heatmap, 0) / max_val
    else:
        heatmap = tf.maximum(heatmap, 0)
        
    return heatmap.numpy()


def create_gradcam_overlay(base_img, heatmap, alpha=GRADCAM_ALPHA):
    """Overlays the heatmap on the base image (expected RGB uint8)."""
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = cv2.resize(jet_heatmap, (base_img.shape[1], base_img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    
    overlay = cv2.addWeighted(base_img, 1 - alpha, jet_heatmap, alpha, 0)
    return base_img, jet_heatmap, overlay


def gradcam_single(model, img_path, save=True):
    """Runs the Grad-CAM pipeline for a single image, feeding preprocessed & segmented lungs to the model."""
    if not os.path.exists(img_path):
        print(f"Error: Could not find image at {img_path}")
        return None, None, None
        
    # 1. Preprocess & Segment to match training data
    try:
        enhanced, _ = preprocess_image(img_path)
        seg_res = segment_lungs(enhanced)
        lung_only = seg_res['lung_only']
    except Exception as e:
        print(f"Error during preprocessing/segmentation for Grad-CAM: {e}")
        return None, None, None
    
    # 2. Format for model input
    # Convert to RGB because VGG16 expects 3 channels
    img_rgb = cv2.cvtColor(lung_only, cv2.COLOR_GRAY2RGB)
    img_array = img_rgb.astype("float32") / 255.0
    
    # Expand dims for batch size of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    preds = model.predict(img_array, verbose=0)
    pred_prob = preds[0][0]
    pred_class = int(pred_prob >= 0.5)
    
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    original, jet_hm, overlay = create_gradcam_overlay(img_rgb, heatmap)
    
    if save:
        os.makedirs(os.path.join(FIGURES_DIR, "gradcam_samples"), exist_ok=True)
        base_name = os.path.basename(img_path)
        out_path = os.path.join(FIGURES_DIR, "gradcam_samples", f"gradcam_{base_name}")
        
        # Save as BGR for cv2
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM overlay to: {out_path}")
        
        # Plotting
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"Original Image")
        plt.imshow(original)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"Grad-CAM Heatmap")
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        class_name = CLASS_NAMES[pred_class]
        plt.title(f"Overlay (Pred: {class_name})")
        plt.imshow(overlay)
        plt.axis('off')
        
        plot_path = os.path.join(FIGURES_DIR, "gradcam_samples", f"plot_{base_name}")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    return overlay, heatmap, pred_prob
