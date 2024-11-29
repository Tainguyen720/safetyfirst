import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  

# Function to normalize image
def normalize_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

# Streamlit interface
st.title("Object Detection and LIME Explanation")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Ensure predict_fn returns fixed-size output
def predict_fn(images):
    """
    Generate predictions for LIME by applying YOLO to perturbed images.
    """
    predictions = []
    for img in images:
        result = model(img)  # YOLO inference
        result_boxes = []
        for single_result in result:
            for box in single_result.boxes:
                result_boxes.append(box.conf.item())  # Confidence as prediction

        # Pad or truncate to ensure uniform size
        fixed_length = 10
        if len(result_boxes) < fixed_length:
            result_boxes += [0] * (fixed_length - len(result_boxes))
        else:
            result_boxes = result_boxes[:fixed_length]

        predictions.append(result_boxes)
    return np.array(predictions)

# LIME explanation function
def explain_with_lime(image, model):
    explainer = LimeImageExplainer()

    # Explain instance
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=5,
        hide_color=None,
        num_samples=1000,
    )

    # Black out part of img that are non-contributing
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
    )
    temp = normalize_image(temp)
    st.image(temp, caption="LIME Explanation - Contributing Towards Prediction", use_column_width=True)

    # Explanation with pros and cons (both positive and negative features)
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
    )
    temp = normalize_image(temp)
    st.image(temp, caption="LIME Explanation - Pros and Cons", use_column_width=True)

    # Heatmap visualization
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    st.subheader("LIME Heatmap")
    plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    st.pyplot(plt)

# Main workflow
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = np.array(img)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display original image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model(img_rgb)

    # Display YOLO visual inference results
    st.subheader("YOLO Visual Inference")
    annotated_image = results[0].plot()  # YOLO's annotated image output
    st.image(annotated_image, caption="YOLO Inference Results", use_column_width=True)

    # Explain with LIME
    st.subheader("LIME Explanation")
    explain_with_lime(img_rgb, model)
