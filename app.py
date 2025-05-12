
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import cm

# Load the model
MODEL_PATH = "unet_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Constants
IMG_HEIGHT = 96
IMG_WIDTH = 128
N_CLASSES = 23

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[..., :3]
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def postprocess_mask(mask: np.ndarray, colorize=True) -> Image.Image:
    mask = np.argmax(mask[0], axis=-1).astype(np.uint8)
    if colorize:
        colormap = cm.get_cmap("nipy_spectral")
        color_mask = colormap(mask / float(N_CLASSES - 1))[:, :, :3]
        color_mask = (color_mask * 255).astype(np.uint8)
        return Image.fromarray(color_mask)
    else:
        return Image.fromarray(mask, mode="L")

def overlay_mask_on_image(image: Image.Image, mask: Image.Image, alpha=0.5) -> Image.Image:
    image = image.resize(mask.size)
    return Image.blend(image.convert("RGBA"), mask.convert("RGBA"), alpha)

def main():
    st.set_page_config(page_title="U-Net Segmentation", layout="centered")
    st.sidebar.title("Instructions")
    st.sidebar.write("1. Upload an image (RGB).\n"
                     "2. Optionally upload the ground truth mask.\n"
                     "3. Click 'Predict' to see the result.\n"
                     "4. The app shows the prediction and overlay.")

    st.title("🧠 U-Net Semantic Segmentation")
    st.write("Upload an image and (optionally) its label mask, then click Predict.")

    uploaded_file = st.file_uploader("Upload input image", type=["jpg", "jpeg", "png"], key="input_image")
    label_file = st.file_uploader("Upload true label mask (optional)", type=["jpg", "jpeg", "png"], key="label_image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        image = None

    if label_file is not None:
        label_image = Image.open(label_file).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(label_image, caption="Ground Truth Mask", use_column_width=True)
    else:
        label_image = None

    if st.button("Predict"):
        if image is None:
            st.warning("Please upload an image before predicting.")
        else:
            with st.spinner("Predicting..."):
                input_array = preprocess_image(image)
                prediction = model.predict(input_array)
                mask_image = postprocess_mask(prediction, colorize=True)
                overlay_image = overlay_mask_on_image(image, mask_image)

            st.image(mask_image, caption="Predicted Segmentation Mask", use_column_width=True)
            st.image(overlay_image, caption="Overlay: Input + Prediction", use_column_width=True)

if __name__ == "__main__":
    main()
