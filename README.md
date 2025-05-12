# U-Net Semantic Segmentation Web App

This Streamlit web app uses a trained **U-Net** model to perform semantic segmentation on uploaded images. It supports **multi-class segmentation** (e.g., 23 classes) and provides visual feedback including a color-coded mask and image overlay.

---

## 🚀 Features

- Upload any image (RGB format)
- Predict segmentation masks using a trained U-Net model
- Color-coded prediction output using a colormap
- Overlay prediction on the original image for easier visual comparison
- Optionally upload a true label mask
- Runs entirely in your browser with Streamlit

---

## 📦 Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- Pillow
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏁 Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL provided by Streamlit in your browser (usually http://localhost:8501).

5. Upload an image to get the segmentation mask prediction. Optionally, upload a true label mask to compare.

---

## 🧠 Model Training Overview

The U-Net model used in this app is a convolutional neural network designed for semantic segmentation tasks. It consists of:

- Encoder blocks that progressively downsample the input image while extracting features.
- Decoder blocks that upsample and combine features to produce pixel-wise class predictions.
- Skip connections between encoder and decoder blocks to preserve spatial information.

The model was trained on a multi-class dataset with 23 classes. Training involved data preprocessing, augmentation, and optimization using TensorFlow/Keras.

For detailed model architecture and training code, refer to the `Unet.ipynb` notebook included in the project.

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit app entry point
├── Unet.ipynb             # Jupyter notebook with model building and training code
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── assets/                # (Optional) Folder for images, colormaps, or other assets
```

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or support, please contact [sakib.hussen@goucher.edu].
