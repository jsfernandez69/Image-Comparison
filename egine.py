import streamlit as st
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

st.title("Image Similarity Scorer")

img1_file = st.file_uploader(
    "Upload Image 1",
    type=["png", "jpg", "jpeg"],
    key="i1"
)

img2_file = st.file_uploader(
    "Upload Image 2",
    type=["png", "jpg", "jpeg"],
    key="i2"
)

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("L")
    img2 = Image.open(img2_file).convert("L")

    # Resize to same dimensions
    width = min(img1.width, img2.width)
    height = min(img1.height, img2.height)

    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Calculate SSIM
    score, diff = ssim(img1_array, img2_array, full=True)

    st.subheader(f"Similarity Score: {score * 100:.2f}%")

    st.image(
        [Image.fromarray(img1_array), Image.fromarray(img2_array)],
        caption=["Image 1", "Image 2"]
    )

    # Create heatmap
    heatmap = (1 - diff) * 255

    st.subheader("Difference Heatmap")
    st.image(heatmap.astype("uint8"))
