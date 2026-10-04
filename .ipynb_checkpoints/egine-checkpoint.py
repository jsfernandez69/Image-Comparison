{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "from skimage.metrics import structural_similarity as ssim\n",
    "\n",
    "st.title(\"Image Similarity Scorer\")\n",
    "\n",
    "img1_file = st.file_uploader(\n",
    "    \"Upload Image 1\",\n",
    "    type=[\"png\", \"jpg\", \"jpeg\"],\n",
    "    key=\"i1\"\n",
    ")\n",
    "\n",
    "img2_file = st.file_uploader(\n",
    "    \"Upload Image 2\",\n",
    "    type=[\"png\", \"jpg\", \"jpeg\"],\n",
    "    key=\"i2\"\n",
    ")\n",
    "\n",
    "if img1_file and img2_file:\n",
    "\n",
    "    img1 = Image.open(img1_file).convert(\"L\")\n",
    "    img2 = Image.open(img2_file).convert(\"L\")\n",
    "\n",
    "    # Resize to same dimensions\n",
    "    width = min(img1.width, img2.width)\n",
    "    height = min(img1.height, img2.height)\n",
    "\n",
    "    img1 = img1.resize((width, height))\n",
    "    img2 = img2.resize((width, height))\n",
    "\n",
    "    img1_array = np.array(img1)\n",
    "    img2_array = np.array(img2)\n",
    "\n",
    "    # Calculate SSIM\n",
    "    score, diff = ssim(img1_array, img2_array, full=True)\n",
    "\n",
    "    st.subheader(f\"Similarity Score: {score * 100:.2f}%\")\n",
    "\n",
    "    st.image(\n",
    "        [Image.fromarray(img1_array), Image.fromarray(img2_array)],\n",
    "        caption=[\"Image 1\", \"Image 2\"]\n",
    "    )\n",
    "\n",
    "    # Create heatmap\n",
    "    heatmap = (1 - diff) * 255\n",
    "\n",
    "    st.subheader(\"Difference Heatmap\")\n",
    "    st.image(heatmap.astype(\"uint8\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (Geochem Pro)",
   "language": "python",
   "name": "geochem_pro"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
