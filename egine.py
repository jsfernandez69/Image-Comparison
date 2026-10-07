import streamlit as st
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Comparison and Color Uniformity", layout="wide")
st.title("Image Comparison, LAB/LCH and Smoothness")
st.caption("Upload one or two images. Measurements are image-based and depend on lighting, camera settings and color calibration.")


def pil_to_rgb(image):
    """Return RGB uint8 image, honoring EXIF rotation."""
    from PIL import ImageOps
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def resize_pair(a, b):
    """Resize both images to a common size without changing numeric range."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = resize(a, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    b2 = resize(b, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return a2, b2


def lab_lch_metrics(rgb):
    """Calculate representative CIE Lab/LCh and image homogeneity metrics."""
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_float)

    # Robust representative color. Median reduces the effect of isolated dust/defects.
    representative = np.median(lab.reshape(-1, 3), axis=0)
    L, a, b = [float(x) for x in representative]
    C = float(np.hypot(a, b))
    h = float((np.degrees(np.arctan2(b, a)) + 360.0) % 360.0)

    # Delta E 2000 of every pixel from the representative color.
    ref = np.broadcast_to(representative, lab.shape)
    de = deltaE_ciede2000(lab, ref)
    mean_de = float(np.mean(de))
    median_de = float(np.median(de))
    p95_de = float(np.percentile(de, 95))

    # Local texture component based on smoothed lightness gradients.
    L_smooth = gaussian_filter(lab[..., 0], sigma=1.0)
    gx = sobel(L_smooth, axis=1, mode="reflect") / 8.0
    gy = sobel(L_smooth, axis=0, mode="reflect") / 8.0
    gradient = np.hypot(gx, gy)
    gradient_p95 = float(np.percentile(gradient, 95))

    # Custom, transparent 0-100 indices. 100 means all pixels are identical.
    # The scale factor 5 makes the score useful for typical photographs.
    color_variation = 0.70 * mean_de + 0.30 * p95_de
    homogeneity = float(np.clip(100.0 * np.exp(-color_variation / 5.0), 0, 100))
    smoothness = float(np.clip(100.0 * np.exp(-(0.80 * color_variation + 0.20 * gradient_p95) / 5.0), 0, 100))

    return {
        "L": L, "a": a, "b": b, "C": C, "h": h,
        "mean_de": mean_de, "median_de": median_de, "p95_de": p95_de,
        "gradient_p95": gradient_p95,
        "homogeneity": homogeneity, "smoothness": smoothness,
        "de_map": de,
    }


def result_panel(name, rgb):
    m = lab_lch_metrics(rgb)
    st.subheader(name)
    st.image(rgb, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("L*", f"{m['L']:.2f}")
    c2.metric("a*", f"{m['a']:.2f}")
    c3.metric("b*", f"{m['b']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("C*", f"{m['C']:.2f}")
    c5.metric("h°", f"{m['h']:.1f}°")
    c6.metric("Smoothness", f"{m['smoothness']:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("Color homogeneity", f"{m['homogeneity']:.1f}%")
    c8.metric("Mean ΔE00", f"{m['mean_de']:.2f}")
    c9.metric("95th percentile ΔE00", f"{m['p95_de']:.2f}")

    fig, ax = plt.subplots()
    im = ax.imshow(m["de_map"], cmap="magma")
    ax.set_title("Color deviation map (ΔE00 from representative color)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="ΔE00")
    st.pyplot(fig, clear_figure=True)
    return m


file1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg", "tif", "tiff"], key="img1")
file2 = st.file_uploader("Upload Image 2 (optional)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="img2")

if file1:
    rgb1 = pil_to_rgb(Image.open(file1))

    if file2:
        rgb2 = pil_to_rgb(Image.open(file2))
        rgb1_compare, rgb2_compare = resize_pair(rgb1, rgb2)

        gray1 = np.dot(rgb1_compare[..., :3], [0.2126, 0.7152, 0.0722])
        gray2 = np.dot(rgb2_compare[..., :3], [0.2126, 0.7152, 0.0722])
        score, diff = ssim(gray1, gray2, data_range=255, full=True)
        st.header(f"Structural similarity: {score * 100:.2f}%")
        st.caption("SSIM is most meaningful when both images show the same scene and are aligned.")

        col1, col2 = st.columns(2)
        with col1:
            metrics1 = result_panel("Image 1", rgb1)
        with col2:
            metrics2 = result_panel("Image 2", rgb2)

        rep1 = np.array([metrics1["L"], metrics1["a"], metrics1["b"]])
        rep2 = np.array([metrics2["L"], metrics2["a"], metrics2["b"]])
        overall_de = float(deltaE_ciede2000(rep1, rep2))
        st.metric("Representative color difference between images (ΔE00)", f"{overall_de:.2f}")

        st.subheader("Structural difference map")
        st.image(((1.0 - diff) * 255.0).clip(0, 255).astype(np.uint8), use_container_width=True)
    else:
        result_panel("Image 1", rgb1)

st.divider()
st.info(
    "LAB/LCH values from a photograph are estimates, not instrument-grade measurements. "
    "For reliable print-color results, use controlled D50 lighting, fixed exposure and white balance, "
    "a neutral reference or color target, and ideally an embedded/correct ICC profile."
)
st.caption(
    "Scoring definition: 100% requires identical pixel color across the analyzed image. "
    "The index decreases exponentially with average and high-percentile ΔE00 variation; "
    "the smoothness score also includes local L* gradients. This is a custom QC index, not an ISO standard."
)
