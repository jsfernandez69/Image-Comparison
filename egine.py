import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Comparison and Color Calibration", layout="wide")
st.title("Image Comparison, LAB/LCH, Smoothness and Calibration")
st.caption(
    "Upload one or two images. Measurements are image-based and depend on lighting, "
    "camera settings, ICC handling and calibration."
)


def pil_to_rgb(image):
    """Return an EXIF-corrected RGB uint8 image."""
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def resize_pair(a, b):
    """Resize both images to a common size without changing numeric range."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = resize(a, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    b2 = resize(b, (h, w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return a2, b2


def lab_to_lch(lab_value):
    """Convert one Lab triplet to LCh, with hue in degrees from 0 to 360."""
    L, a, b = [float(x) for x in lab_value]
    C = float(np.hypot(a, b))
    h = float((np.degrees(np.arctan2(b, a)) + 360.0) % 360.0)
    return L, C, h


def lch_to_lab(L, C, h):
    """Convert one LCh triplet to Lab."""
    angle = np.radians(float(h))
    return np.array([float(L), float(C) * np.cos(angle), float(C) * np.sin(angle)])


def fit_calibration(rows):
    """Fit affine raw-Lab to reference-Lab correction by least squares."""
    raw = np.asarray([row["raw_lab"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_lab"] for row in rows], dtype=float)
    design = np.column_stack([raw, np.ones(len(raw))])
    coefficients, _, _, _ = np.linalg.lstsq(design, reference, rcond=None)
    return coefficients


def apply_calibration(lab_value, coefficients):
    """Apply the fitted affine calibration to one Lab triplet."""
    vector = np.append(np.asarray(lab_value, dtype=float), 1.0)
    corrected = vector @ coefficients
    corrected[0] = np.clip(corrected[0], 0.0, 100.0)
    return corrected


def calibration_error(rows, coefficients):
    """Return calibration-set Delta E 2000 errors."""
    errors = []
    for row in rows:
        predicted = apply_calibration(row["raw_lab"], coefficients)
        reference = np.asarray(row["reference_lab"], dtype=float)
        errors.append(float(deltaE_ciede2000(predicted, reference)))
    return np.asarray(errors)


def lab_lch_metrics(rgb, crop_percent=50, calibration=None):
    """Calculate representative Lab/LCh and image homogeneity metrics."""
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_float)

    # Analyze a centered crop to reduce border, shadow and background influence.
    crop_percent = float(np.clip(crop_percent, 10, 100)) / 100.0
    height, width = lab.shape[:2]
    y_margin = int(height * (1.0 - crop_percent) / 2.0)
    x_margin = int(width * (1.0 - crop_percent) / 2.0)
    crop = lab[y_margin:height - y_margin or height, x_margin:width - x_margin or width]

    raw_representative = np.median(crop.reshape(-1, 3), axis=0)
    representative = (
        apply_calibration(raw_representative, calibration)
        if calibration is not None
        else raw_representative.copy()
    )

    L, C, hue = lab_to_lch(representative)
    raw_L, raw_C, raw_hue = lab_to_lch(raw_representative)

    # Pixel variation remains based on the captured image, not corrected values.
    ref = np.broadcast_to(raw_representative, lab.shape)
    de = deltaE_ciede2000(lab, ref)
    mean_de = float(np.mean(de))
    median_de = float(np.median(de))
    p95_de = float(np.percentile(de, 95))

    L_smooth = gaussian_filter(lab[..., 0], sigma=1.0)
    gx = sobel(L_smooth, axis=1, mode="reflect") / 8.0
    gy = sobel(L_smooth, axis=0, mode="reflect") / 8.0
    gradient_p95 = float(np.percentile(np.hypot(gx, gy), 95))

    color_variation = 0.70 * mean_de + 0.30 * p95_de
    homogeneity = float(np.clip(100.0 * np.exp(-color_variation / 5.0), 0, 100))
    smoothness = float(
        np.clip(
            100.0 * np.exp(-(0.80 * color_variation + 0.20 * gradient_p95) / 5.0),
            0,
            100,
        )
    )

    return {
        "L": L,
        "a": float(representative[1]),
        "b": float(representative[2]),
        "C": C,
        "h": hue,
        "raw_L": raw_L,
        "raw_a": float(raw_representative[1]),
        "raw_b": float(raw_representative[2]),
        "raw_C": raw_C,
        "raw_h": raw_hue,
        "raw_lab": raw_representative,
        "corrected_lab": representative,
        "mean_de": mean_de,
        "median_de": median_de,
        "p95_de": p95_de,
        "gradient_p95": gradient_p95,
        "homogeneity": homogeneity,
        "smoothness": smoothness,
        "de_map": de,
    }


if "calibration_rows" not in st.session_state:
    st.session_state.calibration_rows = []

st.sidebar.header("Measurement settings")
crop_percent = st.sidebar.slider(
    "Centered analysis area (%)", min_value=10, max_value=100, value=50, step=5
)
st.sidebar.caption("Use a smaller centered area when the patch has borders, glare or shadows.")

# A minimum of four points is mathematically required for this affine model.
# More varied samples are strongly preferred for stability and validation.
calibration_coefficients = None
if len(st.session_state.calibration_rows) >= 4:
    calibration_coefficients = fit_calibration(st.session_state.calibration_rows)


def result_panel(name, rgb):
    metrics = lab_lch_metrics(rgb, crop_percent, calibration_coefficients)
    st.subheader(name)
    st.image(rgb, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("L*", f"{metrics['L']:.2f}")
    c2.metric("a*", f"{metrics['a']:.2f}")
    c3.metric("b*", f"{metrics['b']:.2f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("C*", f"{metrics['C']:.2f}")
    c5.metric("h°", f"{metrics['h']:.1f}°")
    c6.metric("Smoothness", f"{metrics['smoothness']:.1f}%")
    c7, c8, c9 = st.columns(3)
    c7.metric("Color homogeneity", f"{metrics['homogeneity']:.1f}%")
    c8.metric("Mean ΔE00", f"{metrics['mean_de']:.2f}")
    c9.metric("95th percentile ΔE00", f"{metrics['p95_de']:.2f}")

    if calibration_coefficients is not None:
        st.caption(
            f"Raw capture: L* {metrics['raw_L']:.2f}, C* {metrics['raw_C']:.2f}, "
            f"h {metrics['raw_h']:.1f}°. Displayed values are calibrated."
        )
    else:
        st.caption("Displayed Lab/LCh values are raw, uncalibrated image estimates.")

    fig, ax = plt.subplots()
    image_plot = ax.imshow(metrics["de_map"], cmap="magma")
    ax.set_title("Color deviation map (ΔE00 from representative color)")
    ax.axis("off")
    fig.colorbar(image_plot, ax=ax, label="ΔE00")
    st.pyplot(fig, clear_figure=True)
    return metrics


file1 = st.file_uploader(
    "Upload Image 1", type=["png", "jpg", "jpeg", "tif", "tiff"], key="img1"
)
file2 = st.file_uploader(
    "Upload Image 2 (optional)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key="img2",
)

metrics1 = None
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
        overall_de = float(
            deltaE_ciede2000(metrics1["corrected_lab"], metrics2["corrected_lab"])
        )
        st.metric("Representative color difference between images (ΔE00)", f"{overall_de:.2f}")
        st.subheader("Structural difference map")
        st.image(
            ((1.0 - diff) * 255.0).clip(0, 255).astype(np.uint8),
            use_container_width=True,
        )
    else:
        metrics1 = result_panel("Image 1", rgb1)


st.divider()
st.header("Microscope calibration")
st.write(
    "Build a correction for this microscope, light and camera setup. Keep illumination, "
    "exposure, white balance, magnification and sample position unchanged."
)

if metrics1 is None:
    st.info("Upload Image 1 to add a calibration sample.")
else:
    st.write(
        f"Current raw reading: L* {metrics1['raw_L']:.2f}, "
        f"C* {metrics1['raw_C']:.2f}, h {metrics1['raw_h']:.1f}°."
    )
    with st.form("add_calibration_sample", clear_on_submit=False):
        sample_name = st.text_input(
            "Sample name", value=f"Sample {len(st.session_state.calibration_rows) + 1}"
        )
        r1, r2, r3 = st.columns(3)
        reference_L = r1.number_input("Reference L*", min_value=0.0, max_value=100.0, value=50.0)
        reference_C = r2.number_input("Reference C*", min_value=0.0, value=55.0)
        reference_h = r3.number_input("Reference h°", min_value=0.0, max_value=360.0, value=247.0)
        submitted = st.form_submit_button("Add calibration sample")
        if submitted:
            st.session_state.calibration_rows.append(
                {
                    "name": sample_name,
                    "raw_lab": metrics1["raw_lab"].tolist(),
                    "reference_lab": lch_to_lab(reference_L, reference_C, reference_h).tolist(),
                }
            )
            st.rerun()

if st.session_state.calibration_rows:
    display_rows = []
    for row in st.session_state.calibration_rows:
        raw_L, raw_C, raw_h = lab_to_lch(row["raw_lab"])
        ref_L, ref_C, ref_h = lab_to_lch(row["reference_lab"])
        display_rows.append(
            {
                "Sample": row["name"],
                "Raw L*": round(raw_L, 2),
                "Raw C*": round(raw_C, 2),
                "Raw h°": round(raw_h, 1),
                "Reference L*": round(ref_L, 2),
                "Reference C*": round(ref_C, 2),
                "Reference h°": round(ref_h, 1),
            }
        )
    st.dataframe(display_rows, use_container_width=True)

    if len(st.session_state.calibration_rows) < 4:
        st.warning(
            f"Add at least {4 - len(st.session_state.calibration_rows)} more varied sample(s) "
            "before calibration can be applied. Eight or more samples are preferable."
        )
    else:
        errors = calibration_error(st.session_state.calibration_rows, calibration_coefficients)
        e1, e2, e3 = st.columns(3)
        e1.metric("Calibration samples", len(errors))
        e2.metric("Mean fitting ΔE00", f"{np.mean(errors):.2f}")
        e3.metric("Maximum fitting ΔE00", f"{np.max(errors):.2f}")
        st.success("Calibration is active. Re-upload or refresh the image results to view corrected values.")
        st.caption(
            "These are fitting errors on the same samples used to create the model, not an independent "
            "accuracy guarantee. Test separate validation patches before relying on the correction."
        )

    if st.button("Clear calibration"):
        st.session_state.calibration_rows = []
        st.rerun()

st.divider()
st.info(
    "LAB/LCH values from a photograph are estimates, not instrument-grade measurements. "
    "For reliable print-color results, use fixed capture settings, controlled illumination, "
    "a neutral reference or color target, and ideally an embedded/correct ICC profile."
)
st.caption(
    "Calibration corrects L*, a* and b* with an affine least-squares model, then calculates "
    "C* and hue from the corrected a* and b*. It is specific to the unchanged capture setup."
)

