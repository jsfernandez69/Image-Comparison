import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, sobel, label
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Image Comparison, Calibration and White Ink Analysis",
    layout="wide",
)
st.title("Image Comparison, LAB/LCH, Calibration and White Ink Quality")
st.caption(
    "Image-based measurements depend on lighting, camera settings, ICC handling, "
    "focus, magnification and calibration. Keep capture conditions unchanged when "
    "comparing samples."
)

SUPPORTED_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]


# -----------------------------------------------------------------------------
# General image and color helpers
# -----------------------------------------------------------------------------
def pil_to_rgb(image):
    """Return an EXIF-corrected RGB uint8 image."""
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def resize_pair(a, b):
    """Resize two RGB images to a common size."""
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    a2 = resize(a, (height, width), preserve_range=True, anti_aliasing=True).astype(
        np.uint8
    )
    b2 = resize(b, (height, width), preserve_range=True, anti_aliasing=True).astype(
        np.uint8
    )
    return a2, b2


def center_crop(array, crop_percent):
    """Return a centered crop from a 2D or 3D array."""
    fraction = float(np.clip(crop_percent, 10, 100)) / 100.0
    height, width = array.shape[:2]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    return array[y0 : y0 + crop_height, x0 : x0 + crop_width]


def rgb_luminance(rgb):
    """Calculate normalized sRGB relative luminance Y in the range 0 to 1."""
    srgb = rgb.astype(np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )


def grayscale_255(rgb):
    """Return perceptual grayscale in the range 0 to 255."""
    return np.dot(
        rgb[..., :3].astype(np.float64), np.array([0.2126, 0.7152, 0.0722])
    )


def lab_to_lch(lab_value):
    """Convert one Lab triplet to LCh, with hue in degrees from 0 to 360."""
    lightness, a_value, b_value = [float(x) for x in lab_value]
    chroma = float(np.hypot(a_value, b_value))
    hue = float((np.degrees(np.arctan2(b_value, a_value)) + 360.0) % 360.0)
    return lightness, chroma, hue


def lch_to_lab(lightness, chroma, hue):
    """Convert one LCh triplet to Lab."""
    angle = np.radians(float(hue))
    return np.array(
        [
            float(lightness),
            float(chroma) * np.cos(angle),
            float(chroma) * np.sin(angle),
        ]
    )


# -----------------------------------------------------------------------------
# Microscope calibration
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# General LAB/LCH and smoothness analysis
# -----------------------------------------------------------------------------
def lab_lch_metrics(rgb, crop_percent=50, calibration=None):
    """Calculate representative Lab/LCh and image homogeneity metrics."""
    rgb_float = rgb.astype(np.float32) / 255.0
    lab = rgb2lab(rgb_float)
    crop = center_crop(lab, crop_percent)

    raw_representative = np.median(crop.reshape(-1, 3), axis=0)
    representative = (
        apply_calibration(raw_representative, calibration)
        if calibration is not None
        else raw_representative.copy()
    )

    lightness, chroma, hue = lab_to_lch(representative)
    raw_lightness, raw_chroma, raw_hue = lab_to_lch(raw_representative)

    reference = np.broadcast_to(raw_representative, lab.shape)
    de_map = deltaE_ciede2000(lab, reference)
    crop_de = center_crop(de_map, crop_percent)

    mean_de = float(np.mean(crop_de))
    median_de = float(np.median(crop_de))
    p95_de = float(np.percentile(crop_de, 95))

    lightness_crop = crop[..., 0]
    lightness_smooth = gaussian_filter(lightness_crop, sigma=1.0)
    gx = sobel(lightness_smooth, axis=1, mode="reflect") / 8.0
    gy = sobel(lightness_smooth, axis=0, mode="reflect") / 8.0
    gradient_p95 = float(np.percentile(np.hypot(gx, gy), 95))

    color_variation = 0.70 * mean_de + 0.30 * p95_de
    homogeneity = float(
        np.clip(100.0 * np.exp(-color_variation / 5.0), 0.0, 100.0)
    )
    smoothness = float(
        np.clip(
            100.0
            * np.exp(
                -(0.80 * color_variation + 0.20 * gradient_p95) / 5.0
            ),
            0.0,
            100.0,
        )
    )

    return {
        "L": lightness,
        "a": float(representative[1]),
        "b": float(representative[2]),
        "C": chroma,
        "h": hue,
        "raw_L": raw_lightness,
        "raw_a": float(raw_representative[1]),
        "raw_b": float(raw_representative[2]),
        "raw_C": raw_chroma,
        "raw_h": raw_hue,
        "raw_lab": raw_representative,
        "corrected_lab": representative,
        "mean_de": mean_de,
        "median_de": median_de,
        "p95_de": p95_de,
        "gradient_p95": gradient_p95,
        "homogeneity": homogeneity,
        "smoothness": smoothness,
        "de_map": de_map,
    }


# -----------------------------------------------------------------------------
# White ink analysis
# -----------------------------------------------------------------------------
def analyze_surface(rgb, crop_percent, pinhole_sensitivity, minimum_pinhole_area):
    """
    Analyze local defects and low-frequency nonuniformity.

    Pinholes are pixels darker than their local background by a user-selected
    luminance difference. Mottle is estimated from the variation remaining in a
    low-frequency luminance field after suppressing fine texture.
    """
    gray = center_crop(grayscale_255(rgb), crop_percent)

    local_background = gaussian_filter(gray, sigma=5.0, mode="reflect")
    dark_difference = local_background - gray
    raw_pinhole_mask = dark_difference > float(pinhole_sensitivity)

    labels, object_count = label(raw_pinhole_mask)
    filtered_mask = np.zeros_like(raw_pinhole_mask, dtype=bool)
    pinhole_count = 0

    for object_id in range(1, object_count + 1):
        object_mask = labels == object_id
        area = int(np.sum(object_mask))
        if area >= int(minimum_pinhole_area):
            filtered_mask |= object_mask
            pinhole_count += 1

    pinhole_area_percent = float(100.0 * np.mean(filtered_mask))
    coverage = float(np.clip(100.0 - pinhole_area_percent, 0.0, 100.0))

    image_scale = max(gray.shape)
    mottle_sigma = max(8.0, image_scale / 35.0)
    mottle_field = gaussian_filter(gray, sigma=mottle_sigma, mode="reflect")
    mottle_field = mottle_field - float(np.mean(mottle_field))
    mottle_index = float(np.std(mottle_field))

    fine_field = gray - gaussian_filter(gray, sigma=2.0, mode="reflect")
    fine_variation = float(np.std(fine_field))

    # Empirical image-quality scores, intended for relative comparison only.
    mottle_score = float(np.clip(100.0 * np.exp(-mottle_index / 6.0), 0.0, 100.0))
    texture_score = float(np.clip(100.0 * np.exp(-fine_variation / 10.0), 0.0, 100.0))
    surface_smoothness = float(
        np.clip(0.55 * mottle_score + 0.30 * texture_score + 0.15 * coverage, 0.0, 100.0)
    )

    return {
        "gray": gray,
        "pinhole_mask": filtered_mask,
        "pinhole_count": pinhole_count,
        "pinhole_area_percent": pinhole_area_percent,
        "coverage": coverage,
        "mottle_field": mottle_field,
        "mottle_index": mottle_index,
        "mottle_score": mottle_score,
        "fine_variation": fine_variation,
        "surface_smoothness": surface_smoothness,
    }


def analyze_white_pair(
    rgb_white_over_white,
    rgb_white_over_black,
    crop_percent,
    pinhole_sensitivity,
    minimum_pinhole_area,
):
    """Analyze one LaNeta white-ink sample from white and black backing images."""
    ww, wb = resize_pair(rgb_white_over_white, rgb_white_over_black)

    y_ww = center_crop(rgb_luminance(ww), crop_percent)
    y_wb = center_crop(rgb_luminance(wb), crop_percent)

    mean_y_ww = float(np.median(y_ww))
    mean_y_wb = float(np.median(y_wb))

    # Contrast ratio. This is an image-derived proxy and not an ISO instrument reading.
    opacity = float(
        np.clip(100.0 * mean_y_wb / max(mean_y_ww, 1e-8), 0.0, 100.0)
    )

    lab_ww = rgb2lab(ww.astype(np.float32) / 255.0)
    lab_wb = rgb2lab(wb.astype(np.float32) / 255.0)
    lightness_ww = float(np.median(center_crop(lab_ww[..., 0], crop_percent)))
    lightness_wb = float(np.median(center_crop(lab_wb[..., 0], crop_percent)))

    surface_ww = analyze_surface(
        ww,
        crop_percent,
        pinhole_sensitivity,
        minimum_pinhole_area,
    )
    surface_wb = analyze_surface(
        wb,
        crop_percent,
        pinhole_sensitivity,
        minimum_pinhole_area,
    )

    # The black backing normally reveals pinholes and noncoverage most clearly.
    surface_smoothness = surface_wb["surface_smoothness"]
    mottle_score = surface_wb["mottle_score"]
    white_quality_index = float(
        np.clip(
            0.40 * opacity + 0.35 * surface_smoothness + 0.25 * mottle_score,
            0.0,
            100.0,
        )
    )

    return {
        "ww": ww,
        "wb": wb,
        "opacity": opacity,
        "Y_ww": mean_y_ww,
        "Y_wb": mean_y_wb,
        "L_ww": lightness_ww,
        "L_wb": lightness_wb,
        "surface_ww": surface_ww,
        "surface_wb": surface_wb,
        "surface_smoothness": surface_smoothness,
        "mottle_score": mottle_score,
        "wqi": white_quality_index,
    }


def white_grade(value):
    """Return a simple comparison grade for a 0 to 100 index."""
    if value >= 90:
        return "A"
    if value >= 80:
        return "B"
    if value >= 70:
        return "C"
    if value >= 60:
        return "D"
    return "F"


def opacity_simulation(white_over_white, white_over_black, opacity_percent):
    """Blend WW and WB images for a visual comparison slider."""
    alpha = float(np.clip(opacity_percent, 0.0, 100.0)) / 100.0
    return np.clip(
        alpha * white_over_white.astype(np.float64)
        + (1.0 - alpha) * white_over_black.astype(np.float64),
        0,
        255,
    ).astype(np.uint8)


def plot_map(data, title, cmap, colorbar_label=None, binary=False):
    """Render an analysis map in Streamlit."""
    fig, ax = plt.subplots()
    if binary:
        image_plot = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    else:
        symmetric_limit = float(np.percentile(np.abs(data), 98))
        if symmetric_limit <= 0:
            symmetric_limit = 1.0
        image_plot = ax.imshow(
            data,
            cmap=cmap,
            vmin=-symmetric_limit,
            vmax=symmetric_limit,
        )
    ax.set_title(title)
    ax.axis("off")
    if colorbar_label:
        fig.colorbar(image_plot, ax=ax, label=colorbar_label)
    st.pyplot(fig, clear_figure=True)


def display_white_sample(sample_name, results, simulation_percent):
    """Display metrics and maps for one white ink sample."""
    st.subheader(sample_name)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(
            results["ww"],
            caption=f"{sample_name}: white over white",
            use_container_width=True,
        )
    with img_col2:
        st.image(
            results["wb"],
            caption=f"{sample_name}: white over black",
            use_container_width=True,
        )

    simulation = opacity_simulation(
        results["ww"], results["wb"], simulation_percent
    )
    st.image(
        simulation,
        caption=f"Visual backing-influence simulation: {simulation_percent}% toward white backing",
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Opacity / contrast ratio", f"{results['opacity']:.1f}%")
    m2.metric("Surface smoothness", f"{results['surface_smoothness']:.1f}%")
    m3.metric("White Quality Index", f"{results['wqi']:.1f}")
    m4.metric("WQI grade", white_grade(results["wqi"]))

    surface = results["surface_wb"]
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Coverage", f"{surface['coverage']:.2f}%")
    m6.metric("Pinhole regions", f"{surface['pinhole_count']}")
    m7.metric("Pinhole area", f"{surface['pinhole_area_percent']:.3f}%")
    m8.metric("Mottle index", f"{surface['mottle_index']:.2f}")

    m9, m10, m11, m12 = st.columns(4)
    m9.metric("Mottle score", f"{results['mottle_score']:.1f}%")
    m10.metric("Fine texture variation", f"{surface['fine_variation']:.2f}")
    m11.metric("L* over white", f"{results['L_ww']:.2f}")
    m12.metric("L* over black", f"{results['L_wb']:.2f}")

    map_col1, map_col2 = st.columns(2)
    with map_col1:
        plot_map(
            surface["pinhole_mask"],
            "Pinhole map, measured over black",
            "hot",
            binary=True,
        )
    with map_col2:
        plot_map(
            surface["mottle_field"],
            "Low-frequency mottle map, measured over black",
            "coolwarm",
            colorbar_label="Relative luminance variation",
        )

    st.caption(
        "Opacity is calculated as the image-derived luminance contrast ratio Y over black / "
        "Y over white. Pinholes and mottle are measured on the black-backed image because "
        "substrate show-through is normally more visible there."
    )


# -----------------------------------------------------------------------------
# Session state and settings
# -----------------------------------------------------------------------------
if "calibration_rows" not in st.session_state:
    st.session_state.calibration_rows = []

st.sidebar.header("Measurement settings")
crop_percent = st.sidebar.slider(
    "Centered analysis area (%)",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
)
st.sidebar.caption(
    "Use a smaller centered area when a patch has borders, glare, card divisions or shadows."
)

st.sidebar.subheader("White ink defect detection")
pinhole_sensitivity = st.sidebar.slider(
    "Pinhole sensitivity, luminance difference",
    min_value=2,
    max_value=50,
    value=15,
    step=1,
    help="Lower values detect more dark spots. Use the same value for every compared sample.",
)
minimum_pinhole_area = st.sidebar.slider(
    "Minimum pinhole area, pixels",
    min_value=1,
    max_value=500,
    value=12,
    step=1,
    help="Increase this value to ignore isolated image noise.",
)
simulation_percent = st.sidebar.slider(
    "Opacity visual simulation (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
)

calibration_coefficients = None
if len(st.session_state.calibration_rows) >= 4:
    calibration_coefficients = fit_calibration(st.session_state.calibration_rows)


# -----------------------------------------------------------------------------
# General image comparison
# -----------------------------------------------------------------------------
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
    c5.metric("h degrees", f"{metrics['h']:.1f}")
    c6.metric("Surface smoothness", f"{metrics['smoothness']:.1f}%")

    c7, c8, c9 = st.columns(3)
    c7.metric("Color homogeneity", f"{metrics['homogeneity']:.1f}%")
    c8.metric("Mean Delta E00", f"{metrics['mean_de']:.2f}")
    c9.metric("95th percentile Delta E00", f"{metrics['p95_de']:.2f}")

    if calibration_coefficients is not None:
        st.caption(
            f"Raw capture: L* {metrics['raw_L']:.2f}, C* {metrics['raw_C']:.2f}, "
            f"h {metrics['raw_h']:.1f} degrees. Displayed LAB/LCH values are calibrated."
        )
    else:
        st.caption("Displayed LAB/LCH values are raw, uncalibrated image estimates.")

    fig, ax = plt.subplots()
    image_plot = ax.imshow(metrics["de_map"], cmap="magma")
    ax.set_title("Color deviation map, Delta E00 from representative color")
    ax.axis("off")
    fig.colorbar(image_plot, ax=ax, label="Delta E00")
    st.pyplot(fig, clear_figure=True)
    return metrics


st.header("General image analysis")
file1 = st.file_uploader(
    "Upload Image 1", type=SUPPORTED_TYPES, key="general_img1"
)
file2 = st.file_uploader(
    "Upload Image 2, optional", type=SUPPORTED_TYPES, key="general_img2"
)

metrics1 = None
if file1:
    rgb1 = pil_to_rgb(Image.open(file1))
    if file2:
        rgb2 = pil_to_rgb(Image.open(file2))
        rgb1_compare, rgb2_compare = resize_pair(rgb1, rgb2)
        gray1 = grayscale_255(rgb1_compare)
        gray2 = grayscale_255(rgb2_compare)
        score, difference = ssim(gray1, gray2, data_range=255, full=True)
        st.subheader(f"Structural similarity: {score * 100:.2f}%")
        st.caption(
            "SSIM is most meaningful when both images show the same scene and are aligned."
        )

        col1, col2 = st.columns(2)
        with col1:
            metrics1 = result_panel("Image 1", rgb1)
        with col2:
            metrics2 = result_panel("Image 2", rgb2)

        overall_de = float(
            deltaE_ciede2000(
                metrics1["corrected_lab"], metrics2["corrected_lab"]
            )
        )
        st.metric(
            "Representative color difference between images, Delta E00",
            f"{overall_de:.2f}",
        )
        st.subheader("Structural difference map")
        st.image(
            ((1.0 - difference) * 255.0).clip(0, 255).astype(np.uint8),
            use_container_width=True,
        )
    else:
        metrics1 = result_panel("Image 1", rgb1)
else:
    st.info("Upload Image 1 to run the general LAB/LCH and smoothness analysis.")


# -----------------------------------------------------------------------------
# White ink quality analysis for one or two samples
# -----------------------------------------------------------------------------
st.divider()
st.header("White ink opacity and surface quality")
st.write(
    "Load matching photographs of the same white ink sample printed on the white and "
    "black areas of a LaNeta card. Sample B is optional. Use consistent exposure, "
    "magnification, focus, crop and illumination."
)

sample_col_a, sample_col_b = st.columns(2)
with sample_col_a:
    st.subheader("Sample A")
    a_ww_file = st.file_uploader(
        "Sample A: white over white",
        type=SUPPORTED_TYPES,
        key="sample_a_ww",
    )
    a_wb_file = st.file_uploader(
        "Sample A: white over black",
        type=SUPPORTED_TYPES,
        key="sample_a_wb",
    )

with sample_col_b:
    st.subheader("Sample B, optional")
    b_ww_file = st.file_uploader(
        "Sample B: white over white",
        type=SUPPORTED_TYPES,
        key="sample_b_ww",
    )
    b_wb_file = st.file_uploader(
        "Sample B: white over black",
        type=SUPPORTED_TYPES,
        key="sample_b_wb",
    )

sample_a_results = None
sample_b_results = None

if a_ww_file and a_wb_file:
    sample_a_results = analyze_white_pair(
        pil_to_rgb(Image.open(a_ww_file)),
        pil_to_rgb(Image.open(a_wb_file)),
        crop_percent,
        pinhole_sensitivity,
        minimum_pinhole_area,
    )
    display_white_sample("Sample A", sample_a_results, simulation_percent)
elif a_ww_file or a_wb_file:
    st.warning("Sample A needs both white-over-white and white-over-black images.")
else:
    st.info("Load both Sample A images to begin white ink analysis.")

if b_ww_file and b_wb_file:
    st.divider()
    sample_b_results = analyze_white_pair(
        pil_to_rgb(Image.open(b_ww_file)),
        pil_to_rgb(Image.open(b_wb_file)),
        crop_percent,
        pinhole_sensitivity,
        minimum_pinhole_area,
    )
    display_white_sample("Sample B", sample_b_results, simulation_percent)
elif b_ww_file or b_wb_file:
    st.warning("Sample B needs both white-over-white and white-over-black images.")

if sample_a_results is not None and sample_b_results is not None:
    st.divider()
    st.subheader("Sample A versus Sample B")

    comparison_rows = [
        {
            "Metric": "Opacity / contrast ratio (%)",
            "Sample A": round(sample_a_results["opacity"], 2),
            "Sample B": round(sample_b_results["opacity"], 2),
            "Preferred": "Higher",
        },
        {
            "Metric": "Surface smoothness (%)",
            "Sample A": round(sample_a_results["surface_smoothness"], 2),
            "Sample B": round(sample_b_results["surface_smoothness"], 2),
            "Preferred": "Higher",
        },
        {
            "Metric": "Coverage (%)",
            "Sample A": round(sample_a_results["surface_wb"]["coverage"], 3),
            "Sample B": round(sample_b_results["surface_wb"]["coverage"], 3),
            "Preferred": "Higher",
        },
        {
            "Metric": "Pinhole regions",
            "Sample A": sample_a_results["surface_wb"]["pinhole_count"],
            "Sample B": sample_b_results["surface_wb"]["pinhole_count"],
            "Preferred": "Lower",
        },
        {
            "Metric": "Pinhole area (%)",
            "Sample A": round(
                sample_a_results["surface_wb"]["pinhole_area_percent"], 4
            ),
            "Sample B": round(
                sample_b_results["surface_wb"]["pinhole_area_percent"], 4
            ),
            "Preferred": "Lower",
        },
        {
            "Metric": "Mottle index",
            "Sample A": round(sample_a_results["surface_wb"]["mottle_index"], 3),
            "Sample B": round(sample_b_results["surface_wb"]["mottle_index"], 3),
            "Preferred": "Lower",
        },
        {
            "Metric": "White Quality Index",
            "Sample A": round(sample_a_results["wqi"], 2),
            "Sample B": round(sample_b_results["wqi"], 2),
            "Preferred": "Higher",
        },
    ]
    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    if abs(sample_a_results["wqi"] - sample_b_results["wqi"]) < 0.05:
        st.info("The two samples have effectively equal White Quality Index values.")
    elif sample_a_results["wqi"] > sample_b_results["wqi"]:
        st.success(
            "Sample A has the higher White Quality Index under the current settings."
        )
    else:
        st.success(
            "Sample B has the higher White Quality Index under the current settings."
        )

    st.caption(
        "The comparison is relative to these images and the selected crop, sensitivity and "
        "minimum-area settings. It is not a substitute for standardized instrumental testing."
    )


# -----------------------------------------------------------------------------
# Microscope calibration interface
# -----------------------------------------------------------------------------
st.divider()
st.header("Microscope calibration")
st.write(
    "Build a correction for this microscope, light and camera setup. Keep illumination, "
    "exposure, white balance, magnification and sample position unchanged."
)

if metrics1 is None:
    st.info("Upload General Image 1 to add a calibration sample.")
else:
    st.write(
        f"Current raw reading: L* {metrics1['raw_L']:.2f}, "
        f"C* {metrics1['raw_C']:.2f}, h {metrics1['raw_h']:.1f} degrees."
    )

    with st.form("add_calibration_sample", clear_on_submit=False):
        sample_name = st.text_input(
            "Calibration sample name",
            value=f"Sample {len(st.session_state.calibration_rows) + 1}",
        )
        r1, r2, r3 = st.columns(3)
        reference_l = r1.number_input(
            "Reference L*", min_value=0.0, max_value=100.0, value=50.0
        )
        reference_c = r2.number_input(
            "Reference C*", min_value=0.0, value=55.0
        )
        reference_h = r3.number_input(
            "Reference h degrees", min_value=0.0, max_value=360.0, value=247.0
        )
        submitted = st.form_submit_button("Add calibration sample")

        if submitted:
            st.session_state.calibration_rows.append(
                {
                    "name": sample_name,
                    "raw_lab": metrics1["raw_lab"].tolist(),
                    "reference_lab": lch_to_lab(
                        reference_l, reference_c, reference_h
                    ).tolist(),
                }
            )
            st.rerun()

if st.session_state.calibration_rows:
    display_rows = []
    for row in st.session_state.calibration_rows:
        raw_l, raw_c, raw_h = lab_to_lch(row["raw_lab"])
        ref_l, ref_c, ref_h = lab_to_lch(row["reference_lab"])
        display_rows.append(
            {
                "Sample": row["name"],
                "Raw L*": round(raw_l, 2),
                "Raw C*": round(raw_c, 2),
                "Raw h": round(raw_h, 1),
                "Reference L*": round(ref_l, 2),
                "Reference C*": round(ref_c, 2),
                "Reference h": round(ref_h, 1),
            }
        )

    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    if len(st.session_state.calibration_rows) < 4:
        st.warning(
            f"Add at least {4 - len(st.session_state.calibration_rows)} more varied "
            "sample(s) before calibration can be applied. Eight or more varied samples "
            "are preferable for stability and validation."
        )
    else:
        errors = calibration_error(
            st.session_state.calibration_rows, calibration_coefficients
        )
        e1, e2, e3 = st.columns(3)
        e1.metric("Calibration samples", len(errors))
        e2.metric("Mean fitting Delta E00", f"{np.mean(errors):.2f}")
        e3.metric("Maximum fitting Delta E00", f"{np.max(errors):.2f}")
        st.success(
            "Calibration is active. Re-upload or refresh image results to view corrected values."
        )
        st.caption(
            "These are fitting errors measured on the same samples used to create the model, "
            "not an independent accuracy guarantee. Validate with separate patches."
        )

    if st.button("Clear calibration"):
        st.session_state.calibration_rows = []
        st.rerun()


# -----------------------------------------------------------------------------
# Measurement notes
# -----------------------------------------------------------------------------
st.divider()
st.info(
    "LAB/LCH, opacity, smoothness, pinhole, mottle and WQI results from photographs are "
    "image-derived estimates. For repeatable comparisons, lock exposure and white balance, "
    "use controlled illumination, keep the camera perpendicular, preserve scale and focus, "
    "and photograph all samples under identical conditions."
)
st.caption(
    "The calibration model corrects L*, a* and b* with affine least squares and then "
    "calculates C* and hue. White-ink opacity is estimated from relative luminance over "
    "black and white backing. WQI is an empirical comparison index: 40% opacity, "
    "35% surface smoothness and 25% mottle score."
)

