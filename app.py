from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from model_utils import prepare_image
import numpy as np
import streamlit as st
from model_utils import load_model
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (150, 150)

MODEL_FILES = {
    "EfficientNetB0": APP_DIR / "efficientnet_transfer_learning_model.keras",
    "ResNet50": APP_DIR / "resnet_transfer_learning_model.keras",
}

CLASS_INFO = [
    ("🏛️", "Buildings"),
    ("🌲", "Forest"),
    ("🧊", "Glacier"),
    ("⛰️", "Mountain"),
    ("🌊", "Sea"),
    ("🛣️", "Street"),
]


st.set_page_config(
    page_title="NISO Internship Project",
    page_icon="🧠",
    layout="wide",
)


def render_html(markup: str) -> None:
    """Render indented HTML without Markdown treating it as a code block."""
    compact_markup = " ".join(
        line.strip()
        for line in dedent(markup).splitlines()
        if line.strip()
    )
    st.markdown(compact_markup, unsafe_allow_html=True)


def load_css(file_name: str) -> None:
    css_path = APP_DIR / file_name

    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load the selected model once and reuse it between Streamlit reruns."""
    import tensorflow as tf

    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.name}")

    custom_objects = None
    if model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input

        custom_objects = {"preprocess_input": preprocess_input}

    load_options = {
        "custom_objects": custom_objects,
        "compile": False,
    }

    try:
        model = tf.keras.models.load_model(
            model_path,
            safe_mode=False,
            **load_options,
        )
    except TypeError as error:
        if "safe_mode" not in str(error):
            raise
        model = tf.keras.models.load_model(model_path, **load_options)

    if model.output_shape[-1] != len(CLASS_INFO):
        raise ValueError(
            f"Expected {len(CLASS_INFO)} output classes, "
            f"but the model has {model.output_shape[-1]}."
        )

    return model





def prepare_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    """Decode an upload and build the 150x150 RGB input batch."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized_image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(resized_image, dtype=np.float32)
    return image, np.expand_dims(image_array, axis=0)


def predict_image(model_name: str, image_batch: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(
        load_model(model_name).predict(image_batch, verbose=0)[0],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 0.0, None)
    probability_sum = float(probabilities.sum())

    if probability_sum <= 0:
        raise ValueError("The model returned invalid probability values.")

    return probabilities / probability_sum


load_css("style.css")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

render_html(
    """
        <div class="header-card">
            <div class="topbar">
                <div class="brand-lockup">
                    <div class="logo-box">N</div>
                    <div>
                        <div class="brand-name">NISO</div>
                        <div class="brand-descriptor">Digital Transformation</div>
                    </div>
                </div>
                <div class="topbar-status">
                    <span class="status-dot"></span>
                    AI workspace online
                </div>
            </div>
            <div class="header-divider"></div>
            <div class="hero-content">
                <div class="hero-copy">
                    <div class="hero-eyebrow">NISO · INTERNSHIP PROGRAM</div>
                    <h1 class="app-title">NISO Internship Project</h1>
                    
                </div>
                <div class="header-badges">
                    <div class="internship-badge">Summer Internship 2026</div>
                    <div class="status-badge">
                        <span class="status-dot"></span>
                        Model ready
                    </div>
                </div>
            </div>
        </div>
    """
)


# ---------------------------------------------------------
# Main layout
# ---------------------------------------------------------

sidebar_column, main_column = st.columns(
    [1, 3],
    gap="large",
)


# ---------------------------------------------------------
# Left information panel
# ---------------------------------------------------------

with sidebar_column:
    render_html(
        """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">⌁</div>
                <div>
                    <div class="sidebar-brand-title">Model Console</div>
                    <div class="sidebar-brand-copy">Workspace configuration</div>
                </div>
            </div>
        """
    )

    selected_model = st.selectbox(
        "Model seçin",
        options=list(MODEL_FILES),
        help="Tahmin için kullanılacak modeli seçin.",
    )


    try:
        model = load_model(selected_model)
        st.success(f"{selected_model} loaded successfully.")
    except Exception as error:
        st.error(f"Model could not be loaded:\n{error}")
        st.stop()

    render_html(
        f"""
            <div class="info-card">
                <div class="section-label">
                    PROJECT INFO
                </div>

                <div class="info-row">
                    <div class="info-label">
                        SELECTED MODEL
                    </div>

                    <div class="info-value">
                        {escape(selected_model)}
                    </div>
                </div>

                <div class="info-row">
                    <div class="info-label">
                        INPUT SIZE
                    </div>

                    <div class="info-value">
                        150 × 150 px
                    </div>
                </div>

                <div class="info-row">
                    <div class="info-label">
                        NUMBER OF CLASSES
                    </div>

                    <div class="info-value">
                        6 categories
                    </div>
                </div>

                <div class="info-row">
                    <div class="info-label">
                        FRAMEWORK
                    </div>

                    <div class="info-value">
                        TensorFlow
                    </div>
                </div>
            </div>
        """
    )

    class_items = "".join(
        f"""
        <div class="class-item">
            <span class="class-icon">{icon}</span>
            <span class="class-name">{escape(class_name)}</span>
        </div>
        """
    for icon, class_name in CLASS_INFO
    )

    render_html(
        f"""
            <div class="info-card">
                <div class="section-label">
                    SCENE CLASSES
                </div>

                {class_items}
            </div>
        """
    )


# ---------------------------------------------------------
# Upload and result area
# ---------------------------------------------------------

current_prediction = None

with main_column:
    render_html(
        f"""
            <div class="workspace-heading">
                <div>
                    <div class="workspace-eyebrow">DEEP LEARNING IMAGE CLASSIFICATION WORKSPACE</div>
                    <h2>Scene Intelligence</h2>
                    <p>Upload an image to classify its environment with the selected model.</p>
                </div>
                <div class="active-model">
                    <div class="active-model-icon" aria-hidden="true"></div>
                    <div class="active-model-copy">
                        <span>ACTIVE MODEL</span>
                        <strong>{escape(selected_model)}</strong>
                    </div>
                </div>
            </div>
        """
    )

    upload_column, result_column = st.columns(
        [1.2, 1],
        gap="large",
    )

    with upload_column:
        render_html(
            """
                <div class="content-card">
                    <div class="section-label">
                        UPLOAD IMAGE
                    </div>

                    <h3 style="margin-top: 0;">
                        Select a natural scene image
                    </h3>

                    <p style="
                        color: #5c7a9a;
                        font-size: 13px;
                    ">
                        Supported formats: JPG, JPEG and PNG
                    </p>
                </div>
            """
        )

        uploaded_file = st.file_uploader(
            "Drag and drop an image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            upload_digest = sha256(image_bytes).hexdigest()

            try:
                uploaded_image, image_batch = prepare_image(image_bytes)
                st.image(
                    uploaded_image,
                    caption="Uploaded Image",
                    use_container_width=True,
                )
                st.write("Input shape:", image_batch.shape)

                if st.button("Classify Image", type="primary"):
                    with st.spinner(f"{selected_model} görüntüyü analiz ediyor..."):
                        probabilities = predict_image(
                            selected_model,
                            image_batch,
                        )

                    st.session_state["prediction"] = {
                        "model": selected_model,
                        "upload_digest": upload_digest,
                        "probabilities": probabilities.tolist(),
                    }
            except (UnidentifiedImageError, OSError):
                st.error("Yüklenen dosya geçerli bir JPG veya PNG görseli değil.")
            except Exception as error:
                st.error(f"Tahmin gerçekleştirilemedi: {error}")

            stored_prediction = st.session_state.get("prediction")
            if (
                stored_prediction
                and stored_prediction["model"] == selected_model
                and stored_prediction["upload_digest"] == upload_digest
            ):
                current_prediction = np.asarray(
                    stored_prediction["probabilities"],
                    dtype=np.float64,
                )

    with result_column:
        if current_prediction is None:
            render_html(
                """
                    <div class="content-card">
                        <div class="section-label">
                            PREDICTION RESULT
                        </div>

                        <div class="result-placeholder">
                            <div>
                                <div class="result-icon ai-scan-icon" aria-hidden="true">
                                    <span class="scan-core"></span>
                                </div>
                                Upload an image and run the model to
                                see the predicted class and confidence score.
                            </div>
                        </div>
                    </div>
                """
            )
        else:
            predicted_index = int(np.argmax(current_prediction))
            predicted_icon, predicted_class = CLASS_INFO[predicted_index]
            confidence = float(current_prediction[predicted_index])

            render_html(
                f"""
                    <div class="content-card">
                        <div class="section-label">
                            PREDICTION RESULT
                        </div>

                        <div class="prediction-result">
                            <div class="prediction-icon">
                                {predicted_icon}
                            </div>
                            <div class="prediction-model">
                                {escape(selected_model)}
                            </div>
                            <div class="prediction-class">
                                {escape(predicted_class)}
                            </div>
                            <div class="confidence-label">
                                CONFIDENCE
                            </div>
                            <div class="confidence-value">
                                {confidence:.1%}
                            </div>
                        </div>
                    </div>
                """
            )


# ---------------------------------------------------------
# Probability section
# ---------------------------------------------------------

if current_prediction is None:
    render_html(
        """
            <div class="content-card" style="margin-top: 8px;">
                <div class="section-label">
                    CLASS PROBABILITIES
                </div>

                <div class="result-placeholder probability-placeholder">
                    Probability bars will appear here after prediction.
                </div>
            </div>
        """
    )
else:
    predicted_index = int(np.argmax(current_prediction))
    probability_rows = ""

    for class_index in np.argsort(current_prediction)[::-1]:
        icon, class_name = CLASS_INFO[int(class_index)]
        probability = float(current_prediction[int(class_index)])
        row_class = (
            "probability-row is-predicted"
            if int(class_index) == predicted_index
            else "probability-row"
        )

        probability_rows += f"""
            <div class="{row_class}">
                <div class="probability-header">
                    <span>{icon}&nbsp; {escape(class_name)}</span>
                    <strong>{probability:.1%}</strong>
                </div>
                <div class="probability-track">
                    <div
                        class="probability-fill"
                        style="width: {probability:.2%};"
                    ></div>
                </div>
            </div>
        """

    render_html(
        f"""
            <div class="content-card probability-card">
                <div class="section-label">
                    CLASS PROBABILITIES · {escape(selected_model)}
                </div>
                <div class="probability-list">
                    {probability_rows}
                </div>
            </div>
        """
    )
