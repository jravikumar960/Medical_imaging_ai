from pathlib import Path
import os

import pandas as pd
from PIL import UnidentifiedImageError
import streamlit as st
import torch
import torchxrayvision as xrv

from audit import (
    MAX_REVIEW_NOTES_CHARS,
    REVIEW_DECISIONS,
    allowed_review_statuses,
    append_review,
    get_prediction_session_id,
    latest_review,
    read_reviews,
    reviews_csv_bytes,
)
from imaging import prepare_image, open_uploaded_image, gradcam, overlay_heatmap
from dicom_io import DicomInputError, dicom_bytes_to_cxr_candidate
from input_suitability import InputSuitabilityError, require_verified_cxr_modality_image
from orthanc_client import OrthancClient, OrthancError

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
AUDIT_PATH = DATA_DIR / "audit_log.csv"
MODEL_NAME = "TorchXRayVision DenseNet121 · densenet121-res224-all"
DISCLAIMER = (
    "EDUCATIONAL / RESEARCH USE ONLY — not for diagnosis or treatment. Outputs may be wrong, "
    "biased, and uncalibrated. Never use them to make patient-care decisions."
)

st.set_page_config(page_title="Chest X-ray Research Demo", page_icon="🩻", layout="wide")


@st.cache_resource(show_spinner="Loading pretrained chest X-ray model…")
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def write_audit(prediction_session_id, finding, score, status, decision, notes):
    """Append a local demo review event linked by an opaque session UUID."""
    append_review(
        AUDIT_PATH,
        prediction_session_id=prediction_session_id,
        finding=finding,
        score=score,
        status=status,
        decision=decision,
        notes=notes,
        model_name=MODEL_NAME,
    )


def clear_review_session():
    st.session_state.pop("_review_source_fingerprint", None)
    st.session_state.pop("_review_prediction_session_id", None)


def clear_uploaded_image():
    """Release the Streamlit uploader's in-memory file from this session."""
    st.session_state.pop("cxr_upload", None)
    clear_review_session()


st.title("Chest X-ray model outputs")
st.warning(DISCLAIMER, icon="⚠️")
st.caption("Local educational prototype. Review findings as model outputs, never as diagnoses.")

with st.sidebar:
    st.header("Model and limitations")
    st.write(MODEL_NAME)
    st.write("TorchXRayVision operating-point-normalized scores are not calibrated probabilities or diagnostic certainty.")
    st.write("Trained across datasets with differing populations, labels, and acquisition methods. Domain shift and dataset bias are likely.")
    st.write("No independent evaluation or clinical validation has been performed by this project.")
    st.header("Input")
    st.write("Single-frame DICOM is eligible only when its tags declare CR/DX, the matching radiography SOP class in both the dataset and file meta, and CHEST/THORAX. PNG/JPEG is processed as a candidate radiograph without verifying modality. Neither raster format nor DICOM tags prove image content.")
    st.write("Use only authorized, de-identified images. The review log does not save image files or filenames.")

    st.header("Orthanc DICOM server")
    try:
        orthanc_secrets = st.secrets.get("orthanc", {})
    except Exception:
        orthanc_secrets = {}
    orthanc_url = orthanc_secrets.get("url", os.environ.get("ORTHANC_URL", "http://127.0.0.1:8042"))
    orthanc_username = orthanc_secrets.get("username", os.environ.get("ORTHANC_USERNAME", ""))
    orthanc_password = orthanc_secrets.get("password", os.environ.get("ORTHANC_PASSWORD", ""))
    orthanc_client = None
    if orthanc_username and orthanc_password:
        try:
            orthanc_client = OrthancClient(orthanc_url, orthanc_username, orthanc_password)
        except ValueError as exc:
            st.error(f"Orthanc configuration is invalid: {exc}")
    if orthanc_client:
        if st.button("List Orthanc studies"):
            try:
                st.session_state["orthanc_studies"] = orthanc_client.list_studies()
                st.session_state.pop("orthanc_instances", None)
                st.session_state.pop("orthanc_dicom_bytes", None)
                st.success(f"Connected to Orthanc; found {len(st.session_state['orthanc_studies'])} studies.")
            except OrthancError as exc:
                st.error(str(exc))
        study_ids = st.session_state.get("orthanc_studies", [])
        if study_ids:
            study_id = st.selectbox("Opaque Orthanc study ID", study_ids, key="orthanc_study_id")
            if st.button("List study instances"):
                try:
                    st.session_state["orthanc_instances"] = orthanc_client.list_instances(study_id)
                    st.session_state.pop("orthanc_dicom_bytes", None)
                except OrthancError as exc:
                    st.error(str(exc))
            instance_ids = st.session_state.get("orthanc_instances", [])
            if instance_ids:
                instance_id = st.selectbox("Opaque Orthanc instance ID", instance_ids, key="orthanc_instance_id")
                if st.button("Load DICOM instance for this session"):
                    try:
                        st.session_state["orthanc_dicom_bytes"] = orthanc_client.download_instance(instance_id)
                        clear_review_session()
                    except OrthancError as exc:
                        st.error(str(exc))
        if st.session_state.get("orthanc_dicom_bytes") and st.button("Clear selected Orthanc image"):
            st.session_state.pop("orthanc_dicom_bytes", None)
            clear_review_session()
    else:
        st.caption("Optional: configure Orthanc credentials in Streamlit secrets or ORTHANC_USERNAME / ORTHANC_PASSWORD environment variables.")

uploaded = st.file_uploader(
    "Choose a chest X-ray image", type=["png", "jpg", "jpeg", "dcm"],
    max_upload_size=25, key="cxr_upload"
)
st.button(
    "Clear uploaded image from this session", on_click=clear_uploaded_image,
    disabled=uploaded is None
)
input_image = None
input_bytes = None
if uploaded:
    try:
        input_bytes = uploaded.getvalue()
        content_type = (
            "application/dicom"
            if uploaded.name.lower().endswith(".dcm")
            else uploaded.type or "application/octet-stream"
        )
        if uploaded.name.lower().endswith(".dcm"):
            input_image = dicom_bytes_to_cxr_candidate(input_bytes)
        else:
            input_image = open_uploaded_image(uploaded)
        verification = require_verified_cxr_modality_image(input_image, content_type)
        st.success(
            f"Modality verified as chest X-ray "
            f"(score {verification.score:.6f}, threshold {verification.threshold:.6f}; "
            "verifier scores are not calibrated probabilities)."
        )
    except InputSuitabilityError as exc:
        st.error(str(exc))
        st.stop()
    except (DicomInputError, UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"This file is not a readable image: {exc}")
        st.stop()
elif st.session_state.get("orthanc_dicom_bytes"):
    try:
        input_bytes = st.session_state["orthanc_dicom_bytes"]
        input_image = dicom_bytes_to_cxr_candidate(input_bytes)
        verification = require_verified_cxr_modality_image(input_image, "application/dicom")
        st.success(
            f"Modality verified as chest X-ray "
            f"(score {verification.score:.6f}, threshold {verification.threshold:.6f}; "
            "verifier scores are not calibrated probabilities)."
        )
    except InputSuitabilityError as exc:
        st.error(str(exc))
        st.session_state.pop("orthanc_dicom_bytes", None)
        st.stop()
    except DicomInputError as exc:
        st.error(f"The selected Orthanc instance is not a supported DICOM image: {exc}")
        st.session_state.pop("orthanc_dicom_bytes", None)
        st.stop()

if input_image:
    prediction_id = get_prediction_session_id(st.session_state, input_bytes)
    input_tensor, gray = prepare_image(input_image)

    left, right = st.columns(2)
    with left:
        st.subheader("Scan preview")
        st.image(
            gray,
            caption="Grayscale, center-cropped, resized to 224 × 224",
            alt="Preprocessed grayscale candidate chest radiograph",
            width="stretch",
        )

    try:
        with st.spinner("Running model inference…"):
            model = load_model()
            with torch.inference_mode():
                output = model(input_tensor)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")
            findings = list(model.pathologies)
            scores = output[0].detach().cpu().numpy()
            if len(findings) != len(scores):
                raise ValueError("Model label count does not match output count.")
            score_label = (
                "Operating-point-normalized score"
                if getattr(model, "op_threshs", None) is not None
                else "Model output score"
            )
            results = pd.DataFrame({"Finding": findings, score_label: scores})
            results = results.sort_values(score_label, ascending=False, kind="stable").reset_index(drop=True)
    except Exception as exc:
        st.error(f"Model could not load or run ({type(exc).__name__}): {exc}")
        st.stop()

    with right:
        st.subheader("Ranked model outputs")
        st.dataframe(results.head(8), hide_index=True, width="stretch")
        st.caption("Scores are normalized to the model's operating points; they are not calibrated probabilities or diagnostic certainty.")

    st.subheader("Grad-CAM visualization")
    chosen = st.selectbox("Finding to visualize", findings)
    try:
        with st.spinner("Generating visualization…"):
            cam, selected_score = gradcam(model, input_tensor, findings.index(chosen))
            fig = overlay_heatmap(gray, cam)
        st.pyplot(fig, clear_figure=True, alt="Grad-CAM sensitivity visualization for the selected finding")
        import matplotlib.pyplot as plt
        plt.close(fig)
        st.caption(f"{chosen} · operating-point-normalized score {selected_score:.4f}. The overlay is a model sensitivity visualization, not proof of reasoning or pathology.")
    except Exception as exc:
        st.error(f"Grad-CAM is unavailable for this run ({type(exc).__name__}): {exc}")
        selected_score = float(results.loc[results["Finding"] == chosen, score_label].iloc[0])

    st.subheader("Human review")
    st.caption(
        "Demonstration workflow only; this is not authenticated clinical approval or a compliant medical audit trail. "
        "Do not enter patient names, IDs, dates of birth, or other identifying details in notes."
    )
    st.caption("Review decisions are stored separately and do not change the model's scores.")
    st.code(f"Demo prediction session: {prediction_id}")
    saved_review = latest_review(AUDIT_PATH, prediction_id, chosen)
    previous_status = saved_review["review_status"] if saved_review else "Pending Review"
    status_options = allowed_review_statuses(previous_status)
    review_key = f"review_{prediction_id}_{findings.index(chosen)}"
    status_key = f"{review_key}_status"
    decision_key = f"{review_key}_decision"
    notes_key = f"{review_key}_notes"
    if status_key not in st.session_state or st.session_state[status_key] not in status_options:
        st.session_state[status_key] = previous_status
    if decision_key not in st.session_state:
        saved_decision = saved_review["review_decision"] if saved_review else ""
        st.session_state[decision_key] = saved_decision if saved_decision in REVIEW_DECISIONS else "Not recorded"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = saved_review["review_notes"] if saved_review else ""

    with st.form(f"{review_key}_form"):
        status = st.segmented_control(
            "Review status", options=status_options, key=status_key, width="stretch"
        )
        decision = st.selectbox(
            "Reviewer decision",
            options=["Not recorded", *REVIEW_DECISIONS],
            key=decision_key,
            help="Choose Agree, Disagree, or Inconclusive for a completed review. Pending Review has no final decision.",
        )
        notes = st.text_area(
            "Optional review notes (no identifying details)",
            key=notes_key,
            max_chars=MAX_REVIEW_NOTES_CHARS,
            help="Saved in the local CSV. Do not include patient names, IDs, dates of birth, or other identifying details.",
        )
        submitted = st.form_submit_button("Save review event")
    if submitted:
        try:
            write_audit(
                prediction_id,
                chosen,
                selected_score,
                status,
                "" if decision == "Not recorded" else decision,
                notes,
            )
            st.success("Review event saved to the local CSV; model scores were not changed.")
        except (OSError, ValueError) as exc:
            st.error(f"Could not save the local review event: {exc}")
    saved_review = latest_review(AUDIT_PATH, prediction_id, chosen)
    if saved_review:
        st.write(f"Saved status: **{saved_review['review_status']}**")
        st.write(f"Saved decision: **{saved_review['review_decision'] or 'Not recorded'}**")
        st.caption(f"Saved notes: {saved_review['review_notes'] or '(none)'}")
    else:
        st.info("No feedback has been saved for this prediction and finding. Current status: Pending Review.")
else:
    st.info("Upload a single-frame DICOM with CR/DX and CHEST/THORAX metadata. Metadata is only a basic input filter, not proof of image content.")

st.divider()
st.subheader("Local review history")
try:
    audit = read_reviews(AUDIT_PATH)
    if audit.empty:
        st.caption("No review events have been saved.")
    else:
        st.dataframe(audit.tail(20), use_container_width=True, hide_index=True)
    st.download_button("Download audit CSV", reviews_csv_bytes(audit), "audit_log.csv", "text/csv")
except (OSError, pd.errors.ParserError, UnicodeError) as exc:
    st.error(f"Could not read the local review history: {exc}")
