import requests
import streamlit as st
from io import BytesIO
from PIL import Image


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Brain Tumor Analysis")
st.write(
    "Upload an MRI image to classify the tumor and "
    "generate a segmentation mask."
)
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Uploaded MRI")

    st.image(
        image,
        caption="Original MRI",
        width = 'stretch'
    )

    if st.button(
        "Analyze MRI",
        type="primary",
        width = 'stretch'
    ):

        # Prepare file for FastAPI
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }
        # Classification
        try:

            with st.spinner("Running classification..."):

                classification_response = requests.post(
                    f"{API_URL}/classify",
                    files=files,
                    timeout=120
                )

            classification_response.raise_for_status()

            classification = classification_response.json()

            predicted_class = classification["class"]
            confidence = classification["confidence"]

            st.subheader("Classification Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction",
                    predicted_class
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

        except requests.RequestException as e:

            st.error(
                f"Could not connect to the classification API.\n\n"
                f"{e}"
            )

            st.stop()

        except (KeyError, ValueError):

            st.error(
                "The classification API returned an invalid response."
            )

            st.stop()
        # Segmentation
        try:

            with st.spinner("Generating segmentation mask..."):

                segmentation_response = requests.post(
                    f"{API_URL}/segment",
                    files=files,
                    timeout=120
                )

            segmentation_response.raise_for_status()

            mask = Image.open(
                BytesIO(segmentation_response.content)
            )

            st.subheader("Tumor Segmentation")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original MRI",
                    width = 'stretch'
                )

            with col2:

                st.image(
                    mask,
                    caption="Segmentation Mask",
                    width = 'stretch'
                )

        except requests.RequestException as e:

            st.error(
                f"Could not connect to the segmentation API.\n\n"
                f"{e}"
            )

        except Exception:

            st.error(
                "The segmentation API returned an invalid image."
            )