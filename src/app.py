import streamlit as st
import tempfile
from PIL import Image

from inference import VisionForge
from evaluation import Benchmark

st.set_page_config(
    page_title="VisionForge AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 VisionForge AI")
st.subheader("Vision Language Model Playground")

engine = VisionForge()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    option = st.selectbox(
        "Choose Task",
        [
            "Image Captioning",
            "OCR",
            "Document Analysis",
            "Object Detection",
            "Image Summary",
            "Visual Question Answering"
        ]
    )

    benchmark = Benchmark()

    benchmark.start()

    benchmark.add_model(engine.model.model_name())

    benchmark.add_task(option)

    if option == "Image Captioning":
        result = engine.caption(image_path)

    elif option == "OCR":
        result = engine.ocr(image_path)

    elif option == "Document Analysis":
        result = engine.document_analysis(image_path)

    elif option == "Object Detection":
        result = engine.detect_objects(image_path)

    elif option == "Image Summary":
        result = engine.summarize(image_path)

    else:
        question = st.text_input("Ask a question")

        if question:
            result = engine.ask_question(image_path, question)
        else:
            result = "Waiting for your question..."

    benchmark.stop()

    st.markdown("---")

    st.subheader("Result")

    st.write(result)

    st.markdown("---")

    st.subheader("Benchmark")

    st.json(benchmark.summary())