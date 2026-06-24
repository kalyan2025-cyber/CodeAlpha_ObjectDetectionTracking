import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯"
)

st.title("🎯 Object Detection using YOLOv8")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Detect Objects"):

        model = YOLO("yolov8n.pt")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)

            results = model(temp_file.name)

            result_image = results[0].plot()

            st.image(
                result_image,
                caption="Detected Objects",
                use_container_width=True
            )

            st.success("Detection Completed Successfully")
            