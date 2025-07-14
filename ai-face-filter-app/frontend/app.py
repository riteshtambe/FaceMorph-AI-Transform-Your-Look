import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AI Face Filter", layout="centered")
st.title("🎨 AI Face Filter App")

uploaded_file = st.file_uploader("Upload your selfie", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", width=300)

    style = st.selectbox("Choose a Filter", ["anime", "sketch", "cartoon", "painting", "pixar"])

    if st.button("Apply Filter"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            f"http://localhost:8000/filter/?style={style}",
            files={"file": uploaded_file}
        )
        if response.status_code == 200:
            output_path = response.json()["output_path"]
            st.image(output_path, caption=f"Filtered: {style}", width=300)
        else:
            st.error("Something went wrong while applying the filter.")
