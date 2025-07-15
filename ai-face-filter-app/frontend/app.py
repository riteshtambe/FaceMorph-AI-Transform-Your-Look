import streamlit as st
from PIL import Image
import requests
import io
import time
import base64

st.set_page_config(page_title="AI Face Filter", layout="centered")
st.title("🧠 FACEMORPH AI")

uploaded_file = st.file_uploader("Upload your selfie", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", width=300)

    style = st.selectbox("Choose a Filter", ["anime", "sketch", "cartoon", "painting", "pixar"])

    if st.button("Apply Filter"):
        try:
            with st.spinner("🔄 Generating image using AI model... Please wait (30-40s on CPU)"):
                start_time = time.time()

                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    f"http://localhost:8000/filter/?style={style}",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()
                    output_path = result["output_path"]

                    with open(output_path, "rb") as f:
                        image_bytes = f.read()
                        output_image = Image.open(io.BytesIO(image_bytes))
                        st.success(f"✅ Filter applied in {int(time.time() - start_time)} seconds!")
                        st.image(output_image, caption="🎨 Styled Output", use_column_width=True)

                    # 🎯 Real Download Button
                    st.download_button(
                        label="📥 Download Styled Image",
                        data=image_bytes,
                        file_name="styled_output.png",
                        mime="image/png"
                    )


                else:
                    st.error(f"❌ Backend error: {response.text}")
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")
