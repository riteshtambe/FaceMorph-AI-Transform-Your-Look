import streamlit as st
from PIL import Image
import requests
import io
import time
import os
import base64
from uuid import uuid4
import cv2
import numpy as np
import mediapipe as mp


# Session state to hold history
if "history" not in st.session_state:
    st.session_state.history = []

# Set page config
st.set_page_config(page_title="AI Face Filter App", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("📌 What It Does")
st.sidebar.markdown("""
Turn your selfies into stunning **AI-stylized portraits** using the power of Hugging Face models.

**Available Filters**:
- Anime 🎌
- Sketch ✏️
- Cartoon 🧑‍🎨
- Pixar 🎥
- Painting 🖼️
""")

st.sidebar.title("🧪 How to Use")
st.sidebar.markdown("""
1. 📤 Upload a photo (portrait/selfie)  
2. 🎨 Choose a style  
3. 🚀 Click *Apply Filter*  
4. 📥 Download the result  
""")

# --- MAIN AREA ---
st.title("🧠 AI Face Filter Generator")

col1, col2 = st.columns([2.5, 1.5])  # Left = main, Right = history

with col1:
    uploaded_file = st.file_uploader("Upload your selfie", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Original Image", width=300)

        style = st.selectbox("🎨 Choose a Filter", ["anime", "sketch", "cartoon", "painting", "pixar"])

        if st.button("✨ Apply Filter"):
            try:
                with st.spinner("🔄 Generating styled image... Please wait (20–40s)"):
                    start_time = time.time()

                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        f"http://localhost:8000/filter/?style={style}",
                        files=files
                    )

                    if response.status_code == 200:
                        result = response.json()
                        output_path = result["output_path"]

                        # Load output image
                        with open(output_path, "rb") as f:
                            image_bytes = f.read()
                            output_image = Image.open(io.BytesIO(image_bytes))

                            st.image(output_image, caption="🎨 Styled Output", use_column_width=True)
                            st.success(f"✅ Done in {int(time.time() - start_time)}s")

                            # Download Button
                            st.download_button(
                                label="📥 Download Image",
                                data=image_bytes,
                                file_name=f"{style}_styled.png",
                                mime="image/png"
                            )

                            # Add to session history
                            st.session_state.history.append((style, image_bytes))

                    else:
                        st.error(f"❌ Backend error: {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- HISTORY COLUMN ---
with col2:
    st.subheader("🕘 Image History")
    if st.session_state.history:
        for idx, (style_used, img_bytes) in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**{idx+1}. {style_used.title()}**")
            st.image(img_bytes, width=200)
            st.download_button(
                label="Download Again",
                data=img_bytes,
                file_name=f"{style_used}_history.png",
                mime="image/png",
                key=f"download_{idx}"
            )
    else:
        st.info("No generated images yet.")
