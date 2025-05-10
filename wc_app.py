import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import base64


# Load the trained model
model = load_model("wc_model.keras")

# Set page config
st.set_page_config(page_title="Eco Waste Classifier", page_icon="♻️", layout="centered")

# Function to set background from local image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("wc_bg.jpg")



# Side bar
st.sidebar.title("About 🌍")
st.sidebar.info("""
This eco-friendly app uses a deep learning model to classify uploaded waste images as **Organic** or **Recyclable**.

- Built with 💚 using TensorFlow and Streamlit
- Designed to help promote sustainability and cleaner environments.
""")
st.sidebar.markdown("[Learn about waste management](https://www.epa.gov/recycle)")

# Title and subtitle
st.markdown(
    "<h1 style='text-align: center; color: #3E8E41;'>🌱 Waste Classification System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #4B6043;'>Upload an image to identify if it's Organic or Recyclable Waste ♻️</h4>",
    unsafe_allow_html=True
)
st.write("")

# Prediction function
def predict(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3]) / 255.0
    prediction = model.predict(img)
    result = np.argmax(prediction)
    confidence = np.max(prediction)
    if result == 0:
        return "🌿 Organic Waste", "#013220", confidence #dark green
    else:
        return "♻️ Recyclable Waste", "#1D2E28", confidence #palm green



# session for history
if "history" not in st.session_state:
    st.session_state.history = []

# File uploader
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

# Main logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    if st.button("Classify Waste"):
        label, color, confidence = predict(image)
        st.session_state.history.append(label)
        

        st.markdown(
            f"<h3 style='color: {color}; text-align: center;'>Prediction: {label}</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align:center;'>Confidence: <strong>{confidence*100:.2f}%</strong></p>",
            unsafe_allow_html=True
        )

        if "Organic" in label:
            st.info("🌱 Tip: Organic waste like food scraps can be composted for fertilizer.")
        else:
            st.info("♻️ Tip: Make sure recyclables are clean and dry to avoid contamination.")

       

# history
if st.session_state.history:
    st.write("### 🔁 Previous Predictions:")
    for i, lbl in enumerate(st.session_state.history[::-1], 1):
        st.write(f"{i}. {lbl}")