import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
MODEL_PATH = 'model/brain-tumor-model.h5'
model = load_model(MODEL_PATH)

# Dictionary for class labels
class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'No tumor', 3: 'Pituitary'}

def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('L')  # 'L' mode converts to grayscale
    img = img.resize((300, 300))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis = -1)  # Shape becomes (300, 300, 1)
    img_array = np.expand_dims(img_array, axis = 0)  # Shape becomes (1, 300, 300, 1)
    return img_array

def predict_tumor(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class, predictions[0]

# Streamlit App 
st.set_page_config(
    page_title = "Brain Tumor Classification App",
)

st.markdown("## :orange[Brain Tumor Classification]")
st.markdown("---")
st.write("Upload an MRI image to classify if there is a brain tumor present and its type.")

# File uploader for image
uploaded_image = st.file_uploader("Choose an MRI Image", type = ["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_image)

    # Display the uploaded image with use_container_width
    st.image(uploaded_image, caption = "Uploaded MRI Image", use_container_width = True)

    # Make the prediction
    predicted_class, prediction_probabilities = predict_tumor(img_array)
    
    # Show the prediction result
    st.write(f"Prediction : {class_names[predicted_class]}")
    st.write(f"Prediction Probability : {prediction_probabilities[predicted_class] * 100:.2f}%")

st.markdown("---")
st.markdown("### :green[Hint for Uploading MRI Images]")
st.write("""
- Please upload an MRI image of the brain. The image should be a high-quality scan of the brain, 
  preferably with visible brain structures.
- The model accepts grayscale (single-channel) MRI images, but you can upload color images, 
  and they will be converted to grayscale automatically.
- **Recommended:** 300x300 pixels size.
""")

st.markdown("### :red[Brain Tumor Classes]")
st.write("""
The model can classify the following types of brain tumors:

1. Glioma ( Class 0 )
2. Meningioma ( Class 1 )
3. No Tumor ( Class 2 )
4. Pituitary Tumor ( Class 3 )
""")
