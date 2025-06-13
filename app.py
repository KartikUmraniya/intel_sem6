import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define image size (should match the size used during training)
IMG_SIZE = 128

# Load the trained autoencoder model
@st.cache_resource
def get_model():
    # Include custom_objects to recognize 'mse'
    model = load_model('model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    # Compile the model after loading if necessary for prediction (though often not needed for inference)
    # model.compile(optimizer='adam', loss='mse') # Uncomment if you encounter issues with predict
    return model

model = get_model()

st.title("Anomaly Detection with Autoencoder")

st.sidebar.header("Upload Image or Use Webcam")
option = st.sidebar.radio("Choose Input Source:", ("Upload Image", "Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = img_to_array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Perform prediction
        reconstructed_img_array = model.predict(img_array)

        # Calculate anomaly score
        anomaly_score = np.mean(np.square(img_array - reconstructed_img_array))

        # Display the reconstructed image and anomaly score
        st.subheader("Reconstructed Image")
        reconstructed_image = Image.fromarray((reconstructed_img_array[0] * 255).astype("uint8"))
        st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

        st.subheader("Anomaly Score")
        st.write(f"The anomaly score for this image is: {anomaly_score:.4f}")

        # Simple anomaly threshold (you may need to adjust this based on your data)
        threshold = 0.005 # Example threshold
        if anomaly_score > threshold:
            st.error("Anomaly Detected!")
        else:
            st.success("No Anomaly Detected")

elif option == "Webcam":
    st.subheader("Real-time Anomaly Detection (Webcam)")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Convert BGR to RGB for model
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess the frame
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Perform prediction
            reconstructed_img_array = model.predict(img_array, verbose=0) # Added verbose=0 to reduce output

            # Calculate anomaly score
            anomaly_score = np.mean(np.square(img_array - reconstructed_img_array))

            # Convert reconstructed image back to BGR for display
            reconstructed_img_display = (reconstructed_img_array[0] * 255).astype("uint8")
            reconstructed_img_display = cv2.cvtColor(reconstructed_img_display, cv2.COLOR_RGB2BGR)

            # Display anomaly score on the frame
            display_img = img # Use original frame for display
            score_text = f"Anomaly Score: {anomaly_score:.4f}"
            color = (0, 255, 0) if anomaly_score <= 0.005 else (0, 0, 255) # Green for normal, Red for anomaly
            cv2.putText(display_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # You can optionally display the reconstructed image next to the original in the Streamlit app
            # For simplicity, we'll just overlay the score on the original frame

            return display_img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    st.info("Select 'Webcam' from the sidebar to use your camera.")
    st.warning("The anomaly threshold (0.005) is an example and may need adjustment.")