import streamlit as st
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd # Import pandas for chart data


# Define image size (should match the size used during training)
IMG_SIZE = 224

# Path to the directory containing the Teachable Machine model files
TM_MODEL_FILE = 'model.json' # Assuming the main model file is named model.json in the root

# Load the trained Teachable Machine model
@st.cache_resource
def get_model():
    try:
        model = tfjs.converters.load_keras_model(TM_MODEL_FILE)
        st.success("Teachable Machine model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading Teachable Machine model from {TM_MODEL_FILE}: {e}")
        st.info("This error indicates the model format or location might be incorrect for `tfjs.converters.load_keras_model()`.")
        st.info("Please ensure 'tensorflowjs' is in requirements.txt and the 'model.json' file (or equivalent main graph file) is in the root of your repository.")
        st.info("Also, verify the exact structure and naming of the exported files from Teachable Machine.")
        return None

model = get_model()

# Only proceed if the model was loaded successfully
if model is not None:
    st.title(" Screw Anomaly Detection with Teachable Machine Model")

    st.sidebar.header("Upload Image or Use Webcam")
    option = st.sidebar.radio("Choose Input Source:", ("Upload Image", "Webcam"))

    input_shape = model.input_shape[1:4] if model and hasattr(model, 'input_shape') and len(model.input_shape) > 3 else (IMG_SIZE, IMG_SIZE, 3)
    st.write(f"Expected input shape for the model: {input_shape}")

    def process_image(image):
        img = image.resize((input_shape[0], input_shape[1]))
        img_array = img_to_array(img)
        img_array = (img_array.astype(np.float32) / 127.5) - 1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_anomaly(image_array):
        predictions = model.predict(image_array)
        return predictions

    def interpret_tm_prediction(predictions, class_names):
        if predictions is None or len(predictions) == 0:
            return "N/A", 0.0

        if len(class_names) != predictions.shape[-1]:
             st.warning(f"Mismatch between number of defined CLASS_NAMES ({len(class_names)}) and model outputs ({predictions.shape[-1]}). Please update CLASS_NAMES.")
             predicted_class_index = np.argmax(predictions)
             predicted_probability = predictions[0][predicted_class_index]
             return f"Class {predicted_class_index}", predicted_probability

        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name, predicted_probability

    # You will need to manually define your class names based on how you set them up in Teachable Machine
    # Example:
    # CLASS_NAMES = ['Good', 'Head Scratch', 'Thread Scratch', 'Neck Scratch', ...]
    # Replace with your actual class names from Teachable Machine
    CLASS_NAMES = ['Good', 'manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top'] # <<< UPDATE THIS LIST AS PER NEED 

    # Check if the user has updated the default class names and display a warning if not
    if CLASS_NAMES == ['Class 1 (Good)', 'Class 2 (Anomaly Type 1)', 'Class 3 (Anomaly Type 2)']:
         st.sidebar.warning("Please update the CLASS_NAMES list in the code with your actual class names from Teachable Machine.")

    # Function to display prediction results including the bar chart
    def display_prediction_results(predictions, class_names):
        predicted_class, predicted_probability = interpret_tm_prediction(predictions, class_names)

        st.subheader("Prediction Results")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{predicted_probability:.4f}**")

        # Add conditional logic based on predicted_class_name (assuming first class is 'Good')
        if predicted_class != CLASS_NAMES[0] and predicted_class != "N/A":
             st.error("Anomaly Detected!")
        elif predicted_class == CLASS_NAMES[0]:
             st.success("No Anomaly Detected")


        st.write("Probabilities per class:")
        # Display raw probabilities as a bar chart if CLASS_NAMES match model output shape
        if len(class_names) == predictions.shape[-1]:
            # Create a pandas DataFrame for the chart
            chart_data = pd.DataFrame({
                'Class': class_names,
                'Probability': predictions[0]
            })
            # Sort by probability for better visualization
            chart_data = chart_data.sort_values(by='Probability', ascending=False)

            # Display as a horizontal bar chart
            st.bar_chart(chart_data.set_index('Class'))
        else:
            st.write("Cannot display probability chart with mismatched CLASS_NAMES.")
            st.write("Raw Predictions (Probabilities per class):")
            st.json(predictions[0].tolist()) # Display as list if mismatch


    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                img_array = process_image(image)
                predictions = predict_anomaly(img_array)

                with col2:
                    display_prediction_results(predictions, CLASS_NAMES)


            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")
                st.error("Please check the image format and ensure the model is loaded correctly.")


    elif option == "Webcam":
        st.subheader("Real-time Anomaly Detection (Webcam)")

        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_array = process_image(img_pil)
                predictions = predict_anomaly(img_array)
                predicted_class, predicted_probability = interpret_tm_prediction(predictions, CLASS_NAMES)

                display_img = img
                score_text = f"Class: {predicted_class} ({predicted_probability:.2f})"
                color = (0, 255, 0) if predicted_class == CLASS_NAMES[0] and predicted_class != "N/A" else (0, 0, 255)
                cv2.putText(display_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # You could potentially display the bar chart in the Streamlit app itself
                # but displaying complex Streamlit elements within the VideoTransformer is tricky.
                # For simplicity, we overlay text on the video feed.

                return display_img

        if model is not None:
             webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
             st.info("Select 'Webcam' from the sidebar to use your camera.")
        else:
             st.warning("Model could not be loaded. Cannot start webcam.")

else:
    st.warning("Model could not be loaded. Please check the error message above.")
