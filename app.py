import streamlit as st
import tensorflow as tf
# We might need tensorflowjs for loading Teachable Machine models
# Ensure 'tensorflowjs' is in your requirements.txt
import tensorflowjs as tfjs # Uncommented and imported

from tensorflow.keras.models import load_model # Still keep this import for now, might be needed by tfjs loading
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define image size (should match the size used during training)
# Teachable Machine default is often 224x224
# Adjust this based on your Teachable Machine model's input size
IMG_SIZE = 224

# Path to the directory containing the Teachable Machine model files
# Assuming metadata, model, weights.bin are in the root of the repo
# For TensorFlow.js models, the main graph file is often model.json
# Teachable Machine might rename 'model' to 'model.json' and export weights in shards.
# Let's assume for now the main model file (model.json) is in the root
TM_MODEL_FILE = 'model.json' # Assuming the main model file is named model.json in the root

# Load the trained Teachable Machine model
@st.cache_resource
def get_model():
    try:
        # **Attempting to load a TensorFlow.js Keras model in Python**
        # This requires 'tensorflowjs' library and uses load_keras_model
        # Assumes the main model file is named model.json and is in the root '.'
        model = tfjs.converters.load_keras_model(TM_MODEL_FILE)


        # **Important Note:** Teachable Machine models are typically classification models.
        # Our original code was for an autoencoder (reconstruction).
        # The prediction and anomaly scoring logic below will need to be completely revised
        # to work with a classification model's output (probabilities for each class).
        # We need the class names from Teachable Machine's metadata.

        st.success("Teachable Machine model loaded successfully (using tensorflowjs)!")
        return model
    except Exception as e:
        st.error(f"Error loading Teachable Machine model from {TM_MODEL_FILE}: {e}")
        st.info("This error indicates the model format or location might be incorrect for `tfjs.converters.load_keras_model()`.")
        st.info("Please ensure 'tensorflowjs' is in requirements.txt and the 'model.json' file (or equivalent main graph file) is in the root of your repository.")
        st.info("Also, verify the exact structure and naming of the exported files from Teachable Machine.")
        return None # Return None if model loading fails

model = get_model()

# Only proceed if the model was loaded successfully
if model is not None:
    st.title("Anomaly Detection with Teachable Machine Model")

    st.sidebar.header("Upload Image or Use Webcam")
    option = st.sidebar.radio("Choose Input Source:", ("Upload Image", "Webcam"))

    # Define the expected input shape for the Teachable Machine model
    # Teachable Machine models typically expect input shape (batch_size, height, width, channels)
    # Get shape from the loaded model if available, otherwise assume a common TM input shape
    input_shape = model.input_shape[1:4] if model and hasattr(model, 'input_shape') and len(model.input_shape) > 3 else (IMG_SIZE, IMG_SIZE, 3)
    st.write(f"Expected input shape for the model: {input_shape}")


    def process_image(image):
        """Preprocesses image for the Teachable Machine model."""
        # Resize to the model's expected input size
        img = image.resize((input_shape[0], input_shape[1]))
        # Convert to numpy array
        img_array = img_to_array(img)
        # Normalize the image (Teachable Machine models often expect values in [0, 1] or [-1, 1])
        # Check Teachable Machine export details for exact normalization
        img_array = (img_array.astype(np.float32) / 127.5) - 1 # Common normalization for TM models
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_anomaly(image_array):
        """Performs prediction using the Teachable Machine model."""
        # Predict probabilities for each class
        predictions = model.predict(image_array)
        return predictions

    def interpret_tm_prediction(predictions, class_names):
        """Interprets the predictions from the Teachable Machine model."""
        # Assuming predictions is a list of probabilities, one for each class
        if predictions is None or len(predictions) == 0:
            return "N/A", 0.0 # Handle empty predictions

        # Ensure the number of class names matches the number of predictions
        if len(class_names) != predictions.shape[-1]:
             st.warning(f"Mismatch between number of defined CLASS_NAMES ({len(class_names)}) and model outputs ({predictions.shape[-1]}). Please update CLASS_NAMES.")
             # Attempt to provide a generic interpretation based on model output shape
             predicted_class_index = np.argmax(predictions)
             predicted_probability = predictions[0][predicted_class_index]
             return f"Class {predicted_class_index}", predicted_probability


        # Find the class with the highest probability
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]


        return predicted_class_name, predicted_probability

    # You will need to manually define your class names based on how you set them up in Teachable Machine
    # Example:
    # CLASS_NAMES = ['Good', 'Head Scratch', 'Thread Scratch', 'Neck Scratch', ...]
    # Replace with your actual class names from Teachable Machine
    CLASS_NAMES = ['Class 1 (Good)', 'Class 2 (Anomaly Type 1)', 'Class 3 (Anomaly Type 2)'] # <<< UPDATE THIS LIST

    # Check if the user has updated the default class names and display a warning if not
    if CLASS_NAMES == ['Class 1 (Good)', 'Class 2 (Anomaly Type 1)', 'Class 3 (Anomaly Type 2)']:
         st.sidebar.warning("Please update the CLASS_NAMES list in the code with your actual class names from Teachable Machine.")


    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file).convert('RGB') # Ensure RGB
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Preprocess the image
                img_array = process_image(image)

                # Perform prediction
                predictions = predict_anomaly(img_array)

                # Interpret and display results
                predicted_class, predicted_probability = interpret_tm_prediction(predictions, CLASS_NAMES)

                st.subheader("Prediction Results")
                st.write(f"Predicted Class: **{predicted_class}**")
                st.write(f"Confidence: **{predicted_probability:.4f}**")

                # You can add conditional logic here based on predicted_class_name
                # For example, if predicted_class_name is one of your anomaly classes:
                # This logic assumes 'Good' is the first class name in CLASS_NAMES
                if predicted_class != CLASS_NAMES[0]: # Assuming first class is 'Good'
                     st.error("Anomaly Detected!")
                else:
                     st.success("No Anomaly Detected")

                st.write("Raw Predictions (Probabilities per class):")
                # Display raw probabilities only if CLASS_NAMES match model output shape
                if len(CLASS_NAMES) == predictions.shape[-1]:
                    st.json({CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))})
                else:
                    st.write("Cannot display raw probabilities with mismatched CLASS_NAMES.")


            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")
                st.error("Please check the image format and ensure the model is loaded correctly.")


    elif option == "Webcam":
        st.subheader("Real-time Anomaly Detection (Webcam)")

        class VideoTransformer(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")

                # Convert BGR to RGB and to PIL Image for processing
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Preprocess the frame
                img_array = process_image(img_pil)

                # Perform prediction
                predictions = predict_anomaly(img_array)

                # Interpret prediction
                predicted_class, predicted_probability = interpret_tm_prediction(predictions, CLASS_NAMES)

                # Display prediction on the frame
                display_img = img # Use original frame for display
                score_text = f"Class: {predicted_class} ({predicted_probability:.2f})"
                # Change color based on prediction (e.g., Green for Good, Red for Anomaly classes)
                # This logic assumes 'Good' is the first class name in CLASS_NAMES
                color = (0, 255, 0) if predicted_class == CLASS_NAMES[0] else (0, 0, 255) # Assuming first class is 'Good'
                cv2.putText(display_img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                return display_img

        # Start the webcam streamer
        if model is not None:
             webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
             st.info("Select 'Webcam' from the sidebar to use your camera.")
        else:
             st.warning("Model could not be loaded. Cannot start webcam.")

else:
    st.warning("Model could not be loaded. Please check the error message above.")
