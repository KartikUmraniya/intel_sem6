## Project Abstract: Anomaly Detection with Teachable Machine and Streamlit

This project implements a web-based anomaly detection application using a model trained with Google's Teachable Machine and deployed with Streamlit. The application is designed to identify anomalies in screw images.

**Key Features:**

*   **Model Training:** Utilizes Teachable Machine for simplified image classification model training, categorizing images into 'good' and various 'anomalous' classes.
*   **Web Interface:** Provides a user-friendly web interface built with Streamlit.
*   **Image Upload:** Allows users to upload images for anomaly detection, displaying the predicted class and confidence score, including a visual bar chart of class probabilities.
*   **Webcam Integration:** Incorporates real-time anomaly detection using the user's webcam feed, overlaying the predicted class and confidence directly onto the video stream.
*   **Deployment:** Designed for deployment on platforms like Streamlit Cloud using a GitHub repository containing the application code (`app.py`), trained model files, and dependency specifications (`requirements.txt`, `packages.txt`).

This project demonstrates the process of training a custom image classification model with a no-code tool like Teachable Machine and integrating it into an interactive Python web application for practical use cases like visual inspection and anomaly detection.
