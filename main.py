import streamlit as st
import cv2 as cv
import numpy as np
import keras
import pickle

# Define label names
label_name = ['Apple Scab', 'Apple Rust', 'Apple Healthy']

# Introduction message
st.write("""
The leaf disease detection model is built using deep learning techniques and leverages a Vision Transformer (ViT) model 
with PCA and SVM for classification. The model is trained on a dataset containing images of over 2000 different Apple leaves. 
For more details on the model architecture and training process, please refer to the documentation.
""")
st.write("Please input only Apple leaf images. Otherwise, the model may not work perfectly.")

# Load pre-trained ViT model for feature extraction
vit_model = keras.models.load_model('vit.h5')

# Load the SVM-PCA model (this assumes you have a pre-trained SVM model saved using pickle)
with open('svm_pca_model.pkl', 'rb') as f:
    svm_pca_model = pickle.load(f)

# Function to process the uploaded image and make predictions
def predict_leaf_disease(image):
    # Pre-process image for the ViT model
    resized_image = cv.resize(cv.cvtColor(image, cv.COLOR_BGR2RGB), (224, 224))  # Adjust based on ViT input size
    normalized_image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize the image

    # Extract features using the ViT model
    vit_features = vit_model.predict(normalized_image)

    # Use PCA to reduce dimensions
    pca_features = svm_pca_model['pca'].transform(vit_features)

    # Use the SVM model to classify the image
    prediction = svm_pca_model['svm'].predict(pca_features)
    
    return prediction[0]  # Return the predicted class

# Upload image section in Streamlit
uploaded_file = st.file_uploader("Upload an Apple leaf image")
if uploaded_file is not None:
    # Read and decode the uploaded image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image_bytes, caption='Uploaded Leaf Image', use_column_width=True)
    
    # Make prediction
    prediction = predict_leaf_disease(img)
    
    # Display the result based on the SVM-ViT model's prediction
    st.write(f"Prediction: {label_name[prediction]}")

