import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st

# Title of the application
st.header('Dental Classification CNN Model')

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the model
model = load_model('Flower_Recog_Model.h5')

def classify_images(image):
    # Resize and preprocess the image
    input_image = tf.keras.utils.img_to_array(image)
    input_image = tf.image.resize(input_image, (256, 256))  # Resize to the expected input size
    input_image_exp_dim = tf.expand_dims(input_image, axis=0)

    # Make predictions
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    
    # Prepare the outcome message
    outcome = f'The image belongs to {class_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%'
    return outcome

# Upload the file
uploaded_file = st.file_uploader('Upload an Image', type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Read the uploaded image
    image = tf.keras.utils.load_img(uploaded_file, target_size=(256, 256))
    
    # Display the image
    st.image(uploaded_file, width=200)

    # Classify the image and display the result
    result = classify_images(image)
    st.markdown(result)
