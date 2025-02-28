import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the pre-trained deep learning model
deep_learning_model = tf.keras.models.load_model('../models/iris_deep_learning_model.keras')

# Load the scaler used during training
scaler = load('../models/scaler.joblib')

# Title of the app
st.title("Iris Flower Species Prediction")

# Create a form
with st.form(key='iris_form'):
    st.header("Enter Flower Measurements")
    
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)
    
    submit_button = st.form_submit_button(label='Predict')

# Process form submission
if submit_button:
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Predict using Deep Learning model (softmax output)
    deep_pred = deep_learning_model.predict(input_scaled)
    deep_species = np.argmax(deep_pred, axis=1)  # Get the class with highest probability
    
    # Map predictions to species names
    species_names = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    
    # Display results
    st.success("Prediction Results:")
    st.subheader("Deep Learning Model Prediction:")
    st.write(f"Predicted Species: {species_names[deep_species[0]]}")
    st.write(f"Confidence (Softmax Probabilities): {deep_pred[0]}")

    # Optional: Visualize input data
    st.write("Input Data:", input_data)

# Add some styling or instructions
st.markdown("""
    **Instructions:** Enter the measurements of an Iris flower (in centimeters) to predict its species.
    The model uses a Deep Learning model with softmax output for classification.
""")