import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('models/new_model.h5')

# Define the Streamlit app
def main():
    st.title("Prediction App")

    # Input fields for user input
    inputs = st.text_input("Enter your inputs (comma separated):")
    
    if st.button("Predict"):
        if inputs:
            # Process the input
            input_array = np.array([float(i) for i in inputs.split(",")]).reshape(1, -1)

            # Perform prediction
            predictions = model.predict(input_array)

            # Convert predictions to list and display the result
            output = predictions.tolist()
            st.write("Predictions:", output)
        else:
            st.error("Please enter valid inputs.")

if __name__ == '__main__':
    main()
