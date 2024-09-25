from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('models/new_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = np.array(data['inputs']).reshape(1, -1)  # Reshape for model input

    # Perform prediction
    predictions = model.predict(inputs)

    # Convert predictions to list and send back as JSON
    output = predictions.tolist()
    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(debug=True)
