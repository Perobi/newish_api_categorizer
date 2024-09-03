from flask import Flask, request, jsonify
from predictor import predict_hierarchy, load_models_and_encoders  # Ensure you have this function in `predictor.py`
import os

app = Flask(__name__)

# Load the models and encoders when the application starts
load_models_and_encoders()

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    title = content.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400
   
    predicted_category, predicted_sub_category, predicted_type = predict_hierarchy(title)
    return jsonify({'category': predicted_category, 'sub_category': predicted_sub_category, 'type': predicted_type}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # default port is 5000 for local development
    app.run(host='0.0.0.0', port=port)  # binds the server to the '0.0.0.0' host and to the port specified by the PORT environment variable
