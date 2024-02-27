from flask import Flask, request, jsonify, send_file
from predictor import predict_hierarchy
import os

app = Flask(__name__)

# Assuming you're uploading the CSV to a specific path
# UPLOAD_FOLDER = './data/uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

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

