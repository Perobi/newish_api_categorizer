from flask import Flask, request, jsonify, send_file
from predictor import train_models, predict_hierarchy
import os
import werkzeug
import pandas as pd
import io

app = Flask(__name__)

# Assuming you're uploading the CSV to a specific path
UPLOAD_FOLDER = './data/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/train', methods=['POST'])
def train():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and werkzeug.utils.secure_filename(file.filename).endswith('.csv'):
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        # Train the models
        train_models(file_path)
        return jsonify({'message': 'Model trained successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    title = content.get('title')
    if not title:
        return jsonify({'error': 'No title provided'}), 400
   
    category, sub_category, type_ = predict_hierarchy(title) 
    return jsonify({'category': category, 'sub_category': sub_category, 'type': type_}), 200

@app.route('/predict_from_csv', methods=['POST'])
def predict_from_csv():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and werkzeug.utils.secure_filename(file.filename).endswith('.csv'):
        filename = werkzeug.utils.secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        # Read CSV file
        df = pd.read_csv(file_path)
        # Predict hierarchy for each row
        predictions = []
        for index, row in df.iterrows():
            category, sub_category, type_ = predict_hierarchy(row['title'], row['description'])
            predictions.append({'title': row['title'], 'description': row['description'], 'category': category, 'sub_category': sub_category, 'type': type_})
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)
        # Save predictions to CSV file
        predictions_file_path = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
        predictions_df.to_csv(predictions_file_path, index=False)

        # Send the file for download
        return send_file(predictions_file_path, as_attachment=True, attachment_filename='predictions.csv', mimetype='text/csv')
    else:
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
