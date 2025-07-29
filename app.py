from flask import Flask, request, jsonify
from predictor import predict_hierarchy
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        content = request.json
        title = content.get('title')
        if not title:
            return jsonify({'error': 'No title provided'}), 400
    
        predicted_category, predicted_sub_category, predicted_type = predict_hierarchy(title)
        
        return jsonify({
            'category': predicted_category, 
            'sub_category': predicted_sub_category, 
            'type': predicted_type
        }), 200
        
    except Exception as e:
        app.logger.error(f'Error occurred: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'hierarchical_multi_label'}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) 