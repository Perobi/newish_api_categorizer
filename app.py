from flask import Flask, request, jsonify
from predictor import predict_hierarchy, load_models_and_encoders
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag for model initialization
_models_initialized = False

def initialize_models():
    """Initialize models on first request to reduce cold start time."""
    global _models_initialized
    if not _models_initialized:
        try:
            logger.info("🚀 Initializing models on first request...")
            load_models_and_encoders()
            _models_initialized = True
            logger.info("✅ Models initialized successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Model initialization failed, will lazy load: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        content = request.json
        if not content:
            return jsonify({'error': 'No JSON content provided'}), 400
            
        title = content.get('title')
        if not title or not title.strip():
            return jsonify({'error': 'No title provided or title is empty'}), 400
    
        # Initialize models if needed
        initialize_models()
        
        # Make prediction
        predicted_category, predicted_sub_category, predicted_type = predict_hierarchy(title.strip())
        
        return jsonify({
            'category': predicted_category, 
            'sub_category': predicted_sub_category, 
            'type': predicted_type,
            'input_title': title.strip()
        }), 200
        
    except Exception as e:
        logger.error(f'Error occurred during prediction: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        import psutil
        memory_info = {
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except ImportError:
        memory_info = {'note': 'psutil not available'}
    
    return jsonify({
        'status': 'healthy', 
        'model': 'hierarchical_multi_label',
        'memory': memory_info
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) 