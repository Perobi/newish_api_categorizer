from flask import Flask, request, jsonify
from predictor import predict_hierarchy, load_models_and_encoders, unload_models, check_model_timeout
import os
import logging
import threading
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global flag for model initialization
_models_initialized = False
_memory_cleanup_thread = None
_cleanup_running = False

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

def memory_cleanup_worker():
    """Background worker to periodically clean up memory and unload inactive models."""
    global _cleanup_running
    _cleanup_running = True
    
    while _cleanup_running:
        try:
            # Check for model timeout every 30 seconds
            check_model_timeout()
            
            # Force garbage collection
            gc.collect()
            
            # Sleep for 30 seconds
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in memory cleanup worker: {e}")
            time.sleep(60)  # Wait longer on error

def start_memory_cleanup():
    """Start the background memory cleanup thread."""
    global _memory_cleanup_thread
    
    if _memory_cleanup_thread is None or not _memory_cleanup_thread.is_alive():
        _memory_cleanup_thread = threading.Thread(target=memory_cleanup_worker, daemon=True)
        _memory_cleanup_thread.start()
        logger.info("🧹 Started memory cleanup worker")

def stop_memory_cleanup():
    """Stop the background memory cleanup thread."""
    global _cleanup_running
    _cleanup_running = False

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
    """Health check endpoint with enhanced memory monitoring."""
    try:
        import psutil
        memory_info = {
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
        
        # Memory warning if usage is high
        if memory_info['memory_percent'] > 80:
            logger.warning(f"High memory usage: {memory_info['memory_percent']:.1f}%")
            # Force memory cleanup
            gc.collect()
            check_model_timeout()
        
        # Check if we're approaching Heroku's 1GB limit
        if memory_info['memory_used_mb'] > 900:
            logger.warning(f"Approaching Heroku memory limit: {memory_info['memory_used_mb']:.1f} MB")
            # Force model unloading if memory is critical
            if memory_info['memory_used_mb'] > 950:
                logger.warning("Critical memory usage, forcing model unload")
                unload_models()
            
    except ImportError:
        memory_info = {'note': 'psutil not available'}
    
    return jsonify({
        'status': 'healthy', 
        'model': 'hierarchical_multi_label',
        'memory': memory_info,
        'models_loaded': _models_initialized,
        'memory_optimized': True,
        'cleanup_worker_running': _memory_cleanup_thread.is_alive() if _memory_cleanup_thread else False
    }), 200

@app.route('/memory/cleanup', methods=['POST'])
def force_memory_cleanup():
    """Force memory cleanup endpoint."""
    try:
        logger.info("🧹 Forcing memory cleanup...")
        
        # Force garbage collection
        gc.collect()
        
        # Check for model timeout
        check_model_timeout()
        
        # Get memory info after cleanup
        import psutil
        memory_info = {
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
        
        return jsonify({
            'status': 'cleanup_completed',
            'memory_after_cleanup': memory_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/unload', methods=['POST'])
def force_model_unload():
    """Force model unloading endpoint."""
    try:
        logger.info("🗑️ Forcing model unload...")
        unload_models()
        
        return jsonify({
            'status': 'models_unloaded',
            'message': 'Models have been unloaded to free memory'
        }), 200
        
    except Exception as e:
        logger.error(f"Error during model unload: {e}")
        return jsonify({'error': str(e)}), 500

# Startup event to initialize memory cleanup
@app.before_first_request
def setup_memory_cleanup():
    """Initialize memory cleanup on first request."""
    start_memory_cleanup()

# Shutdown event to clean up resources
@app.teardown_appcontext
def cleanup_on_shutdown(error):
    """Clean up resources when the app context is torn down."""
    if error:
        logger.error(f"App context error: {error}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port) 