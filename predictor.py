import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Input, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from clean_data import clean_title
import gc
import time
import threading

# Global constants
DATA_PATH = './data/raw_data/kashew_supervised_products.csv'
MODEL_DIR = './model'
MAX_WORDS = 1500  # Further reduced to save memory
MAX_LEN = 20      # Further reduced to save memory

# Global variables for models and encoders
hierarchical_model = None
tokenizer = None
mlb_category = None
mlb_sub_category = None
mlb_type = None
_models_loaded = False
_loading_lock = threading.Lock()  # Thread-safe lock
_models_unloaded = True  # Track if models are unloaded
_last_used_time = 0  # Track when models were last used
_model_timeout = 300  # Unload models after 5 minutes of inactivity

def parse_multi_labels(label_string):
    """Parse multi-label strings into lists."""
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split(',') if label.strip()]

def create_hierarchical_model(output_dims):
    """Create a memory-efficient hierarchical model."""
    # Input layer
    input_layer = Input(shape=(MAX_LEN,))
    
    # Smaller embedding
    embedding = Embedding(MAX_WORDS, 16)(input_layer)  # Reduced from 32 to 16
    dropout1 = SpatialDropout1D(0.1)(embedding)
    
    # Single LSTM layer (removed the second LSTM)
    lstm_features = LSTM(32, dropout=0.1, recurrent_dropout=0.1)(dropout1)  # Reduced from 64 to 32
    
    # Shared dense layer for all outputs
    shared_dense = Dense(32, activation='relu')(lstm_features)  # Reduced from 64 to 32
    shared_dropout = Dropout(0.2)(shared_dense)
    
    # Category prediction (parent level)
    category_output = Dense(output_dims['category'], activation='sigmoid')(shared_dropout)
    
    # Sub-category prediction (child level)
    sub_category_output = Dense(output_dims['sub_category'], activation='sigmoid')(shared_dropout)
    
    # Type prediction (grandchild level)
    type_output = Dense(output_dims['type'], activation='sigmoid')(shared_dropout)
    
    model = Model(inputs=input_layer, outputs=[category_output, sub_category_output, type_output])
    
    # Simplified loss weights
    model.compile(
        loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        loss_weights=[1.0, 1.0, 1.0],  # Equal weights
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def save_model_as_tf(model, model_name):
    model.save(f'{MODEL_DIR}/{model_name}', save_format='tf')

def save_object(obj, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(f'{MODEL_DIR}/{filename}', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_hierarchical_model(padded_sequences, category_labels, sub_category_labels, type_labels):
    """Train a hierarchical model that considers all levels together."""
    output_dims = {
        'category': category_labels.shape[1],
        'sub_category': sub_category_labels.shape[1],
        'type': type_labels.shape[1]
    }
    
    model = create_hierarchical_model(output_dims)
    
    print(f"Training hierarchical model with {len(padded_sequences)} samples")
    print(f"Category classes: {category_labels.shape[1]}")
    print(f"Sub-category classes: {sub_category_labels.shape[1]}")
    print(f"Type classes: {type_labels.shape[1]}")
    
    model.fit(
        padded_sequences,
        [category_labels, sub_category_labels, type_labels],
        epochs=20,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.001, restore_best_weights=True)],
        verbose=1
    )
    
    save_model_as_tf(model, 'hierarchical_model')

def check_model_timeout():
    """Check if models should be unloaded due to inactivity."""
    global _last_used_time, _models_loaded
    
    if _models_loaded and time.time() - _last_used_time > _model_timeout:
        print(f"🕐 Models inactive for {_model_timeout} seconds, unloading...")
        unload_models()

def load_models_and_encoders():
    """Load models and encoders - now with lazy loading and thread safety."""
    global hierarchical_model, tokenizer, mlb_category, mlb_sub_category, mlb_type, _models_loaded, _models_unloaded, _last_used_time
    
    # Check if models should be unloaded due to timeout
    check_model_timeout()
    
    if _models_loaded:
        _last_used_time = time.time()  # Update last used time
        return  # Already loaded
    
    # Use thread lock to prevent multiple simultaneous loads
    with _loading_lock:
        # Double-check after acquiring lock
        if _models_loaded:
            _last_used_time = time.time()
            return
        
        try:
            print("🚀 Loading hierarchical models and encoders...")
            
            # Aggressive memory optimization
            tf.keras.backend.clear_session()  # Clear any existing models
            gc.collect()  # Force garbage collection
            
            # Set TensorFlow memory growth to prevent memory issues
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU memory growth setting failed: {e}")
            
            # Load hierarchical model with memory optimization
            hierarchical_model = tf.keras.models.load_model(
                f'{MODEL_DIR}/hierarchical_model', 
                custom_objects={'Adam': tf.keras.optimizers.Adam},
                compile=False  # Don't compile to save memory
            )
            
            # Load tokenizer and binarizers
            with open(f'{MODEL_DIR}/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            with open(f'{MODEL_DIR}/mlb_category.pickle', 'rb') as handle:
                mlb_category = pickle.load(handle)
            with open(f'{MODEL_DIR}/mlb_sub_category.pickle', 'rb') as handle:
                mlb_sub_category = pickle.load(handle)
            with open(f'{MODEL_DIR}/mlb_type.pickle', 'rb') as handle:
                mlb_type = pickle.load(handle)
            
            # Force garbage collection after loading
            gc.collect()
            
            _models_loaded = True
            _models_unloaded = False
            _last_used_time = time.time()
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            _models_loaded = False
            raise

def unload_models():
    """Unload models to free memory."""
    global hierarchical_model, tokenizer, mlb_category, mlb_sub_category, mlb_type, _models_loaded, _models_unloaded
    
    if _models_unloaded:
        return  # Already unloaded
    
    try:
        print("🗑️ Unloading models to free memory...")
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Delete model references
        del hierarchical_model
        del tokenizer
        del mlb_category
        del mlb_sub_category
        del mlb_type
        
        # Set to None
        hierarchical_model = None
        tokenizer = None
        mlb_category = None
        mlb_sub_category = None
        mlb_type = None
        
        # Update flags
        _models_loaded = False
        _models_unloaded = True
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Models unloaded successfully!")
        
    except Exception as e:
        print(f"⚠️ Error unloading models: {e}")
        # Reset flags anyway
        _models_loaded = False
        _models_unloaded = True

def train_models(data_path):
    """Train hierarchical multi-label classification models."""
    tf.keras.backend.clear_session()
    df = pd.read_csv(data_path)
    df['clean_title'] = df['title'].apply(clean_title)

    # Parse multi-labels
    df['categories'] = df['category'].apply(parse_multi_labels)
    df['sub_categories'] = df['sub_category'].apply(parse_multi_labels)
    df['types'] = df['type'].apply(parse_multi_labels)

    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(df['clean_title'])
    sequences = tokenizer.texts_to_sequences(df['clean_title'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

    # Create MultiLabelBinarizers for multi-label classification
    mlb_category = MultiLabelBinarizer()
    mlb_sub_category = MultiLabelBinarizer()
    mlb_type = MultiLabelBinarizer()

    # Transform labels
    category_labels = mlb_category.fit_transform(df['categories'])
    sub_category_labels = mlb_sub_category.fit_transform(df['sub_categories'])
    type_labels = mlb_type.fit_transform(df['types'])

    print(f"Category classes: {len(mlb_category.classes_)}")
    print(f"Sub-category classes: {len(mlb_sub_category.classes_)}")
    print(f"Type classes: {len(mlb_type.classes_)}")

    # Train hierarchical model
    train_hierarchical_model(padded_sequences, category_labels, sub_category_labels, type_labels)

    # Save tokenizer and binarizers
    save_object(tokenizer, 'tokenizer.pickle')
    save_object(mlb_category, 'mlb_category.pickle')
    save_object(mlb_sub_category, 'mlb_sub_category.pickle')
    save_object(mlb_type, 'mlb_type.pickle')

def predict_hierarchy(title):
    """Predict hierarchical categories using the trained model."""
    global _models_loaded, _models_unloaded, _last_used_time
    
    # Load models if not already loaded
    if not _models_loaded:
        load_models_and_encoders()
    else:
        _last_used_time = time.time()  # Update last used time
    
    # Verify models are loaded
    if not _models_loaded or hierarchical_model is None:
        raise RuntimeError("Models failed to load properly")
    
    try:
        # Preprocess the title
        clean_title_text = clean_title(title)
        sequence = tokenizer.texts_to_sequences([clean_title_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
        
        # Make predictions with memory optimization
        category_pred, sub_category_pred, type_pred = hierarchical_model.predict(
            padded_sequence, 
            verbose=0  # Reduce logging output
        )
        
        # Convert predictions to labels with adaptive thresholds
        category_threshold = 0.25
        sub_category_threshold = 0.25
        type_threshold = 0.25
        
        # Get predicted categories
        category_indices = np.where(category_pred[0] > category_threshold)[0]
        predicted_categories = mlb_category.classes_[category_indices]
        
        # Get predicted sub-categories
        sub_category_indices = np.where(sub_category_pred[0] > sub_category_threshold)[0]
        predicted_sub_categories = mlb_sub_category.classes_[sub_category_indices]
        
        # Get predicted types
        type_indices = np.where(type_pred[0] > type_threshold)[0]
        predicted_types = mlb_type.classes_[type_indices]
        
        # Join multiple predictions
        category_str = ', '.join(predicted_categories) if len(predicted_categories) > 0 else 'None'
        sub_category_str = ', '.join(predicted_sub_categories) if len(predicted_sub_categories) > 0 else 'None'
        type_str = ', '.join(predicted_types) if len(predicted_types) > 0 else 'None'
        
        return category_str, sub_category_str, type_str
        
    except Exception as e:
        # If prediction fails, try to reload models and retry once
        if _models_loaded:
            print(f"⚠️ Prediction failed, attempting model reload: {e}")
            unload_models()
            load_models_and_encoders()
            
            # Retry prediction
            clean_title_text = clean_title(title)
            sequence = tokenizer.texts_to_sequences([clean_title_text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
            
            category_pred, sub_category_pred, type_pred = hierarchical_model.predict(padded_sequence, verbose=0)
            
            # Process predictions (same logic as above)
            category_indices = np.where(category_pred[0] > category_threshold)[0]
            predicted_categories = mlb_category.classes_[category_indices]
            
            sub_category_indices = np.where(sub_category_pred[0] > sub_category_threshold)[0]
            predicted_sub_categories = mlb_sub_category.classes_[sub_category_indices]
            
            type_indices = np.where(type_pred[0] > type_threshold)[0]
            predicted_types = mlb_type.classes_[type_indices]
            
            category_str = ', '.join(predicted_categories) if len(predicted_categories) > 0 else 'None'
            sub_category_str = ', '.join(predicted_sub_categories) if len(predicted_sub_categories) > 0 else 'None'
            type_str = ', '.join(predicted_types) if len(predicted_types) > 0 else 'None'
            
            return category_str, sub_category_str, type_str
        else:
            raise e

# Uncomment to train models
# train_models(DATA_PATH)

