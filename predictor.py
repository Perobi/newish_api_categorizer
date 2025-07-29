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

# Global constants
DATA_PATH = './data/raw_data/kashew_supervised_products.csv'
MODEL_DIR = './model'
MAX_WORDS = 3000  # Reduced from 5000 to save memory
MAX_LEN = 30      # Reduced from 50 to save memory

# Global variables for models and encoders
hierarchical_model = None
tokenizer = None
mlb_category = None
mlb_sub_category = None
mlb_type = None
_models_loaded = False
_loading_lock = False  # Prevent multiple simultaneous loads

def parse_multi_labels(label_string):
    """Parse multi-label strings into lists."""
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split(',') if label.strip()]

def create_hierarchical_model(output_dims):
    """Create a hierarchical model that considers parent-child relationships."""
    # Input layer
    input_layer = Input(shape=(MAX_LEN,))
    
    # Shared embedding and LSTM layers (reduced sizes for memory)
    embedding = Embedding(MAX_WORDS, 32)(input_layer)  # Reduced from 64 to 32
    dropout1 = SpatialDropout1D(0.2)(embedding)
    lstm = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dropout1)  # Reduced from 128 to 64
    
    # Extract features from LSTM
    lstm_features = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(lstm)  # Reduced from 64 to 32
    
    # Category prediction (parent level)
    category_dense = Dense(64, activation='relu')(lstm_features)  # Reduced from 128 to 64
    category_dropout = Dropout(0.3)(category_dense)
    category_output = Dense(output_dims['category'], activation='sigmoid')(category_dropout)
    
    # Sub-category prediction (child level) - influenced by category predictions
    sub_category_dense = Dense(64, activation='relu')(lstm_features)  # Reduced from 128 to 64
    # Concatenate with category predictions for hierarchy
    category_embedding = Dense(32, activation='relu')(category_output)  # Reduced from 64 to 32
    sub_category_combined = Concatenate()([sub_category_dense, category_embedding])
    sub_category_dropout = Dropout(0.3)(sub_category_combined)
    sub_category_output = Dense(output_dims['sub_category'], activation='sigmoid')(sub_category_dropout)
    
    # Type prediction (grandchild level) - influenced by both category and sub-category
    type_dense = Dense(64, activation='relu')(lstm_features)  # Reduced from 128 to 64
    sub_category_embedding = Dense(32, activation='relu')(sub_category_output)  # Reduced from 64 to 32
    type_combined = Concatenate()([type_dense, category_embedding, sub_category_embedding])
    type_dropout = Dropout(0.3)(type_combined)
    type_output = Dense(output_dims['type'], activation='sigmoid')(type_dropout)
    
    model = Model(inputs=input_layer, outputs=[category_output, sub_category_output, type_output])
    
    # Use different loss weights to prioritize hierarchy
    model.compile(
        loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        loss_weights=[1.0, 1.2, 0.8],  # Give more weight to sub-category prediction
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

def load_models_and_encoders():
    """Load models and encoders - now with lazy loading and thread safety."""
    global hierarchical_model, tokenizer, mlb_category, mlb_sub_category, mlb_type, _models_loaded, _loading_lock
    
    if _models_loaded:
        return  # Already loaded
    
    # Prevent multiple simultaneous loads
    if _loading_lock:
        # Wait for another thread to finish loading
        import time
        while _loading_lock:
            time.sleep(0.1)
        return  # Models should now be loaded
    
    _loading_lock = True
    
    try:
        print("Loading hierarchical models and encoders...")
        
        # Aggressive memory optimization
        tf.keras.backend.clear_session()  # Clear any existing models
        import gc
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
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        _models_loaded = False
        raise
    finally:
        _loading_lock = False

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
    global _models_loaded
    
    # Lazy load models if not already loaded
    if not _models_loaded:
        load_models_and_encoders()
    
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
        
        # Memory cleanup after prediction
        import gc
        gc.collect()
        
        return category_str, sub_category_str, type_str
        
    except Exception as e:
        # If prediction fails, try to reload models and retry once
        if _models_loaded:
            print(f"⚠️ Prediction failed, attempting model reload: {e}")
            _models_loaded = False
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
            
            # Memory cleanup
            import gc
            gc.collect()
            
            return category_str, sub_category_str, type_str
        else:
            raise e

# Uncomment to train models
# train_models(DATA_PATH) 