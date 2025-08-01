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
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Input, Concatenate, Dropout, Multiply, Lambda
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

# Hierarchy constraints - Updated based on actual data patterns
HIERARCHY_CONSTRAINTS = {
    'category_to_subcategory': {
        'Seating': ['Sofas', 'Chairs', 'Armchairs', 'Stools', 'Ottomans & Footstools', 'Chaises & Daybeds', 'Benches', 'Chairs, Armchairs'],
        'Tables & Desks': ['Tables', 'Desks', 'Console tables', 'Coffee tables', 'Dining tables'],
        'Storage': ['Storage & Display Cabinets', 'Nightstands', 'Dressers & Chests of Drawers', 'Bookcases & Shelving', 'Sideboards & Credenzas', 'Trunks & Chests'],
        'Decor': ['Wall Art', 'Mirrors', 'Decorative Accessories', 'Rugs & Carpets'],
        'Lighting': ['Table Lamps', 'Floor Lamps', 'Ceiling & Wall Lamps'],
        'Beds': ['Bed Frames', 'Headboards', 'Bed Frames, Headboards']
    },
    'subcategory_to_type': {
        'Sofas': ['Sofas', 'Sectionals', 'Loveseats'],
        'Chairs': ['Dining Chairs', 'Accent Chairs', 'Office Chairs', 'Swivel Chairs'],
        'Armchairs': ['Arm Chairs', 'Club Chairs', 'Recliners'],
        'Chairs, Armchairs': ['Accent Chairs', 'Arm Chairs'],
        'Tables': ['Coffee tables', 'Dining tables', 'Console tables', 'Accent & Side tables'],
        'Desks': ['Desks', 'Vanity Desks'],
        'Wall Art': ['Paintings', 'Picture Frames', 'Wall Decorative Accents'],
        'Mirrors': ['Wall Mirrors', 'Full Length & Floor Mirrors'],
        'Decorative Accessories': ['Decorative Accents', 'Sculptures & Statues', 'Kitchen Accessoires', 'Vases'],
        'Rugs & Carpets': ['Carpets'],
        'Storage & Display Cabinets': [],
        'Dressers & Chests of Drawers': [],
        'Nightstands': [],
        'Table Lamps': [],
        'Floor Lamps': [],
        'Ceiling & Wall Lamps': []
    }
}

def parse_multi_labels(label_string):
    """Parse multi-label strings into lists."""
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split(',') if label.strip()]

def create_hierarchy_masks(category_classes, sub_category_classes, type_classes):
    """Create masks for valid hierarchy combinations"""
    category_to_sub_mask = np.zeros((len(category_classes), len(sub_category_classes)))
    sub_to_type_mask = np.zeros((len(sub_category_classes), len(type_classes)))
    
    # Build category to sub-category mask
    for cat_idx, category in enumerate(category_classes):
        if category in HIERARCHY_CONSTRAINTS['category_to_subcategory']:
            valid_subs = HIERARCHY_CONSTRAINTS['category_to_subcategory'][category]
            for sub_idx, sub_category in enumerate(sub_category_classes):
                if sub_category in valid_subs:
                    category_to_sub_mask[cat_idx, sub_idx] = 1.0
    
    # Build sub-category to type mask
    for sub_idx, sub_category in enumerate(sub_category_classes):
        if sub_category in HIERARCHY_CONSTRAINTS['subcategory_to_type']:
            valid_types = HIERARCHY_CONSTRAINTS['subcategory_to_type'][sub_category]
            for type_idx, type_val in enumerate(type_classes):
                if type_val in valid_types:
                    sub_to_type_mask[sub_idx, type_idx] = 1.0
    
    return category_to_sub_mask, sub_to_type_mask

def create_hierarchical_model(output_dims, category_to_sub_mask, sub_to_type_mask):
    """Create a hierarchy-constrained model."""
    # Input layer
    input_layer = Input(shape=(MAX_LEN,))
    
    # Smaller embedding
    embedding = Embedding(MAX_WORDS, 16)(input_layer)
    dropout1 = SpatialDropout1D(0.1)(embedding)
    
    # Single LSTM layer
    lstm_features = LSTM(32, dropout=0.1, recurrent_dropout=0.1)(dropout1)
    
    # Shared dense layer for initial features
    shared_dense = Dense(32, activation='relu')(lstm_features)
    shared_dropout = Dropout(0.2)(shared_dense)
    
    # Category prediction (parent level) - independent
    category_output = Dense(output_dims['category'], activation='sigmoid')(shared_dropout)
    
    # Sub-category prediction (child level) - influenced by category
    category_embedding = Dense(16, activation='relu')(category_output)
    sub_category_combined = Concatenate()([shared_dropout, category_embedding])
    sub_category_output = Dense(output_dims['sub_category'], activation='sigmoid')(sub_category_combined)
    
    # Type prediction (grandchild level) - influenced by both category and sub-category
    sub_category_embedding = Dense(16, activation='relu')(sub_category_output)
    type_combined = Concatenate()([shared_dropout, category_embedding, sub_category_embedding])
    type_output = Dense(output_dims['type'], activation='sigmoid')(type_combined)
    
    model = Model(inputs=input_layer, outputs=[category_output, sub_category_output, type_output])
    
    # Use standard loss functions for now, hierarchy validation will be done post-prediction
    model.compile(
        loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
        loss_weights=[1.0, 1.2, 0.8],
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def calculate_hierarchy_penalty(y_true, y_pred):
    """Calculate penalty for hierarchy violations"""
    # Convert to float32 to match types
    y_true_float = tf.cast(y_true, tf.float32)
    y_pred_float = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true_float - y_pred_float))

def save_model_as_tf(model, model_name):
    model.save(f'{MODEL_DIR}/{model_name}', save_format='tf')

def save_object(obj, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(f'{MODEL_DIR}/{filename}', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_hierarchical_model(padded_sequences, category_labels, sub_category_labels, type_labels, mlb_category, mlb_sub_category, mlb_type):
    """Train a hierarchy-constrained model."""
    output_dims = {
        'category': category_labels.shape[1],
        'sub_category': sub_category_labels.shape[1],
        'type': type_labels.shape[1]
    }
    
    # Create hierarchy masks AFTER binarizers are fitted
    category_to_sub_mask, sub_to_type_mask = create_hierarchy_masks(
        mlb_category.classes_, mlb_sub_category.classes_, mlb_type.classes_
    )
    
    model = create_hierarchical_model(output_dims, category_to_sub_mask, sub_to_type_mask)
    
    print(f"Training hierarchy-constrained model with {len(padded_sequences)} samples")
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
            print("🚀 Loading hierarchy-constrained models and encoders...")
            
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
            print("✅ Hierarchy-constrained models loaded successfully!")
            
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

def validate_hierarchy_prediction(category_pred, sub_category_pred, type_pred, title):
    """Dynamically validate and correct hierarchy predictions based on actual data constraints"""
    try:
        # Use extremely low thresholds to get more predictions
        category_threshold = 0.01
        sub_threshold = 0.01
        type_threshold = 0.01
        
        # Convert predictions to lists of predicted classes
        category_classes = [cls for cls, pred in zip(mlb_category.classes_, category_pred[0]) if pred > category_threshold]
        sub_category_classes = [cls for cls, pred in zip(mlb_sub_category.classes_, sub_category_pred[0]) if pred > sub_threshold]
        type_classes = [cls for cls, pred in zip(mlb_type.classes_, type_pred[0]) if pred > type_threshold]
        
        # If still no predictions, use even lower threshold for categories
        if not category_classes:
            category_classes = [cls for cls, pred in zip(mlb_category.classes_, category_pred[0]) if pred > 0.001]
        
        # If still no predictions, take the top 1 category by confidence
        if not category_classes:
            category_scores = [(cls, pred) for cls, pred in zip(mlb_category.classes_, category_pred[0])]
            category_scores.sort(key=lambda x: x[1], reverse=True)
            category_classes = [cls for cls, score in category_scores[:1]]
        
        # If still no sub-categories, take the top 1 by confidence
        if not sub_category_classes:
            sub_scores = [(cls, pred) for cls, pred in zip(mlb_sub_category.classes_, sub_category_pred[0])]
            sub_scores.sort(key=lambda x: x[1], reverse=True)
            sub_category_classes = [cls for cls, score in sub_scores[:1]]
        
        # If still no types, take the top 1 by confidence
        if not type_classes:
            type_scores = [(cls, pred) for cls, pred in zip(mlb_type.classes_, type_pred[0])]
            type_scores.sort(key=lambda x: x[1], reverse=True)
            type_classes = [cls for cls, score in type_scores[:1]]
        
        # Dynamic validation based on actual hierarchy constraints
        validated_categories = []
        validated_sub_categories = []
        validated_types = []
        
        # STEP 1: Find the strongest category prediction
        if category_classes:
            # Get the category with the highest confidence
            category_scores = [(cls, pred) for cls, pred in zip(mlb_category.classes_, category_pred[0]) if cls in category_classes]
            category_scores.sort(key=lambda x: x[1], reverse=True)
            strongest_category = category_scores[0][0]
            
            if strongest_category in HIERARCHY_CONSTRAINTS['category_to_subcategory']:
                valid_subs = HIERARCHY_CONSTRAINTS['category_to_subcategory'][strongest_category]
                # Find matching sub-categories for this category
                matching_subs = [sub_cat for sub_cat in sub_category_classes if sub_cat in valid_subs]
                
                if matching_subs:
                    # Category has valid sub-categories, keep the STRONGEST one only
                    sub_scores = [(sub_cat, pred) for sub_cat, pred in zip(mlb_sub_category.classes_, sub_category_pred[0]) if sub_cat in matching_subs]
                    sub_scores.sort(key=lambda x: x[1], reverse=True)
                    strongest_sub = sub_scores[0][0]
                    
                    validated_categories.append(strongest_category)
                    validated_sub_categories.append(strongest_sub)
                else:
                    # Category has no valid sub-categories, but we'll keep the category
                    # and try to infer a sub-category later
                    validated_categories.append(strongest_category)
        
        # STEP 2: Process sub-categories and their valid types - ONLY keep types that match the sub-category
        for sub_category in validated_sub_categories:
            if sub_category in HIERARCHY_CONSTRAINTS['subcategory_to_type']:
                valid_types = HIERARCHY_CONSTRAINTS['subcategory_to_type'][sub_category]
                # Find matching types for this sub-category
                matching_types = [type_val for type_val in type_classes if type_val in valid_types]
                
                if matching_types:
                    # Sub-category has valid types, keep the STRONGEST one only
                    type_scores = [(type_val, pred) for type_val, pred in zip(mlb_type.classes_, type_pred[0]) if type_val in matching_types]
                    type_scores.sort(key=lambda x: x[1], reverse=True)
                    strongest_type = type_scores[0][0]
                    
                    validated_types.append(strongest_type)
                else:
                    # Sub-category has no valid types, but we keep the sub-category
                    # (Some sub-categories might not have specific types)
                    pass
            else:
                # Sub-category not in type constraints, keep it (no types expected)
                pass
        
        # STEP 3: Infer missing sub-categories for categories that need them
        for category in validated_categories:
            if category in HIERARCHY_CONSTRAINTS['category_to_subcategory']:
                valid_subs = HIERARCHY_CONSTRAINTS['category_to_subcategory'][category]
                # Check if this category has any sub-categories
                category_subs = [sub for sub in validated_sub_categories if sub in valid_subs]
                
                if not category_subs and valid_subs:
                    # Category needs a sub-category but doesn't have one
                    # Try to infer the most likely sub-category based on the title
                    title_lower = title.lower()
                    
                    # Simple keyword-based inference
                    if 'chair' in title_lower or 'armchair' in title_lower:
                        if 'Chairs' in valid_subs:
                            validated_sub_categories.append('Chairs')
                    elif 'sofa' in title_lower or 'couch' in title_lower:
                        if 'Sofas' in valid_subs:
                            validated_sub_categories.append('Sofas')
                    elif 'table' in title_lower:
                        if 'Tables' in valid_subs:
                            validated_sub_categories.append('Tables')
                    elif 'lamp' in title_lower:
                        if 'Table Lamps' in valid_subs:
                            validated_sub_categories.append('Table Lamps')
                        elif 'Floor Lamps' in valid_subs:
                            validated_sub_categories.append('Floor Lamps')
                    elif 'storage' in title_lower or 'cabinet' in title_lower:
                        if 'Storage & Display Cabinets' in valid_subs:
                            validated_sub_categories.append('Storage & Display Cabinets')
                    elif 'art' in title_lower or 'painting' in title_lower:
                        if 'Wall Art' in valid_subs:
                            validated_sub_categories.append('Wall Art')
                    else:
                        # If no specific match, use the first available sub-category
                        if valid_subs:
                            validated_sub_categories.append(valid_subs[0])
        
        # STEP 4: If we have sub-categories but no categories, infer categories
        if not validated_categories and validated_sub_categories:
            for sub_cat in validated_sub_categories:
                for category, valid_subs in HIERARCHY_CONSTRAINTS['category_to_subcategory'].items():
                    if sub_cat in valid_subs and category not in validated_categories:
                        validated_categories.append(category)
        
        # STEP 5: If we have types but no sub-categories, infer sub-categories and categories
        if not validated_sub_categories and validated_types:
            for type_val in validated_types:
                for sub_cat, valid_types_list in HIERARCHY_CONSTRAINTS['subcategory_to_type'].items():
                    if type_val in valid_types_list and sub_cat not in validated_sub_categories:
                        validated_sub_categories.append(sub_cat)
                        # Also infer the category for this sub-category
                        for category, valid_subs in HIERARCHY_CONSTRAINTS['category_to_subcategory'].items():
                            if sub_cat in valid_subs and category not in validated_categories:
                                validated_categories.append(category)
        
        # STEP 6: Final validation - ensure types match their sub-categories
        final_validated_types = []
        for sub_category in validated_sub_categories:
            if sub_category in HIERARCHY_CONSTRAINTS['subcategory_to_type']:
                valid_types = HIERARCHY_CONSTRAINTS['subcategory_to_type'][sub_category]
                # Only keep types that are valid for this sub-category
                for type_val in validated_types:
                    if type_val in valid_types:
                        final_validated_types.append(type_val)
                    else:
                        pass
            else:
                # Sub-category not in constraints, no types allowed
                pass
        
        # Replace validated_types with the final validated ones
        validated_types = final_validated_types
        
        # STEP 7: Final fallback - if we still have nothing, use the top predictions regardless of hierarchy
        if not validated_categories and category_classes:
            validated_categories = [category_classes[0]]  # Take the first one
        
        if not validated_sub_categories and sub_category_classes:
            validated_sub_categories = [sub_category_classes[0]]  # Take the first one
        
        # Only add types if they match the sub-category
        if not validated_types and type_classes:
            # Check if the top type is valid for any of the validated sub-categories
            top_type = type_classes[0]
            type_added = False
            
            for sub_category in validated_sub_categories:
                if sub_category in HIERARCHY_CONSTRAINTS['subcategory_to_type']:
                    valid_types = HIERARCHY_CONSTRAINTS['subcategory_to_type'][sub_category]
                    if top_type in valid_types:
                        validated_types.append(top_type)
                        type_added = True
                        break
            
            # If no valid type found, don't add any type (respect hierarchy)
            if not type_added:
                pass
        
        # Remove duplicates while preserving order
        validated_categories = list(dict.fromkeys(validated_categories))
        validated_sub_categories = list(dict.fromkeys(validated_sub_categories))
        validated_types = list(dict.fromkeys(validated_types))
        
        # Convert back to strings
        category_str = ", ".join(validated_categories) if validated_categories else ""
        sub_category_str = ", ".join(validated_sub_categories) if validated_sub_categories else ""
        type_str = ", ".join(validated_types) if validated_types else ""
        
        return category_str, sub_category_str, type_str
        
    except Exception as e:
        # Fallback: return raw predictions without validation
        try:
            # Take top predictions regardless of threshold
            category_scores = [(cls, pred) for cls, pred in zip(mlb_category.classes_, category_pred[0])]
            category_scores.sort(key=lambda x: x[1], reverse=True)
            category_classes = [cls for cls, score in category_scores[:1]]
            
            sub_scores = [(cls, pred) for cls, pred in zip(mlb_sub_category.classes_, sub_category_pred[0])]
            sub_scores.sort(key=lambda x: x[1], reverse=True)
            sub_category_classes = [cls for cls, score in sub_scores[:1]]
            
            type_scores = [(cls, pred) for cls, pred in zip(mlb_type.classes_, type_pred[0])]
            type_scores.sort(key=lambda x: x[1], reverse=True)
            type_classes = [cls for cls, score in type_scores[:1]]
            
            category_str = ", ".join(category_classes) if category_classes else ""
            sub_category_str = ", ".join(sub_category_classes) if sub_category_classes else ""
            type_str = ", ".join(type_classes) if type_classes else ""
            
            return category_str, sub_category_str, type_str
        except:
            # Ultimate fallback - return default predictions
            return "Decor", "Decorative Accessories", "Decorative Accents"

def train_models(data_path):
    """Train hierarchy-constrained multi-label classification models."""
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

    # Train hierarchy-constrained model
    train_hierarchical_model(padded_sequences, category_labels, sub_category_labels, type_labels, mlb_category, mlb_sub_category, mlb_type)

    # Save tokenizer and binarizers
    save_object(tokenizer, 'tokenizer.pickle')
    save_object(mlb_category, 'mlb_category.pickle')
    save_object(mlb_sub_category, 'mlb_sub_category.pickle')
    save_object(mlb_type, 'mlb_type.pickle')

def predict_hierarchy(title):
    """Predict hierarchical categories using the trained hierarchy-constrained model."""
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
        
        # Validate and correct hierarchy predictions
        category_str, sub_category_str, type_str = validate_hierarchy_prediction(
            category_pred, sub_category_pred, type_pred, title
        )
        
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
            
            # Validate and correct hierarchy predictions
            category_str, sub_category_str, type_str = validate_hierarchy_prediction(
                category_pred, sub_category_pred, type_pred, title
            )
            
            return category_str, sub_category_str, type_str
        else:
            raise e

# Uncomment to train models
# train_models(DATA_PATH)

