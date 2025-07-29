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
MAX_WORDS = 5000
MAX_LEN = 50

# Global variables for models and encoders
hierarchical_model = None
tokenizer = None
mlb_category = None
mlb_sub_category = None
mlb_type = None

def parse_multi_labels(label_string):
    """Parse multi-label strings into lists."""
    if pd.isna(label_string) or label_string == '':
        return []
    return [label.strip() for label in str(label_string).split(',') if label.strip()]

def create_hierarchical_model(output_dims):
    """Create a hierarchical model that considers parent-child relationships."""
    # Input layer
    input_layer = Input(shape=(MAX_LEN,))
    
    # Shared embedding and LSTM layers
    embedding = Embedding(MAX_WORDS, 64)(input_layer)
    dropout1 = SpatialDropout1D(0.2)(embedding)
    lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(dropout1)
    
    # Extract features from LSTM
    lstm_features = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(lstm)
    
    # Category prediction (parent level)
    category_dense = Dense(128, activation='relu')(lstm_features)
    category_dropout = Dropout(0.3)(category_dense)
    category_output = Dense(output_dims['category'], activation='sigmoid')(category_dropout)
    
    # Sub-category prediction (child level) - influenced by category predictions
    sub_category_dense = Dense(128, activation='relu')(lstm_features)
    # Concatenate with category predictions for hierarchy
    category_embedding = Dense(64, activation='relu')(category_output)
    sub_category_combined = Concatenate()([sub_category_dense, category_embedding])
    sub_category_dropout = Dropout(0.3)(sub_category_combined)
    sub_category_output = Dense(output_dims['sub_category'], activation='sigmoid')(sub_category_dropout)
    
    # Type prediction (grandchild level) - influenced by both category and sub-category
    type_dense = Dense(128, activation='relu')(lstm_features)
    sub_category_embedding = Dense(64, activation='relu')(sub_category_output)
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
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.001, restore_best_weights=True)],
        verbose=1
    )
    
    save_model_as_tf(model, 'hierarchical_model')

def load_models_and_encoders():
    global hierarchical_model, tokenizer, mlb_category, mlb_sub_category, mlb_type

    # Load hierarchical model
    hierarchical_model = tf.keras.models.load_model(f'{MODEL_DIR}/hierarchical_model', custom_objects={'Adam': tf.keras.optimizers.Adam})

    # Load tokenizer and binarizers
    with open(f'{MODEL_DIR}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'{MODEL_DIR}/mlb_category.pickle', 'rb') as handle:
        mlb_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/mlb_sub_category.pickle', 'rb') as handle:
        mlb_sub_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/mlb_type.pickle', 'rb') as handle:
        mlb_type = pickle.load(handle)

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
    # Preprocess the title
    clean_title_text = clean_title(title)
    sequence = tokenizer.texts_to_sequences([clean_title_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Make predictions
    category_pred, sub_category_pred, type_pred = hierarchical_model.predict(padded_sequence)
    
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

# Uncomment to train models
# train_models(DATA_PATH) 