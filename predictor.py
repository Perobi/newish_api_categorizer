import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from clean_data import clean_title, clean_description  

# Global constants
DATA_PATH = './data/raw_data/kashew_ml_products.csv'
MODEL_DIR = './models'
MAX_WORDS = 5000
MAX_LEN = 50

schema = [
    {"category": "Seating", "subCategories": [
        {"name": "Chairs", "types": [
            {"name": "Accent Chairs"}, {"name": "Office Chairs"},
            {"name": "Swivel Chairs"}, {"name": "Dining Chairs"}]},
        {"name": "Sofas", "types": [
            {"name": "Loveseats"}, {"name": "Sofas"}, {"name": "Sectionals"}]},
        {"name": "Armchairs", "types": [
            {"name": "Arm Chairs"}, {"name": "Rocking Chairs"},
            {"name": "Club Chairs"}, {"name": "Recliners"}]},
        {"name": "Benches"}, {"name": "Stools", "types": [
            {"name": "Low Stools"}, {"name": "Bar Stools"}]},
        {"name": "Ottomans & Footstools"}, {"name": "Chaises & Daybeds"}]},
    {"category": "Tables & Desks", "subCategories": [
        {"name": "Tables", "types": [
            {"name": "Coffee tables"}, {"name": "Console tables"},
            {"name": "Accent & Side tables"}, {"name": "Dining tables"}]},
        {"name": "Desks", "types": [
            {"name": "Desks"}, {"name": "Secretary Desks"}, {"name": "Vanity Desks"}]}]},
    {"category": "Storage", "subCategories": [
        {"name": "Dressers & Chests of Drawers"}, {"name": "Nightstands"},
        {"name": "Armoires & Wardrobes"}, {"name": "Sideboards & Credenzas"},
        {"name": "Bar Carts"}, {"name": "Storage & Display Cabinets"},
        {"name": "Bookcases & Shelving"}, {"name": "Trunks & Chests"}]},
    {"category": "Beds", "subCategories": [
        {"name": "Headboards"}, {"name": "Bed Frames"}]},
    {"category": "Decor", "subCategories": [
        {"name": "Decorative Accessories", "types": [
            {"name": "Vases"}, {"name": "Sculptures & Statues"},
            {"name": "Decorative Accents"}, {"name": "Kitchen Accessoires"}]},
        {"name": "Room Dividers"}, {"name": "Mirrors", "types": [
            {"name": "Wall Mirrors"}, {"name": "Full Length & Floor Mirrors"}]},
        {"name": "Rugs & Carpets", "types": [
            {"name": "Runners"}, {"name": "Carpets"}]},
        {"name": "Wall Art", "types": [
            {"name": "Paintings"}, {"name": "Picture Frames"},
            {"name": "Wall Decorative Accents"}]}]},
    {"category": "Lighting", "subCategories": [
        {"name": "Table Lamps"}, {"name": "Desk Lamps"},
        {"name": "Ceiling & Wall Lamps"}, {"name": "Floor Lamps"}]},
]



def train_models(data_path):
    # # Reset TensorFlow session
    tf.keras.backend.clear_session()

    df = pd.read_csv(data_path)
    df['clean_title'] = df['title'].apply(clean_title)
 
    # Tokenize combined text
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(df['clean_title'])
    sequences = tokenizer.texts_to_sequences(df['clean_title'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

    # Encode labels
    label_encoder_category = LabelEncoder()
    label_encoder_sub_category = LabelEncoder()
    label_encoder_type = LabelEncoder()

    category_encoded = label_encoder_category.fit_transform(df['category'])
    sub_category_encoded = label_encoder_sub_category.fit_transform(df['sub_category'])
    type_encoded = label_encoder_type.fit_transform(df['type'])

    # Convert to one-hot
    category_labels = to_categorical(category_encoded)
    sub_category_labels = to_categorical(sub_category_encoded)
    type_labels = to_categorical(type_encoded)

    # Train models if the model files do not exist
    train_and_save_model(padded_sequences, category_labels, 'category')

    train_and_save_model(padded_sequences, sub_category_labels, 'sub_category')

    train_and_save_model(padded_sequences, type_labels, 'type')

    # Save tokenizer and label encoders
    save_object(tokenizer, 'tokenizer.pickle')
    save_object(label_encoder_category, 'label_encoder_category.pickle')
    save_object(label_encoder_sub_category, 'label_encoder_sub_category.pickle')
    save_object(label_encoder_type, 'label_encoder_type.pickle')

def train_and_save_model(padded_sequences, labels, model_name):
    model = create_model(labels.shape[1])
    model.fit(padded_sequences, labels, batch_size=4, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    model.save(f'{MODEL_DIR}/{model_name}.keras')

def create_model(output_dim):
    model = Sequential([
        Embedding(MAX_WORDS, 40, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_object(obj, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(f'{MODEL_DIR}/{filename}', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def adjust_predictions_based_on_schema(category, sub_category, type_):
    for cat in schema:
        if cat['category'] == category:
            valid_sub_categories = [sub['name'] for sub in cat['subCategories']]
            if sub_category not in valid_sub_categories:
                sub_category = valid_sub_categories[0]  # Default to the first valid subcategory
                
            for sub in cat['subCategories']:
                if sub['name'] == sub_category:
                    if 'types' in sub:  # If there are types for this subcategory
                        valid_types = [t['name'] for t in sub['types']]
                        if type_ not in valid_types:
                            if valid_types:  # If there are valid types, adjust the type prediction
                                type_ = valid_types[0]
                    else:
                        type_ = 'None'  # If no types are defined for the subcategory, set type to None
            break
    return category, sub_category, type_

def predict_hierarchy(title):
    # Load necessary objects
    model_category = tf.keras.models.load_model(f'{MODEL_DIR}/category.keras')
    model_sub_category = tf.keras.models.load_model(f'{MODEL_DIR}/sub_category.keras')
    model_type = tf.keras.models.load_model(f'{MODEL_DIR}/type.keras')
    with open(f'{MODEL_DIR}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_category.pickle', 'rb') as handle:
        label_encoder_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_sub_category.pickle', 'rb') as handle:
        label_encoder_sub_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_type.pickle', 'rb') as handle:
        label_encoder_type = pickle.load(handle)

    # Clean title and description
    title_cleaned = clean_title(title) 

    # Tokenize combined text
    seq = tokenizer.texts_to_sequences([title_cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    # Predict
    category_pred = model_category.predict(padded)
    sub_category_pred = model_sub_category.predict(padded)
    type_pred = model_type.predict(padded)
    
    category = label_encoder_category.inverse_transform([np.argmax(category_pred)])
    sub_category = label_encoder_sub_category.inverse_transform([np.argmax(sub_category_pred)])
    type_ = label_encoder_type.inverse_transform([np.argmax(type_pred)])
    
   # Adjust predictions based on schema
    category, sub_category, type_ = adjust_predictions_based_on_schema(category[0], sub_category[0], type_[0])
    
    return category, sub_category, type_
        
