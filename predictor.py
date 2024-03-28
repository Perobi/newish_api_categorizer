import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from clean_data import clean_title  # Ensure this function is correctly defined in your module

# Global constants
DATA_PATH = './data/raw_data/kashew_ml_products_mar_2024.csv'
MODEL_DIR = './model'
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

def create_model(output_dim):
    model = Sequential([
        Embedding(MAX_WORDS, 40),
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

def train_and_save_model(padded_sequences, labels, model_name, y_integers):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = create_model(labels.shape[1])
    model.fit(padded_sequences, labels, batch_size=4, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)], class_weight=class_weight_dict)
    model.save(f'{MODEL_DIR}/{model_name}.keras')

def adjust_predictions_based_on_schema(category, sub_category, type_):
    for cat in schema:
        if cat['category'] == category:
            valid_sub_categories = [sub['name'] for sub in cat.get('subCategories', [])]
            if sub_category not in valid_sub_categories:
                sub_category = valid_sub_categories[0] if valid_sub_categories else 'None'
                
            valid_types = []
            for sub in cat.get('subCategories', []):
                if sub['name'] == sub_category and 'types' in sub:
                    valid_types = [t['name'] for t in sub['types']]
                    break
                    
            if type_ not in valid_types:
                type_ = valid_types[0] if valid_types else 'None'
            break
    return category, sub_category, type_

def train_models(data_path):
    tf.keras.backend.clear_session()
    df = pd.read_csv(data_path)
    df['clean_title'] = df['title'].apply(clean_title)

    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(df['clean_title'])
    sequences = tokenizer.texts_to_sequences(df['clean_title'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

    label_encoder_category = LabelEncoder()
    label_encoder_sub_category = LabelEncoder()
    label_encoder_type = LabelEncoder()

    df['category_encoded'] = label_encoder_category.fit_transform(df['category'])
    df['sub_category_encoded'] = label_encoder_sub_category.fit_transform(df['sub_category'])
    df['type_encoded'] = label_encoder_type.fit_transform(df['type'].fillna('None'))  # Handle NaN types

    category_labels = to_categorical(df['category_encoded'])
    sub_category_labels = to_categorical(df['sub_category_encoded'])
    type_labels = to_categorical(df['type_encoded'])

    # Pass integer-encoded labels for class weight computation
    train_and_save_model(padded_sequences, category_labels, 'category', df['category_encoded'])
    train_and_save_model(padded_sequences, sub_category_labels, 'sub_category', df['sub_category_encoded'])
    train_and_save_model(padded_sequences, type_labels, 'type', df['type_encoded'])

    save_object(tokenizer, 'tokenizer.pickle')
    save_object(label_encoder_category, 'label_encoder_category.pickle')
    save_object(label_encoder_sub_category, 'label_encoder_sub_category.pickle')
    save_object(label_encoder_type, 'label_encoder_type.pickle')

def predict_hierarchy(title):
    # Load the tokenizer and label encoders
    with open(f'{MODEL_DIR}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_category.pickle', 'rb') as handle:
        label_encoder_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_sub_category.pickle', 'rb') as handle:
        label_encoder_sub_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_type.pickle', 'rb') as handle:
        label_encoder_type = pickle.load(handle)
    
    # Preprocess the title
    clean_title_text = clean_title(title)
    sequence = tokenizer.texts_to_sequences([clean_title_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Load models and make predictions
    category_model = tf.keras.models.load_model(f'{MODEL_DIR}/category.keras')
    sub_category_model = tf.keras.models.load_model(f'{MODEL_DIR}/sub_category.keras')
    type_model = tf.keras.models.load_model(f'{MODEL_DIR}/type.keras')
    
    category_pred = category_model.predict(padded_sequence)
    sub_category_pred = sub_category_model.predict(padded_sequence)
    type_pred = type_model.predict(padded_sequence)
    
    # Convert predictions to labels
    category = label_encoder_category.inverse_transform([np.argmax(category_pred)])[0]
    sub_category = label_encoder_sub_category.inverse_transform([np.argmax(sub_category_pred)])[0]
    type_ = label_encoder_type.inverse_transform([np.argmax(type_pred)])[0]
    
    # Adjust predictions based on schema
    adjusted_category, adjusted_sub_category, adjusted_type = adjust_predictions_based_on_schema(category, sub_category, type_)
    
    return adjusted_category, adjusted_sub_category, adjusted_type

# make a prediction using the trained model and print the result
predicted_category, predicted_sub_category, predicted_type = predict_hierarchy('La-Z-Boy Burgundy Recliner')
print(f'CAT: {predicted_category}, SUB: {predicted_sub_category}, TYP: {predicted_type}' )


