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
DATA_PATH = './data/raw_data/kashew_supervised_products.csv'
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

# Global variables for models and encoders
category_model = None
sub_category_model = None
type_model = None
tokenizer = None
label_encoder_category = None
label_encoder_sub_category = None
label_encoder_type = None

def create_model(output_dim):
    model = Sequential([
        Embedding(MAX_WORDS, 40),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_model_as_tf(model, model_name):
    model.save(f'{MODEL_DIR}/{model_name}', save_format='tf')

def save_object(obj, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(f'{MODEL_DIR}/{filename}', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_and_save_model(padded_sequences, labels, model_name, y_integers):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = create_model(labels.shape[1])
    model.fit(padded_sequences, labels, batch_size=4, epochs=10, validation_split=0.2, 
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)], 
              class_weight=class_weight_dict)
    save_model_as_tf(model, model_name)

def load_models_and_encoders():
    global category_model, sub_category_model, type_model
    global tokenizer, label_encoder_category, label_encoder_sub_category, label_encoder_type

    # Load models
    category_model = tf.keras.models.load_model(f'{MODEL_DIR}/category', custom_objects={'Adam': tf.keras.optimizers.Adam})
    sub_category_model = tf.keras.models.load_model(f'{MODEL_DIR}/sub_category', custom_objects={'Adam': tf.keras.optimizers.Adam})
    type_model = tf.keras.models.load_model(f'{MODEL_DIR}/type', custom_objects={'Adam': tf.keras.optimizers.Adam})

    # Load tokenizer and encoders
    with open(f'{MODEL_DIR}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_category.pickle', 'rb') as handle:
        label_encoder_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_sub_category.pickle', 'rb') as handle:
        label_encoder_sub_category = pickle.load(handle)
    with open(f'{MODEL_DIR}/label_encoder_type.pickle', 'rb') as handle:
        label_encoder_type = pickle.load(handle)

def adjust_predictions_based_on_schema(category, sub_category, type_):
    # print(f"Original Prediction - Category: {category}, Sub-Category: {sub_category}, Type: {type_}")

    def find_valid_category(category_name):
        for cat in schema:
            if cat['category'] == category_name:
                return cat
        return None

    def find_valid_sub_categories(category_schema, sub_category_names):
        valid_sub_categories = []
        for sub_name in sub_category_names:
            for sub in category_schema.get('subCategories', []):
                if sub['name'] == sub_name:
                    valid_sub_categories.append(sub)
                    break
        return valid_sub_categories

    def find_valid_types(sub_category_schema, type_names):
        valid_types = []
        if 'types' in sub_category_schema:
            available_types = [t['name'] for t in sub_category_schema['types']]
            for type_name in type_names:
                if type_name in available_types:
                    valid_types.append(type_name)
                elif available_types:
                    # Use the first available type if none match
                    valid_types.append(available_types[0])
        return valid_types or ['None']

    # Split the predictions into lists
    categories = [cat.strip() for cat in category.split(',') if cat.strip()]
    sub_categories = [sub.strip() for sub in sub_category.split(',') if sub.strip()]
    types = [typ.strip() for typ in type_.split(',') if typ.strip()]

    adjusted_categories = []
    adjusted_sub_categories = []
    adjusted_types = []

    for cat in categories:
        category_schema = find_valid_category(cat)
        if category_schema:
            # Check for valid sub-categories
            valid_sub_categories = find_valid_sub_categories(category_schema, sub_categories)
            if valid_sub_categories:
                for sub_cat in valid_sub_categories:
                    # Check for valid types for each valid sub-category
                    valid_types = find_valid_types(sub_cat, types)
                    adjusted_categories.append(cat)
                    adjusted_sub_categories.append(sub_cat['name'])
                    adjusted_types.append(', '.join(valid_types))
            else:
                # If no valid sub-category found, skip this category
                continue
        else:
            continue

    # If no valid categories are found, set all to 'None'
    if not adjusted_categories:
        adjusted_categories.append('None')
        adjusted_sub_categories.append('None')
        adjusted_types.append('None')

    # Remove duplicate values
    adjusted_categories = list(set(adjusted_categories))
    adjusted_sub_categories = list(set(adjusted_sub_categories))
    adjusted_types = list(set(adjusted_types))

    # print(f"Adjusted Prediction - Category: {', '.join(adjusted_categories)}, Sub-Category: {', '.join(adjusted_sub_categories)}, Type: {', '.join(adjusted_types)}")

    return ', '.join(adjusted_categories), ', '.join(adjusted_sub_categories), ', '.join(adjusted_types)


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
    # Preprocess the title
    clean_title_text = clean_title(title)
    sequence = tokenizer.texts_to_sequences([clean_title_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    # Make predictions
    category_pred = category_model.predict(padded_sequence)
    sub_category_pred = sub_category_model.predict(padded_sequence)
    type_pred = type_model.predict(padded_sequence)
    
    # Convert predictions to labels
    category = label_encoder_category.inverse_transform([np.argmax(category_pred)])[0]
    sub_category = label_encoder_sub_category.inverse_transform([np.argmax(sub_category_pred)])[0]
    type_ = label_encoder_type.inverse_transform([np.argmax(type_pred)])[0]
    
    # # Print raw predictions
    # print(f'Raw Predictions - Category: {category}, Sub-Category: {sub_category}, Type: {type_}')
    
    # Adjust predictions based on schema
    adjusted_category, adjusted_sub_category, adjusted_type = adjust_predictions_based_on_schema(category, sub_category, type_)
    
    return adjusted_category, adjusted_sub_category, adjusted_type


# Uncomment the following lines to train models and make predictions
# train_models(DATA_PATH)
# Uncomment below lines only if you need to test the predictions after training
# load_models_and_encoders()
# # Test with examples
# titles = [
#     'La-Z-Boy Burgundy Recliner',  # Example title
#     'Crate & Barrel Woven Rattan Side Dining Chairs, Set of Four',
#     'Crate & Barrel Avalon Dining Table with 2 IKEA Dining Chairs',
#     'IKEA Malm Dresser',
#     'IKEA Billy Bookcase',
#     'Knoll International by Charles Pollock Executive Armchair Brown Tweed Hopsacking on Casters'

# ]

# for title in titles:
#     predicted_category, predicted_sub_category, predicted_type = predict_hierarchy(title)
#     print(f'Predicted - CAT: {predicted_category}, SUB: {predicted_sub_category}, TYP: {predicted_type}')
