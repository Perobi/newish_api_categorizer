# Furniture Categorization API

A hierarchical multi-label classification system for categorizing furniture items based on their titles. The system predicts three levels of categorization: **Category**, **Sub-category**, and **Type**.

## 🏗️ Architecture

The system uses a **hierarchical multi-label classification** approach with:

- **Single Neural Network**: One model predicts all three levels simultaneously
- **Multi-label Support**: Items can belong to multiple categories/sub-categories/types
- **Hierarchical Consistency**: Maintains parent-child relationships between levels

## 📊 Model Performance

- **Category Accuracy**: ~96% (training), ~95% (validation)
- **Sub-category Accuracy**: ~90% (training), ~88% (validation)
- **Type Accuracy**: ~61% (training), ~57% (validation)
- **Average Prediction Time**: ~0.024 seconds per prediction

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment with uv
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Run the API

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### 3. Make Predictions

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "IKEA Billy Bookcase"}'
```

Response:

```json
{
  "category": "Storage",
  "sub_category": "Bookcases & Shelving",
  "type": "None"
}
```

## 📁 Project Structure

```
├── app.py                 # Flask API server
├── predictor.py           # Core model logic and training
├── clean_data.py          # Text preprocessing utilities
├── train.py              # Training script
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
├── runtime.txt           # Python runtime version
├── data/
│   ├── raw_data/         # Training data
│   └── cleaned_data/     # Processed data
└── model/                # Trained models and encoders
    ├── hierarchical_model/
    ├── tokenizer.pickle
    ├── mlb_category.pickle
    ├── mlb_sub_category.pickle
    └── mlb_type.pickle
```

## 🔄 Retraining the Model

To retrain the model on updated data:

1. **Add new data** to `data/raw_data/kashew_supervised_products.csv`
   > **Note**: Data files are not included in this repository due to size. You'll need to provide your own training data.
2. **Run training**:
   ```bash
   python train.py
   ```
3. **Restart the API** to use the new model

## 📈 Supported Categories

### Categories (6)

- **Beds**
- **Decor**
- **Lighting**
- **Seating**
- **Storage**
- **Tables & Desks**

### Sub-categories (25)

- Armchairs, Armoires & Wardrobes, Bar Carts, Bed Frames, Bookcases & Shelving
- Ceiling & Wall Lamps, Chairs, Chaises & Daybeds, Decorative Accessories
- Desk Lamps, Desks, Dressers & Chests of Drawers, Floor Lamps, Headboards
- Mirrors, Nightstands, Ottomans & Footstools, Rugs & Carpets
- Sideboards & Credenzas, Sofas, Stools, Storage & Display Cabinets
- Table Lamps, Tables, Wall Art

### Types (27)

- Accent & Side tables, Accent Chairs, Arm Chairs, Bar Stools, Carpets
- Club Chairs, Coffee tables, Console tables, Decorative Accents, Desks
- Dining Chairs, Dining tables, Full Length & Floor Mirrors, Kitchen Accessoires
- Loveseats, Office Chairs, Paintings, Picture Frames, Recliners
- Sculptures & Statues, Secretary Desks, Sectionals, Sofas, Swivel Chairs
- Vases, Wall Decorative Accents, Wall Mirrors

## 🔧 API Endpoints

### POST /predict

Predicts hierarchical categories for a furniture item title.

**Request:**

```json
{
  "title": "Leather Armchair Brown"
}
```

**Response:**

```json
{
  "category": "Seating",
  "sub_category": "Armchairs",
  "type": "Arm Chairs"
}
```

**Multi-label Example:**

```json
{
  "title": "Swivel Chair Leather"
}
```

**Response:**

```json
{
  "category": "Seating",
  "sub_category": "Armchairs, Chairs",
  "type": "Accent Chairs, Arm Chairs, Swivel Chairs"
}
```

## 🛠️ Development

### Key Features

1. **Hierarchical Multi-label Classification**: Single model predicts all three levels
2. **Multi-label Support**: Items can belong to multiple categories simultaneously
3. **Text Preprocessing**: Cleans and normalizes input titles
4. **Adaptive Thresholds**: Dynamic confidence thresholds for multi-label predictions
5. **Production Ready**: Optimized for deployment with proper error handling

### Model Architecture

- **Input**: Text sequences (max 50 words)
- **Embedding**: 128-dimensional word embeddings
- **LSTM**: 128 units with dropout
- **Output**: Three dense layers for category, sub-category, and type prediction
- **Activation**: Sigmoid for multi-label classification

### Data Format

Training data should be in CSV format with columns:

- `title`: Product title
- `category`: Comma-separated categories
- `sub_category`: Comma-separated sub-categories
- `type`: Comma-separated types

**Data Requirements:**

- Place your training data in `data/raw_data/kashew_supervised_products.csv`
- The model expects hierarchical multi-label data
- Each item can have multiple categories, sub-categories, and types
- Data files are not included in this repository (see .gitignore)

## 🚀 Deployment

### Heroku

The app is configured for Heroku deployment with:

- `Procfile`: Defines the web process
- `runtime.txt`: Specifies Python version
- `requirements.txt`: Lists dependencies

### Local Production

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.
