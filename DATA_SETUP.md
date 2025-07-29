# Data Setup Guide

## 📁 Required Data Files

This repository requires training data to function. The data files are not included due to size constraints.

### Required File Structure

```
data/
├── raw_data/
│   ├── .gitkeep
│   └── kashew_supervised_products.csv  # ← Add your training data here
└── cleaned_data/
    └── .gitkeep
```

### Data Format

Your `kashew_supervised_products.csv` file should have these columns:

| Column         | Description                    | Example                                   |
| -------------- | ------------------------------ | ----------------------------------------- |
| `title`        | Product title                  | "IKEA Billy Bookcase"                     |
| `category`     | Comma-separated categories     | "Storage" or "Seating, Storage"           |
| `sub_category` | Comma-separated sub-categories | "Bookcases & Shelving"                    |
| `type`         | Comma-separated types          | "None" or "Coffee tables, Console tables" |

### Example Data

```csv
title,category,sub_category,type
"IKEA Billy Bookcase",Storage,"Bookcases & Shelving",None
"Leather Armchair Brown",Seating,Armchairs,"Arm Chairs"
"Coffee Table Glass","Tables & Desks",Tables,"Coffee tables"
```

### Getting Started

1. **Add your data file** to `data/raw_data/kashew_supervised_products.csv`
2. **Train the model**: `python train.py`
3. **Run the API**: `python app.py`

### Data Requirements

- **Format**: CSV with UTF-8 encoding
- **Hierarchical**: Categories → Sub-categories → Types
- **Multi-label**: Items can have multiple categories/sub-categories/types
- **Size**: The model works best with 10,000+ items

### Notes

- Data files are excluded from git via `.gitignore`
- The model will automatically clean and preprocess your data
- Processed data will be saved to `data/cleaned_data/` during training
