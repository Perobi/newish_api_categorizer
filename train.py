#!/usr/bin/env python3
"""
Training script for the hierarchical furniture categorization model.
Run this script to retrain the model on updated data.
"""

import os
import sys
from predictor import train_models, DATA_PATH

def main():
    """Main function to train the hierarchical model."""
    print("🚀 Starting hierarchical model training...")
    print(f"📁 Using data from: {DATA_PATH}")
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    
    try:
        # Train the models
        print("🔄 Training hierarchical multi-label classification model...")
        train_models(DATA_PATH)
        print("✅ Model training completed successfully!")
        print("📦 Models saved to ./model/ directory")
        print("\n🎯 The model is now ready for production use!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 