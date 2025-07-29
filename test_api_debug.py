#!/usr/bin/env python3
import requests
import json
import time

def test_api_debug():
    """Debug test to see what the API is actually returning"""
    
    test_cases = [
        "Antique Meji New Year Day 1892 Triptych",
        "Fatboy Usa Original Slim Bean Bag Chair", 
        "Trombone Floor Lamp with Marble Table",
        "Stanley Round Commode Table"
    ]
    
    for title in test_cases:
        print(f"\n🔍 Testing: {title}")
        print("-" * 50)
        
        try:
            response = requests.post(
                'http://localhost:8080/predict',
                json={'title': title},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"API Response:")
                print(f"  Category: '{result.get('category', 'N/A')}'")
                print(f"  Sub-category: '{result.get('sub_category', 'N/A')}'")
                print(f"  Type: '{result.get('type', 'N/A')}'")
                
                # Check if empty
                if not result.get('category') and not result.get('sub_category') and not result.get('type'):
                    print(f"  ❌ EMPTY PREDICTION")
                else:
                    print(f"  ✅ HAS PREDICTIONS")
                    
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🐛 API Debug Test")
    print("=" * 50)
    test_api_debug() 