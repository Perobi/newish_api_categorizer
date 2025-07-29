#!/usr/bin/env python3
import requests
import json
import time

def test_api_prediction(title):
    """Test a single prediction with the API"""
    try:
        response = requests.post(
            'http://localhost:8080/predict',
            json={'title': title},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {title}")
            print(f"   Category: {result.get('category', 'N/A')}")
            print(f"   Sub-category: {result.get('sub_category', 'N/A')}")
            print(f"   Type: {result.get('type', 'N/A')}")
            
            # Check for empty sub-categories
            category = result.get('category', '')
            sub_category = result.get('sub_category', '')
            
            if category and not sub_category:
                print(f"   ❌ ISSUE: Category '{category}' has empty sub-category!")
            elif category and sub_category:
                print(f"   ✅ GOOD: Category '{category}' has sub-category '{sub_category}'")
            else:
                print(f"   ⚠️  No category predicted")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def main():
    print("🧪 Testing API Hierarchy Validation")
    print("=" * 50)
    
    # Test cases that previously had issues
    test_cases = [
        "Antique Meji New Year Day 1892 Triptych",
        "Sunrise Home Lee Industries Nol.1074 Brown Velvet",
        "Stanley Round Commode Table",
        "Fatboy Usa Original Slim Bean Bag Chair",
        "2 Section Laundry Hamper",
        "Baker Furniture Co Mahogany Canterbury or Bottle C",
        "Trombone Floor Lamp with Marble Table",
        "Four Piece White Wicker Bedroom Set",
        "New International Atelier Northpoint Nightand",
        "Vintage Port Storage Book Box"
    ]
    
    for i, title in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        test_api_prediction(title)
        time.sleep(1)  # Small delay between requests
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main() 