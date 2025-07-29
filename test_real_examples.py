#!/usr/bin/env python3
"""
Test script for API categorization using real examples from raw_data
"""

import requests
import json
import time
from typing import Dict, List, Tuple

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

# Real examples from raw_data with expected categories
REAL_EXAMPLES = [
    # Lighting examples
    {
        "title": "IKEA lamp",
        "expected_category": "Lighting",
        "expected_sub_category": "Table Lamps",
        "expected_type": ""
    },
    
    # Seating examples
    {
        "title": "Living Spaces Harper Beige Microfiber 4 Piece Sectional With Left Arm Facing Chaise",
        "expected_category": "Seating",
        "expected_sub_category": "Sofas",
        "expected_type": "Sectionals"
    },
    {
        "title": "Transitional Velvet Lounge Chair with Dark Wood Frame",
        "expected_category": "Seating",
        "expected_sub_category": "Chairs",
        "expected_type": "Accent Chairs"
    },
    {
        "title": "Room & Board Dublin Upholstered Lounge Chair",
        "expected_category": "Seating",
        "expected_sub_category": "Armchairs",
        "expected_type": "Club Chairs"
    },
    
    # Tables & Desks examples
    {
        "title": "Scandinavian Designs Glass Top Console Table",
        "expected_category": "Tables & Desks",
        "expected_sub_category": "Tables",
        "expected_type": "Console tables"
    },
    {
        "title": "Furniture of America Mona Coffee Table",
        "expected_category": "Tables & Desks",
        "expected_sub_category": "Tables",
        "expected_type": "Coffee tables"
    },
    {
        "title": "Devonshire Farmhouse Table by Winners Only New In Box",
        "expected_category": "Tables & Desks",
        "expected_sub_category": "Tables",
        "expected_type": "Dining tables"
    },
    
    # Decor examples
    {
        "title": "Hem Pro Glitch Throw (New)",
        "expected_category": "Decor",
        "expected_sub_category": "Decorative Accessories",
        "expected_type": "Decorative Accents"
    },
    {
        "title": "Christopher Guy (Harrison Gil) Ornate Wall Mirror",
        "expected_category": "Decor",
        "expected_sub_category": "Mirrors",
        "expected_type": "Wall Mirrors"
    },
    {
        "title": "Abstract \"Third Journey 4\" by Marilu Hartnett",
        "expected_category": "Decor",
        "expected_sub_category": "Wall Art",
        "expected_type": "Paintings"
    },
    
    # Storage examples
    {
        "title": "Wood Media Cabinet with Ribbed Glass Paneled Cabinet Doors",
        "expected_category": "Storage",
        "expected_sub_category": "Storage & Display Cabinets",
        "expected_type": ""
    },
    {
        "title": "10-Drawer Dresser by American Drew's American Independence Collection",
        "expected_category": "Storage",
        "expected_sub_category": "Dressers & Chests of Drawers",
        "expected_type": ""
    },
    
    # Edge cases and variations
    {
        "title": "One Drawer End Table",
        "expected_category": "Storage, Tables & Desks",
        "expected_sub_category": "Nightstands",
        "expected_type": ""
    },
    {
        "title": "CB2 Leather Venice Accent Chair",
        "expected_category": "Seating",
        "expected_sub_category": "Chairs, Armchairs",
        "expected_type": ""
    },
    {
        "title": "Chinese Carved Bamboo Happy Buddha Figure",
        "expected_category": "Decor",
        "expected_sub_category": "Decorative Accessories",
        "expected_type": "Decorative Accents"
    }
]

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_single_prediction(title: str, expected_category: str, expected_sub_category: str, expected_type: str) -> Dict:
    """Test a single prediction and return results"""
    try:
        payload = {"title": title}
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predicted_category = result.get('category', 'None')
            predicted_sub_category = result.get('sub_category', 'None')
            predicted_type = result.get('type', 'None')
            
            # Check if predictions match expected (allowing for partial matches)
            category_match = expected_category.lower() in predicted_category.lower() or predicted_category.lower() in expected_category.lower()
            sub_category_match = expected_sub_category.lower() in predicted_sub_category.lower() or predicted_sub_category.lower() in expected_sub_category.lower()
            type_match = expected_type.lower() in predicted_type.lower() or predicted_type.lower() in expected_type.lower() if expected_type else True
            
            return {
                "success": True,
                "title": title,
                "expected": {
                    "category": expected_category,
                    "sub_category": expected_sub_category,
                    "type": expected_type
                },
                "predicted": {
                    "category": predicted_category,
                    "sub_category": predicted_sub_category,
                    "type": predicted_type
                },
                "matches": {
                    "category": category_match,
                    "sub_category": sub_category_match,
                    "type": type_match
                }
            }
        else:
            return {
                "success": False,
                "title": title,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "title": title,
            "error": str(e)
        }

def run_comprehensive_test():
    """Run comprehensive test with all real examples"""
    print("🚀 Starting comprehensive API test with real examples...")
    print("=" * 80)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ Health check failed. Make sure the API is running on port 8080")
        return
    
    print("\n📊 Testing predictions with real examples from raw_data...")
    print("=" * 80)
    
    results = []
    total_tests = len(REAL_EXAMPLES)
    successful_tests = 0
    
    for i, example in enumerate(REAL_EXAMPLES, 1):
        print(f"\n🔍 Test {i}/{total_tests}: {example['title'][:50]}...")
        
        result = test_single_prediction(
            example['title'],
            example['expected_category'],
            example['expected_sub_category'],
            example['expected_type']
        )
        
        results.append(result)
        
        if result['success']:
            matches = result['matches']
            category_status = "✅" if matches['category'] else "❌"
            sub_category_status = "✅" if matches['sub_category'] else "❌"
            type_status = "✅" if matches['type'] else "❌"
            
            print(f"   Category: {category_status} Expected: {example['expected_category']} | Predicted: {result['predicted']['category']}")
            print(f"   Sub-Category: {sub_category_status} Expected: {example['expected_sub_category']} | Predicted: {result['predicted']['sub_category']}")
            print(f"   Type: {type_status} Expected: {example['expected_type']} | Predicted: {result['predicted']['type']}")
            
            if all(matches.values()):
                successful_tests += 1
                print("   🎉 All predictions match!")
            else:
                print("   ⚠️  Some predictions don't match expected values")
        else:
            print(f"   ❌ Test failed: {result['error']}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Successful API calls: {len([r for r in results if r['success']])}")
    print(f"Perfect matches: {successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Detailed breakdown
    print("\n📊 Detailed Results:")
    for i, result in enumerate(results, 1):
        if result['success']:
            matches = result['matches']
            status = "✅" if all(matches.values()) else "⚠️"
            print(f"{status} Test {i}: {result['title'][:40]}...")
        else:
            print(f"❌ Test {i}: {result['title'][:40]}... - {result['error']}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 Testing edge cases...")
    print("=" * 80)
    
    edge_cases = [
        {"title": "", "description": "Empty title"},
        {"title": "   ", "description": "Whitespace only"},
        {"title": "A" * 1000, "description": "Very long title"},
        {"title": "123456789", "description": "Numbers only"},
        {"title": "!@#$%^&*()", "description": "Special characters only"},
        {"title": "Mixed 123 !@# Furniture", "description": "Mixed content"},
    ]
    
    for case in edge_cases:
        print(f"\n🔍 Testing: {case['description']}")
        try:
            payload = {"title": case['title']}
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success - Category: {result.get('category', 'None')}")
                print(f"   Sub-Category: {result.get('sub_category', 'None')}")
                print(f"   Type: {result.get('type', 'None')}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 API Categorization Test Suite")
    print("Using real examples from kashew_supervised_products.csv")
    print("=" * 80)
    
    try:
        run_comprehensive_test()
        test_edge_cases()
        print("\n🎉 Test suite completed!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 