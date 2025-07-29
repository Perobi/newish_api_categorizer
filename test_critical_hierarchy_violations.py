#!/usr/bin/env python3
"""
Critical Hierarchy Violations Test: Identify and analyze serious hierarchy issues
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
from collections import defaultdict, Counter

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

def test_critical_violations():
    """Test for critical hierarchy violations"""
    print("🚨 Critical Hierarchy Violations Test")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test cases that should reveal hierarchy issues
    critical_test_cases = [
        {
            'title': 'Antique Meji New Year\'s Day 1892 Triptych',
            'expected': {'category': 'Decor', 'sub_category': 'Wall Art', 'type': None},
            'description': 'Art piece incorrectly classified as seating'
        },
        {
            'title': 'Vintage Barrel Back Arm Chair, Newly Reupholstered',
            'expected': {'category': 'Seating', 'sub_category': 'Armchairs', 'type': None},
            'description': 'Should be single sub-category, not combined'
        },
        {
            'title': 'Single Drawer Square Wood Side Table Nightstand TR260-4',
            'expected': {'category': 'Storage, Tables & Desks', 'sub_category': 'Nightstands, Tables', 'type': None},
            'description': 'Multi-category item'
        },
        {
            'title': 'Vintage Tiffany Alarm Clock',
            'expected': {'category': 'Decor', 'sub_category': 'Decorative Accessories', 'type': None},
            'description': 'Clock should be decor, not functional item'
        },
        {
            'title': 'Waterford Crystal Vase',
            'expected': {'category': 'Decor', 'sub_category': 'Decorative Accessories', 'type': None},
            'description': 'Vase should be decor'
        },
        {
            'title': 'Round Leather Ottoman with Wood Base',
            'expected': {'category': 'Seating', 'sub_category': 'Ottomans & Footstools', 'type': None},
            'description': 'Ottoman should be seating'
        },
        {
            'title': 'Vintage Baker Furniture Co. Brown Coffee Table Onlay w Pull',
            'expected': {'category': 'Tables & Desks', 'sub_category': 'Tables', 'type': 'Coffee tables'},
            'description': 'Coffee table should be tables'
        }
    ]
    
    violations = []
    
    for i, test_case in enumerate(critical_test_cases, 1):
        title = test_case['title']
        expected = test_case['expected']
        description = test_case['description']
        
        print(f"\n🔍 Test Case {i}: {description}")
        print(f"   Title: {title}")
        print(f"   Expected: {expected['category']} > {expected['sub_category']} > {expected['type']}")
        
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                prediction = response.json()
                predicted_cat = prediction.get('category', '')
                predicted_sub = prediction.get('sub_category', '')
                predicted_type = prediction.get('type')
                
                print(f"   Predicted: {predicted_cat} > {predicted_sub} > {predicted_type}")
                
                # Check for critical violations
                violations_found = []
                
                # 1. Type should not be a sub-category name
                if predicted_type and predicted_type in ['Sofas', 'Chairs', 'Tables', 'Lamps', 'Mirrors']:
                    violations_found.append(f"Type '{predicted_type}' should be sub-category")
                
                # 2. Sub-category should not be "None" when type exists
                if predicted_sub == "None" and predicted_type:
                    violations_found.append("Sub-category 'None' with non-empty type")
                
                # 3. Category should not be multi-category when item is clearly single category
                if ',' in predicted_cat and ',' not in expected['category']:
                    violations_found.append(f"Multi-category '{predicted_cat}' for single-category item")
                
                # 4. Impossible category combinations
                if 'Seating' in predicted_cat and 'Table' in predicted_sub:
                    violations_found.append("Seating category with table sub-category")
                if 'Tables' in predicted_cat and 'Chair' in predicted_sub:
                    violations_found.append("Tables category with chair sub-category")
                if 'Decor' in predicted_cat and 'Sofa' in predicted_type:
                    violations_found.append("Decor category with sofa type")
                
                # 5. Empty sub-category with non-empty type
                if not predicted_sub or predicted_sub == "None":
                    if predicted_type and predicted_type not in ["None", None]:
                        violations_found.append("Empty sub-category with non-empty type")
                
                if violations_found:
                    print(f"   ❌ CRITICAL VIOLATIONS: {', '.join(violations_found)}")
                    violations.append({
                        'title': title,
                        'expected': expected,
                        'predicted': prediction,
                        'violations': violations_found,
                        'response_time': response_time
                    })
                else:
                    print(f"   ✅ No critical violations detected")
                
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Analyze violations
    print("\n" + "=" * 60)
    print("🚨 CRITICAL VIOLATION ANALYSIS")
    print("=" * 60)
    
    print(f"Total test cases: {len(critical_test_cases)}")
    print(f"Cases with violations: {len(violations)}")
    print(f"Violation rate: {(len(violations)/len(critical_test_cases))*100:.1f}%")
    
    if violations:
        print(f"\n❌ Violation Types:")
        violation_types = defaultdict(int)
        for v in violations:
            for violation in v['violations']:
                violation_types[violation] += 1
        
        for violation_type, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {violation_type}: {count} times")
        
        print(f"\n🔍 Detailed Violations:")
        for i, violation in enumerate(violations, 1):
            print(f"\n   Case {i}: {violation['title']}")
            print(f"      Expected: {violation['expected']['category']} > {violation['expected']['sub_category']} > {violation['expected']['type']}")
            print(f"      Predicted: {violation['predicted']['category']} > {violation['predicted']['sub_category']} > {violation['predicted']['type']}")
            print(f"      Violations: {', '.join(violation['violations'])}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"critical_violations_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_cases': len(critical_test_cases),
                'violations_found': len(violations)
            },
            'violations': violations
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_type_subcategory_confusion():
    """Test specifically for type/sub-category confusion"""
    print("\n🎯 Type vs Sub-category Confusion Test")
    print("=" * 50)
    
    # Items that should have specific sub-categories, not types
    type_confusion_cases = [
        'Vintage Barrel Back Arm Chair, Newly Reupholstered',
        'Taylor King Blue Club Chair',
        'Round Leather Ottoman with Wood Base',
        'Vintage Baker Furniture Co. Brown Coffee Table Onlay w Pull',
        'Waterford Crystal Vase',
        'Vintage Tiffany Alarm Clock'
    ]
    
    for title in type_confusion_cases:
        print(f"\n🔍 Testing: {title}")
        
        try:
            payload = {"title": title}
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                prediction = response.json()
                cat = prediction.get('category', '')
                sub = prediction.get('sub_category', '')
                type_val = prediction.get('type')
                
                print(f"   Category: {cat}")
                print(f"   Sub-category: {sub}")
                print(f"   Type: {type_val}")
                
                # Check for type/sub-category confusion
                issues = []
                
                # Common sub-categories that shouldn't be types
                sub_category_names = ['Sofas', 'Chairs', 'Tables', 'Lamps', 'Mirrors', 'Ottomans', 'Stools']
                
                if type_val in sub_category_names:
                    issues.append(f"Type '{type_val}' should be sub-category")
                
                if sub == "None" and type_val and type_val not in ["None", None]:
                    issues.append("Empty sub-category with non-empty type")
                
                if issues:
                    print(f"   ❌ Issues: {', '.join(issues)}")
                else:
                    print(f"   ✅ No type/sub-category confusion")
                    
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🚨 Critical Hierarchy Violations Test")
    print("Identifying serious hierarchy relationship issues")
    print("=" * 70)
    
    try:
        test_critical_violations()
        test_type_subcategory_confusion()
        print("\n🎉 Critical violation testing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 