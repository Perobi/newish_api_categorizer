#!/usr/bin/env python3
"""
Hierarchy Validation Test: Check if category > sub-category > type relationships are correct
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
from collections import defaultdict, Counter

# API endpoint
API_URL = "http://localhost:8080"

def load_hierarchy_test_data():
    """Load data with known hierarchical relationships"""
    print("📊 Loading hierarchy test data...")
    
    # Read the CSV file
    df = pd.read_csv('data/raw_data/kashew_supervised_products.csv')
    
    # Remove rows with missing categories
    df = df.dropna(subset=['category', 'sub_category'])
    
    # Create a comprehensive test set covering different hierarchies
    test_cases = []
    
    # Sample from each major category to test hierarchies
    categories = df['category'].unique()
    
    for category in categories:
        category_data = df[df['category'] == category]
        if len(category_data) > 0:
            # Sample 1-2 items from each category (for 30 total)
            sample_size = min(2, len(category_data))
            samples = category_data.sample(n=sample_size, random_state=42)
            
            for _, row in samples.iterrows():
                test_cases.append({
                    'title': row['title'],
                    'expected_hierarchy': {
                        'category': row['category'],
                        'sub_category': row['sub_category'],
                        'type': row['type'] if pd.notna(row['type']) else None
                    }
                })
    
    print(f"✅ Loaded {len(test_cases)} test cases across {len(categories)} categories")
    return test_cases

def validate_hierarchy_structure(predicted_hierarchy, expected_hierarchy):
    """Validate if the predicted hierarchy structure is correct"""
    issues = []
    
    # Check if all required fields are present
    required_fields = ['category', 'sub_category']
    for field in required_fields:
        if field not in predicted_hierarchy:
            issues.append(f"Missing {field} field")
    
    # Check if category is not None/empty
    if not predicted_hierarchy.get('category'):
        issues.append("Category is empty or None")
    
    # Check if sub_category is not None/empty
    if not predicted_hierarchy.get('sub_category'):
        issues.append("Sub-category is empty or None")
    
    return issues

def validate_hierarchy_relationships(predicted_hierarchy, expected_hierarchy):
    """Validate if the hierarchical relationships make sense"""
    issues = []
    
    predicted_cat = predicted_hierarchy.get('category', '')
    predicted_sub = predicted_hierarchy.get('sub_category', '')
    predicted_type = predicted_hierarchy.get('type')
    
    expected_cat = expected_hierarchy.get('category', '')
    expected_sub = expected_hierarchy.get('sub_category', '')
    expected_type = expected_hierarchy.get('type')
    
    # Check for common hierarchy violations
    common_violations = [
        # Category should not be a sub-category of itself
        (predicted_cat == predicted_sub, f"Category '{predicted_cat}' equals sub-category '{predicted_sub}'"),
        
        # Sub-category should not be empty when category exists
        (predicted_cat and not predicted_sub, f"Category '{predicted_cat}' has empty sub-category"),
        
        # Check for impossible combinations
        (predicted_cat == 'Seating' and 'Table' in predicted_sub, f"Seating category with table sub-category: {predicted_sub}"),
        (predicted_cat == 'Tables & Desks' and 'Chair' in predicted_sub, f"Tables category with chair sub-category: {predicted_sub}"),
        (predicted_cat == 'Storage' and 'Lighting' in predicted_sub, f"Storage category with lighting sub-category: {predicted_sub}"),
    ]
    
    for condition, message in common_violations:
        if condition:
            issues.append(message)
    
    return issues

def test_hierarchy_validation():
    """Test hierarchical relationships"""
    print("🎯 Hierarchy Validation Test")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load test data
    test_cases = load_hierarchy_test_data()
    
    results = []
    hierarchy_issues = []
    structure_issues = []
    relationship_issues = []
    
    print(f"\n🔍 Testing {len(test_cases)} items for hierarchy validation...")
    
    for i, test_case in enumerate(test_cases, 1):
        title = test_case['title']
        expected = test_case['expected_hierarchy']
        
        print(f"\n📝 Item {i:3d}/{len(test_cases)}: {title[:50]}...")
        
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                prediction = response.json()
                
                # Validate hierarchy structure
                structure_problems = validate_hierarchy_structure(prediction, expected)
                if structure_problems:
                    structure_issues.extend(structure_problems)
                    print(f"   ❌ Structure issues: {', '.join(structure_problems)}")
                
                # Validate hierarchy relationships
                relationship_problems = validate_hierarchy_relationships(prediction, expected)
                if relationship_problems:
                    relationship_issues.extend(relationship_problems)
                    print(f"   ❌ Relationship issues: {', '.join(relationship_problems)}")
                
                # Check if hierarchy is logically consistent
                hierarchy_problems = []
                
                # Check for multi-category items that should be single category
                if ',' in prediction.get('category', ''):
                    hierarchy_problems.append("Multi-category detected where single expected")
                
                # Check for inconsistent sub-category patterns
                sub_cat = prediction.get('sub_category', '')
                if sub_cat and ',' in sub_cat and len(sub_cat.split(',')) > 3:
                    hierarchy_problems.append("Too many sub-categories in one field")
                
                if hierarchy_problems:
                    hierarchy_issues.extend(hierarchy_problems)
                    print(f"   ⚠️  Hierarchy issues: {', '.join(hierarchy_problems)}")
                
                # Determine overall status
                if structure_problems or relationship_problems or hierarchy_problems:
                    status = "❌"
                    print(f"   {status} {response_time:.2f}s | Hierarchy validation failed")
                else:
                    status = "✅"
                    print(f"   {status} {response_time:.2f}s | Hierarchy validation passed")
                
                result = {
                    'title': title,
                    'success': True,
                    'response_time': response_time,
                    'expected': expected,
                    'predicted': prediction,
                    'structure_issues': structure_problems,
                    'relationship_issues': relationship_problems,
                    'hierarchy_issues': hierarchy_problems,
                    'valid': not (structure_problems or relationship_problems or hierarchy_problems)
                }
                
            else:
                result = {
                    'title': title,
                    'success': False,
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'expected': expected
                }
                print(f"   ❌ {response_time:.2f}s | Failed: {response.status_code}")
            
            results.append(result)
            
        except Exception as e:
            result = {
                'title': title,
                'success': False,
                'response_time': 0,
                'error': str(e),
                'expected': expected
            }
            results.append(result)
            print(f"   ❌ Error: {e}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📈 HIERARCHY VALIDATION ANALYSIS")
    print("=" * 60)
    
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    valid_hierarchies = [r for r in successful_requests if r.get('valid', False)]
    
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {len(failed_requests)}")
    print(f"Valid hierarchies: {len(valid_hierarchies)}")
    print(f"Invalid hierarchies: {len(successful_requests) - len(valid_hierarchies)}")
    print(f"Hierarchy validity rate: {(len(valid_hierarchies)/len(successful_requests))*100:.1f}%")
    
    # Issue breakdown
    print(f"\n❌ Issue Breakdown:")
    print(f"   Structure issues: {len(structure_issues)}")
    print(f"   Relationship issues: {len(relationship_issues)}")
    print(f"   Hierarchy issues: {len(hierarchy_issues)}")
    
    # Category-wise hierarchy validation
    print(f"\n📂 Category-wise Hierarchy Validation:")
    category_stats = defaultdict(lambda: {'valid': 0, 'total': 0})
    
    for r in successful_requests:
        cat = r['predicted'].get('category', 'Unknown')
        category_stats[cat]['total'] += 1
        if r.get('valid', False):
            category_stats[cat]['valid'] += 1
    
    for cat, stats in sorted(category_stats.items()):
        validity_rate = (stats['valid'] / stats['total']) * 100
        print(f"   {cat}: {validity_rate:.1f}% ({stats['valid']}/{stats['total']})")
    
    # Common hierarchy issues
    if structure_issues or relationship_issues or hierarchy_issues:
        print(f"\n🔍 Common Issues:")
        
        # Structure issues
        if structure_issues:
            print(f"   Structure Issues:")
            issue_counts = Counter(structure_issues)
            for issue, count in issue_counts.most_common(5):
                print(f"      {issue}: {count} times")
        
        # Relationship issues
        if relationship_issues:
            print(f"   Relationship Issues:")
            issue_counts = Counter(relationship_issues)
            for issue, count in issue_counts.most_common(5):
                print(f"      {issue}: {count} times")
        
        # Hierarchy issues
        if hierarchy_issues:
            print(f"   Hierarchy Issues:")
            issue_counts = Counter(hierarchy_issues)
            for issue, count in issue_counts.most_common(5):
                print(f"      {issue}: {count} times")
    
    # Performance metrics
    if successful_requests:
        response_times = [r['response_time'] for r in successful_requests]
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average response time: {sum(response_times)/len(response_times):.2f}s")
        print(f"   Median response time: {sorted(response_times)[len(response_times)//2]:.2f}s")
        print(f"   Fastest: {min(response_times):.2f}s")
        print(f"   Slowest: {max(response_times):.2f}s")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"hierarchy_validation_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'valid_hierarchies': len(valid_hierarchies),
                'structure_issues': len(structure_issues),
                'relationship_issues': len(relationship_issues),
                'hierarchy_issues': len(hierarchy_issues)
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_specific_hierarchy_cases():
    """Test specific problematic hierarchy cases"""
    print("\n🎯 Specific Hierarchy Case Testing")
    print("=" * 40)
    
    # Test cases that are known to have hierarchy issues
    problematic_cases = [
        {
            'title': 'Single Drawer Square Wood Side Table Nightstand TR260-4',
            'expected': {'category': 'Storage, Tables & Desks', 'sub_category': 'Nightstands, Tables'},
            'description': 'Multi-category item'
        },
        {
            'title': 'Vintage Barrel Back Arm Chair, Newly Reupholstered',
            'expected': {'category': 'Seating', 'sub_category': 'Armchairs'},
            'description': 'Should be single sub-category'
        },
        {
            'title': 'Taylor King Blue Club Chair',
            'expected': {'category': 'Seating', 'sub_category': 'Armchairs'},
            'description': 'Should be single sub-category'
        }
    ]
    
    for i, case in enumerate(problematic_cases, 1):
        print(f"\n🔍 Test Case {i}: {case['description']}")
        print(f"   Title: {case['title']}")
        print(f"   Expected: {case['expected']['category']} > {case['expected']['sub_category']}")
        
        try:
            payload = {"title": case['title']}
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                prediction = response.json()
                print(f"   Predicted: {prediction.get('category')} > {prediction.get('sub_category')}")
                
                # Check for hierarchy issues
                issues = []
                if ',' in prediction.get('category', ''):
                    issues.append("Multi-category detected")
                if ',' in prediction.get('sub_category', '') and len(prediction.get('sub_category', '').split(',')) > 2:
                    issues.append("Too many sub-categories")
                
                if issues:
                    print(f"   ❌ Issues: {', '.join(issues)}")
                else:
                    print(f"   ✅ No hierarchy issues detected")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 Hierarchy Validation Test")
    print("Validating category > sub-category > type relationships")
    print("=" * 70)
    
    try:
        test_hierarchy_validation()
        test_specific_hierarchy_cases()
        print("\n🎉 Hierarchy validation completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 