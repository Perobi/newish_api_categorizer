#!/usr/bin/env python3
"""
Cross-reference test: Compare API predictions with actual data labels
"""

import requests
import pandas as pd
import time
import statistics
import json
from datetime import datetime
from collections import defaultdict, Counter
import random

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

def load_sample_data(sample_size=100):
    """Load a random sample from the actual dataset"""
    print(f"📊 Loading {sample_size} random samples from dataset...")
    
    # Read the CSV file
    df = pd.read_csv('data/raw_data/kashew_supervised_products.csv')
    
    # Remove rows with missing categories
    df = df.dropna(subset=['category', 'sub_category'])
    
    # Take a random sample
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"✅ Loaded {len(sample_df)} samples")
    return sample_df

def test_crossreference_accuracy(sample_size=100):
    """Test API accuracy against actual data labels"""
    print("🎯 Cross-Reference Accuracy Test")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load sample data
    sample_df = load_sample_data(sample_size)
    
    results = []
    total_start_time = time.time()
    
    print(f"\n🔍 Testing {len(sample_df)} items...")
    
    for idx, row in sample_df.iterrows():
        title = row['title']
        actual_category = row['category']
        actual_sub_category = row['sub_category']
        actual_type = row['type'] if pd.notna(row['type']) else None
        
        print(f"\n📝 Item {len(results)+1:3d}/{len(sample_df)}: {title[:60]}...")
        
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                prediction = response.json()
                predicted_category = prediction.get('category')
                predicted_sub_category = prediction.get('sub_category')
                predicted_type = prediction.get('type')
                
                # Calculate accuracy
                category_correct = predicted_category == actual_category
                sub_category_correct = predicted_sub_category == actual_sub_category
                type_correct = predicted_type == actual_type if actual_type else True
                
                # Overall accuracy (all levels must be correct)
                overall_correct = category_correct and sub_category_correct and type_correct
                
                result = {
                    'title': title,
                    'success': True,
                    'response_time': response_time,
                    'actual': {
                        'category': actual_category,
                        'sub_category': actual_sub_category,
                        'type': actual_type
                    },
                    'predicted': {
                        'category': predicted_category,
                        'sub_category': predicted_sub_category,
                        'type': predicted_type
                    },
                    'accuracy': {
                        'category_correct': category_correct,
                        'sub_category_correct': sub_category_correct,
                        'type_correct': type_correct,
                        'overall_correct': overall_correct
                    }
                }
                
                results.append(result)
                
                # Print result with color coding
                if overall_correct:
                    print(f"   ✅ {response_time:.2f}s | Perfect match!")
                elif category_correct:
                    print(f"   🟡 {response_time:.2f}s | Category correct, sub-category wrong")
                else:
                    print(f"   ❌ {response_time:.2f}s | Category wrong")
                    print(f"      Expected: {actual_category} > {actual_sub_category}")
                    print(f"      Got: {predicted_category} > {predicted_sub_category}")
                
            else:
                result = {
                    'title': title,
                    'success': False,
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'actual': {
                        'category': actual_category,
                        'sub_category': actual_sub_category,
                        'type': actual_type
                    }
                }
                results.append(result)
                print(f"   ❌ {response_time:.2f}s | Failed: {response.status_code}")
                
        except Exception as e:
            result = {
                'title': title,
                'success': False,
                'response_time': 0,
                'error': str(e),
                'actual': {
                    'category': actual_category,
                    'sub_category': actual_sub_category,
                    'type': actual_type
                }
            }
            results.append(result)
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - total_start_time
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📈 ACCURACY ANALYSIS")
    print("=" * 60)
    
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {len(failed_requests)}")
    print(f"Success rate: {(len(successful_requests)/len(results))*100:.1f}%")
    print(f"Total test time: {total_time:.2f}s")
    print(f"Average time per request: {total_time/len(results):.2f}s")
    
    if successful_requests:
        # Calculate accuracy metrics
        category_correct = [r for r in successful_requests if r['accuracy']['category_correct']]
        sub_category_correct = [r for r in successful_requests if r['accuracy']['sub_category_correct']]
        overall_correct = [r for r in successful_requests if r['accuracy']['overall_correct']]
        
        category_accuracy = len(category_correct) / len(successful_requests) * 100
        sub_category_accuracy = len(sub_category_correct) / len(successful_requests) * 100
        overall_accuracy = len(overall_correct) / len(successful_requests) * 100
        
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Category accuracy: {category_accuracy:.1f}% ({len(category_correct)}/{len(successful_requests)})")
        print(f"   Sub-category accuracy: {sub_category_accuracy:.1f}% ({len(sub_category_correct)}/{len(successful_requests)})")
        print(f"   Overall accuracy: {overall_accuracy:.1f}% ({len(overall_correct)}/{len(successful_requests)})")
        
        # Performance metrics
        response_times = [r['response_time'] for r in successful_requests]
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average response time: {statistics.mean(response_times):.2f}s")
        print(f"   Median response time: {statistics.median(response_times):.2f}s")
        print(f"   Fastest: {min(response_times):.2f}s")
        print(f"   Slowest: {max(response_times):.2f}s")
        
        # Category-wise accuracy
        print(f"\n📂 Category-wise Accuracy:")
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for r in successful_requests:
            actual_cat = r['actual']['category']
            category_stats[actual_cat]['total'] += 1
            if r['accuracy']['category_correct']:
                category_stats[actual_cat]['correct'] += 1
        
        for cat, stats in sorted(category_stats.items()):
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"   {cat}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
        
        # Error analysis
        print(f"\n❌ Error Analysis:")
        category_errors = defaultdict(int)
        sub_category_errors = defaultdict(int)
        
        for r in successful_requests:
            if not r['accuracy']['category_correct']:
                actual = r['actual']['category']
                predicted = r['predicted']['category']
                category_errors[f"{actual} → {predicted}"] += 1
            
            if not r['accuracy']['sub_category_correct']:
                actual = f"{r['actual']['category']} > {r['actual']['sub_category']}"
                predicted = f"{r['predicted']['category']} > {r['predicted']['sub_category']}"
                sub_category_errors[f"{actual} → {predicted}"] += 1
        
        print(f"   Top category misclassifications:")
        for error, count in sorted(category_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {error}: {count} times")
        
        print(f"   Top sub-category misclassifications:")
        for error, count in sorted(sub_category_errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {error}: {count} times")
    
    if failed_requests:
        print(f"\n❌ Failed Requests:")
        for r in failed_requests:
            print(f"   {r['title'][:50]}...: {r['error']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"crossreference_accuracy_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'sample_size': sample_size,
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'total_time': total_time
            },
            'accuracy_metrics': {
                'category_accuracy': category_accuracy if successful_requests else 0,
                'sub_category_accuracy': sub_category_accuracy if successful_requests else 0,
                'overall_accuracy': overall_accuracy if successful_requests else 0
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_category_distribution():
    """Test if our sample covers all major categories"""
    print("\n📊 Category Distribution Analysis")
    print("=" * 40)
    
    df = pd.read_csv('data/raw_data/kashew_supervised_products.csv')
    df = df.dropna(subset=['category'])
    
    category_counts = df['category'].value_counts()
    print("Dataset category distribution:")
    for cat, count in category_counts.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"   {cat}: {count} items ({percentage:.1f}%)")

if __name__ == "__main__":
    print("🎯 Cross-Reference Accuracy Test")
    print("Comparing API predictions with actual dataset labels")
    print("=" * 70)
    
    try:
        # Test with 50 samples
        sample_size = 50
        print(f"\n{'='*20} Testing with {sample_size} samples {'='*20}")
        test_crossreference_accuracy(sample_size)
        
        test_category_distribution()
        print("\n🎉 Cross-reference testing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc() 