#!/usr/bin/env python3
"""
Comprehensive performance test with 50 real titles from raw_data
"""

import requests
import time
import statistics
import json
from datetime import datetime

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

# 50 real titles from our raw_data
REAL_TITLES = [
    "IKEA lamp",
    "Living Spaces Harper Beige Microfiber 4 Piece Sectional With Left Arm Facing Chaise",
    "Scandinavian Designs Glass Top Console Table",
    "Hem Pro Glitch Throw (New)",
    "Furniture of America Mona Coffee Table",
    "One Drawer End Table",
    "Transitional Velvet Lounge Chair with Dark Wood Frame",
    "Wood Media Cabinet with Ribbed Glass Paneled Cabinet Doors",
    "Room & Board Dublin Upholstered Lounge Chair",
    "Denver Modern Vail White Boucle Counter Stool",
    "Castlery Casa Sideboard",
    "Console Table with Mirror",
    "Ashley Furniture Bedroom Mirror",
    "Amazon Arch Mirror with Angle Stand",
    "Calmart International Round Metal Mirror",
    "Wood Framed Mirror",
    "Distressed Frame Mirror",
    "Target Wood Framed Wall Mirror",
    "Mario Bellini Modern Modular Sofa in Chenille Helios-Evergreen",
    "Kartell Piuma Chair in Mustard (New - in box)",
    "Marlo Round Gold Mirror",
    "Three Drawer Nightstand with Fluted Trim",
    "Faux Leather Dining Chair",
    "Transitional Tan Faux Leather Bar Stool with Dark Wood Frame",
    "Benchmade Modern Skinny Fat 37\" Ottoman",
    "Carved Elephant Heads Accent Table",
    "Eames Style Mid Century Modern Lounge Chair and Ottoman",
    "West Elm Rustic Industrial Counter Stools",
    "Two Drawer Vanity Table with Pop-Up Mirror",
    "Round Leather Ottoman with Wood Base",
    "King Kong USA 5560-Galaxy-D3000 Air Massage Chair",
    "Two Door Storage Cabinet with Diamond Pattern Doors",
    "Contemporary 7-Piece Dining Set by Najarian Furniture",
    "Three Drawer Nightstand",
    "Beige Loveseat with Contemporary Design",
    "Western Style High-Back Lounge Chair with Cowhide and Leather",
    "Tan Velvet Lounge Chair with Rolled Arms",
    "Traditional Cream Velvet Loveseat",
    "Classic Grey 3-Seat Sofa with Accent Pillows",
    "Wishbone Dining Chair",
    "Classic Leather Tufted Square Ottoman",
    "Traditional 3-Seat Sofa with Decorative Pillows",
    "Vintage Style 3-Seat Velvet Sofa with Decorative Pillows",
    "Lulu & Georgia Philana Nightstand",
    "Lulu & Georgia Philana Nightstand",
    "Curved Back 3-Seat Sofa with Accent Pillows",
    "Album Cover Collage Art Piece",
    "Grey Sleeper Sofa with Plush Cushions and Mattress",
    "Soleil Antiqued Brass floor Lamp by Phoenix Day"
]

def test_50_real_titles():
    """Test 50 real titles from our dataset"""
    print("🚀 Testing 50 Real Titles from Raw Data")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = []
    total_start_time = time.time()
    
    for i, title in enumerate(REAL_TITLES, 1):
        print(f"\n🔍 Request {i:2d}/50: {title[:50]}...")
        
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'request_id': i,
                    'title': title,
                    'success': True,
                    'response_time': response_time,
                    'category': result.get('category'),
                    'sub_category': result.get('sub_category'),
                    'type': result.get('type')
                })
                
                print(f"   ✅ {response_time:.2f}s | {result.get('category')} | {result.get('sub_category')} | {result.get('type')}")
            else:
                results.append({
                    'request_id': i,
                    'title': title,
                    'success': False,
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
                print(f"   ❌ {response_time:.2f}s | Failed: {response.status_code}")
                
        except Exception as e:
            results.append({
                'request_id': i,
                'title': title,
                'success': False,
                'response_time': 0,
                'error': str(e)
            })
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - total_start_time
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📈 COMPREHENSIVE PERFORMANCE ANALYSIS")
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
        response_times = [r['response_time'] for r in successful_requests]
        
        print(f"\n📊 Response Time Statistics:")
        print(f"   Average: {statistics.mean(response_times):.2f}s")
        print(f"   Median: {statistics.median(response_times):.2f}s")
        print(f"   Fastest: {min(response_times):.2f}s")
        print(f"   Slowest: {max(response_times):.2f}s")
        print(f"   Standard deviation: {statistics.stdev(response_times):.2f}s")
        
        # Performance categories
        blazing_fast = [t for t in response_times if t < 1.0]
        fast = [t for t in response_times if 1.0 <= t < 3.0]
        moderate = [t for t in response_times if 3.0 <= t < 5.0]
        slow = [t for t in response_times if t >= 5.0]
        
        print(f"\n⚡ Performance Breakdown:")
        print(f"   Blazing fast (< 1s): {len(blazing_fast)} requests ({len(blazing_fast)/len(response_times)*100:.1f}%)")
        if blazing_fast:
            print(f"      Average: {statistics.mean(blazing_fast):.2f}s")
        
        print(f"   Fast (1-3s): {len(fast)} requests ({len(fast)/len(response_times)*100:.1f}%)")
        if fast:
            print(f"      Average: {statistics.mean(fast):.2f}s")
        
        print(f"   Moderate (3-5s): {len(moderate)} requests ({len(moderate)/len(response_times)*100:.1f}%)")
        if moderate:
            print(f"      Average: {statistics.mean(moderate):.2f}s")
        
        print(f"   Slow (≥ 5s): {len(slow)} requests ({len(slow)/len(response_times)*100:.1f}%)")
        if slow:
            print(f"      Average: {statistics.mean(slow):.2f}s")
        
        # Speed improvement calculation
        if blazing_fast and slow:
            improvement = statistics.mean(slow) / statistics.mean(blazing_fast)
            print(f"\n🚀 Speed improvement: {improvement:.1f}x faster for warm requests!")
        
        # Category distribution
        categories = {}
        for r in successful_requests:
            cat = r['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📂 Category Distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {count} items")
    
    if failed_requests:
        print(f"\n❌ Failed Requests:")
        for r in failed_requests:
            print(f"   Request {r['request_id']}: {r['error']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"performance_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_requests': len(results),
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'total_time': total_time
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_memory_trend():
    """Test memory usage trend during the test"""
    print("\n🧠 Memory Usage Trend")
    print("=" * 40)
    
    # Get initial memory
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            initial_memory = response.json().get('memory', {})
            print(f"Initial memory: {initial_memory.get('memory_percent', 'N/A')}% ({initial_memory.get('memory_used_mb', 'N/A'):.1f} MB)")
    except:
        print("Could not get initial memory info")
    
    # Get final memory after a few requests
    for i in range(5):
        try:
            payload = {"title": REAL_TITLES[i]}
            requests.post(f"{API_URL}/predict", json=payload)
        except:
            pass
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            final_memory = response.json().get('memory', {})
            print(f"Final memory: {final_memory.get('memory_percent', 'N/A')}% ({final_memory.get('memory_used_mb', 'N/A'):.1f} MB)")
    except:
        print("Could not get final memory info")

if __name__ == "__main__":
    print("🎯 50 Real Titles Performance Test")
    print("Testing with actual data from kashew_supervised_products.csv")
    print("=" * 70)
    
    try:
        test_50_real_titles()
        test_memory_trend()
        print("\n🎉 Comprehensive test completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 