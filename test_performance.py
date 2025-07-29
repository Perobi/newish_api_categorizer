#!/usr/bin/env python3
"""
Focused performance test to measure speed improvements
"""

import requests
import time
import statistics

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

def test_sequential_performance():
    """Test sequential requests to see performance improvement"""
    print("🚀 Testing Sequential Performance")
    print("=" * 50)
    
    test_titles = [
        "IKEA lamp",
        "Living Spaces Harper Beige Microfiber 4 Piece Sectional",
        "Scandinavian Designs Glass Top Console Table",
        "Hem Pro Glitch Throw (New)",
        "Christopher Guy Ornate Wall Mirror",
        "Furniture of America Mona Coffee Table",
        "Wood Media Cabinet with Ribbed Glass Paneled Cabinet Doors",
        "Room & Board Dublin Upholstered Lounge Chair"
    ]
    
    response_times = []
    
    for i, title in enumerate(test_titles, 1):
        print(f"\n🔍 Request {i}: {title[:40]}...")
        
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success in {response_time:.2f}s")
                print(f"   Category: {result.get('category')}")
                print(f"   Sub-Category: {result.get('sub_category')}")
                print(f"   Type: {result.get('type')}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Analyze results
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    if response_times:
        print(f"Total requests: {len(response_times)}")
        print(f"Average response time: {statistics.mean(response_times):.2f}s")
        print(f"Median response time: {statistics.median(response_times):.2f}s")
        print(f"Fastest response: {min(response_times):.2f}s")
        print(f"Slowest response: {max(response_times):.2f}s")
        
        # Identify fast vs slow requests
        fast_requests = [t for t in response_times if t < 5.0]
        slow_requests = [t for t in response_times if t >= 5.0]
        
        print(f"\n⚡ Fast requests (< 5s): {len(fast_requests)}")
        if fast_requests:
            print(f"   Average: {statistics.mean(fast_requests):.2f}s")
        
        print(f"🐌 Slow requests (≥ 5s): {len(slow_requests)}")
        if slow_requests:
            print(f"   Average: {statistics.mean(slow_requests):.2f}s")
        
        # Performance improvement estimate
        if fast_requests and slow_requests:
            improvement = statistics.mean(slow_requests) / statistics.mean(fast_requests)
            print(f"\n🚀 Estimated speed improvement: {improvement:.1f}x faster for warm requests!")

def test_memory_usage():
    """Test memory usage over multiple requests"""
    print("\n🧠 Testing Memory Usage")
    print("=" * 50)
    
    # Get initial memory
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            initial_memory = response.json().get('memory', {})
            print(f"Initial memory usage: {initial_memory.get('memory_percent', 'N/A')}%")
            print(f"Initial memory used: {initial_memory.get('memory_used_mb', 'N/A'):.1f} MB")
    except:
        print("Could not get initial memory info")
    
    # Make several requests
    test_titles = ["IKEA lamp", "Coffee table", "Sofa", "Mirror", "Chair"]
    
    for i, title in enumerate(test_titles, 1):
        try:
            payload = {"title": title}
            response = requests.post(f"{API_URL}/predict", json=payload)
            print(f"Request {i}: {title} - Status: {response.status_code}")
        except Exception as e:
            print(f"Request {i}: Error - {e}")
    
    # Get final memory
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            final_memory = response.json().get('memory', {})
            print(f"\nFinal memory usage: {final_memory.get('memory_percent', 'N/A')}%")
            print(f"Final memory used: {final_memory.get('memory_used_mb', 'N/A'):.1f} MB")
    except:
        print("Could not get final memory info")

if __name__ == "__main__":
    print("🎯 Performance Test - Measuring Speed Improvements")
    print("=" * 60)
    
    try:
        test_sequential_performance()
        test_memory_usage()
        print("\n🎉 Performance test completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 