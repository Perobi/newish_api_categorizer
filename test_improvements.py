#!/usr/bin/env python3
"""
Test script to verify the improvements work correctly
"""

import requests
import time
import json

# API endpoint
API_URL = "https://categorizer-api-2d06dff61638.herokuapp.com"

def test_health_with_memory():
    """Test the improved health endpoint with memory info"""
    print("🔍 Testing improved health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {data.get('status')}")
            print(f"   Model: {data.get('model')}")
            if 'memory' in data:
                memory = data['memory']
                print(f"   Memory Usage: {memory.get('memory_percent', 'N/A')}%")
                print(f"   Memory Used: {memory.get('memory_used_mb', 'N/A'):.1f} MB")
                print(f"   Memory Available: {memory.get('memory_available_mb', 'N/A'):.1f} MB")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_concurrent_requests():
    """Test multiple concurrent requests to see if models reload properly"""
    print("\n🚀 Testing concurrent requests...")
    
    test_titles = [
        "IKEA lamp",
        "Living Spaces Harper Beige Microfiber 4 Piece Sectional",
        "Scandinavian Designs Glass Top Console Table",
        "Hem Pro Glitch Throw (New)",
        "Christopher Guy Ornate Wall Mirror"
    ]
    
    import threading
    import queue
    
    results = queue.Queue()
    
    def make_request(title, request_id):
        try:
            payload = {"title": title}
            start_time = time.time()
            response = requests.post(f"{API_URL}/predict", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                results.put({
                    'request_id': request_id,
                    'title': title,
                    'success': True,
                    'response_time': end_time - start_time,
                    'category': result.get('category'),
                    'input_title': result.get('input_title')
                })
            else:
                results.put({
                    'request_id': request_id,
                    'title': title,
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
        except Exception as e:
            results.put({
                'request_id': request_id,
                'title': title,
                'success': False,
                'error': str(e)
            })
    
    # Start concurrent requests
    threads = []
    for i, title in enumerate(test_titles):
        thread = threading.Thread(target=make_request, args=(title, i))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    all_results = []
    while not results.empty():
        all_results.append(results.get())
    
    # Sort by request ID
    all_results.sort(key=lambda x: x['request_id'])
    
    # Print results
    print(f"📊 Concurrent request results ({len(all_results)} requests):")
    for result in all_results:
        if result['success']:
            print(f"   ✅ Request {result['request_id']}: {result['title'][:30]}...")
            print(f"      Response time: {result['response_time']:.2f}s")
            print(f"      Category: {result['category']}")
        else:
            print(f"   ❌ Request {result['request_id']}: {result['error']}")
    
    # Check if any requests failed
    failed_requests = [r for r in all_results if not r['success']]
    if failed_requests:
        print(f"⚠️ {len(failed_requests)} requests failed")
    else:
        print("🎉 All concurrent requests succeeded!")

def test_error_handling():
    """Test improved error handling"""
    print("\n🧪 Testing error handling...")
    
    error_cases = [
        {"payload": None, "description": "No JSON content"},
        {"payload": {}, "description": "Empty JSON"},
        {"payload": {"title": ""}, "description": "Empty title"},
        {"payload": {"title": "   "}, "description": "Whitespace title"},
        {"payload": {"wrong_field": "test"}, "description": "Wrong field name"},
    ]
    
    for case in error_cases:
        print(f"\n🔍 Testing: {case['description']}")
        try:
            response = requests.post(f"{API_URL}/predict", json=case['payload'])
            print(f"   Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   Response: {response.text}")
            else:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    print("🎯 Testing API Improvements")
    print("=" * 50)
    
    try:
        # Test health endpoint with memory info
        test_health_with_memory()
        
        # Test concurrent requests
        test_concurrent_requests()
        
        # Test error handling
        test_error_handling()
        
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 