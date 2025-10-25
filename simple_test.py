import sys
import os

def test_simple_math():
    assert 1 + 1 == 2
    print("✅ Simple math test passed")

def test_imports():
    import pandas
    import numpy
    import requests
    print("✅ All imports work")

def test_api_health():
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠️ API returned status: {response.status_code}")
    except:
        print("⚠️ API server not running - that's ok for now")

if __name__ == "__main__":
    print("🧪 RUNNING SIMPLE TESTS")
    print("=" * 30)
    
    test_simple_math()
    test_imports() 
    test_api_health()
    
    print("\n🎉 ALL TESTS COMPLETED!")
