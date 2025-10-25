import requests
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def test_complete_system():
    print('🚀 COMPLETE FRAUD DETECTION SYSTEM TEST')
    print('=' * 60)
    
    # 1. Test API Backend
    print('\\n1️⃣ Testing Backend API...')
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f'   ✅ API Status: {data.get("status")}')
            print(f'   📊 Version: {data.get("version")}')
            print(f'   🤖 ML Models: {data.get("ml_models", 0)}')
        else:
            print(f'   ❌ API Error: HTTP {response.status_code}')
            return False
    except Exception as e:
        print(f'   ❌ API Connection failed: {e}')
        return False
    
    # 2. Test Monitoring Services
    print('\\n2️⃣ Testing Monitoring Services...')
    
    # Grafana
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        print(f'   ✅ Grafana: Available (HTTP {response.status_code})')
    except:
        print('   ⚠️ Grafana: Not available')
    
    # Prometheus
    try:
        response = requests.get('http://localhost:9090', timeout=5)
        print(f'   ✅ Prometheus: Available (HTTP {response.status_code})')
    except:
        print('   ⚠️ Prometheus: Not available')
    
    # MLflow
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        print(f'   ✅ MLflow: Available (HTTP {response.status_code})')
    except:
        print('   ⚠️ MLflow: Not available')
    
    # 3. Test Prediction Endpoint
    print('\\n3️⃣ Testing Fraud Prediction...')
    test_transaction = {
        'transaction_id': 'SYSTEM_TEST_001',
        'user_id': 'TEST_USER_001',
        'merchant_id': 'TEST_MERCHANT_001',
        'amount': 999.99,
        'payment_method': 'credit_card',
        'merchant_category': 'retail',
        'country': 'US',
        'device_type': 'desktop',
        'timestamp': '2025-10-19T11:21:00Z'
    }
    
    try:
        response = requests.post(
            'http://localhost:8000/api/v1/predict',
            json=test_transaction,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f'   ✅ Prediction successful!')
            print(f'   📊 Risk Level: {result.get("risk_level")}')
            print(f'   🎯 Fraud Probability: {result.get("fraud_probability", 0):.4f}')
            print(f'   🤖 Model Used: {result.get("model_used")}')
        else:
            print(f'   ❌ Prediction failed: HTTP {response.status_code}')
    except Exception as e:
        print(f'   ❌ Prediction error: {e}')
    
    # 4. Check Frontend Files
    print('\\n4️⃣ Testing Frontend Files...')
    frontend_files = [
        'frontend/templates/dashboard.html',
        'frontend/static/css/dashboard.css',
        'frontend/static/js/dashboard.js',
        'frontend_server.py'
    ]
    
    for file_path in frontend_files:
        if Path(file_path).exists():
            print(f'   ✅ {file_path}')
        else:
            print(f'   ❌ {file_path} missing')
    
    return True

def launch_complete_system():
    print('\\n🎯 LAUNCHING COMPLETE FRAUD DETECTION SYSTEM')
    print('=' * 60)
    
    # Check if system is ready
    if not test_complete_system():
        print('❌ System not ready. Please fix issues first.')
        return
    
    print('\\n🚀 STARTING FRONTEND SERVER...')
    
    # Start frontend server in separate process
    try:
        frontend_process = subprocess.Popen([
            sys.executable, 'frontend_server.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print('⏳ Waiting for frontend server to start...')
        time.sleep(5)
        
        # Test frontend
        try:
            response = requests.get('http://localhost:3001', timeout=5)
            if response.status_code == 200:
                print('✅ Frontend server started successfully!')
            else:
                print(f'⚠️ Frontend server issue: HTTP {response.status_code}')
        except:
            print('⚠️ Frontend server not responding yet')
        
        print('\\n🌐 SYSTEM READY! Access Points:')
        print('=' * 40)
        print('🖥️ Frontend Dashboard: http://localhost:3001')
        print('📖 API Documentation: http://localhost:8000/docs')
        print('🏥 API Health: http://localhost:8000/health')
        print('📊 Grafana: http://localhost:3000 (admin/admin123)')
        print('🔥 Prometheus: http://localhost:9090')
        print('🤖 MLflow: http://localhost:5000')
        
        print('\\n🎉 OPENING FRONTEND DASHBOARD IN BROWSER...')
        time.sleep(2)
        
        # Open browser
        try:
            webbrowser.open('http://localhost:3001')
            print('✅ Dashboard opened in browser!')
        except:
            print('⚠️ Could not open browser automatically')
        
        print('\\n✨ YOUR COMPLETE FRAUD DETECTION SYSTEM IS NOW RUNNING! ✨')
        print('Press Ctrl+C to stop all services')
        
        # Keep frontend running
        try:
            frontend_process.wait()
        except KeyboardInterrupt:
            print('\\n👋 Shutting down frontend server...')
            frontend_process.terminate()
            
    except Exception as e:
        print(f'❌ Failed to start frontend: {e}')

if __name__ == '__main__':
    launch_complete_system()
