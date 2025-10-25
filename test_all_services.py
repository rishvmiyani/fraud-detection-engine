import requests
import time
import json

def test_all_services():
    print('🔍 COMPREHENSIVE SERVICE TESTING')
    print('=' * 50)
    
    services = [
        {
            'name': 'Fraud Detection API',
            'url': 'http://localhost:8000/health',
            'expected': 'healthy'
        },
        {
            'name': 'Grafana',
            'url': 'http://localhost:3000/api/health',
            'expected': 'ok'
        },
        {
            'name': 'Prometheus',
            'url': 'http://localhost:9090/-/ready',
            'expected': 'ready'
        },
        {
            'name': 'MLflow',
            'url': 'http://localhost:5000/health',
            'expected': None  # MLflow doesn't have standard health endpoint
        }
    ]
    
    results = []
    
    for service in services:
        print(f'\n🔍 Testing {service["name"]}...')
        try:
            response = requests.get(service['url'], timeout=10)
            if response.status_code == 200:
                print(f'   ✅ Status: ONLINE (HTTP {response.status_code})')
                
                # Try to parse JSON response for more info
                try:
                    data = response.json()
                    if service['name'] == 'Fraud Detection API':
                        print(f'   📊 Version: {data.get("version", "unknown")}')
                        print(f'   📊 Status: {data.get("status", "unknown")}')
                except:
                    print(f'   📄 Response: {response.text[:100]}...')
                
                results.append({'service': service['name'], 'status': 'online'})
            else:
                print(f'   ⚠️ Status: ONLINE but HTTP {response.status_code}')
                results.append({'service': service['name'], 'status': 'warning'})
                
        except requests.exceptions.ConnectionError:
            print(f'   ❌ Status: OFFLINE (Connection refused)')
            results.append({'service': service['name'], 'status': 'offline'})
        except requests.exceptions.Timeout:
            print(f'   ❌ Status: TIMEOUT')
            results.append({'service': service['name'], 'status': 'timeout'})
        except Exception as e:
            print(f'   ❌ Status: ERROR ({str(e)})')
            results.append({'service': service['name'], 'status': 'error'})
    
    # Summary
    print('\n' + '=' * 50)
    print('📊 SERVICE STATUS SUMMARY:')
    print('=' * 50)
    
    online_count = sum(1 for r in results if r['status'] == 'online')
    total_count = len(results)
    
    for result in results:
        status_icon = {
            'online': '✅',
            'warning': '⚠️',
            'offline': '❌',
            'timeout': '⏱️',
            'error': '❌'
        }.get(result['status'], '❓')
        
        print(f'   {status_icon} {result["service"]}: {result["status"].upper()}')
    
    print(f'\n📈 Overall Status: {online_count}/{total_count} services online')
    
    if online_count == total_count:
        print('🎉 ALL SERVICES ARE RUNNING SUCCESSFULLY!')
    elif online_count >= total_count * 0.75:
        print('⚠️ Most services are running, check offline services')
    else:
        print('❌ Multiple services are offline, check configuration')
    
    # Quick access guide
    print('\n🌐 QUICK ACCESS GUIDE:')
    print('=' * 30)
    print('📖 API Docs: http://localhost:8000/docs')
    print('📊 Grafana: http://localhost:3000 (admin/admin123)')
    print('🔥 Prometheus: http://localhost:9090')
    print('🤖 MLflow: http://localhost:5000')

if __name__ == '__main__':
    test_all_services()
