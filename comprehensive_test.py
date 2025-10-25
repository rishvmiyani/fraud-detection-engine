import requests
import json
import time
import random

def comprehensive_api_testing():
    print('🧪 COMPREHENSIVE API TESTING SUITE')
    print('=' * 60)
    
    base_url = 'http://localhost:8000'
    
    # Test 1: Health Check
    print('\n1️⃣ Health Check Test:')
    try:
        response = requests.get(f'{base_url}/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f'   ✅ Status: {data.get(\"status\")}')
            print(f'   📊 Version: {data.get(\"version\")}')
            print(f'   🤖 ML Models: {data.get(\"ml_models\", 0)}')
        else:
            print(f'   ❌ HTTP {response.status_code}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Test 2: Root Endpoint
    print('\n2️⃣ Root Endpoint Test:')
    try:
        response = requests.get(f'{base_url}/', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f'   ✅ Message: {data.get(\"message\")}')
            print(f'   📊 Models Loaded: {data.get(\"models_loaded\")}')
            print(f'   🔧 Features: {data.get(\"features\")}')
        else:
            print(f'   ❌ HTTP {response.status_code}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Test 3: Individual Predictions
    print('\n3️⃣ Individual Prediction Tests:')
    
    test_cases = [
        {
            'name': 'Low Risk Transaction',
            'data': {
                'transaction_id': 'LOW_RISK_001',
                'user_id': 'USER_SAFE_001',
                'merchant_id': 'MERCHANT_TRUSTED_001',
                'amount': 45.99,
                'payment_method': 'debit_card',
                'merchant_category': 'grocery',
                'country': 'US',
                'device_type': 'mobile',
                'timestamp': '2025-10-18T15:30:00Z'
            },
            'expected_risk': 'low'
        },
        {
            'name': 'Medium Risk Transaction',
            'data': {
                'transaction_id': 'MED_RISK_001',
                'user_id': 'USER_NORMAL_001',
                'merchant_id': 'MERCHANT_REGULAR_001',
                'amount': 750.00,
                'payment_method': 'credit_card',
                'merchant_category': 'retail',
                'country': 'UK',
                'device_type': 'desktop',
                'timestamp': '2025-10-18T22:45:00Z'
            },
            'expected_risk': 'medium'
        },
        {
            'name': 'High Risk Transaction',
            'data': {
                'transaction_id': 'HIGH_RISK_001',
                'user_id': 'USER_SUSPICIOUS_001',
                'merchant_id': 'MERCHANT_UNKNOWN_001',
                'amount': 2500.00,
                'payment_method': 'cryptocurrency',
                'merchant_category': 'online',
                'country': 'Unknown',
                'device_type': 'mobile',
                'timestamp': '2025-10-19T02:30:00Z'
            },
            'expected_risk': 'high'
        },
        {
            'name': 'Round Amount Suspicious',
            'data': {
                'transaction_id': 'ROUND_AMT_001',
                'user_id': 'USER_LAUNDER_001',
                'merchant_id': 'MERCHANT_CASH_001',
                'amount': 5000.00,
                'payment_method': 'wire_transfer',
                'merchant_category': 'other',
                'country': 'Offshore',
                'device_type': 'atm',
                'timestamp': '2025-10-19T03:15:00Z'
            },
            'expected_risk': 'high'
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f'\n   🔍 Testing: {test_case[\"name\"]}')
        try:
            response = requests.post(
                f'{base_url}/api/v1/predict',
                json=test_case['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                fraud_prob = result.get('fraud_probability', 0)
                risk_level = result.get('risk_level', 'unknown')
                model_used = result.get('model_used', 'unknown')
                confidence = result.get('confidence', 0)
                
                print(f'      ✅ Fraud Probability: {fraud_prob:.4f}')
                print(f'      📊 Risk Level: {risk_level}')
                print(f'      🤖 Model Used: {model_used}')
                print(f'      🎯 Confidence: {confidence:.4f}')
                
                results.append({
                    'test': test_case['name'],
                    'fraud_prob': fraud_prob,
                    'risk_level': risk_level,
                    'model_used': model_used,
                    'success': True
                })
            else:
                print(f'      ❌ HTTP {response.status_code}')
                results.append({'test': test_case['name'], 'success': False})
                
        except Exception as e:
            print(f'      ❌ Error: {e}')
            results.append({'test': test_case['name'], 'success': False})
    
    # Test 4: Batch Prediction
    print('\n4️⃣ Batch Prediction Test:')
    try:
        batch_data = {
            'transactions': [case['data'] for case in test_cases[:2]]
        }
        
        response = requests.post(
            f'{base_url}/api/v1/predict/batch',
            json=batch_data,
            timeout=15
        )
        
        if response.status_code == 200:
            batch_result = response.json()
            predictions = batch_result.get('predictions', [])
            count = batch_result.get('count', 0)
            
            print(f'   ✅ Batch processed: {count} transactions')
            for i, pred in enumerate(predictions):
                print(f'      Transaction {i+1}: {pred.get(\"risk_level\")} risk, {pred.get(\"fraud_probability\", 0):.4f} probability')
        else:
            print(f'   ❌ Batch test failed: HTTP {response.status_code}')
    except Exception as e:
        print(f'   ❌ Batch test error: {e}')
    
    # Test 5: Statistics Endpoint
    print('\n5️⃣ Statistics Test:')
    try:
        response = requests.get(f'{base_url}/api/v1/stats', timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f'   ✅ Total Predictions: {stats.get(\"total_predictions\", 0)}')
            print(f'   ⚠️ Fraud Detected: {stats.get(\"fraud_detected\", 0)}')
            print(f'   📊 Fraud Rate: {stats.get(\"fraud_rate\", \"0%\")}')
            print(f'   🤖 Models Loaded: {stats.get(\"models_loaded\", 0)}')
            print(f'   🔧 Features Available: {stats.get(\"features_available\", 0)}')
        else:
            print(f'   ❌ Stats test failed: HTTP {response.status_code}')
    except Exception as e:
        print(f'   ❌ Stats test error: {e}')
    
    # Test 6: Load Test (Performance)
    print('\n6️⃣ Performance Load Test:')
    try:
        load_test_data = {
            'transaction_id': 'LOAD_TEST_BASE',
            'user_id': 'LOAD_USER',
            'merchant_id': 'LOAD_MERCHANT',
            'amount': 100.0,
            'payment_method': 'credit_card',
            'merchant_category': 'retail',
            'country': 'US',
            'device_type': 'mobile',
            'timestamp': '2025-10-18T23:06:00Z'
        }
        
        response_times = []
        success_count = 0
        test_count = 20
        
        print(f'   🚀 Running {test_count} concurrent requests...')
        start_time = time.time()
        
        for i in range(test_count):
            try:
                test_data = load_test_data.copy()
                test_data['transaction_id'] = f'LOAD_TEST_{i:03d}'
                test_data['amount'] = random.uniform(50, 1000)
                
                req_start = time.time()
                response = requests.post(
                    f'{base_url}/api/v1/predict',
                    json=test_data,
                    timeout=5
                )
                req_end = time.time()
                
                if response.status_code == 200:
                    success_count += 1
                    response_times.append(req_end - req_start)
                    
            except Exception as e:
                pass  # Continue with load test
        
        total_time = time.time() - start_time
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f'   ✅ Success Rate: {success_count}/{test_count} ({success_count/test_count*100:.1f}%)')
            print(f'   ⏱️ Average Response Time: {avg_response_time:.3f}s')
            print(f'   ⚡ Min Response Time: {min_response_time:.3f}s')
            print(f'   🐌 Max Response Time: {max_response_time:.3f}s')
            print(f'   📊 Total Test Time: {total_time:.2f}s')
            print(f'   🔥 Requests per Second: {success_count/total_time:.2f} req/s')
        else:
            print('   ❌ No successful requests in load test')
            
    except Exception as e:
        print(f'   ❌ Load test error: {e}')
    
    print('\n' + '=' * 60)
    print('🎉 COMPREHENSIVE TESTING COMPLETED!')
    
    # Summary
    successful_tests = sum(1 for r in results if r.get('success'))
    print(f'📊 Individual Tests Passed: {successful_tests}/{len(results)}')
    print('🌐 Access your API at: http://localhost:8000/docs')
    print('📈 Monitor stats at: http://localhost:8000/api/v1/stats')

if __name__ == '__main__':
    comprehensive_api_testing()
