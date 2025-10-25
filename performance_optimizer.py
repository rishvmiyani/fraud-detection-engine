import time
import requests
import json
import concurrent.futures
import statistics
from datetime import datetime


class PerformanceOptimizer:
    def __init__(self, api_url='http://localhost:8000'):
        self.api_url = api_url
        
    def benchmark_single_prediction(self, iterations=100):
        """Benchmark single prediction performance"""
        print(f'🔥 Benchmarking Single Predictions ({iterations} iterations)...')
        
        test_transaction = {
            'transaction_id': 'BENCHMARK_001',
            'user_id': 'BENCH_USER',
            'merchant_id': 'BENCH_MERCHANT',
            'amount': 299.99,
            'payment_method': 'credit_card',
            'country': 'US',
            'timestamp': datetime.now().isoformat()
        }
        
        response_times = []
        successful_requests = 0
        
        for i in range(iterations):
            test_transaction['transaction_id'] = f'BENCHMARK_{i:04d}'
            
            try:
                start_time = time.time()
                response = requests.post(
                    f'{self.api_url}/api/v1/predict',
                    json=test_transaction,
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                    successful_requests += 1
                    
            except Exception as e:
                print(f'Request {i} failed: {e}')
        
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            p95_time = sorted(response_times)[int(0.95 * len(response_times))]
            
            print(f'📊 SINGLE PREDICTION BENCHMARK RESULTS:')
            print(f'   ✅ Success Rate: {successful_requests}/{iterations} ({successful_requests/iterations*100:.1f}%)')
            print(f'   ⚡ Average Response Time: {avg_time*1000:.1f}ms')
            print(f'   📊 Median Response Time: {median_time*1000:.1f}ms')
            print(f'   🚀 Fastest Response: {min_time*1000:.1f}ms')
            print(f'   🐌 Slowest Response: {max_time*1000:.1f}ms')
            print(f'   📈 95th Percentile: {p95_time*1000:.1f}ms')
            print(f'   🔥 Throughput: {successful_requests/sum(response_times):.1f} predictions/sec')
        
        return response_times
    
    def benchmark_concurrent_predictions(self, concurrent_users=10, requests_per_user=10):
        """Benchmark concurrent prediction performance"""
        print(f'🚀 Benchmarking Concurrent Predictions ({concurrent_users} users, {requests_per_user} requests each)...')
        
        def make_predictions(user_id):
            response_times = []
            successful_requests = 0
            
            for i in range(requests_per_user):
                test_transaction = {
                    'transaction_id': f'CONCURRENT_U{user_id:02d}_R{i:03d}',
                    'user_id': f'USER_{user_id:04d}',
                    'merchant_id': f'MERCHANT_{i%50:03d}',
                    'amount': 100.0 + (i * 50),
                    'payment_method': ['credit_card', 'debit_card', 'paypal'][i % 3],
                    'country': 'US',
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f'{self.api_url}/api/v1/predict',
                        json=test_transaction,
                        timeout=15
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append(end_time - start_time)
                        successful_requests += 1
                        
                except Exception as e:
                    pass  # Continue with other requests
            
            return response_times, successful_requests
        
        # Run concurrent tests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_predictions, user_id) for user_id in range(concurrent_users)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Aggregate results
        all_response_times = []
        total_successful_requests = 0
        
        for response_times, successful_requests in results:
            all_response_times.extend(response_times)
            total_successful_requests += successful_requests
        
        if all_response_times:
            avg_time = statistics.mean(all_response_times)
            median_time = statistics.median(all_response_times)
            p95_time = sorted(all_response_times)[int(0.95 * len(all_response_times))]
            
            total_requests = concurrent_users * requests_per_user
            
            print(f'📊 CONCURRENT PREDICTION BENCHMARK RESULTS:')
            print(f'   ✅ Success Rate: {total_successful_requests}/{total_requests} ({total_successful_requests/total_requests*100:.1f}%)')
            print(f'   ⚡ Average Response Time: {avg_time*1000:.1f}ms')
            print(f'   📊 Median Response Time: {median_time*1000:.1f}ms')
            print(f'   📈 95th Percentile: {p95_time*1000:.1f}ms')
            print(f'   ⏱️  Total Test Time: {total_time:.2f}s')
            print(f'   🔥 Overall Throughput: {total_successful_requests/total_time:.1f} predictions/sec')
    
    def benchmark_health_endpoint(self, iterations=50):
        """Benchmark health endpoint performance"""
        print(f'🏥 Benchmarking Health Endpoint ({iterations} iterations)...')
        
        response_times = []
        successful_requests = 0
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = requests.get(f'{self.api_url}/health', timeout=5)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                    successful_requests += 1
                    
            except Exception as e:
                print(f'Health check {i} failed: {e}')
        
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f'📊 HEALTH ENDPOINT BENCHMARK RESULTS:')
            print(f'   ✅ Success Rate: {successful_requests}/{iterations} ({successful_requests/iterations*100:.1f}%)')
            print(f'   ⚡ Average Response Time: {avg_time*1000:.1f}ms')
            print(f'   🚀 Fastest Response: {min_time*1000:.1f}ms')
            print(f'   🐌 Slowest Response: {max_time*1000:.1f}ms')
    
    def stress_test_api(self, duration_seconds=30, concurrent_users=5):
        """Run stress test for specified duration"""
        print(f'💪 Running Stress Test ({duration_seconds}s with {concurrent_users} concurrent users)...')
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        total_successful = 0
        total_errors = 0
        response_times = []
        
        def stress_worker():
            nonlocal total_requests, total_successful, total_errors
            
            while time.time() < end_time:
                test_transaction = {
                    'transaction_id': f'STRESS_{int(time.time() * 1000000)}',
                    'user_id': f'STRESS_USER_{total_requests % 1000}',
                    'merchant_id': f'STRESS_MERCHANT_{total_requests % 100}',
                    'amount': 50.0 + (total_requests % 500),
                    'payment_method': ['credit_card', 'debit_card', 'paypal'][total_requests % 3],
                    'country': 'US',
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    request_start = time.time()
                    response = requests.post(
                        f'{self.api_url}/api/v1/predict',
                        json=test_transaction,
                        timeout=10
                    )
                    request_end = time.time()
                    
                    total_requests += 1
                    
                    if response.status_code == 200:
                        total_successful += 1
                        response_times.append(request_end - request_start)
                    else:
                        total_errors += 1
                        
                except Exception as e:
                    total_requests += 1
                    total_errors += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
        
        # Run stress test with multiple workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(stress_worker) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        actual_duration = time.time() - start_time
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
            
            print(f'📊 STRESS TEST RESULTS:')
            print(f'   ⏱️  Duration: {actual_duration:.1f}s')
            print(f'   📊 Total Requests: {total_requests}')
            print(f'   ✅ Successful: {total_successful}')
            print(f'   ❌ Errors: {total_errors}')
            print(f'   📈 Success Rate: {total_successful/total_requests*100:.1f}%')
            print(f'   ⚡ Avg Response Time: {avg_response_time*1000:.1f}ms')
            print(f'   📈 95th Percentile: {p95_response_time*1000:.1f}ms')
            print(f'   🔥 Requests/Second: {total_requests/actual_duration:.1f}')
            print(f'   🎯 Successful/Second: {total_successful/actual_duration:.1f}')
    
    def run_comprehensive_benchmark(self):
        """Run all performance benchmarks"""
        print('🏁 COMPREHENSIVE PERFORMANCE BENCHMARK')
        print('=' * 60)
        print(f'🎯 Target API: {self.api_url}')
        print(f'⏰ Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print()
        
        # Check API availability first
        try:
            response = requests.get(f'{self.api_url}/health', timeout=5)
            if response.status_code != 200:
                print('❌ API not available for benchmarking')
                return
        except Exception as e:
            print(f'❌ Cannot connect to API for benchmarking: {e}')
            return
        
        print('✅ API is available, starting benchmarks...')
        print()
        
        try:
            # Run benchmarks
            self.benchmark_health_endpoint(25)
            print()
            
            self.benchmark_single_prediction(50)
            print()
            
            self.benchmark_concurrent_predictions(3, 5)
            print()
            
            self.stress_test_api(15, 3)
            print()
            
            print('🎉 COMPREHENSIVE BENCHMARKING COMPLETED!')
            print('=' * 60)
            
        except KeyboardInterrupt:
            print('\n⚠️  Benchmarking interrupted by user')
        except Exception as e:
            print(f'\n❌ Benchmarking error: {e}')


def main():
    """Main function to run benchmarks"""
    print("🚀 Performance Optimizer Starting...")
    
    optimizer = PerformanceOptimizer()
    
    # Check if API is running
    try:
        response = requests.get(f'{optimizer.api_url}/health', timeout=5)
        if response.status_code == 200:
            print("✅ API server detected - starting benchmarks...")
            optimizer.run_comprehensive_benchmark()
        else:
            print("❌ API server not responding properly")
    except Exception as e:
        print("❌ API server is not running!")
        print("📝 Please start the API server first:")
        print("   python quick_server.py")

if __name__ == '__main__':
    optimizer = PerformanceOptimizer()
    optimizer.run_comprehensive_benchmark()
