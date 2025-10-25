import requests
import time
import json
from datetime import datetime
import threading
import os
import random


class FraudDetectionMonitor:
    def __init__(self, api_url='http://localhost:8000'):
        self.api_url = api_url
        self.running = True
        self.stats_history = []
        
    def get_current_stats(self):
        """Get current API statistics"""
        try:
            response = requests.get(f'{self.api_url}/api/v1/stats', timeout=5)
            if response.status_code == 200:
                stats = response.json()
                stats['timestamp'] = datetime.now().isoformat()
                return stats
        except Exception as e:
            print(f"Stats error: {e}")
        return None
    
    def get_health_status(self):
        """Get API health status"""
        try:
            response = requests.get(f'{self.api_url}/health', timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Health check error: {e}")
        return None
    
    def simulate_transactions(self):
        """Simulate transactions for demo"""
        print("🤖 Transaction simulation started...")
        
        while self.running:
            try:
                # Generate random transaction
                transaction = {
                    'transaction_id': f'DEMO_{int(time.time())}_{random.randint(1000,9999)}',
                    'user_id': f'USER_{random.randint(1,1000):04d}',
                    'merchant_id': f'MERCHANT_{random.randint(1,100):03d}',
                    'amount': round(random.lognormal(5, 1.5), 2),
                    'payment_method': random.choice(['credit_card', 'debit_card', 'paypal']),
                    'country': random.choice(['US', 'UK', 'CA', 'IN']),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send transaction to API
                response = requests.post(
                    f'{self.api_url}/api/v1/predict',
                    json=transaction,
                    timeout=3
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Transaction {transaction['transaction_id']}: {result.get('decision', 'unknown')}")
                
                # Random delay between transactions (2-8 seconds)
                time.sleep(random.uniform(2, 8))
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {e}")
                time.sleep(5)  # Wait if API not available
            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(5)
    
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        while self.running:
            try:
                # Clear screen
                self.clear_screen()
                
                print('🚨 FRAUD DETECTION ENGINE - LIVE MONITORING DASHBOARD')
                print('=' * 80)
                print(f'📅 Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                print('=' * 80)
                
                # Get current stats
                stats = self.get_current_stats()
                health = self.get_health_status()
                
                # Display API Status
                if health:
                    status = health.get('status', 'unknown')
                    status_icon = '🟢' if status == 'healthy' else '🔴'
                    print(f'{status_icon} API STATUS: {status.upper()}')
                    print(f'📊 Version: {health.get("version", "unknown")}')
                    
                    # Display service health
                    services = health.get('services', {})
                    if services:
                        print(f'🗄️  Database: {services.get("database", "unknown")}')
                        print(f'📦 Redis: {services.get("redis", "unknown")}')
                else:
                    print('🔴 API STATUS: OFFLINE')
                
                print('')
                
                # Display Statistics
                if stats:
                    print('📈 REAL-TIME STATISTICS:')
                    print('-' * 40)
                    print(f'📊 Total Transactions: {stats.get("total_transactions", 0):,}')
                    print(f'⚠️  Fraud Transactions: {stats.get("fraud_transactions", 0):,}')
                    print(f'📊 Fraud Rate: {stats.get("fraud_rate_percent", 0):.2f}%')
                    print(f'💰 Avg Amount: ${stats.get("average_amount", 0):.2f}')
                    
                    # Add to history
                    self.stats_history.append(stats)
                    if len(self.stats_history) > 20:
                        self.stats_history.pop(0)
                    
                    # Show trend
                    if len(self.stats_history) >= 2:
                        prev_stats = self.stats_history[-2]
                        current_total = stats.get('total_transactions', 0)
                        prev_total = prev_stats.get('total_transactions', 0)
                        
                        if current_total > prev_total:
                            trend = f'📈 +{current_total - prev_total} new transactions'
                        else:
                            trend = '📊 No new activity'
                        
                        print(f'📊 Activity: {trend}')
                
                else:
                    print('❌ Unable to fetch statistics - API may be offline')
                
                print('')
                print('🌐 QUICK ACCESS LINKS:')
                print('-' * 40)
                print(f'📖 API Documentation: {self.api_url}/docs')
                print(f'🏥 Health Check: {self.api_url}/health')
                print(f'📊 Stats Endpoint: {self.api_url}/api/v1/stats')
                
                print('')
                print('⚡ PERFORMANCE METRICS:')
                print('-' * 40)
                
                # Performance test
                try:
                    start_time = time.time()
                    response = requests.get(f'{self.api_url}/health', timeout=2)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        print(f'⚡ API Response Time: {response_time:.1f}ms')
                        if response_time < 100:
                            print('🟢 Performance: Excellent')
                        elif response_time < 500:
                            print('🟡 Performance: Good')
                        else:
                            print('🔴 Performance: Slow')
                    else:
                        print('❌ API Performance: Error')
                except requests.exceptions.Timeout:
                    print('❌ API Performance: Timeout')
                except Exception as e:
                    print(f'❌ Performance check failed: {e}')
                
                print('')
                print('🔄 Auto-refreshing every 5 seconds... Press Ctrl+C to stop')
                print('🤖 Demo transactions are being generated automatically')
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print('\n👋 Dashboard stopped by user')
                self.running = False
                break
            except Exception as e:
                print(f'Dashboard error: {e}')
                time.sleep(5)
    
    def start_monitoring(self):
        """Start the monitoring dashboard"""
        print('🚀 Starting Fraud Detection Monitoring Dashboard...')
        print('🤖 Starting transaction simulation...')
        
        # Start transaction simulation in background
        simulation_thread = threading.Thread(target=self.simulate_transactions, daemon=True)
        simulation_thread.start()
        
        # Start dashboard display
        try:
            self.display_dashboard()
        except KeyboardInterrupt:
            print('\n👋 Monitoring stopped')
        finally:
            self.running = False


def main():
    """Main function to run the monitor"""
    print("🔍 Fraud Detection Monitor Starting...")
    
    # Check if API is available
    monitor = FraudDetectionMonitor()
    
    health = monitor.get_health_status()
    if not health:
        print("❌ API server is not running!")
        print("📝 Please start the API server first:")
        print("   python quick_server.py")
        return
    
    print("✅ API server detected - starting monitor...")
    monitor.start_monitoring()


if __name__ == '__main__':
    main()
