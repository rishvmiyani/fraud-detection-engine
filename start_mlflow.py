import subprocess
import sys
import os
import time

def start_mlflow_local():
    print('🤖 Starting MLflow as local process...')
    
    # Install MLflow if not installed
    try:
        import mlflow
        print('✅ MLflow already installed')
    except ImportError:
        print('📦 Installing MLflow...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mlflow'])
    
    # Create mlruns directory
    os.makedirs('mlruns', exist_ok=True)
    
    # Start MLflow server
    print('🚀 Starting MLflow server on http://localhost:5000')
    
    # Run MLflow server
    cmd = [
        sys.executable, '-m', 'mlflow', 'server',
        '--host', '0.0.0.0',
        '--port', '5000',
        '--backend-store-uri', 'sqlite:///mlflow.db',
        '--default-artifact-root', './mlruns'
    ]
    
    try:
        subprocess.Popen(cmd, cwd=os.getcwd())
        print('✅ MLflow server started successfully!')
        print('🌐 Access MLflow at: http://localhost:5000')
        return True
    except Exception as e:
        print(f'❌ Failed to start MLflow: {e}')
        return False

if __name__ == '__main__':
    start_mlflow_local()
