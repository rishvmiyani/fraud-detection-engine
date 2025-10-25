#!/usr/bin/env python
"""
Fraud Detection Engine Setup Script
Complete environment setup and configuration
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import yaml
import json
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetectionSetup:
    """Complete setup and configuration for fraud detection engine"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "config" / "setup_config.yaml"
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load setup configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'python_version': '3.12',
            'environment_name': 'fraud-detection',
            'requirements_file': 'requirements.txt',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'fraud_detection',
                'user': 'fraud_user',
                'password': 'fraud_password'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'kafka': {
                'bootstrap_servers': 'localhost:9092',
                'topics': ['raw-transactions', 'fraud-alerts']
            },
            'model_store': {
                'backend': 'local',
                'path': './models'
            },
            'monitoring': {
                'prometheus_port': 9090,
                'grafana_port': 3000
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        required_version = self.config['python_version']
        
        if python_version != required_version:
            logger.warning(f"Python version mismatch: {python_version} vs {required_version}")
        
        # Check required commands
        required_commands = ['pip', 'git', 'docker', 'docker-compose']
        missing_commands = []
        
        for cmd in required_commands:
            if not self.command_exists(cmd):
                missing_commands.append(cmd)
        
        if missing_commands:
            logger.error(f"Missing required commands: {missing_commands}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def command_exists(self, command: str) -> bool:
        """Check if command exists in system PATH"""
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create and setup Python virtual environment"""
        env_name = self.config['environment_name']
        logger.info(f"🐍 Creating virtual environment: {env_name}")
        
        try:
            # Create conda environment
            subprocess.run([
                'conda', 'create', '-n', env_name, 
                f"python={self.config['python_version']}", '-y'
            ], check=True)
            
            logger.info(f"✅ Virtual environment '{env_name}' created")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / self.config['requirements_file']
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Install from requirements.txt
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                str(requirements_file)
            ], check=True)
            
            # Install development dependencies
            dev_packages = [
                'jupyter', 'jupyterlab', 'black', 'flake8', 'pytest',
                'pre-commit', 'nbconvert', 'ipywidgets'
            ]
            
            subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + dev_packages, check=True)
            
            logger.info("✅ Python dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_database(self) -> bool:
        """Setup PostgreSQL database"""
        logger.info("🐘 Setting up PostgreSQL database...")
        
        db_config = self.config['database']
        
        # Create database initialization script
        init_script = self.project_root / "scripts" / "database" / "init_database.py"
        
        try:
            subprocess.run([
                sys.executable, str(init_script),
                '--host', db_config['host'],
                '--port', str(db_config['port']),
                '--database', db_config['database'],
                '--user', db_config['user'],
                '--password', db_config['password']
            ], check=True)
            
            logger.info("✅ Database setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def setup_redis(self) -> bool:
        """Setup Redis cache"""
        logger.info("🔴 Setting up Redis cache...")
        
        redis_config = self.config['redis']
        
        try:
            import redis
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            r.ping()
            logger.info("✅ Redis connection verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis setup failed: {e}")
            return False
    
    def setup_kafka(self) -> bool:
        """Setup Kafka message broker"""
        logger.info("📨 Setting up Kafka...")
        
        kafka_config = self.config['kafka']
        
        # Create Kafka topics
        topics_script = self.project_root / "scripts" / "data_processing" / "create_kafka_topics.py"
        
        try:
            subprocess.run([
                sys.executable, str(topics_script),
                '--bootstrap-servers', kafka_config['bootstrap_servers'],
                '--topics'
            ] + kafka_config['topics'], check=True)
            
            logger.info("✅ Kafka setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Kafka setup failed: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring stack"""
        logger.info("📊 Setting up monitoring...")
        
        monitoring_script = self.project_root / "monitoring" / "scripts" / "setup_monitoring.sh"
        
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['bash', str(monitoring_script)], check=True)
            else:
                subprocess.run(['bash', str(monitoring_script)], check=True)
            
            logger.info("✅ Monitoring setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        logger.info("📁 Creating project directories...")
        
        directories = [
            'data/raw', 'data/processed', 'data/external',
            'models/trained', 'models/artifacts', 'models/experiments',
            'logs', 'outputs', 'reports', 'notebooks/outputs',
            'config/environments', 'tests/unit', 'tests/integration'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep_file = dir_path / '.gitkeep'
            if not any(dir_path.iterdir()):
                gitkeep_file.touch()
        
        logger.info("✅ Project directories created")
        return True
    
    def setup_git_hooks(self) -> bool:
        """Setup Git pre-commit hooks"""
        logger.info("🔧 Setting up Git hooks...")
        
        try:
            # Initialize pre-commit
            subprocess.run([sys.executable, '-m', 'pre_commit', 'install'], 
                         check=True, cwd=self.project_root)
            
            logger.info("✅ Git hooks setup completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git hooks setup failed: {e}")
            return False
    
    def create_environment_files(self) -> bool:
        """Create environment configuration files"""
        logger.info("⚙️ Creating environment files...")
        
        # Create .env file
        env_content = f"""
# Fraud Detection Engine Configuration
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://{self.config['database']['user']}:{self.config['database']['password']}@{self.config['database']['host']}:{self.config['database']['port']}/{self.config['database']['database']}

# Redis Configuration
REDIS_URL=redis://{self.config['redis']['host']}:{self.config['redis']['port']}/{self.config['redis']['db']}

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS={self.config['kafka']['bootstrap_servers']}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production

# Model Configuration
MODEL_STORE_BACKEND={self.config['model_store']['backend']}
MODEL_STORE_PATH={self.config['model_store']['path']}

# Monitoring
PROMETHEUS_PORT={self.config['monitoring']['prometheus_port']}
GRAFANA_PORT={self.config['monitoring']['grafana_port']}
METRICS_ENABLED=true
LOG_LEVEL=INFO
"""
        
        env_file = self.project_root / '.env'
        with open(env_file, 'w') as f:
            f.write(env_content.strip())
        
        # Create environment-specific configs
        for env in ['development', 'staging', 'production']:
            config_dir = self.project_root / 'config' / 'environments'
            config_file = config_dir / f'{env}.yaml'
            
            env_config = self.config.copy()
            env_config['environment'] = env
            env_config['debug'] = env == 'development'
            
            with open(config_file, 'w') as f:
                yaml.dump(env_config, f, default_flow_style=False)
        
        logger.info("✅ Environment files created")
        return True
    
    def validate_setup(self) -> bool:
        """Validate the complete setup"""
        logger.info("✅ Validating setup...")
        
        validation_script = self.project_root / "scripts" / "testing" / "validate_setup.py"
        
        try:
            subprocess.run([sys.executable, str(validation_script)], check=True)
            logger.info("✅ Setup validation passed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Setup validation failed: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 Starting fraud detection engine setup...")
        
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Directories", self.create_directories),
            ("Dependencies", self.install_dependencies),
            ("Database", self.setup_database),
            ("Redis", self.setup_redis),
            ("Kafka", self.setup_kafka),
            ("Environment Files", self.create_environment_files),
            ("Git Hooks", self.setup_git_hooks),
            ("Monitoring", self.setup_monitoring),
            ("Validation", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Setup failed at step: {step_name}")
                return False
        
        logger.info("🎉 Fraud detection engine setup completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("   1. Activate the virtual environment:")
        logger.info(f"      conda activate {self.config['environment_name']}")
        logger.info("   2. Start the services:")
        logger.info("      docker-compose -f infrastructure/docker/docker-compose.yml up -d")
        logger.info("   3. Run the API server:")
        logger.info("      python -m uvicorn src.main:app --reload")
        logger.info("   4. Access the documentation:")
        logger.info("      http://localhost:8000/docs")
        
        return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Fraud Detection Engine Setup")
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--step', type=str, help="Run specific setup step")
    parser.add_argument('--validate-only', action='store_true', help="Only run validation")
    
    args = parser.parse_args()
    
    setup = FraudDetectionSetup(args.config)
    
    if args.validate_only:
        success = setup.validate_setup()
    elif args.step:
        step_methods = {
            'prerequisites': setup.check_prerequisites,
            'environment': setup.create_virtual_environment,
            'dependencies': setup.install_dependencies,
            'database': setup.setup_database,
            'redis': setup.setup_redis,
            'kafka': setup.setup_kafka,
            'directories': setup.create_directories,
            'git-hooks': setup.setup_git_hooks,
            'monitoring': setup.setup_monitoring,
            'env-files': setup.create_environment_files,
            'validate': setup.validate_setup
        }
        
        if args.step in step_methods:
            success = step_methods[args.step]()
        else:
            logger.error(f"Unknown step: {args.step}")
            logger.info(f"Available steps: {list(step_methods.keys())}")
            success = False
    else:
        success = setup.run_full_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
