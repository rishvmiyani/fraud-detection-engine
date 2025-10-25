#!/usr/bin/env python
"""
Database Initialization Script
Create and configure PostgreSQL database for fraud detection
"""

import psycopg2
import psycopg2.extras
import argparse
import logging
import sys
from pathlib import Path
import yaml
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Initialize and configure PostgreSQL database"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            # First connect to default database to create our database
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database='postgres',  # Connect to default database first
                user=self.config['user'],
                password=self.config['password']
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            logger.info(f"✅ Connected to PostgreSQL at {self.config['host']}:{self.config['port']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def create_database(self) -> bool:
        """Create fraud detection database if it doesn't exist"""
        try:
            # Check if database exists
            self.cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config['database'],)
            )
            
            if self.cursor.fetchone():
                logger.info(f"Database '{self.config['database']}' already exists")
                return True
            
            # Create database
            self.cursor.execute(f"CREATE DATABASE {self.config['database']}")
            logger.info(f"✅ Created database '{self.config['database']}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create database: {e}")
            return False
    
    def connect_to_app_database(self) -> bool:
        """Connect to the application database"""
        try:
            if self.connection:
                self.connection.close()
            
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            logger.info(f"✅ Connected to application database '{self.config['database']}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to application database: {e}")
            return False
    
    def create_extensions(self) -> bool:
        """Create necessary PostgreSQL extensions"""
        extensions = [
            'uuid-ossp',  # For UUID generation
            'pgcrypto',   # For cryptographic functions
            'pg_trgm',    # For similarity searches
            'btree_gin',  # For GIN indexes
        ]
        
        try:
            for extension in extensions:
                self.cursor.execute(f"CREATE EXTENSION IF NOT EXISTS \"{extension}\"")
                logger.info(f"✅ Created extension: {extension}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create extensions: {e}")
            return False
    
    def create_schemas(self) -> bool:
        """Create database schemas"""
        schemas = [
            'fraud_detection',  # Main application schema
            'analytics',        # Analytics and reporting
            'audit',           # Audit logs
            'monitoring'       # Monitoring and metrics
        ]
        
        try:
            for schema in schemas:
                self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                logger.info(f"✅ Created schema: {schema}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create schemas: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create database tables"""
        try:
            # Users table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    status VARCHAR(20) DEFAULT 'active',
                    role VARCHAR(20) DEFAULT 'viewer',
                    is_active BOOLEAN DEFAULT true,
                    is_verified BOOLEAN DEFAULT false,
                    phone VARCHAR(20),
                    department VARCHAR(100),
                    job_title VARCHAR(100),
                    last_login_at TIMESTAMP WITH TIME ZONE,
                    login_count INTEGER DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    api_key VARCHAR(255) UNIQUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_by VARCHAR(100),
                    updated_by VARCHAR(100),
                    metadata_json TEXT
                )
            """)
            
            # Merchants table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.merchants (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    merchant_id VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(500) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    risk_level VARCHAR(20) DEFAULT 'medium',
                    country VARCHAR(2),
                    website VARCHAR(500),
                    established_date DATE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata_json TEXT
                )
            """)
            
            # Transactions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.transactions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    transaction_id VARCHAR(100) UNIQUE NOT NULL,
                    external_transaction_id VARCHAR(100),
                    user_id VARCHAR(100) NOT NULL,
                    merchant_id VARCHAR(100) NOT NULL,
                    amount DECIMAL(15,2) NOT NULL,
                    currency VARCHAR(3) DEFAULT 'USD',
                    payment_method VARCHAR(50) NOT NULL,
                    payment_processor VARCHAR(100),
                    card_last_four VARCHAR(4),
                    card_brand VARCHAR(50),
                    card_country VARCHAR(2),
                    transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    country VARCHAR(2),
                    region VARCHAR(100),
                    city VARCHAR(100),
                    ip_address INET,
                    user_agent TEXT,
                    device_id VARCHAR(255),
                    session_id VARCHAR(255),
                    fraud_score FLOAT,
                    risk_level VARCHAR(20),
                    decision VARCHAR(20),
                    model_version VARCHAR(50),
                    model_confidence FLOAT,
                    risk_factors JSONB,
                    feature_values JSONB,
                    requires_review BOOLEAN DEFAULT false,
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    is_fraud BOOLEAN,
                    actual_fraud_status BOOLEAN,
                    ground_truth_source VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata_json TEXT
                )
            """)
            
            # Alerts table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    alert_id VARCHAR(100) UNIQUE NOT NULL,
                    alert_type VARCHAR(100) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    severity VARCHAR(20) NOT NULL,
                    status VARCHAR(20) DEFAULT 'open',
                    priority INTEGER DEFAULT 5,
                    transaction_id UUID REFERENCES fraud_detection.transactions(id),
                    user_id VARCHAR(100),
                    merchant_id VARCHAR(100),
                    risk_score FLOAT,
                    risk_level VARCHAR(20),
                    assigned_to UUID REFERENCES fraud_detection.users(id),
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    resolution_status VARCHAR(100),
                    resolution_notes TEXT,
                    is_false_positive BOOLEAN,
                    alert_generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Model metadata table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.model_metadata (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_name VARCHAR(200) UNIQUE NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) DEFAULT 'training',
                    accuracy FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    f1_score FLOAT,
                    roc_auc FLOAT,
                    training_data_size INTEGER,
                    feature_count INTEGER,
                    hyperparameters JSONB,
                    metrics JSONB,
                    model_path VARCHAR(500),
                    deployed_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_by VARCHAR(100)
                )
            """)
            
            # Blacklist table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS fraud_detection.blacklists (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entry_type VARCHAR(50) NOT NULL,
                    value VARCHAR(500) NOT NULL,
                    reason VARCHAR(50) NOT NULL,
                    description TEXT,
                    is_active BOOLEAN DEFAULT true,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_by VARCHAR(100)
                )
            """)
            
            # Audit logs table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit.audit_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES fraud_detection.users(id),
                    action VARCHAR(50) NOT NULL,
                    entity_type VARCHAR(100) NOT NULL,
                    entity_id VARCHAR(100) NOT NULL,
                    details TEXT,
                    ip_address INET,
                    user_agent TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            logger.info("✅ Database tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create database indexes for performance"""
        indexes = [
            # Transaction indexes
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON fraud_detection.transactions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON fraud_detection.transactions(merchant_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_date ON fraud_detection.transactions(transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_fraud_score ON fraud_detection.transactions(fraud_score)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON fraud_detection.transactions(risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_status ON fraud_detection.transactions(status)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON fraud_detection.transactions(user_id, transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_merchant_date ON fraud_detection.transactions(merchant_id, transaction_date)",
            
            # Alert indexes
            "CREATE INDEX IF NOT EXISTS idx_alerts_status ON fraud_detection.alerts(status)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON fraud_detection.alerts(severity)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON fraud_detection.alerts(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_assigned_to ON fraud_detection.alerts(assigned_to)",
            
            # User indexes
            "CREATE INDEX IF NOT EXISTS idx_users_username ON fraud_detection.users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON fraud_detection.users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_status ON fraud_detection.users(status)",
            "CREATE INDEX IF NOT EXISTS idx_users_api_key ON fraud_detection.users(api_key)",
            
            # Merchant indexes
            "CREATE INDEX IF NOT EXISTS idx_merchants_merchant_id ON fraud_detection.merchants(merchant_id)",
            "CREATE INDEX IF NOT EXISTS idx_merchants_category ON fraud_detection.merchants(category)",
            "CREATE INDEX IF NOT EXISTS idx_merchants_risk_level ON fraud_detection.merchants(risk_level)",
            
            # Blacklist indexes
            "CREATE INDEX IF NOT EXISTS idx_blacklists_type_value ON fraud_detection.blacklists(entry_type, value)",
            "CREATE INDEX IF NOT EXISTS idx_blacklists_active ON fraud_detection.blacklists(is_active)",
            
            # JSONB indexes for better performance
            "CREATE INDEX IF NOT EXISTS idx_transactions_risk_factors ON fraud_detection.transactions USING gin(risk_factors)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_features ON fraud_detection.transactions USING gin(feature_values)",
        ]
        
        try:
            for index_sql in indexes:
                self.cursor.execute(index_sql)
            
            logger.info(f"✅ Created {len(indexes)} database indexes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            return False
    
    def create_functions(self) -> bool:
        """Create database functions and triggers"""
        try:
            # Function to update timestamp
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create triggers for updated_at
            tables_with_updated_at = [
                'fraud_detection.users',
                'fraud_detection.merchants', 
                'fraud_detection.transactions',
                'fraud_detection.alerts',
                'fraud_detection.model_metadata',
                'fraud_detection.blacklists'
            ]
            
            for table in tables_with_updated_at:
                trigger_name = f"{table.split('.')[1]}_updated_at_trigger"
                self.cursor.execute(f"""
                    CREATE TRIGGER {trigger_name}
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
                """)
            
            # Function to calculate fraud risk score
            self.cursor.execute("""
                CREATE OR REPLACE FUNCTION calculate_user_risk_score(user_id_param VARCHAR)
                RETURNS FLOAT AS $$
                DECLARE
                    fraud_count INTEGER;
                    total_count INTEGER;
                    avg_amount DECIMAL;
                    risk_score FLOAT;
                BEGIN
                    -- Get user transaction stats
                    SELECT 
                        COUNT(*) FILTER (WHERE is_fraud = true),
                        COUNT(*),
                        AVG(amount)
                    INTO fraud_count, total_count, avg_amount
                    FROM fraud_detection.transactions 
                    WHERE user_id = user_id_param
                      AND transaction_date >= NOW() - INTERVAL '30 days';
                    
                    -- Calculate basic risk score
                    IF total_count = 0 THEN
                        risk_score := 0.5; -- Neutral for new users
                    ELSE
                        risk_score := LEAST(fraud_count::FLOAT / total_count::FLOAT * 10, 1.0);
                    END IF;
                    
                    RETURN risk_score;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Database functions and triggers created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create functions: {e}")
            return False
    
    def insert_sample_data(self) -> bool:
        """Insert sample data for testing"""
        try:
            # Insert sample admin user
            self.cursor.execute("""
                INSERT INTO fraud_detection.users (
                    username, email, first_name, last_name, password_hash, 
                    role, status, is_active, is_verified
                ) VALUES (
                    'admin', 'admin@frauddetection.com', 'Admin', 'User', 
                    '/LewwWyCoDO.4L4YMO', -- password: admin123
                    'admin', 'active', true, true
                ) ON CONFLICT (username) DO NOTHING
            """)
            
            # Insert sample merchant categories
            sample_merchants = [
                ('MERCHANT_001', 'Electronics Store', 'electronics', 'low'),
                ('MERCHANT_002', 'Fashion Boutique', 'clothing', 'medium'),
                ('MERCHANT_003', 'Online Pharmacy', 'healthcare', 'high'),
                ('MERCHANT_004', 'Gaming Platform', 'entertainment', 'medium'),
                ('MERCHANT_005', 'Crypto Exchange', 'cryptocurrency', 'high')
            ]
            
            for merchant_data in sample_merchants:
                self.cursor.execute("""
                    INSERT INTO fraud_detection.merchants (
                        merchant_id, name, category, risk_level
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (merchant_id) DO NOTHING
                """, merchant_data)
            
            # Insert sample blacklist entries
            sample_blacklist = [
                ('ip_address', '192.168.1.100', 'fraud', 'Known fraudulent IP'),
                ('email', 'fraud@example.com', 'fraud', 'Fraudulent email address'),
                ('device_id', 'DEVICE_FRAUD_001', 'suspicious_activity', 'Suspicious device')
            ]
            
            for blacklist_data in sample_blacklist:
                self.cursor.execute("""
                    INSERT INTO fraud_detection.blacklists (
                        entry_type, value, reason, description
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, blacklist_data)
            
            logger.info("✅ Sample data inserted")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            return False
    
    def run_database_setup(self) -> bool:
        """Run complete database setup"""
        logger.info("🐘 Starting database initialization...")
        
        steps = [
            ("Connect to PostgreSQL", self.connect),
            ("Create Database", self.create_database),
            ("Connect to App Database", self.connect_to_app_database),
            ("Create Extensions", self.create_extensions),
            ("Create Schemas", self.create_schemas),
            ("Create Tables", self.create_tables),
            ("Create Indexes", self.create_indexes),
            ("Create Functions", self.create_functions),
            ("Insert Sample Data", self.insert_sample_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            if not step_func():
                logger.error(f"❌ Database setup failed at: {step_name}")
                return False
        
        logger.info("🎉 Database initialization completed successfully!")
        return True
    
    def cleanup(self):
        """Clean up database connections"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Initialize fraud detection database")
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--database', default='fraud_detection', help='Database name')
    parser.add_argument('--user', default='fraud_user', help='Database user')
    parser.add_argument('--password', default='fraud_password', help='Database password')
    
    args = parser.parse_args()
    
    config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    db_init = DatabaseInitializer(config)
    
    try:
        success = db_init.run_database_setup()
        sys.exit(0 if success else 1)
    finally:
        db_init.cleanup()

if __name__ == "__main__":
    main()
