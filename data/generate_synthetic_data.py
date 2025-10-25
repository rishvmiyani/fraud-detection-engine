"""
Advanced Synthetic Data Generation for Fraud Detection
Generates realistic transaction data with proper fraud patterns
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import json
from typing import Dict, List, Tuple
import hashlib

class FraudDataGenerator:
    """
    Advanced synthetic fraud data generator
    Creates realistic transaction patterns with embedded fraud scenarios
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the fraud data generator"""
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        # Define fraud patterns and configurations
        self.fraud_patterns = {
            'high_amount': {'probability': 0.15, 'amount_multiplier': 5.0},
            'velocity_attack': {'probability': 0.20, 'transaction_count': 5},
            'geographic_anomaly': {'probability': 0.10, 'distance_threshold': 1000},
            'unusual_time': {'probability': 0.08, 'hour_range': (2, 5)},
            'blacklisted_merchant': {'probability': 0.12},
            'stolen_card': {'probability': 0.25, 'different_location': True},
            'account_takeover': {'probability': 0.10, 'new_device': True}
        }
        
        # Payment methods with fraud likelihoods
        self.payment_methods = {
            'credit_card': {'weight': 0.45, 'fraud_rate': 0.025},
            'debit_card': {'weight': 0.25, 'fraud_rate': 0.015},
            'paypal': {'weight': 0.15, 'fraud_rate': 0.008},
            'apple_pay': {'weight': 0.08, 'fraud_rate': 0.005},
            'google_pay': {'weight': 0.07, 'fraud_rate': 0.005}
        }
        
        # Merchant categories with risk levels
        self.merchant_categories = {
            'electronics': {'weight': 0.20, 'avg_amount': 250, 'fraud_rate': 0.035},
            'grocery': {'weight': 0.15, 'avg_amount': 75, 'fraud_rate': 0.008},
            'gas_station': {'weight': 0.12, 'avg_amount': 50, 'fraud_rate': 0.012},
            'restaurant': {'weight': 0.10, 'avg_amount': 45, 'fraud_rate': 0.010},
            'clothing': {'weight': 0.10, 'avg_amount': 120, 'fraud_rate': 0.018},
            'jewelry': {'weight': 0.08, 'avg_amount': 800, 'fraud_rate': 0.065},
            'financial': {'weight': 0.05, 'avg_amount': 2000, 'fraud_rate': 0.080},
            'travel': {'weight': 0.12, 'avg_amount': 450, 'fraud_rate': 0.025},
            'entertainment': {'weight': 0.08, 'avg_amount': 85, 'fraud_rate': 0.015}
        }
        
        # Country risk profiles
        self.countries = {
            'US': {'weight': 0.60, 'fraud_rate': 0.020, 'timezone': -5},
            'CA': {'weight': 0.10, 'fraud_rate': 0.015, 'timezone': -5},
            'UK': {'weight': 0.08, 'fraud_rate': 0.018, 'timezone': 0},
            'DE': {'weight': 0.06, 'fraud_rate': 0.012, 'timezone': 1},
            'FR': {'weight': 0.04, 'fraud_rate': 0.016, 'timezone': 1},
            'AU': {'weight': 0.03, 'fraud_rate': 0.014, 'timezone': 10},
            'JP': {'weight': 0.05, 'fraud_rate': 0.008, 'timezone': 9},
            'BR': {'weight': 0.04, 'fraud_rate': 0.035, 'timezone': -3}
        }
        
    def generate_user_profiles(self, num_users: int) -> pd.DataFrame:
        """Generate realistic user profiles"""
        users = []
        
        for i in range(num_users):
            # Generate basic user info
            user_id = f"USR{str(i+1).zfill(6)}"
            email = self.fake.email()
            
            # Age distribution (18-75, weighted toward 25-45)
            age = int(np.random.beta(2, 5) * 57 + 18)
            
            # Country selection based on weights
            country = np.random.choice(
                list(self.countries.keys()),
                p=[self.countries[c]['weight'] for c in self.countries.keys()]
            )
            
            # Registration date (past 3 years)
            reg_date = self.fake.date_between(
                start_date='-3y',
                end_date='-30d'
            )
            
            # Risk score calculation
            base_risk = np.random.beta(2, 8)  # Most users low risk
            country_risk_modifier = self.countries[country]['fraud_rate']
            age_risk_modifier = 0.1 if age < 25 or age > 65 else 0.05
            
            risk_score = min(1.0, base_risk + country_risk_modifier + age_risk_modifier)
            
            # Account status based on risk score
            if risk_score > 0.8:
                status = np.random.choice(['flagged', 'suspended'], p=[0.7, 0.3])
            elif risk_score > 0.6:
                status = np.random.choice(['active', 'flagged'], p=[0.8, 0.2])
            else:
                status = 'active'
            
            users.append({
                'user_id': user_id,
                'email': email,
                'phone': self.fake.phone_number(),
                'registration_date': reg_date,
                'country': country,
                'age': age,
                'account_status': status,
                'risk_score': round(risk_score, 3),
                'timezone': self.countries[country]['timezone']
            })
            
        return pd.DataFrame(users)
    
    def generate_merchants(self, num_merchants: int) -> pd.DataFrame:
        """Generate merchant profiles"""
        merchants = []
        
        for i in range(num_merchants):
            merchant_id = f"MRC{str(i+1).zfill(6)}"
            
            # Select category based on weights
            category = np.random.choice(
                list(self.merchant_categories.keys()),
                p=[self.merchant_categories[c]['weight'] 
                   for c in self.merchant_categories.keys()]
            )
            
            # Country for merchant
            country = np.random.choice(
                list(self.countries.keys()),
                p=[self.countries[c]['weight'] for c in self.countries.keys()]
            )
            
            # Risk level based on category fraud rate
            fraud_rate = self.merchant_categories[category]['fraud_rate']
            if fraud_rate > 0.05:
                risk_level = 'high'
            elif fraud_rate > 0.02:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            merchants.append({
                'merchant_id': merchant_id,
                'merchant_name': self.fake.company(),
                'category': category,
                'country': country,
                'risk_level': risk_level,
                'registration_date': self.fake.date_between(start_date='-5y', end_date='-1y'),
                'fraud_rate': fraud_rate,
                'avg_amount': self.merchant_categories[category]['avg_amount']
            })
            
        return pd.DataFrame(merchants)
    
    def generate_transactions(self, users_df: pd.DataFrame, 
                            merchants_df: pd.DataFrame, 
                            num_transactions: int,
                            fraud_rate: float = 0.02) -> pd.DataFrame:
        """Generate realistic transactions with fraud patterns"""
        transactions = []
        
        # Calculate number of fraud transactions
        num_fraud = int(num_transactions * fraud_rate)
        num_legitimate = num_transactions - num_fraud
        
        # Generate legitimate transactions
        for i in range(num_legitimate):
            transaction = self._generate_legitimate_transaction(
                i, users_df, merchants_df
            )
            transaction['is_fraud'] = 0
            transactions.append(transaction)
        
        # Generate fraudulent transactions
        for i in range(num_fraud):
            transaction = self._generate_fraud_transaction(
                num_legitimate + i, users_df, merchants_df
            )
            transaction['is_fraud'] = 1
            transactions.append(transaction)
        
        # Shuffle transactions to mix legitimate and fraud
        random.shuffle(transactions)
        
        # Add sequential transaction IDs and timestamps
        base_time = datetime.now() - timedelta(days=30)
        for idx, transaction in enumerate(transactions):
            transaction['transaction_id'] = f"TXN{str(idx+1).zfill(8)}"
            # Add random time intervals between transactions
            time_offset = timedelta(
                minutes=np.random.exponential(2),  # Average 2 minutes between transactions
                seconds=np.random.randint(0, 60)
            )
            transaction['timestamp'] = base_time + time_offset
            base_time = transaction['timestamp']
        
        return pd.DataFrame(transactions)
    
    def _generate_legitimate_transaction(self, idx: int, 
                                       users_df: pd.DataFrame,
                                       merchants_df: pd.DataFrame) -> Dict:
        """Generate a legitimate transaction"""
        # Select random user (weighted by activity level)
        user = users_df.sample(1).iloc[0]
        
        # Select merchant based on user's country preference
        same_country_merchants = merchants_df[
            merchants_df['country'] == user['country']
        ]
        
        if len(same_country_merchants) > 0 and np.random.random() < 0.8:
            merchant = same_country_merchants.sample(1).iloc[0]
        else:
            merchant = merchants_df.sample(1).iloc[0]
        
        # Generate amount based on merchant category
        base_amount = merchant['avg_amount']
        amount = max(0.01, np.random.lognormal(
            mean=np.log(base_amount),
            sigma=0.5
        ))
        
        # Select payment method
        payment_method = np.random.choice(
            list(self.payment_methods.keys()),
            p=[self.payment_methods[pm]['weight'] 
               for pm in self.payment_methods.keys()]
        )
        
        # Generate realistic IP address
        ip_address = self.fake.ipv4()
        
        # Device ID (consistent for user)
        device_hash = hashlib.md5(f"{user['user_id']}_primary".encode()).hexdigest()
        device_id = f"DEV{device_hash[:8].upper()}"
        
        return {
            'user_id': user['user_id'],
            'merchant_id': merchant['merchant_id'],
            'amount': round(amount, 2),
            'currency': 'USD',  # Simplified for this example
            'payment_method': payment_method,
            'country': merchant['country'],
            'ip_address': ip_address,
            'device_id': device_id,
            'merchant_category': merchant['category'],
            'user_age': user['age'],
            'user_risk_score': user['risk_score']
        }
    
    def _generate_fraud_transaction(self, idx: int,
                                  users_df: pd.DataFrame,
                                  merchants_df: pd.DataFrame) -> Dict:
        """Generate a fraudulent transaction with realistic patterns"""
        # Select fraud pattern
        pattern = np.random.choice(list(self.fraud_patterns.keys()))
        
        # Start with legitimate transaction base
        transaction = self._generate_legitimate_transaction(idx, users_df, merchants_df)
        
        # Apply fraud pattern modifications
        if pattern == 'high_amount':
            transaction['amount'] *= self.fraud_patterns[pattern]['amount_multiplier']
            
        elif pattern == 'geographic_anomaly':
            # Transaction from different country than user
            user_country = users_df[users_df['user_id'] == transaction['user_id']]['country'].iloc[0]
            different_countries = [c for c in self.countries.keys() if c != user_country]
            transaction['country'] = np.random.choice(different_countries)
            
        elif pattern == 'unusual_time':
            # Transaction at unusual hours (2-5 AM)
            hour_range = self.fraud_patterns[pattern]['hour_range']
            unusual_hour = np.random.randint(hour_range[0], hour_range[1])
            # Note: timestamp will be set later in main function
            transaction['unusual_hour'] = unusual_hour
            
        elif pattern == 'stolen_card':
            # Different location and new device
            user = users_df[users_df['user_id'] == transaction['user_id']].iloc[0]
            if self.fraud_patterns[pattern]['different_location']:
                different_country = np.random.choice([c for c in self.countries.keys() 
                                                    if c != user['country']])
                transaction['country'] = different_country
            
            # New device ID
            device_hash = hashlib.md5(f"FRAUD_{idx}".encode()).hexdigest()
            transaction['device_id'] = f"DEV{device_hash[:8].upper()}"
            
        elif pattern == 'blacklisted_merchant':
            # Select high-risk merchant
            high_risk_merchants = merchants_df[merchants_df['risk_level'] == 'high']
            if len(high_risk_merchants) > 0:
                merchant = high_risk_merchants.sample(1).iloc[0]
                transaction['merchant_id'] = merchant['merchant_id']
                transaction['merchant_category'] = merchant['category']
        
        # Add fraud indicators
        transaction['fraud_pattern'] = pattern
        transaction['amount'] = round(transaction['amount'], 2)
        
        return transaction
    
    def generate_complete_dataset(self, num_users: int = 1000,
                                num_merchants: int = 500,
                                num_transactions: int = 10000,
                                fraud_rate: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete fraud detection dataset"""
        print(f"Generating {num_users} users...")
        users_df = self.generate_user_profiles(num_users)
        
        print(f"Generating {num_merchants} merchants...")
        merchants_df = self.generate_merchants(num_merchants)
        
        print(f"Generating {num_transactions} transactions with {fraud_rate*100}% fraud rate...")
        transactions_df = self.generate_transactions(
            users_df, merchants_df, num_transactions, fraud_rate
        )
        
        return users_df, merchants_df, transactions_df

def main():
    """Generate and save synthetic fraud detection dataset"""
    generator = FraudDataGenerator(seed=42)
    
    # Generate complete dataset
    users_df, merchants_df, transactions_df = generator.generate_complete_dataset(
        num_users=2000,
        num_merchants=800,
        num_transactions=50000,
        fraud_rate=0.025  # 2.5% fraud rate
    )
    
    # Save to CSV files
    users_df.to_csv('raw/synthetic_users.csv', index=False)
    merchants_df.to_csv('raw/synthetic_merchants.csv', index=False)
    transactions_df.to_csv('raw/synthetic_transactions.csv', index=False)
    
    # Generate summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Users: {len(users_df)}")
    print(f"Merchants: {len(merchants_df)}")
    print(f"Transactions: {len(transactions_df)}")
    print(f"Fraud transactions: {transactions_df['is_fraud'].sum()}")
    print(f"Fraud rate: {transactions_df['is_fraud'].mean()*100:.2f}%")
    
    # Category breakdown
    print("\n=== MERCHANT CATEGORIES ===")
    print(merchants_df['category'].value_counts())
    
    # Country breakdown
    print("\n=== COUNTRY DISTRIBUTION ===")
    print(users_df['country'].value_counts())
    
    # Payment method breakdown
    print("\n=== PAYMENT METHODS ===")
    print(transactions_df['payment_method'].value_counts())
    
    print("\nSynthetic dataset generated successfully!")

if __name__ == "__main__":
    main()
