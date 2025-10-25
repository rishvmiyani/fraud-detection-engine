import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data directory
os.makedirs('data/processed', exist_ok=True)

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
    'user_id': [f'USER_{np.random.randint(1, 101):03d}' for _ in range(n_samples)],
    'merchant_id': [f'MERCHANT_{np.random.randint(1, 51):02d}' for _ in range(n_samples)],
    'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples).round(2),
    'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal'], n_samples),
    'timestamp': [datetime.now() - timedelta(minutes=np.random.randint(0, 43200)) for _ in range(n_samples)],
    'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
}

df = pd.DataFrame(data)
df.to_csv('data/processed/sample_transactions.csv', index=False)

print(f'✅ Generated {len(df)} sample transactions')
print(f'✅ Fraud rate: {df["is_fraud"].mean():.2%}')
print(f'✅ Average amount: ${df["amount"].mean():.2f}')
print('✅ Data saved to: data/processed/sample_transactions.csv')

# Show sample
print('\n📊 Sample data:')
print(df.head())
