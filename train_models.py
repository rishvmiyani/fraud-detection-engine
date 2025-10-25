import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
import os
from datetime import datetime

def train_ml_models():
    print('🤖 Starting Advanced ML Model Training...')
    
    # Load training data
    print('📊 Loading training data...')
    train_df = pd.read_csv('data/training/train_data.csv')
    test_df = pd.read_csv('data/training/test_data.csv')
    
    print(f'Training set: {len(train_df):,} samples')
    print(f'Test set: {len(test_df):,} samples')
    
    # Prepare features
    print('🔧 Engineering features...')
    
    # Categorical encoding
    categorical_cols = ['payment_method', 'merchant_category', 'country', 'device_type']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].astype(str))
        test_df[f'{col}_encoded'] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    
    # Feature columns for ML
    feature_cols = [
        'amount', 'amount_log', 'amount_vs_user_avg',
        'hour', 'day_of_week', 'month',
        'is_weekend', 'is_night', 'is_business_hours', 'is_round_amount',
        'payment_method_encoded', 'merchant_category_encoded',
        'country_encoded', 'device_type_encoded'
    ]
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['is_fraud']
    X_test = test_df[feature_cols]
    y_test = test_df['is_fraud']
    
    print(f'Features: {len(feature_cols)}')
    print(f'Training fraud rate: {y_train.mean():.2%}')
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models (simplified for faster training)
    models = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50),
            'params': {
                'max_depth': [10, 20],
                'min_samples_split': [5, 10]
            },
            'use_scaled': False
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            },
            'use_scaled': True
        }
    }
    
    # Train models
    model_results = {}
    best_models = {}
    
    for name, config in models.items():
        print(f'\n🤖 Training {name}...')
        
        # Select data
        X_train_model = X_train_scaled if config['use_scaled'] else X_train
        X_test_model = X_test_scaled if config['use_scaled'] else X_test
        
        # Grid search
        print('   🔍 Hyperparameter tuning...')
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_model, y_train)
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_proba = best_model.predict_proba(X_test_model)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f'   ✅ Best ROC AUC: {roc_auc:.4f}')
        print(f'   ⚙️ Best params: {grid_search.best_params_}')
        
        # Save model
        model_path = f'models/production/{name}_model.pkl'
        joblib.dump(best_model, model_path)
        
        if config['use_scaled']:
            joblib.dump(scaler, f'models/production/{name}_scaler.pkl')
        
        model_results[name] = {
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'model_path': model_path,
            'use_scaled': config['use_scaled']
        }
        
        best_models[name] = best_model
    
    # Save artifacts
    joblib.dump(encoders, 'models/production/encoders.pkl')
    joblib.dump(scaler, 'models/production/scaler.pkl')
    
    with open('models/production/model_comparison.json', 'w') as f:
        json.dump(model_results, f, indent=2, default=str)
    
    feature_info = {
        'feature_columns': feature_cols,
        'categorical_columns': categorical_cols,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'fraud_rate': float(y_train.mean())
    }
    
    with open('models/production/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Results
    best_model_name = max(model_results.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_score = model_results[best_model_name]['roc_auc']
    
    print('\n🏆 TRAINING SUMMARY:')
    print('=' * 50)
    print(f'Best Model: {best_model_name}')
    print(f'Best ROC AUC: {best_score:.4f}')
    print(f'Models trained: {len(model_results)}')
    print(f'Features used: {len(feature_cols)}')
    print('✅ All models saved to models/production/')
    
    # Feature importance
    if hasattr(best_models[best_model_name], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_models[best_model_name].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print('\n📊 TOP 10 IMPORTANT FEATURES:')
        for _, row in importance_df.head(10).iterrows():
            print(f'   {row["feature"]}: {row["importance"]:.4f}')
    
    print('\n🎉 ML training completed successfully!')
    return model_results, best_model_name

if __name__ == "__main__":
    train_ml_models()
