#!/usr/bin/env python
"""
Model Training Pipeline
Automated training pipeline for fraud detection models
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Automated model training pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_path = Path(config['data_path'])
        self.model_output_path = Path(config['model_output_path'])
        self.experiment_name = config.get('experiment_name', 'fraud_detection')
        self.random_state = config.get('random_state', 42)
        
        # Ensure output directory exists
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)
        
        self.models = {}
        self.results = {}
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        logger.info(f"📂 Loading training data from: {self.data_path}")
        
        if self.data_path.is_file():
            if self.data_path.suffix == '.parquet':
                df = pd.read_parquet(self.data_path)
            elif self.data_path.suffix == '.csv':
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        else:
            # Load most recent processed file
            parquet_files = list(self.data_path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {self.data_path}")
            
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            logger.info(f"   Loaded latest file: {latest_file}")
        
        logger.info(f"   Data shape: {df.shape}")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Prepare features and target
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # Handle target variable
        if 'is_fraud' in df.columns:
            y = df['is_fraud'].copy()
        elif 'fraud_label' in df.columns:
            y = df['fraud_label'].copy()
        elif 'label' in df.columns:
            y = df['label'].copy()
        else:
            # Use anomaly detection as proxy
            if 'is_anomaly' in df.columns:
                y = df['is_anomaly'].copy()
                logger.warning("Using 'is_anomaly' as target variable (no fraud labels found)")
            else:
                raise ValueError("No target variable found (is_fraud, fraud_label, label, is_anomaly)")
        
        # Data quality checks
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Fraud rate: {y.mean():.2%}")
        logger.info(f"   Class distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Found missing values, filling with median")
            X = X.fillna(X.median())
        
        return X, y
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for training"""
        exclude_columns = [
            'transaction_id', 'user_id', 'merchant_id', 'timestamp', 
            'is_fraud', 'fraud_label', 'label', 'is_anomaly',
            'processed_at', 'data_source'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_columns]
        
        # Only include numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"   Selected {len(numeric_cols)} numeric features")
        return numeric_cols
    
    def prepare_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split training data"""
        logger.info("🔄 Preparing training data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.get('test_size', 0.2),
            random_state=self.random_state, stratify=y
        )
        
        # Scale features if configured
        if self.config.get('scale_features', True):
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            scaler_path = self.model_output_path / 'feature_scaler.joblib'
            joblib.dump(scaler, scaler_path)
            logger.info(f"   Scaler saved: {scaler_path}")
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Handle class imbalance
        if self.config.get('handle_imbalance', True):
            logger.info("   Applying SMOTE for class balancing...")
            original_counts = np.bincount(y_train)
            
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            new_counts = np.bincount(y_train_balanced)
            logger.info(f"   Class distribution: {original_counts} → {new_counts}")
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
        
        logger.info(f"   Training set: {X_train_balanced.shape}")
        logger.info(f"   Test set: {X_test_scaled.shape}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def define_models(self) -> Dict[str, Any]:
        """Define models to train"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                scale_pos_weight=1,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            )
        }
        
        return models
    
    def train_model(self, model, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train a single model with MLflow tracking"""
        logger.info(f"🤖 Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['training_time_seconds'] = training_time
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            mlflow.log_metrics({
                'cv_auc_mean': metrics['cv_auc_mean'],
                'cv_auc_std': metrics['cv_auc_std']
            })
            
            # Save model
            model_path = self.model_output_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': range(len(feature_importance)),
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                importance_path = self.model_output_path / f"{model_name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            logger.info(f"   ✅ {model_name} trained - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return {
                'model': model,
                'metrics': metrics,
                'model_path': str(model_path),
                'predictions': {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            }
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp), 
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        return metrics
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train all defined models"""
        logger.info("🚀 Starting model training pipeline...")
        
        models_to_train = self.define_models()
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                result = self.train_model(model, model_name, X_train, y_train, X_test, y_test)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {str(e)}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select best performing model"""
        if not results:
            raise ValueError("No models trained successfully")
        
        # Sort by ROC AUC score
        model_scores = {}
        for model_name, result in results.items():
            auc_score = result['metrics'].get('roc_auc', 0)
            model_scores[model_name] = auc_score
        
        best_model_name = max(model_scores, key=model_scores.get)
        best_result = results[best_model_name]
        
        logger.info(f"🏆 Best model: {best_model_name} (AUC: {model_scores[best_model_name]:.4f})")
        
        return best_model_name, best_result
    
    def generate_model_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive model comparison report"""
        logger.info("📊 Generating model comparison report...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_path = self.model_output_path / 'model_comparison_report.csv'
        comparison_df.to_csv(report_path, index=False)
        
        # Generate detailed report
        report = {
            'experiment_name': self.experiment_name,
            'training_timestamp': datetime.utcnow().isoformat(),
            'models_trained': len(results),
            'best_model': self.select_best_model(results)[0],
            'training_config': self.config,
            'model_comparison': comparison_df.to_dict('records'),
            'summary_statistics': {
                'mean_auc': comparison_df['roc_auc'].mean(),
                'std_auc': comparison_df['roc_auc'].std(),
                'best_auc': comparison_df['roc_auc'].max(),
                'worst_auc': comparison_df['roc_auc'].min()
            }
        }
        
        report_json_path = self.model_output_path / 'training_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"   Report saved: {report_json_path}")
        return str(report_json_path)
    
    def run_training_pipeline(self) -> bool:
        """Run complete training pipeline"""
        try:
            logger.info("🔥 Starting fraud detection model training pipeline...")
            
            # Load data
            X, y = self.load_training_data()
            
            # Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(X, y)
            
            # Train all models
            results = self.train_all_models(X_train, X_test, y_train, y_test)
            
            if not results:
                logger.error("❌ No models trained successfully")
                return False
            
            # Select best model and create deployment package
            best_model_name, best_result = self.select_best_model(results)
            
            # Generate reports
            self.generate_model_comparison_report(results)
            
            # Save best model metadata
            best_model_metadata = {
                'model_name': best_model_name,
                'model_path': best_result['model_path'],
                'metrics': best_result['metrics'],
                'training_config': self.config,
                'feature_count': X.shape[1],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'deployment_ready': True,
                'created_at': datetime.utcnow().isoformat()
            }
            
            metadata_path = self.model_output_path / 'best_model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(best_model_metadata, f, indent=2, default=str)
            
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"   Best model: {best_model_name}")
            logger.info(f"   AUC Score: {best_result['metrics']['roc_auc']:.4f}")
            logger.info(f"   F1 Score: {best_result['metrics']['f1_score']:.4f}")
            logger.info(f"   Model saved: {best_result['model_path']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument('--data-path', required=True, help='Path to training data')
    parser.add_argument('--output-path', required=True, help='Output directory for models')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--experiment-name', default='fraud_detection', help='MLflow experiment name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--scale-features', action='store_true', help='Scale features')
    parser.add_argument('--handle-imbalance', action='store_true', help='Handle class imbalance')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        'data_path': args.data_path,
        'model_output_path': args.output_path,
        'experiment_name': args.experiment_name,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'scale_features': args.scale_features,
        'handle_imbalance': args.handle_imbalance
    })
    
    # Run training
    trainer = ModelTrainer(config)
    success = trainer.run_training_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
