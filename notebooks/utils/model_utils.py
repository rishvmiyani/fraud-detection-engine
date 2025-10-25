"""
Model Utilities for Fraud Detection
Comprehensive model training, evaluation, and deployment utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, log_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import pickle
import joblib
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    def get_classification_metrics(self) -> Dict[str, float]:
        """Get comprehensive classification metrics"""
        metrics = {
            'accuracy': (self.y_pred == self.y_test).mean(),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, self.y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(self.y_test, self.y_pred)
        }
        
        if self.y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_pred_proba)
            metrics['avg_precision'] = average_precision_score(self.y_test, self.y_pred_proba)
            metrics['log_loss'] = log_loss(self.y_test, self.y_pred_proba)
        
        return metrics
    
    def get_confusion_matrix_metrics(self) -> Dict[str, Union[int, float]]:
        """Get confusion matrix and derived metrics"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        return metrics
    
    def get_business_metrics(self, fraud_cost_avg: float = 100, 
                           false_positive_cost: float = 5) -> Dict[str, float]:
        """Calculate business impact metrics"""
        cm_metrics = self.get_confusion_matrix_metrics()
        
        # Business calculations
        fraud_prevented = cm_metrics['true_positives'] * fraud_cost_avg
        fraud_missed = cm_metrics['false_negatives'] * fraud_cost_avg
        false_positive_cost_total = cm_metrics['false_positives'] * false_positive_cost
        
        net_benefit = fraud_prevented - false_positive_cost_total
        total_potential_fraud = (cm_metrics['true_positives'] + cm_metrics['false_negatives']) * fraud_cost_avg
        
        return {
            'fraud_prevented_amount': fraud_prevented,
            'fraud_missed_amount': fraud_missed,
            'false_positive_cost': false_positive_cost_total,
            'net_benefit': net_benefit,
            'fraud_prevention_rate': fraud_prevented / total_potential_fraud if total_potential_fraud > 0 else 0,
            'cost_ratio': false_positive_cost_total / fraud_prevented if fraud_prevented > 0 else float('inf')
        }
    
    def get_threshold_metrics(self, thresholds: List[float]) -> pd.DataFrame:
        """Evaluate model at different probability thresholds"""
        if self.y_pred_proba is None:
            raise ValueError("Model does not support probability predictions")
        
        results = []
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            # Basic metrics
            precision = precision_score(self.y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(self.y_test, y_pred_thresh, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_thresh, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred_thresh)
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def print_evaluation_report(self):
        """Print comprehensive evaluation report"""
        print(f"\n📊 EVALUATION REPORT - {self.model_name}")
        print("=" * 60)
        
        # Classification metrics
        class_metrics = self.get_classification_metrics()
        print(f"\n🎯 Classification Metrics:")
        for metric, value in class_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Confusion matrix metrics
        cm_metrics = self.get_confusion_matrix_metrics()
        print(f"\n📋 Confusion Matrix:")
        print(f"   True Negatives: {cm_metrics['true_negatives']:,}")
        print(f"   False Positives: {cm_metrics['false_positives']:,}")
        print(f"   False Negatives: {cm_metrics['false_negatives']:,}")
        print(f"   True Positives: {cm_metrics['true_positives']:,}")
        print(f"   Sensitivity (Recall): {cm_metrics['sensitivity']:.4f}")
        print(f"   Specificity: {cm_metrics['specificity']:.4f}")
        print(f"   False Positive Rate: {cm_metrics['false_positive_rate']:.4f}")
        
        # Business metrics
        business_metrics = self.get_business_metrics()
        print(f"\n💼 Business Impact:")
        print(f"   Fraud Prevented: ")
        print(f"   Fraud Missed: ")
        print(f"   False Positive Cost: ")
        print(f"   Net Benefit: ")
        print(f"   Fraud Prevention Rate: {business_metrics['fraud_prevention_rate']:.2%}")


def handle_class_imbalance(X: np.ndarray, y: np.ndarray, 
                          method: str = 'smote',
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using various resampling techniques
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Resampling method ('smote', 'adasyn', 'undersampling', 'smoteenn', 'smotetomek')
        random_state: Random seed
        
    Returns:
        Tuple of resampled (X, y)
    """
    print(f"🔄 Applying {method} for class imbalance...")
    print(f"   Original distribution: {np.bincount(y)}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif method == 'undersampling':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smoteenn':
        sampler = SMOTEENN(random_state=random_state)
    elif method == 'smotetomek':
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    print(f"   Resampled distribution: {np.bincount(y_resampled)}")
    print(f"   New dataset size: {X_resampled.shape}")
    
    return X_resampled, y_resampled


def perform_cross_validation(model, X: np.ndarray, y: np.ndarray,
                            cv_folds: int = 5,
                            scoring_metrics: List[str] = None,
                            random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive cross-validation
    
    Args:
        model: Machine learning model
        X: Feature matrix
        y: Target variable
        cv_folds: Number of cross-validation folds
        scoring_metrics: List of scoring metrics
        random_state: Random seed
        
    Returns:
        Dictionary of cross-validation results
    """
    if scoring_metrics is None:
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}
    
    print(f"🔄 Performing {cv_folds}-fold cross-validation...")
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"   {metric.upper()}: {scores.mean():.4f} (±{scores.std():.4f})")
        except Exception as e:
            print(f"   ⚠️ Error calculating {metric}: {str(e)}")
    
    return cv_results


def hyperparameter_tuning(model, param_grid: Dict[str, List],
                         X: np.ndarray, y: np.ndarray,
                         search_type: str = 'grid',
                         cv_folds: int = 5,
                         n_iter: int = 100,
                         scoring: str = 'roc_auc',
                         random_state: int = 42,
                         n_jobs: int = -1) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning
    
    Args:
        model: Base model to tune
        param_grid: Parameter grid for search
        X: Feature matrix
        y: Target variable
        search_type: Type of search ('grid' or 'random')
        cv_folds: Number of CV folds
        n_iter: Number of iterations for random search
        scoring: Scoring metric
        random_state: Random seed
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (best_model, search_results)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    print(f"🔍 Performing {search_type} search for hyperparameter tuning...")
    print(f"   Parameter grid size: {np.prod([len(v) for v in param_grid.values()]):,}")
    
    if search_type == 'grid':
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=n_jobs, verbose=1, return_train_score=True
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
            n_jobs=n_jobs, verbose=1, random_state=random_state, return_train_score=True
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    
    # Fit search
    search.fit(X, y)
    
    print(f"✅ Hyperparameter tuning completed!")
    print(f"   Best score: {search.best_score_:.4f}")
    print(f"   Best parameters: {search.best_params_}")
    
    # Prepare results
    search_results = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'cv_results': pd.DataFrame(search.cv_results_)
    }
    
    return search.best_estimator_, search_results


def calibrate_model_probabilities(model, X_cal: np.ndarray, y_cal: np.ndarray,
                                 method: str = 'isotonic',
                                 cv_folds: int = 3) -> Any:
    """
    Calibrate model probability predictions
    
    Args:
        model: Trained model
        X_cal: Calibration feature matrix
        y_cal: Calibration target variable
        method: Calibration method ('isotonic' or 'sigmoid')
        cv_folds: Number of CV folds for calibration
        
    Returns:
        Calibrated model
    """
    print(f"🎯 Calibrating model probabilities using {method} method...")
    
    calibrated_model = CalibratedClassifierCV(
        model, method=method, cv=cv_folds
    )
    
    calibrated_model.fit(X_cal, y_cal)
    
    print(f"✅ Model calibration completed!")
    
    return calibrated_model


def analyze_model_calibration(model, X_test: np.ndarray, y_test: np.ndarray,
                             n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze model probability calibration
    
    Args:
        model: Trained model with probability predictions
        X_test: Test feature matrix
        y_test: Test target variable
        n_bins: Number of bins for calibration analysis
        
    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=n_bins
    )
    
    # Calculate calibration metrics
    calibration_error = np.abs(fraction_of_positives - mean_predicted_value).mean()
    
    print(f"📊 Calibration Analysis:")
    print(f"   Mean Calibration Error: {calibration_error:.4f}")
    print(f"   Perfect calibration error: 0.0000")
    
    return fraction_of_positives, mean_predicted_value


def save_model(model, filepath: str, metadata: Dict[str, Any] = None) -> None:
    """
    Save model with metadata
    
    Args:
        model: Trained model
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': pd.Timestamp.now(),
        'model_type': type(model).__name__
    }
    
    if filepath.endswith('.joblib'):
        joblib.dump(model_data, filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    else:
        # Default to joblib
        joblib.dump(model_data, f"{filepath}.joblib")
    
    print(f"💾 Model saved to: {filepath}")


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model with metadata
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        Tuple of (model, metadata)
    """
    if filepath.endswith('.joblib') or 'joblib' in filepath:
        model_data = joblib.load(filepath)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    else:
        # Try joblib first
        try:
            model_data = joblib.load(filepath)
        except:
            # Fallback to pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
    
    print(f"📂 Model loaded from: {filepath}")
    print(f"   Model type: {model_data.get('model_type', 'Unknown')}")
    print(f"   Saved at: {model_data.get('saved_at', 'Unknown')}")
    
    return model_data['model'], model_data.get('metadata', {})


def compare_models(models_results: Dict[str, Dict[str, float]],
                  metrics: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple models across various metrics
    
    Args:
        models_results: Dictionary with model names and their metrics
        metrics: List of metrics to compare
        
    Returns:
        DataFrame with model comparison
    """
    if metrics is None:
        metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    
    comparison_data = []
    for model_name, results in models_results.items():
        row = {'Model': model_name}
        for metric in metrics:
            if metric in results:
                row[metric.replace('_', ' ').title()] = results[metric]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric (first in the list)
    if len(metrics) > 0 and metrics[0].replace('_', ' ').title() in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            metrics[0].replace('_', ' ').title(), 
            ascending=False
        ).reset_index(drop=True)
    
    return comparison_df


def create_ensemble_model(models: List[Any], 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         method: str = 'voting',
                         weights: List[float] = None) -> Any:
    """
    Create ensemble model from multiple base models
    
    Args:
        models: List of trained models
        X_train: Training feature matrix
        y_train: Training target variable
        method: Ensemble method ('voting', 'stacking')
        weights: Voting weights for models
        
    Returns:
        Ensemble model
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    
    if method == 'voting':
        # Create voting classifier
        estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability predictions
            weights=weights
        )
        
    elif method == 'stacking':
        from sklearn.ensemble import StackingClassifier
        
        # Create stacking classifier
        estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Fit ensemble
    print(f"🎭 Creating {method} ensemble with {len(models)} models...")
    ensemble.fit(X_train, y_train)
    print(f"✅ Ensemble model created successfully!")
    
    return ensemble


if __name__ == "__main__":
    # Example usage
    print("🤖 Testing model utilities...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=1, flip_y=0.01,
                              class_sep=0.8, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator(model, X_test, y_test, "Random Forest")
    evaluator.print_evaluation_report()
    
    # Test cross-validation
    cv_results = perform_cross_validation(model, X_train, y_train)
    
    print("\n✅ Model utilities test completed!")
