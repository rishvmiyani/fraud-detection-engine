"""
Comprehensive Data Validation for Fraud Detection Pipeline
Validates data quality, schema compliance, and business rules
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

class DataValidator:
    """
    Advanced data validation for fraud detection pipeline
    Handles schema validation, data quality checks, and business rule validation
    """
    
    def __init__(self, schemas_path: str = "schemas/"):
        """Initialize data validator"""
        self.schemas_path = Path(schemas_path)
        self.validation_errors = []
        self.validation_warnings = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load schema definition from JSON file"""
        schema_file = self.schemas_path / f"{schema_name}_schema.json"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r') as f:
            return json.load(f)
    
    def validate_schema_compliance(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate dataframe against schema"""
        errors = []
        
        # Check required fields
        schema_fields = [field['name'] for field in schema['fields']]
        missing_fields = set(schema_fields) - set(df.columns)
        
        if missing_fields:
            errors.append({
                'type': 'missing_fields',
                'message': f"Missing required fields: {missing_fields}"
            })
        
        # Check extra fields
        extra_fields = set(df.columns) - set(schema_fields)
        if extra_fields:
            self.validation_warnings.append({
                'type': 'extra_fields',
                'message': f"Extra fields found: {extra_fields}"
            })
        
        # Validate each field
        for field in schema['fields']:
            field_name = field['name']
            
            if field_name not in df.columns:
                continue
            
            # Check data type
            field_errors = self._validate_field(df[field_name], field, field_name)
            errors.extend(field_errors)
        
        # Validate constraints
        constraint_errors = self._validate_constraints(df, schema.get('constraints', []))
        errors.extend(constraint_errors)
        
        return errors
    
    def _validate_field(self, series: pd.Series, field_config: Dict[str, Any], field_name: str) -> List[Dict[str, str]]:
        """Validate individual field against its configuration"""
        errors = []
        
        # Check required fields for null values
        if field_config.get('required', False):
            null_count = series.isnull().sum()
            if null_count > 0:
                errors.append({
                    'type': 'null_values',
                    'field': field_name,
                    'message': f"Field '{field_name}' has {null_count} null values but is required"
                })
        
        # Type validation
        expected_type = field_config['type']
        if not self._check_data_type(series, expected_type):
            errors.append({
                'type': 'data_type',
                'field': field_name,
                'message': f"Field '{field_name}' should be of type {expected_type}"
            })
        
        # Pattern validation
        if 'pattern' in field_config:
            pattern_errors = self._validate_pattern(series, field_config['pattern'], field_name)
            errors.extend(pattern_errors)
        
        # Range validation
        if 'min_value' in field_config or 'max_value' in field_config:
            range_errors = self._validate_range(series, field_config, field_name)
            errors.extend(range_errors)
        
        # Length validation
        if 'max_length' in field_config:
            length_errors = self._validate_length(series, field_config['max_length'], field_name)
            errors.extend(length_errors)
        
        # Allowed values validation
        if 'allowed_values' in field_config:
            value_errors = self._validate_allowed_values(series, field_config['allowed_values'], field_name)
            errors.extend(value_errors)
        
        return errors
    
    def _check_data_type(self, series: pd.Series, expected_type: str) -> bool:
        """Check if series matches expected data type"""
        if expected_type == 'string':
            return series.dtype == 'object'
        elif expected_type == 'integer':
            return pd.api.types.is_integer_dtype(series)
        elif expected_type == 'float':
            return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
        elif expected_type == 'datetime':
            return pd.api.types.is_datetime64_any_dtype(series)
        elif expected_type == 'date':
            return pd.api.types.is_datetime64_any_dtype(series)
        return True
    
    def _validate_pattern(self, series: pd.Series, pattern: str, field_name: str) -> List[Dict[str, str]]:
        """Validate field values against regex pattern"""
        errors = []
        
        # Skip null values
        non_null_series = series.dropna()
        
        try:
            regex = re.compile(pattern)
            invalid_count = 0
            
            for value in non_null_series:
                if not regex.match(str(value)):
                    invalid_count += 1
            
            if invalid_count > 0:
                errors.append({
                    'type': 'pattern_mismatch',
                    'field': field_name,
                    'message': f"Field '{field_name}' has {invalid_count} values that don't match pattern: {pattern}"
                })
        
        except re.error as e:
            errors.append({
                'type': 'invalid_pattern',
                'field': field_name,
                'message': f"Invalid regex pattern for field '{field_name}': {e}"
            })
        
        return errors
    
    def _validate_range(self, series: pd.Series, field_config: Dict[str, Any], field_name: str) -> List[Dict[str, str]]:
        """Validate numeric field ranges"""
        errors = []
        
        # Skip non-numeric series
        if not pd.api.types.is_numeric_dtype(series):
            return errors
        
        min_val = field_config.get('min_value')
        max_val = field_config.get('max_value')
        
        if min_val is not None:
            below_min = (series < min_val).sum()
            if below_min > 0:
                errors.append({
                    'type': 'range_violation',
                    'field': field_name,
                    'message': f"Field '{field_name}' has {below_min} values below minimum {min_val}"
                })
        
        if max_val is not None:
            above_max = (series > max_val).sum()
            if above_max > 0:
                errors.append({
                    'type': 'range_violation',
                    'field': field_name,
                    'message': f"Field '{field_name}' has {above_max} values above maximum {max_val}"
                })
        
        return errors
    
    def _validate_length(self, series: pd.Series, max_length: int, field_name: str) -> List[Dict[str, str]]:
        """Validate string field lengths"""
        errors = []
        
        # Skip null values and convert to string
        non_null_series = series.dropna().astype(str)
        
        too_long = (non_null_series.str.len() > max_length).sum()
        if too_long > 0:
            errors.append({
                'type': 'length_violation',
                'field': field_name,
                'message': f"Field '{field_name}' has {too_long} values exceeding max length {max_length}"
            })
        
        return errors
    
    def _validate_allowed_values(self, series: pd.Series, allowed_values: List[Any], field_name: str) -> List[Dict[str, str]]:
        """Validate field values against allowed values list"""
        errors = []
        
        invalid_values = set(series.dropna()) - set(allowed_values)
        
        if invalid_values:
            errors.append({
                'type': 'invalid_values',
                'field': field_name,
                'message': f"Field '{field_name}' has invalid values: {invalid_values}"
            })
        
        return errors
    
    def _validate_constraints(self, df: pd.DataFrame, constraints: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate business rule constraints"""
        errors = []
        
        for constraint in constraints:
            constraint_type = constraint['type']
            
            if constraint_type == 'unique':
                unique_errors = self._validate_uniqueness(df, constraint)
                errors.extend(unique_errors)
            
            elif constraint_type == 'business_rule':
                rule_errors = self._validate_business_rule(df, constraint)
                errors.extend(rule_errors)
        
        return errors
    
    def _validate_uniqueness(self, df: pd.DataFrame, constraint: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate uniqueness constraints"""
        errors = []
        
        fields = constraint['fields']
        duplicates = df.duplicated(subset=fields).sum()
        
        if duplicates > 0:
            errors.append({
                'type': 'uniqueness_violation',
                'message': f"Uniqueness constraint violated for fields {fields}: {duplicates} duplicates found"
            })
        
        return errors
    
    def _validate_business_rule(self, df: pd.DataFrame, constraint: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate business rules"""
        errors = []
        
        rule = constraint['rule']
        
        try:
            # Simple rule evaluation (can be extended for complex rules)
            if 'amount > 0' in rule:
                if 'amount' in df.columns:
                    violations = (df['amount'] <= 0).sum()
                    if violations > 0:
                        errors.append({
                            'type': 'business_rule_violation',
                            'message': f"Business rule '{rule}' violated: {violations} records"
                        })
            
            elif 'timestamp <= NOW()' in rule:
                if 'timestamp' in df.columns:
                    future_dates = (pd.to_datetime(df['timestamp']) > datetime.now()).sum()
                    if future_dates > 0:
                        errors.append({
                            'type': 'business_rule_violation',
                            'message': f"Business rule '{rule}' violated: {future_dates} records"
                        })
        
        except Exception as e:
            errors.append({
                'type': 'business_rule_error',
                'message': f"Error evaluating business rule '{rule}': {e}"
            })
        
        return errors
    
    def generate_data_quality_report(self, df: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'dataset_info': {
                'total_records': len(df),
                'total_fields': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'completeness': {},
            'validity': {},
            'consistency': {},
            'uniqueness': {},
            'timeliness': {}
        }
        
        # Completeness analysis
        for col in df.columns:
            null_count = df[col].isnull().sum()
            report['completeness'][col] = {
                'null_count': null_count,
                'null_percentage': (null_count / len(df)) * 100,
                'completeness_score': ((len(df) - null_count) / len(df)) * 100
            }
        
        # Validity analysis
        schema = self.load_schema(schema_name)
        schema_errors = self.validate_schema_compliance(df, schema)
        report['validity']['schema_errors'] = len(schema_errors)
        report['validity']['error_details'] = schema_errors
        
        # Uniqueness analysis
        for col in df.columns:
            unique_count = df[col].nunique()
            report['uniqueness'][col] = {
                'unique_count': unique_count,
                'duplicate_count': len(df) - unique_count,
                'uniqueness_ratio': unique_count / len(df)
            }
        
        # Data distribution analysis
        report['distribution'] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            report['distribution'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        return report
    
    def validate_transaction_data(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Specific validation for transaction data"""
        validation_report = {
            'schema_validation': [],
            'business_rules': [],
            'data_quality': {},
            'fraud_analysis': {}
        }
        
        # Schema validation
        schema = self.load_schema('transaction')
        schema_errors = self.validate_schema_compliance(transactions_df, schema)
        validation_report['schema_validation'] = schema_errors
        
        # Fraud-specific business rules
        if 'is_fraud' in transactions_df.columns:
            fraud_rate = transactions_df['is_fraud'].mean()
            validation_report['fraud_analysis'] = {
                'fraud_rate': fraud_rate,
                'total_fraud': transactions_df['is_fraud'].sum(),
                'total_legitimate': (transactions_df['is_fraud'] == 0).sum()
            }
            
            # Check fraud rate reasonableness
            if fraud_rate > 0.1:  # More than 10% fraud
                validation_report['business_rules'].append({
                    'type': 'fraud_rate_high',
                    'message': f"Fraud rate is unusually high: {fraud_rate*100:.2f}%"
                })
            elif fraud_rate < 0.001:  # Less than 0.1% fraud
                validation_report['business_rules'].append({
                    'type': 'fraud_rate_low',
                    'message': f"Fraud rate is unusually low: {fraud_rate*100:.3f}%"
                })
        
        # Amount validation
        if 'amount' in transactions_df.columns:
            # Check for negative amounts
            negative_amounts = (transactions_df['amount'] < 0).sum()
            if negative_amounts > 0:
                validation_report['business_rules'].append({
                    'type': 'negative_amounts',
                    'message': f"Found {negative_amounts} transactions with negative amounts"
                })
            
            # Check for extremely high amounts
            high_amount_threshold = transactions_df['amount'].quantile(0.999)
            extreme_amounts = (transactions_df['amount'] > high_amount_threshold).sum()
            validation_report['fraud_analysis']['extreme_amounts'] = extreme_amounts
        
        # Timestamp validation
        if 'timestamp' in transactions_df.columns:
            # Convert to datetime if not already
            timestamps = pd.to_datetime(transactions_df['timestamp'])
            
            # Check for future dates
            future_dates = (timestamps > datetime.now()).sum()
            if future_dates > 0:
                validation_report['business_rules'].append({
                    'type': 'future_timestamps',
                    'message': f"Found {future_dates} transactions with future timestamps"
                })
            
            # Check for old dates (more than 5 years)
            old_threshold = datetime.now() - timedelta(days=5*365)
            old_dates = (timestamps < old_threshold).sum()
            if old_dates > 0:
                validation_report['business_rules'].append({
                    'type': 'very_old_timestamps',
                    'message': f"Found {old_dates} transactions older than 5 years"
                })
        
        return validation_report

def main():
    """Main validation function"""
    validator = DataValidator()
    
    # Validate transaction data
    print("Loading transaction data...")
    transactions_df = pd.read_csv('raw/transactions.csv')
    
    print("Validating transaction data...")
    validation_report = validator.validate_transaction_data(transactions_df)
    
    # Print validation results
    print("\n=== VALIDATION REPORT ===")
    print(f"Schema validation errors: {len(validation_report['schema_validation'])}")
    print(f"Business rule violations: {len(validation_report['business_rules'])}")
    
    if validation_report['schema_validation']:
        print("\nSchema Errors:")
        for error in validation_report['schema_validation']:
            print(f"  - {error['type']}: {error['message']}")
    
    if validation_report['business_rules']:
        print("\nBusiness Rule Violations:")
        for rule in validation_report['business_rules']:
            print(f"  - {rule['type']}: {rule['message']}")
    
    if 'fraud_analysis' in validation_report and validation_report['fraud_analysis']:
        print("\nFraud Analysis:")
        for key, value in validation_report['fraud_analysis'].items():
            print(f"  - {key}: {value}")
    
    # Generate comprehensive data quality report
    print("\nGenerating data quality report...")
    quality_report = validator.generate_data_quality_report(transactions_df, 'transaction')
    
    print("\n=== DATA QUALITY SUMMARY ===")
    print(f"Total records: {quality_report['dataset_info']['total_records']}")
    print(f"Memory usage: {quality_report['dataset_info']['memory_usage_mb']:.2f} MB")
    
    print("\nCompleteness Scores:")
    for field, stats in quality_report['completeness'].items():
        print(f"  - {field}: {stats['completeness_score']:.1f}%")
    
    print("\nData validation completed successfully!")

if __name__ == "__main__":
    main()
