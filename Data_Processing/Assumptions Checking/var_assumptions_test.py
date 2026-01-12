"""
Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
Date: Dec 12th 2025
Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
Class: 02-613 at Caregie Mellon University

VAR Assumptions Testing Script

Tests statistical assumptions for VAR model validity:
- Stationarity (ADF and KPSS tests)
- Linearity (RESET test)
- Multicollinearity (VIF)

Applies log transformations to variables that fail assumptions.

Usage:
    python var_assumptions_test.py --country Singapore
    python var_assumptions_test.py --all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Country configurations
COUNTRY_CONFIG = {
    'Singapore': {
        'prefix': 'SG',
        'folder': 'Singapore',
    },
    'Qatar': {
        'prefix': 'Qatar',
        'folder': 'Qatar',
    },
    'NewJersey': {
        'prefix': 'NJ',
        'folder': 'NewJersey',
    },
}

# Columns that should NOT be log-transformed
NO_LOG_COLS = ['LOG_INF_A', 'LOG_INF_B', 'temperature', 'dew_point_temperature', 'wet_bulb_temperature']


def test_stationarity(series, col_name):
    """Test stationarity using ADF and KPSS tests."""
    results = {'column': col_name}
    series = series.dropna()
    
    if len(series) < 20:
        return {
            'column': col_name,
            'adf_pvalue': np.nan,
            'kpss_pvalue': np.nan,
            'adf_stationary': False,
            'kpss_stationary': False,
            'error': 'Insufficient data'
        }
    
    # ADF Test
    try:
        adf_result = adfuller(series, autolag='AIC')
        results['adf_stat'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
        results['adf_stationary'] = adf_result[1] < 0.05
    except Exception as e:
        results['adf_pvalue'] = np.nan
        results['adf_stationary'] = False
    
    # KPSS Test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        results['kpss_stat'] = kpss_result[0]
        results['kpss_pvalue'] = kpss_result[1]
        results['kpss_stationary'] = kpss_result[1] >= 0.05
    except Exception as e:
        results['kpss_pvalue'] = np.nan
        results['kpss_stationary'] = False
    
    return results


def test_linearity(series, col_name):
    """Test linearity using RESET test."""
    results = {'column': col_name}
    series = series.dropna().values
    n = len(series)
    
    if n < 20:
        return {
            'column': col_name,
            'reset_pvalue': np.nan,
            'is_linear': False,
            'error': 'Insufficient data'
        }
    
    try:
        X = np.arange(n).reshape(-1, 1)
        y = series
        
        X_const = add_constant(X)
        model_linear = OLS(y, X_const).fit()
        
        y_hat = model_linear.fittedvalues
        X_squared = np.column_stack([X_const, y_hat**2])
        model_extended = OLS(y, X_squared).fit()
        
        rss_linear = model_linear.ssr
        rss_extended = model_extended.ssr
        
        df_num = 1
        df_denom = n - model_extended.df_model - 1
        
        if rss_extended > 0 and df_denom > 0:
            f_stat = ((rss_linear - rss_extended) / df_num) / (rss_extended / df_denom)
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)
        else:
            f_stat = np.nan
            p_value = np.nan
        
        results['reset_f_stat'] = f_stat
        results['reset_pvalue'] = p_value
        results['is_linear'] = p_value >= 0.05 if not np.isnan(p_value) else False
        
    except Exception:
        results['reset_pvalue'] = np.nan
        results['is_linear'] = False
    
    return results


def calculate_vif(df):
    """Calculate VIF for all columns."""
    vif_data = []
    df_clean = df.dropna()
    
    if len(df_clean) < 10:
        return pd.DataFrame({'column': df.columns, 'VIF': [np.nan] * len(df.columns)})
    
    for i, col in enumerate(df_clean.columns):
        try:
            vif = variance_inflation_factor(df_clean.values, i)
            vif_data.append({'column': col, 'VIF': vif})
        except Exception:
            vif_data.append({'column': col, 'VIF': np.nan})
    
    return pd.DataFrame(vif_data)


def apply_log_transform(df, columns_to_transform):
    """Apply log(x + 1) transformation to specified columns."""
    df_transformed = df.copy()
    transformed_cols = []
    
    for col in columns_to_transform:
        if col in df.columns and col not in NO_LOG_COLS:
            min_val = df[col].min()
            if min_val >= 0:
                new_col = f"LOG_{col.upper().replace(' ', '_')}"
                df_transformed[new_col] = np.log1p(df[col])
                transformed_cols.append((col, new_col))
                print(f"    Created {new_col} from {col}")
            else:
                print(f"    Skipping {col} (has negative values)")
    
    return df_transformed, transformed_cols


def process_dataset(data_path, flu_type, output_dir):
    """Process a single dataset - test assumptions and apply transformations."""
    print(f"\n{'-'*60}")
    print(f"Processing: {os.path.basename(data_path)}")
    print(f"{'-'*60}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    
    all_cols = df.columns.tolist()
    
    # Test stationarity
    print("\nStationarity Tests:")
    non_stationary_cols = []
    for col in all_cols:
        result = test_stationarity(df[col], col)
        adf_ok = result.get('adf_stationary', False)
        kpss_ok = result.get('kpss_stationary', False)
        
        if adf_ok and kpss_ok:
            status = "Stationary"
        elif adf_ok or kpss_ok:
            status = "Partially stationary"
        else:
            status = "Non-stationary"
            non_stationary_cols.append(col)
        
        adf_p = result.get('adf_pvalue', np.nan)
        kpss_p = result.get('kpss_pvalue', np.nan)
        print(f"  {col:25s} ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f} -> {status}")
    
    # Test linearity
    print("\nLinearity Tests:")
    non_linear_cols = []
    for col in all_cols:
        result = test_linearity(df[col], col)
        is_linear = result.get('is_linear', False)
        p_val = result.get('reset_pvalue', np.nan)
        
        if is_linear:
            status = "Linear"
        else:
            status = "Non-linear"
            non_linear_cols.append(col)
        
        print(f"  {col:25s} p={p_val:.4f} -> {status}")
    
    # Test multicollinearity
    print("\nMulticollinearity (VIF):")
    vif_df = calculate_vif(df)
    high_vif_cols = []
    for _, row in vif_df.iterrows():
        col = row['column']
        vif = row['VIF']
        
        if np.isnan(vif):
            status = "Could not calculate"
        elif vif < 5:
            status = "OK"
        elif vif < 10:
            status = "Moderate"
        else:
            status = "High"
            high_vif_cols.append(col)
        
        print(f"  {col:25s} VIF={vif:10.2f} -> {status}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Non-stationary: {non_stationary_cols if non_stationary_cols else 'None'}")
    print(f"  Non-linear: {non_linear_cols if non_linear_cols else 'None'}")
    print(f"  High VIF: {high_vif_cols if high_vif_cols else 'None'}")
    
    # Apply transformations if needed
    cols_needing_transform = list(set(non_stationary_cols + non_linear_cols))
    cols_to_transform = [c for c in cols_needing_transform if c not in NO_LOG_COLS]
    
    if cols_to_transform:
        print(f"\nApplying log transformations to: {cols_to_transform}")
        df_transformed, transformed_cols = apply_log_transform(df, cols_to_transform)
        
        # Save transformed data
        output_prefix = os.path.basename(data_path).replace('.csv', '')
        output_path = os.path.join(output_dir, f"{output_prefix}_transformed.csv")
        df_transformed.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        return df_transformed, True
    else:
        print("\nNo transformations needed.")
        return df, False


def process_country(country_name, config):
    """Process both INF_A and INF_B datasets for a country."""
    print(f"\n{'='*60}")
    print(f"Country: {country_name}")
    print(f"{'='*60}")
    
    base_dir = os.path.join(PROJECT_ROOT, "Files", "Final_Training_Data", config['folder'])
    prefix = config['prefix']
    
    # Process INF_A
    path_a = os.path.join(base_dir, f"{prefix}_Training_Data_INF_A.csv")
    if os.path.exists(path_a):
        process_dataset(path_a, 'INF_A', base_dir)
    else:
        print(f"  File not found: {path_a}")
    
    # Process INF_B
    path_b = os.path.join(base_dir, f"{prefix}_Training_Data_INF_B.csv")
    if os.path.exists(path_b):
        process_dataset(path_b, 'INF_B', base_dir)
    else:
        print(f"  File not found: {path_b}")


def main():
    parser = argparse.ArgumentParser(
        description="Test VAR assumptions and apply transformations"
    )
    parser.add_argument(
        "--country",
        choices=list(COUNTRY_CONFIG.keys()),
        help="Country to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all countries"
    )
    
    args = parser.parse_args()
    
    if not args.country and not args.all:
        parser.print_help()
        sys.exit(1)
    
    countries = list(COUNTRY_CONFIG.keys()) if args.all else [args.country]
    
    print("=" * 60)
    print("VAR Assumptions Testing")
    print("=" * 60)
    
    for country in countries:
        config = COUNTRY_CONFIG[country]
        process_country(country, config)
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

