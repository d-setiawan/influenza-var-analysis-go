"""
Authors: Rohan Adla, Arrio Gonsalves, Shreyan Nalwad, Dylan Setiawan
Date: Dec 12th 2025
Project: A VAR-based Computational Analysis of Influenza and Weather Dynamics
Class: 02-613 at Caregie Mellon University
Data Integration Script for VAR Analysis

Merges weather data with FluNet influenza data for VAR model training.
Supports multiple countries/regions with configurable parameters.

Usage:
    python integration.py --country Singapore
    python integration.py --country Qatar
    python integration.py --country NewJersey
    python integration.py --all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Get the project root directory (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Country configurations
COUNTRY_CONFIG = {
    'Singapore': {
        'country_code': 'SGP',
        'weather_file': 'Singapore/singapore_weather_weekly_last25yrs.csv',
        'output_prefix': 'SG',
        'output_folder': 'Singapore',
        'include_visibility': True,
    },
    'Qatar': {
        'country_code': 'QAT',
        'weather_file': 'Qatar/Qatar_weather_weekly_last25yrs.csv',
        'output_prefix': 'Qatar',
        'output_folder': 'Qatar',
        'include_visibility': True,
    },
    'NewJersey': {
        'country_code': 'USA',
        'weather_file': 'newjersey/newjersey_weather_weekly_last25yrs.csv',
        'output_prefix': 'NJ',
        'output_folder': 'NewJersey',
        'include_visibility': False,  # High NaN rate for NJ visibility
    },
}

# Columns to drop from weather data
DROP_COLS = [
    "wind_gust",
    "pressure_3hr_change",
    "station_id",
    "Station_ID",
    "Station_name",
    "wind_dir",
    "wind_dir_deg",
    "wind_direction",
    "Latitude",
    "Longitude",
    "Elevation",
    "station_level_pressure",
    "altimeter",
    "isoyw",
    "iso_weekstartdate",
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_15_hour",
    "precipitation_18_hour",
    "precipitation_21_hour",
]

# Date columns to remove after merge
DATE_COLS = [
    "iso_weekstartdate",
    "ISO_WEEKSTARTDATE",
    "mmwr_weekstartdate",
    "mmwr_year",
    "mmwr_week",
    "isoyw",
    "mmwryw",
    "iso_year",
    "iso_week",
    "ISO_YEAR",
    "ISO_WEEK",
    "COUNTRY_CODE",
    "country_code",
    "COUNTRY_AREA_TERRITORY",
]


def get_weather_columns(include_visibility=True):
    """Get list of weather columns for final output."""
    cols = [
        "temperature",
        "dew_point_temperature",
        "relative_humidity",
        "wind_speed",
        "sea_level_pressure",
        "wet_bulb_temperature",
        "precipitation",
        "precipitation_24_hour",
    ]
    if include_visibility:
        cols.insert(5, "visibility")
    return cols


def load_and_clean_weather(weather_path, include_visibility=True):
    """Load and clean weather data."""
    weather = pd.read_csv(weather_path)
    
    # Drop unnecessary columns
    weather = weather.drop(columns=[c for c in DROP_COLS if c in weather.columns])
    
    # Build list of relevant columns
    relevant = ["iso_year", "iso_week"] + get_weather_columns(include_visibility)
    weather = weather[[c for c in relevant if c in weather.columns]]
    
    # Sort and interpolate
    if "iso_year" in weather.columns and "iso_week" in weather.columns:
        weather = weather.sort_values(["iso_year", "iso_week"])
    
    # Interpolate numeric columns
    for col in get_weather_columns(include_visibility):
        if col in weather.columns:
            weather[col] = weather[col].interpolate(method="linear", limit_direction="both")
    
    return weather


def load_and_filter_flu(flu_path, country_code):
    """Load and filter FluNet data for a specific country."""
    flu = pd.read_csv(flu_path)
    
    # Handle both uppercase and lowercase column names
    country_col = "COUNTRY_CODE" if "COUNTRY_CODE" in flu.columns else "country_code"
    
    if country_col in flu.columns:
        flu = flu[flu[country_col].astype(str).str.upper() == country_code].copy()
        print(f"  Flu rows after {country_code} filter: {flu.shape[0]}")
    else:
        print("  [WARN] Country code column not found; no filter applied.")
    
    return flu


def merge_data(flu, weather):
    """Merge flu and weather data on ISO year/week."""
    # Handle column name variations
    iso_year_col = "ISO_YEAR" if "ISO_YEAR" in flu.columns else "iso_year"
    iso_week_col = "ISO_WEEK" if "ISO_WEEK" in flu.columns else "iso_week"
    
    # Prepare flu data
    flu["iso_year"] = pd.to_numeric(flu[iso_year_col], errors="coerce")
    flu["iso_week"] = pd.to_numeric(flu[iso_week_col], errors="coerce")
    flu = flu[flu["iso_year"].notna() & flu["iso_week"].notna()].copy()
    flu["iso_year"] = flu["iso_year"].astype(int)
    flu["iso_week"] = flu["iso_week"].astype(int)
    flu = flu.sort_values(["iso_year", "iso_week"])
    
    # Interpolate flu counts
    inf_a_col = "INF_A" if "INF_A" in flu.columns else "inf_a"
    inf_b_col = "INF_B" if "INF_B" in flu.columns else "inf_b"
    
    for col in [inf_a_col, inf_b_col]:
        if col in flu.columns:
            flu[col] = flu[col].interpolate(method="linear", limit_direction="both")
    
    # Prepare weather data
    weather["iso_year"] = pd.to_numeric(weather["iso_year"], errors="coerce")
    weather["iso_week"] = pd.to_numeric(weather["iso_week"], errors="coerce")
    weather = weather[weather["iso_year"].notna() & weather["iso_week"].notna()].copy()
    weather["iso_year"] = weather["iso_year"].astype(int)
    weather["iso_week"] = weather["iso_week"].astype(int)
    
    # Aggregate weather to one row per week
    value_cols = [c for c in weather.columns if c not in ["iso_year", "iso_week"]]
    weather_weekly = weather.groupby(["iso_year", "iso_week"], as_index=False)[value_cols].mean()
    
    # Merge
    merged = pd.merge(
        flu,
        weather_weekly,
        on=["iso_year", "iso_week"],
        how="inner",
        validate="many_to_one"
    )
    
    # Remove date columns
    merged = merged.drop(columns=[c for c in DATE_COLS if c in merged.columns])
    
    return merged, inf_a_col, inf_b_col


def create_training_datasets(merged, inf_a_col, inf_b_col, include_visibility=True):
    """Create separate training datasets for INF_A and INF_B."""
    # Handle LOG columns
    if "LOG_INF_A" not in merged.columns and inf_a_col in merged.columns:
        merged["LOG_INF_A"] = np.log1p(merged[inf_a_col])
    
    if "LOG_INF_B" not in merged.columns and inf_b_col in merged.columns:
        merged["LOG_INF_B"] = np.log1p(merged[inf_b_col])
    
    weather_cols = get_weather_columns(include_visibility)
    
    # Create INF_A dataset
    cols_a = ["LOG_INF_A"] + weather_cols
    merged_a = merged[[c for c in cols_a if c in merged.columns]].copy()
    
    # Create INF_B dataset
    cols_b = ["LOG_INF_B"] + weather_cols
    merged_b = merged[[c for c in cols_b if c in merged.columns]].copy()
    
    return merged_a, merged_b


def process_country(country_name, config):
    """Process data for a single country."""
    print(f"\n{'='*60}")
    print(f"Processing: {country_name}")
    print(f"{'='*60}")
    
    # Build paths
    weather_path = os.path.join(PROJECT_ROOT, "Files", "Raw Data", config['weather_file'])
    flu_path = os.path.join(PROJECT_ROOT, "Files", "Raw Data", "Final_FluNet.csv")
    output_dir = os.path.join(PROJECT_ROOT, "Files", "Final_Training_Data", config['output_folder'])
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"  Loading weather from: {config['weather_file']}")
    weather = load_and_clean_weather(weather_path, config['include_visibility'])
    print(f"  Weather columns: {weather.columns.tolist()}")
    
    print(f"  Loading flu data for: {config['country_code']}")
    flu = load_and_filter_flu(flu_path, config['country_code'])
    
    # Merge
    print("  Merging datasets...")
    merged, inf_a_col, inf_b_col = merge_data(flu, weather)
    print(f"  Merged shape: {merged.shape}")
    
    # Create training datasets
    merged_a, merged_b = create_training_datasets(
        merged, inf_a_col, inf_b_col, config['include_visibility']
    )
    
    # Save outputs
    prefix = config['output_prefix']
    
    path_a = os.path.join(output_dir, f"{prefix}_Training_Data_INF_A.csv")
    merged_a.to_csv(path_a, index=False)
    print(f"  Saved INF_A: {path_a} ({merged_a.shape})")
    
    path_b = os.path.join(output_dir, f"{prefix}_Training_Data_INF_B.csv")
    merged_b.to_csv(path_b, index=False)
    print(f"  Saved INF_B: {path_b} ({merged_b.shape})")
    
    return merged_a, merged_b


def main():
    parser = argparse.ArgumentParser(
        description="Integrate weather and flu data for VAR analysis"
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
    print("Data Integration for VAR Analysis")
    print("=" * 60)
    
    for country in countries:
        config = COUNTRY_CONFIG[country]
        process_country(country, config)
    
    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

