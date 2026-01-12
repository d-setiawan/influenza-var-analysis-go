"""
Create Top 20 Countries Interpolated Dataset

This script selects the top 20 countries by data quality and applies
interpolation to fill NaN values in the influenza time series data.

Usage:
    python create_top20_interpolated.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Get project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File paths relative to project root
COUNTRY_RANKING_PATH = PROJECT_ROOT / "Files" / "Raw Data" / "country_ranking_by_nan.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "Files" / "Raw Data" / "clean_influenza_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "Files" / "Raw Data" / "top20_countries_interpolated.csv"

# Load country ranking
print("\n1. Loading country ranking...")
ranking_df = pd.read_csv(COUNTRY_RANKING_PATH)
print(f"   Loaded ranking for {len(ranking_df)} countries")

# Get top 20 countries
top20_countries = ranking_df.head(20)[['COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY', 'Data_Quality_Score']].copy()
print(f"\n2. Top 20 countries by data quality:")
print("-" * 80)
for idx, row in top20_countries.iterrows():
    print(f"   {idx+1:2d}. {row['COUNTRY_AREA_TERRITORY']:40s} (Quality: {row['Data_Quality_Score']:6.2f}%)")

# Load clean influenza data
print(f"\n3. Loading clean influenza data...")
df = pd.read_csv(CLEAN_DATA_PATH)
print(f"   Loaded {len(df)} rows")

# Filter for top 20 countries
print(f"\n4. Filtering data for top 20 countries...")
country_codes_top20 = top20_countries['COUNTRY_CODE'].tolist()
df_top20 = df[df['COUNTRY_CODE'].isin(country_codes_top20)].copy()
print(f"   Filtered to {len(df_top20)} rows")

# Convert date column to datetime for proper sorting
print(f"\n5. Sorting by time (preserving chronological order)...")
df_top20['ISO_WEEKSTARTDATE'] = pd.to_datetime(df_top20['ISO_WEEKSTARTDATE'])
df_top20 = df_top20.sort_values(['COUNTRY_CODE', 'ISO_YEAR', 'ISO_WEEK']).reset_index(drop=True)
print(f"   Data sorted by country and time")

# Count NaNs before interpolation
nan_before = {
    'INF_A': df_top20['INF_A'].isna().sum(),
    'INF_B': df_top20['INF_B'].isna().sum(),
    'INF_ALL': df_top20['INF_ALL'].isna().sum()
}
total_nan_before = sum(nan_before.values())

print(f"\n6. NaN counts before interpolation:")
print(f"   - INF_A:   {nan_before['INF_A']:5d} NaNs")
print(f"   - INF_B:   {nan_before['INF_B']:5d} NaNs")
print(f"   - INF_ALL: {nan_before['INF_ALL']:5d} NaNs")
print(f"   - Total:   {total_nan_before:5d} NaNs")

# Apply interpolation by country
print(f"\n7. Applying interpolation to fill NaNs...")
print("   (Using linear interpolation within each country's time series)")

# Group by country and interpolate
inf_columns = ['INF_A', 'INF_B', 'INF_ALL']
for col in inf_columns:
    df_top20[col] = df_top20.groupby('COUNTRY_CODE')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    # Round to whole numbers
    df_top20[col] = df_top20[col].round(0).astype(float)

# Count NaNs after interpolation
nan_after = {
    'INF_A': df_top20['INF_A'].isna().sum(),
    'INF_B': df_top20['INF_B'].isna().sum(),
    'INF_ALL': df_top20['INF_ALL'].isna().sum()
}
total_nan_after = sum(nan_after.values())

print(f"\n8. NaN counts after interpolation:")
print(f"   - INF_A:   {nan_after['INF_A']:5d} NaNs (removed {nan_before['INF_A'] - nan_after['INF_A']})")
print(f"   - INF_B:   {nan_after['INF_B']:5d} NaNs (removed {nan_before['INF_B'] - nan_after['INF_B']})")
print(f"   - INF_ALL: {nan_after['INF_ALL']:5d} NaNs (removed {nan_before['INF_ALL'] - nan_after['INF_ALL']})")
print(f"   - Total:   {total_nan_after:5d} NaNs (removed {total_nan_before - total_nan_after})")

# If there are still NaNs (at the boundaries), fill with forward/backward fill
if total_nan_after > 0:
    print(f"\n9. Applying forward/backward fill for remaining boundary NaNs...")
    for col in inf_columns:
        df_top20[col] = df_top20.groupby('COUNTRY_CODE')[col].transform(
            lambda x: x.ffill().bfill()
        )
        # Round to whole numbers
        df_top20[col] = df_top20[col].round(0).astype(float)
    
    # Final NaN count
    nan_final = {
        'INF_A': df_top20['INF_A'].isna().sum(),
        'INF_B': df_top20['INF_B'].isna().sum(),
        'INF_ALL': df_top20['INF_ALL'].isna().sum()
    }
    total_nan_final = sum(nan_final.values())
    
    print(f"   Final NaN count: {total_nan_final}")
    
    # If still NaNs remain (countries with all NaN columns), fill with 0
    if total_nan_final > 0:
        print(f"\n10. Filling remaining NaNs with 0 (isolated values)...")
        for col in inf_columns:
            df_top20[col] = df_top20[col].fillna(0)
        print(f"    All NaNs filled")

# Convert date back to string format for CSV export
df_top20['ISO_WEEKSTARTDATE'] = df_top20['ISO_WEEKSTARTDATE'].dt.strftime('%Y-%m-%d')

# Save to CSV
print(f"\n11. Saving interpolated data...")
try:
    df_top20.to_csv(OUTPUT_PATH, index=False)
    print(f"    Data saved to: {OUTPUT_PATH}")
except Exception as e:
    print(f"    Error saving data: {e}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total rows in output: {len(df_top20)}")
print(f"Number of countries: {df_top20['COUNTRY_CODE'].nunique()}")
print(f"Date range: {df_top20['ISO_WEEKSTARTDATE'].min()} to {df_top20['ISO_WEEKSTARTDATE'].max()}")
print(f"\nData completeness after interpolation:")
print(f"  - INF_A:   {100 - (df_top20['INF_A'].isna().sum() / len(df_top20) * 100):.2f}% complete")
print(f"  - INF_B:   {100 - (df_top20['INF_B'].isna().sum() / len(df_top20) * 100):.2f}% complete")
print(f"  - INF_ALL: {100 - (df_top20['INF_ALL'].isna().sum() / len(df_top20) * 100):.2f}% complete")

print("\nCountry breakdown:")
country_summary = df_top20.groupby('COUNTRY_AREA_TERRITORY').agg({
    'ISO_WEEKSTARTDATE': 'count',
    'INF_A': lambda x: f"{((~x.isna()).sum() / len(x) * 100):.1f}%",
    'INF_B': lambda x: f"{((~x.isna()).sum() / len(x) * 100):.1f}%",
    'INF_ALL': lambda x: f"{((~x.isna()).sum() / len(x) * 100):.1f}%"
}).rename(columns={'ISO_WEEKSTARTDATE': 'Records'})

print(country_summary.to_string())

print("\n" + "=" * 80)
print("Processing complete!")
print("=" * 80)
