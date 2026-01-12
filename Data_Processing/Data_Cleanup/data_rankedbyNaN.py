"""
FluNet Data Cleaning and Country Ranking by NaN Count

This script processes raw WHO FluNet data, cleans it, and ranks countries 
by data quality (NaN count). Outputs are saved to Files/Raw Data/.

Usage:
    python data_rankedbyNaN.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Get project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File paths relative to project root
RAW_DATA_PATH = PROJECT_ROOT / "Files" / "Raw Data" / "VIW_FNT.csv"
OUTPUT_CLEAN_DATA = PROJECT_ROOT / "Files" / "Raw Data" / "clean_influenza_data.csv"
OUTPUT_COUNTRY_RANKING = PROJECT_ROOT / "Files" / "Raw Data" / "country_ranking_by_nan.csv"

print("=" * 80)
print("WHO FluNet Data Cleaning and Country Ranking by NaN Count")
print("=" * 80)

# Load the raw data
print("\n1. Loading raw WHO FluNet data...")
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Loaded {len(df)} rows and {len(df.columns)} columns")
except FileNotFoundError:
    print(f"   Error: File not found at {RAW_DATA_PATH}")
    exit(1)

# Select relevant columns
print("\n2. Selecting relevant columns...")
relevant_columns = [
    # Identifiers
    'COUNTRY_CODE',
    'COUNTRY_AREA_TERRITORY',
    'ISO_WEEKSTARTDATE',
    'ISO_YEAR',
    'ISO_WEEK',
    # Influenza data
    'INF_A',
    'INF_B',
    'INF_ALL'
]

# Check if all columns exist
missing_cols = [col for col in relevant_columns if col not in df.columns]
if missing_cols:
    print(f"   Error: Missing columns in dataset: {missing_cols}")
    exit(1)

df_clean = df[relevant_columns].copy()
print(f"   Selected {len(relevant_columns)} relevant columns")
print(f"   Columns: {', '.join(relevant_columns)}")

# Display data info
print("\n3. Dataset Overview:")
print(f"   - Total rows: {len(df_clean)}")
print(f"   - Date range: {df_clean['ISO_WEEKSTARTDATE'].min()} to {df_clean['ISO_WEEKSTARTDATE'].max()}")
print(f"   - Number of unique countries: {df_clean['COUNTRY_AREA_TERRITORY'].nunique()}")

# Count NaNs by country
print("\n4. Counting NaNs by country...")

# Group by country and count NaNs for INF columns
country_nan_stats = df_clean.groupby(['COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY']).agg({
    'INF_A': lambda x: x.isna().sum(),
    'INF_B': lambda x: x.isna().sum(),
    'INF_ALL': lambda x: x.isna().sum()
}).reset_index()

# Rename columns for clarity
country_nan_stats.columns = ['COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY', 
                              'INF_A_NaN_Count', 'INF_B_NaN_Count', 'INF_ALL_NaN_Count']

# Calculate total NaN count and total records per country
country_totals = df_clean.groupby(['COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY']).size().reset_index(name='Total_Records')
country_nan_stats = country_nan_stats.merge(country_totals, on=['COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY'])

# Calculate total NaN count across all INF columns
country_nan_stats['Total_NaN_Count'] = (country_nan_stats['INF_A_NaN_Count'] + 
                                         country_nan_stats['INF_B_NaN_Count'] + 
                                         country_nan_stats['INF_ALL_NaN_Count'])

# Calculate percentage of NaNs
country_nan_stats['NaN_Percentage'] = (country_nan_stats['Total_NaN_Count'] / 
                                       (country_nan_stats['Total_Records'] * 3) * 100).round(2)

# Calculate data quality score (100 - NaN percentage)
country_nan_stats['Data_Quality_Score'] = (100 - country_nan_stats['NaN_Percentage']).round(2)

# Sort by total NaN count (ascending = better data quality)
country_nan_stats_sorted = country_nan_stats.sort_values('Total_NaN_Count', ascending=True).reset_index(drop=True)
country_nan_stats_sorted['Rank'] = range(1, len(country_nan_stats_sorted) + 1)

# Reorder columns for better readability
country_nan_stats_sorted = country_nan_stats_sorted[[
    'Rank', 'COUNTRY_CODE', 'COUNTRY_AREA_TERRITORY', 'Total_Records',
    'INF_A_NaN_Count', 'INF_B_NaN_Count', 'INF_ALL_NaN_Count', 'Total_NaN_Count',
    'NaN_Percentage', 'Data_Quality_Score'
]]

print(f"   Analyzed {len(country_nan_stats_sorted)} countries")

# Display top 10 countries with best data quality (lowest NaN count)
print("\n5. Top 10 Countries with Best Data Quality (Lowest NaN Count):")
print("-" * 80)
for idx, row in country_nan_stats_sorted.head(10).iterrows():
    print(f"   {int(row['Rank']):3d}. {row['COUNTRY_AREA_TERRITORY']:30s} | "
          f"Records: {int(row['Total_Records']):5d} | "
          f"NaNs: {int(row['Total_NaN_Count']):5d} | "
          f"Quality: {row['Data_Quality_Score']:6.2f}%")

# Display bottom 10 countries with worst data quality (highest NaN count)
print("\n6. Bottom 10 Countries with Worst Data Quality (Highest NaN Count):")
print("-" * 80)
for idx, row in country_nan_stats_sorted.tail(10).iterrows():
    print(f"   {int(row['Rank']):3d}. {row['COUNTRY_AREA_TERRITORY']:30s} | "
          f"Records: {int(row['Total_Records']):5d} | "
          f"NaNs: {int(row['Total_NaN_Count']):5d} | "
          f"Quality: {row['Data_Quality_Score']:6.2f}%")

# Save cleaned data
print("\n7. Saving cleaned data...")
try:
    df_clean.to_csv(OUTPUT_CLEAN_DATA, index=False)
    print(f"   Cleaned data saved to: {OUTPUT_CLEAN_DATA}")
except Exception as e:
    print(f"   Error saving cleaned data: {e}")

# Save country ranking
print("\n8. Saving country ranking...")
try:
    country_nan_stats_sorted.to_csv(OUTPUT_COUNTRY_RANKING, index=False)
    print(f"   Country ranking saved to: {OUTPUT_COUNTRY_RANKING}")
except Exception as e:
    print(f"   Error saving country ranking: {e}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total countries analyzed: {len(country_nan_stats_sorted)}")
print(f"\nData Quality Distribution:")
print(f"  - Excellent (>90% complete): {len(country_nan_stats_sorted[country_nan_stats_sorted['Data_Quality_Score'] > 90])} countries")
print(f"  - Good (75-90% complete):    {len(country_nan_stats_sorted[(country_nan_stats_sorted['Data_Quality_Score'] > 75) & (country_nan_stats_sorted['Data_Quality_Score'] <= 90)])} countries")
print(f"  - Fair (50-75% complete):    {len(country_nan_stats_sorted[(country_nan_stats_sorted['Data_Quality_Score'] > 50) & (country_nan_stats_sorted['Data_Quality_Score'] <= 75)])} countries")
print(f"  - Poor (<50% complete):      {len(country_nan_stats_sorted[country_nan_stats_sorted['Data_Quality_Score'] <= 50])} countries")

print(f"\nAverage Data Quality Score: {country_nan_stats_sorted['Data_Quality_Score'].mean():.2f}%")
print(f"Median Data Quality Score: {country_nan_stats_sorted['Data_Quality_Score'].median():.2f}%")

print("\n" + "=" * 80)
print("Processing complete!")
print("=" * 80)
