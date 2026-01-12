import pandas as pd
import numpy as np

# -----------------------------
# 0. Config
# -----------------------------

# Station pipe separated values file is not included in the repository due to size. Please refer to NOAA website to download station files
# Station number is in the PSV file name though, so use that if reproducing results
file_path = "./GHCNh_QAI0000OTBD_por.psv"

# Only keep columns we actually use
usecols = [
    # Time/meta
    "Station_ID", "Station_name", "Year", "Month", "Day", "Hour", "Minute",
    "Latitude", "Longitude", "Elevation",
    # Core weather
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "sea_level_pressure",
    "station_level_pressure",
    # Optional weather
    "visibility",
    "wind_gust",
    "wet_bulb_temperature",
    "altimeter",
    "pressure_3hr_change",
    # Optional precip windows
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_15_hour",
    "precipitation_18_hour",
    "precipitation_21_hour",
    "precipitation_24_hour",
    # Optional cloud cover
    "sky_cover_1",
    "sky_cover_2",
    "sky_cover_3",
]

# -----------------------------
# 1. Read in chunks, keep last ~25 years (Year >= 2000), files are too big otherwise
# -----------------------------
chunks = []
chunksize = 200_000  # tweak according to memory needs 

for chunk in pd.read_csv(
    file_path,
    sep="|",
    usecols=lambda c: c in usecols or c in ["Year", "Month", "Day", "Hour", "Minute"],
    low_memory=False,
    chunksize=chunksize,
):
    # Keep only recent years to reduce memory
    chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
    chunk = chunk[chunk["Year"] >= 2000]

    if not chunk.empty:
        chunks.append(chunk)

if not chunks:
    raise ValueError("No rows found for Year >= 2000. Check the data or the year filter.")

df = pd.concat(chunks, ignore_index=True)

# -----------------------------
# 2. Build datetime
# -----------------------------
for col in ["Month", "Day", "Hour", "Minute"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["datetime"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=df["Day"],
        hour=df["Hour"],
        minute=df["Minute"],
    ),
    errors="coerce",
)

# 1. Drop rows with unparseable datetime only
df = df[~df["datetime"].isna()].sort_values("datetime")

# 2. Build complete hourly index
full_index = pd.date_range(start=df["datetime"].min(),
                           end=df["datetime"].max(),
                           freq="H")

# 3. Reindex → missing hours become NA rows (essential for continuity)
df = df.set_index("datetime").reindex(full_index)

# Rename back to datetime index name
df.index.name = "datetime"

# -----------------------------
# 3. Select only useful columns that actually exist
# -----------------------------
core_and_optional = usecols
keep_cols = [c for c in core_and_optional if c in df.columns]


df = df[keep_cols].copy()

# -----------------------------
# 4. Convert units
# -----------------------------
for col in ["temperature", "dew_point_temperature", "wet_bulb_temperature"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0  # tenths °C → °C

for col in [
    "precipitation",
    "precipitation_3_hour", "precipitation_6_hour",
    "precipitation_9_hour", "precipitation_12_hour",
    "precipitation_15_hour", "precipitation_18_hour",
    "precipitation_21_hour", "precipitation_24_hour",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in [
    "relative_humidity",
    "wind_speed", "wind_gust",
    "sea_level_pressure", "station_level_pressure",
    "visibility",
    "altimeter",
    "pressure_3hr_change",
    "wind_direction",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 5. Aggregate hourly → weekly
# -----------------------------


agg_dict = {}

# Mean-type variables
for col in [
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "sea_level_pressure",
    "station_level_pressure",
    "visibility",
    "wind_gust",
    "wet_bulb_temperature",
    "altimeter",
    "pressure_3hr_change",
]:
    if col in df.columns:
        agg_dict[col] = "mean"

# Sum-type variables
for col in [
    "precipitation",
    "precipitation_3_hour", "precipitation_6_hour",
    "precipitation_9_hour", "precipitation_12_hour",
    "precipitation_15_hour", "precipitation_18_hour",
    "precipitation_21_hour", "precipitation_24_hour",
]:
    if col in df.columns:
        agg_dict[col] = "sum"

# Station metadata: first of each week
for col in ["Station_ID", "Station_name", "Latitude", "Longitude", "Elevation"]:
    if col in df.columns:
        agg_dict[col] = "first"

weekly = df.resample("W-MON").agg(agg_dict).reset_index()
weekly = weekly.rename(columns={"datetime": "iso_weekstartdate"})

# -----------------------------
# 6. ISO year-week & NaN summary
# -----------------------------
iso = weekly["iso_weekstartdate"].dt.isocalendar()
weekly["iso_year"] = iso["year"]
weekly["iso_week"] = iso["week"]
weekly["isoyw"] = weekly["iso_year"] * 100 + weekly["iso_week"]

cols_order = (
    ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"] +
    [c for c in weekly.columns if c not in ["iso_weekstartdate", "iso_year", "iso_week", "isoyw"]]
)
weekly = weekly[cols_order]

print("Weekly head:")
print(weekly.head())

nan_counts = weekly.isna().sum()
nan_frac = weekly.isna().mean()

nan_summary = pd.DataFrame({
    "NaN_count": nan_counts,
    "NaN_fraction": nan_frac
}).sort_values("NaN_fraction", ascending=False)

print("\nNaN summary per column:")
print(nan_summary)

weekly.to_csv("Qatar_weather_weekly_last25yrs.csv", index=False)
nan_summary.to_csv("Qatar_weather_weekly_last25yrs_nan_summary.csv")
