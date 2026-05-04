import pandas as pd
import numpy as np

# ============================================================
# CLEAN AND MERGE ALL THREE DATASETS INTO ONE MASTER DATASET
# Inputs:
#   - data/raw/tbbt_raw_dialogue.csv
#   - data/raw/tbbt_imdb_ratings.csv
#   - data/raw/tbbt_ww_trends.csv
# Outputs:
#   - data/processed/tbbt_scripts_clean.csv
#   - data/processed/tbbt_imdb_clean.csv
#   - data/processed/tbbt_trends_clean.csv
#   - data/processed/tbbt_master_by_season.csv
#   - data/processed/tbbt_master_by_episode.csv
# ============================================================

# STEP 1: LOAD ALL THREE RAW DATASETS
# ============================================================

print("Loading raw datasets...")

tbbt_raw_scripts = pd.read_csv("data/raw/tbbt_raw_dialogue.csv")
tbbt_raw_imdb = pd.read_csv("data/raw/tbbt_imdb_ratings.csv")
tbbt_raw_trends = pd.read_csv("data/raw/tbbt_ww_trends.csv")

print(f"Scripts: {tbbt_raw_scripts.shape[0]:,} rows")
print(f"IMDb ratings: {tbbt_raw_imdb.shape[0]} rows")
print(f"Google Trends: {tbbt_raw_trends.shape[0]} rows")


# STEP 2: CLEAN THE SCRIPTS DATASET
# ============================================================

print("\nCleaning scripts dataset...")

tbbt_scripts_clean = tbbt_raw_scripts.copy()

# Sort by season and episode so data is in correct order
tbbt_scripts_clean = tbbt_scripts_clean.sort_values(
    ["season", "episode"]
).reset_index(drop=True)

# Verify season and episode are integers
tbbt_scripts_clean["season"] = pd.to_numeric(
    tbbt_scripts_clean["season"], errors="coerce"
).astype("Int64")
tbbt_scripts_clean["episode"] = pd.to_numeric(
    tbbt_scripts_clean["episode"], errors="coerce"
).astype("Int64")

# Drop any rows with missing values
tbbt_scripts_clean = tbbt_scripts_clean.dropna(
    subset=["season", "episode", "character", "dialogue"]
)

# Ensure character names are uppercase and stripped of whitespace
tbbt_scripts_clean["character"] = (
    tbbt_scripts_clean["character"].str.upper().str.strip()
)

# Ensure dialogue is string and stripped of whitespace
tbbt_scripts_clean["dialogue"] = (
    tbbt_scripts_clean["dialogue"].astype(str).str.strip()
)

# Remove any dialogue lines that are suspiciously short
# (likely scraping artifacts rather than real dialogue)
tbbt_scripts_clean = tbbt_scripts_clean[
    tbbt_scripts_clean["dialogue"].str.len() > 2
]

# Only keep seasons 1-10
tbbt_scripts_clean = tbbt_scripts_clean[
    tbbt_scripts_clean["season"] <= 10
]

print(f"Scripts after cleaning: {len(tbbt_scripts_clean):,} rows")
print(f"Seasons covered: {sorted(tbbt_scripts_clean['season'].unique().tolist())}")
print(f"Top 10 characters:")
print(tbbt_scripts_clean["character"].value_counts().head(10))

# STEP 3: CLEAN THE IMDB DATASET
# ============================================================

print("\nCleaning IMDb dataset...")

tbbt_imdb_clean = tbbt_raw_imdb.copy()

# Drop the episode_imdb_id column — not needed for analysis
tbbt_imdb_clean = tbbt_imdb_clean.drop(columns=["episode_imdb_id"])

# Verify data types
tbbt_imdb_clean["season"] = pd.to_numeric(
    tbbt_imdb_clean["season"], errors="coerce"
).astype("Int64")
tbbt_imdb_clean["episode"] = pd.to_numeric(
    tbbt_imdb_clean["episode"], errors="coerce"
).astype("Int64")
tbbt_imdb_clean["imdb_rating"] = pd.to_numeric(
    tbbt_imdb_clean["imdb_rating"], errors="coerce"
)
tbbt_imdb_clean["vote_count"] = pd.to_numeric(
    tbbt_imdb_clean["vote_count"], errors="coerce"
)

# Drop any rows with missing ratings
missing_ratings = tbbt_imdb_clean["imdb_rating"].isna().sum()
if missing_ratings > 0:
    print(f"  Dropping {missing_ratings} episodes with missing ratings")
    tbbt_imdb_clean = tbbt_imdb_clean.dropna(subset=["imdb_rating"])

# Sort by season and episode
tbbt_imdb_clean = tbbt_imdb_clean.sort_values(
    ["season", "episode"]
).reset_index(drop=True)

print(f"IMDb data after cleaning: {len(tbbt_imdb_clean)} rows")
print(f"Rating range: {tbbt_imdb_clean['imdb_rating'].min()} - {tbbt_imdb_clean['imdb_rating'].max()}")

# Aggregate to season level
tbbt_imdb_by_season = tbbt_imdb_clean.groupby("season").agg(
    avg_imdb_rating=("imdb_rating", "mean"),
    min_imdb_rating=("imdb_rating", "min"),
    max_imdb_rating=("imdb_rating", "max"),
    avg_vote_count=("vote_count", "mean"),
    total_votes=("vote_count", "sum")
).reset_index()

tbbt_imdb_by_season["avg_imdb_rating"] = tbbt_imdb_by_season["avg_imdb_rating"].round(2)
tbbt_imdb_by_season["avg_vote_count"] = tbbt_imdb_by_season["avg_vote_count"].round(0)

print(f"\nIMDb by season:")
print(tbbt_imdb_by_season[["season", "avg_imdb_rating", "avg_vote_count"]])


# STEP 4: CLEAN THE GOOGLE TRENDS DATASET
# ============================================================

print("\nCleaning Google Trends dataset...")

tbbt_trends_clean = tbbt_raw_trends.copy()

# Rename columns to clean names
tbbt_trends_clean.columns = ["date", "search_interest"]

# Convert date to datetime
tbbt_trends_clean["date"] = pd.to_datetime(tbbt_trends_clean["date"])

# Remove any rows where search interest is not numeric
tbbt_trends_clean["search_interest"] = pd.to_numeric(
    tbbt_trends_clean["search_interest"], errors="coerce"
)
tbbt_trends_clean = tbbt_trends_clean.dropna(subset=["search_interest"])

# Map each monthly date to a season number based on air dates
# Season air date ranges (approximate — first of month boundaries)
season_air_dates = {
    1:  ("2007-09-01", "2008-05-31"),
    2:  ("2008-09-01", "2009-05-31"),
    3:  ("2009-09-01", "2010-05-31"),
    4:  ("2010-09-01", "2011-05-31"),
    5:  ("2011-09-01", "2012-05-31"),
    6:  ("2012-09-01", "2013-05-31"),
    7:  ("2013-09-01", "2014-05-31"),
    8:  ("2014-09-01", "2015-05-31"),
    9:  ("2015-09-01", "2016-05-31"),
    10: ("2016-09-01", "2017-05-31")
}

def map_date_to_season(date):
    """Map a date to a TBBT season number based on air date ranges"""
    for season_num, (start_date, end_date) in season_air_dates.items():
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            return season_num
    return None

tbbt_trends_clean["season"] = tbbt_trends_clean["date"].apply(
    map_date_to_season
)

# Drop dates that don't fall within any season
# (summer breaks between seasons)
tbbt_trends_with_season = tbbt_trends_clean.dropna(subset=["season"])
tbbt_trends_with_season = tbbt_trends_with_season.copy()
tbbt_trends_with_season["season"] = tbbt_trends_with_season["season"].astype(int)

# Average search interest per season
tbbt_trends_by_season = tbbt_trends_with_season.groupby("season").agg(
    avg_search_interest=("search_interest", "mean"),
    peak_search_interest=("search_interest", "max")
).reset_index()

tbbt_trends_by_season["avg_search_interest"] = (
    tbbt_trends_by_season["avg_search_interest"].round(1)
)

print(f"Google Trends by season:")
print(tbbt_trends_by_season)


# STEP 5: CREATE MASTER SEASON-LEVEL DATASET
# ============================================================

print("\nCreating master season-level dataset...")

# Merge IMDb and Google Trends at season level
tbbt_master_by_season = tbbt_imdb_by_season.merge(
    tbbt_trends_by_season,
    on="season",
    how="inner"
)

print(f"Master dataset shape: {tbbt_master_by_season.shape}")
print(f"\nMaster dataset preview:")
print(tbbt_master_by_season)


# STEP 6: CREATE EPISODE-LEVEL DATASET
# (scripts merged with IMDb ratings per episode)
# ============================================================

print("\nCreating episode-level dataset...")

# Aggregate scripts to episode level for word count
tbbt_scripts_by_episode = tbbt_scripts_clean.groupby(
    ["season", "episode"]
).agg(
    total_lines=("dialogue", "count"),
    total_words=("dialogue", lambda x: x.str.split().str.len().sum()),
    unique_characters=("character", "nunique")
).reset_index()

# Merge with IMDb episode ratings
tbbt_master_by_episode = tbbt_scripts_by_episode.merge(
    tbbt_imdb_clean,
    on=["season", "episode"],
    how="inner"
)

print(f"Episode-level dataset shape: {tbbt_master_by_episode.shape}")
print(f"\nEpisode-level preview:")
print(tbbt_master_by_episode.head(5))


# STEP 7: SAVE ALL CLEANED DATASETS
# ============================================================

print("\nSaving cleaned datasets...")

tbbt_scripts_clean.to_csv(
    "data/processed/tbbt_scripts_clean.csv", index=False
)
tbbt_imdb_clean.to_csv(
    "data/processed/tbbt_imdb_clean.csv", index=False
)
tbbt_trends_by_season.to_csv(
    "data/processed/tbbt_trends_by_season.csv", index=False
)
tbbt_master_by_season.to_csv(
    "data/processed/tbbt_master_by_season.csv", index=False
)
tbbt_master_by_episode.to_csv(
    "data/processed/tbbt_master_by_episode.csv", index=False
)

print("All cleaned datasets saved to data/processed/")
print("\nFiles saved:")
print("  - tbbt_scripts_clean.csv")
print("  - tbbt_imdb_clean.csv")
print("  - tbbt_trends_by_season.csv")
print("  - tbbt_master_by_season.csv")
print("  - tbbt_master_by_episode.csv")