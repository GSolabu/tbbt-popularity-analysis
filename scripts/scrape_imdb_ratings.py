import pandas as pd
import gzip
import shutil
import os

# ============================================================
# EXTRACT TBBT EPISODE RATINGS FROM IMDB OFFICIAL DATASETS
# Source: datasets.imdbws.com (officially provided by IMDb)
# Files needed:
#   - data/raw/title.episode.tsv.gz  (episode/season structure)
#   - data/raw/title.ratings.tsv.gz  (ratings and vote counts)
#
# TBBT IMDb ID: tt0898266
# We extract all episodes for seasons 1-10 and merge with ratings
# ============================================================

TBBT_IMDB_ID = "tt0898266"


# STEP 1: DECOMPRESS AND LOAD THE EPISODE DATASET
# ============================================================

print("Loading IMDb episode dataset...")
print("(This file is large — may take 30-60 seconds to load)")

imdb_all_episodes = pd.read_csv(
    "data/raw/title.episode.tsv",
    sep="\t",
    na_values="\\N",
    low_memory=False
)

print(f"Total episodes in IMDb database: {len(imdb_all_episodes):,}")
print(f"Columns: {imdb_all_episodes.columns.tolist()}")
print(f"\nSample rows:")
print(imdb_all_episodes.head(3))


# STEP 2: FILTER TO TBBT EPISODES ONLY
# ============================================================

print(f"\nFiltering to TBBT episodes (parent ID: {TBBT_IMDB_ID})...")

tbbt_episodes_only = imdb_all_episodes[
    imdb_all_episodes["parentTconst"] == TBBT_IMDB_ID
].copy()

print(f"TBBT episodes found: {len(tbbt_episodes_only)}")

# Clean up season and episode numbers
tbbt_episodes_only["seasonNumber"] = pd.to_numeric(
    tbbt_episodes_only["seasonNumber"], errors="coerce"
)
tbbt_episodes_only["episodeNumber"] = pd.to_numeric(
    tbbt_episodes_only["episodeNumber"], errors="coerce"
)

# Drop rows with missing season or episode numbers
tbbt_episodes_only = tbbt_episodes_only.dropna(
    subset=["seasonNumber", "episodeNumber"]
)

# Convert to integers
tbbt_episodes_only["seasonNumber"] = tbbt_episodes_only["seasonNumber"].astype(int)
tbbt_episodes_only["episodeNumber"] = tbbt_episodes_only["episodeNumber"].astype(int)

# Filter to seasons 1-10 only (matching our transcript data)
tbbt_episodes_s1_s10 = tbbt_episodes_only[
    tbbt_episodes_only["seasonNumber"] <= 10
].copy()

print(f"TBBT episodes in seasons 1-10: {len(tbbt_episodes_s1_s10)}")
print(f"\nEpisodes per season:")
print(tbbt_episodes_s1_s10.groupby("seasonNumber")["episodeNumber"].count())


# STEP 3: LOAD THE RATINGS DATASET
# ============================================================

print("\nLoading IMDb ratings dataset...")

imdb_all_ratings = pd.read_csv(
    "data/raw/title.ratings.tsv",
    sep="\t",
    na_values="\\N",
    low_memory=False
)


print(f"Total titles with ratings: {len(imdb_all_ratings):,}")
print(f"Columns: {imdb_all_ratings.columns.tolist()}")


# STEP 4: MERGE EPISODE LIST WITH RATINGS
# ============================================================

print("\nMerging episode data with ratings...")

# The episode dataset has a "tconst" column which is the 
# individual episode's IMDb ID — we use this to join with ratings
tbbt_imdb_ratings = tbbt_episodes_s1_s10.merge(
    imdb_all_ratings,
    on="tconst",
    how="left"
)

# Rename columns to be more readable
tbbt_imdb_ratings = tbbt_imdb_ratings.rename(columns={
    "tconst": "episode_imdb_id",
    "parentTconst": "show_imdb_id",
    "seasonNumber": "season",
    "episodeNumber": "episode",
    "averageRating": "imdb_rating",
    "numVotes": "vote_count"
})

# Keep only the columns we need
tbbt_imdb_ratings = tbbt_imdb_ratings[[
    "season",
    "episode",
    "episode_imdb_id",
    "imdb_rating",
    "vote_count"
]].sort_values(["season", "episode"]).reset_index(drop=True)


# STEP 5: VERIFY AND PREVIEW RESULTS
# ============================================================

print(f"\n{'='*50}")
print(f"IMDb data extraction complete!")
print(f"Total episodes: {len(tbbt_imdb_ratings)}")
print(f"\nEpisodes per season:")
print(tbbt_imdb_ratings.groupby("season")["episode"].count())
print(f"\nAverage IMDb rating per season:")
print(tbbt_imdb_ratings.groupby("season")["imdb_rating"].mean().round(2))
print(f"\nAverage vote count per season:")
print(tbbt_imdb_ratings.groupby("season")["vote_count"].mean().round(0))
print(f"\nPreview of first 10 rows:")
print(tbbt_imdb_ratings.head(10))

# Check for any missing ratings
missing_ratings = tbbt_imdb_ratings["imdb_rating"].isna().sum()
if missing_ratings > 0:
    print(f"\nWARNING: {missing_ratings} episodes have no rating")
    print(tbbt_imdb_ratings[tbbt_imdb_ratings["imdb_rating"].isna()])


# STEP 6: SAVE TO CSV

tbbt_imdb_ratings.to_csv("data/raw/tbbt_imdb_ratings.csv", index=False)
print(f"\nSaved to data/raw/tbbt_imdb_ratings.csv")


