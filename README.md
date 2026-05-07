# The Big Bang Theory and the Price of Popularity
### Did Commercial Success Cost the Show Its Scientific Soul?

A data-driven investigation into whether The Big Bang Theory's extraordinary 
rise to mainstream popularity came at the cost of its scientific identity — 
analysing 50,137 lines of dialogue across 10 seasons (2007–2017).

**Read the full blog post here:** [HackMD Article](https://hackmd.io/@CZKBF_vDREWssAhnAk7W_g/Sy3Y3vYAZe)

---

## Project Overview

This project uses three data sources — scraped TBBT transcripts, IMDb episode 
ratings and Google Trends search data — to measure how the show's scientific 
language density changed as its mainstream popularity grew. The analysis includes 
scientific language scoring, vocabulary complexity measurement, character-level 
breakdown and regression modelling.

**Key findings:**
- Scientific language density declined by nearly 70% from Season 1 to Season 9
- The decline is statistically significant (R² = 0.597, p = 0.009)
- Episodes with more scientific content scored higher on IMDb (p = 0.028)
- Sheldon Cooper's scientific language declined 38.3% — the largest absolute drop
- Penny's scientific language declined 46.5% — the largest percentage drop

---

## Repository Structure
tbbt-popularity-analysis/
├── README.md                          # This file
├── blog.ipynb                         # Main blog post (Jupyter Notebook)
├── data/
│   ├── raw/                           # Original unmodified data files
│   │   ├── tbbt_raw_dialogue.csv      # Scraped TBBT transcripts S1-10
│   │   ├── tbbt_imdb_ratings.csv      # IMDb episode ratings S1-10
│   │   └── tbbt_ww_trends.csv         # Google Trends worldwide data
│   └── processed/                     # Cleaned and merged data files
│       ├── tbbt_scripts_clean.csv
│       ├── tbbt_imdb_clean.csv
│       ├── tbbt_trends_by_season.csv
│       ├── tbbt_master_final.csv
│       ├── tbbt_master_by_episode.csv
│       ├── tbbt_scored_by_episode.csv
│       ├── tbbt_scored_by_season.csv
│       ├── tbbt_character_comparison.csv
│       ├── tbbt_master_episodes_final.csv
│       └── tbbt_regression_results.txt
├── scripts/
│   ├── scrape_tbbt_transcripts.py     # Scrapes dialogue from bigbangtrans.wordpress.com
│   ├── scrape_imdb_ratings.py         # Extracts ratings from IMDb official datasets
│   ├── clean_and_merge_data.py        # Cleans and merges all three datasets
│   ├── scientific_language_analysis.py # Scientific language scoring and analysis
│   └── visualisations_and_regression.py # Generates all charts and regression output
└── charts/                            # All visualisation outputs
├── 01_summary_table.png
├── 02_sci_density_by_season.png
├── 03_sci_vs_trends.png
├── 04_imdb_ratings.png
├── 05_regression_scatter.png
└── 06_character_breakdown.png


## Data Sources

### 1. TBBT Scripts
- **Source:** Scraped from [bigbangtrans.wordpress.com](https://bigbangtrans.wordpress.com)
- **Method:** Python BeautifulSoup scraper (`scripts/scrape_tbbt_transcripts.py`)
- **Coverage:** Seasons 1–10, 207 episodes, 50,137 lines of dialogue
- **File:** `data/raw/tbbt_raw_dialogue.csv`
- **Columns:** season, episode, character, dialogue

### 2. IMDb Episode Ratings
- **Source:** Official IMDb datasets at [datasets.imdbws.com](https://datasets.imdbws.com)
- **Method:** Downloaded `title.episode.tsv` and `title.ratings.tsv`, extracted using Python (`scripts/scrape_imdb_ratings.py`)
- **Coverage:** 231 episodes across seasons 1–10
- **File:** `data/raw/tbbt_imdb_ratings.csv`
- **Columns:** season, episode, imdb_rating, vote_count
- **Note:** The raw IMDb source files (title.episode.tsv and title.ratings.tsv) exceed GitHub's 100MB file size limit and are not included in this repository. They can be downloaded directly from datasets.imdbws.com.

### 3. Google Trends
- **Source:** [trends.google.com](https://trends.google.com)
- **Search term:** "Big Bang Theory", Worldwide, September 2007 – May 2017
- **Method:** Direct CSV download from Google Trends
- **File:** `data/raw/tbbt_ww_trends.csv`
- **Columns:** time, search_interest (0–100 relative scale)

---

## How to Replicate

### Requirements
- Python 3.8 or above
- pip or conda package manager

### Step 1 — Install dependencies

pip install pandas numpy matplotlib seaborn requests beautifulsoup4 statsmodels jupyter spacy
python -m spacy download en_core_web_sm


### Step 2 — Download IMDb source files
Download the following two files from [datasets.imdbws.com](https://datasets.imdbws.com) and save them to `data/raw/`:
- `title.episode.tsv`
- `title.ratings.tsv`

### Step 3 — Run the pipeline in order

python scripts/scrape_tbbt_transcripts.py
python scripts/scrape_imdb_ratings.py
python scripts/clean_and_merge_data.py
python scripts/scientific_language_analysis.py
python scripts/visualisations_and_regression.py


### Step 4 — View the blog post
```bash
jupyter notebook blog.ipynb
```

---

## Methodology

**Scientific Language Scoring:** Every line of dialogue was scored against a 
curated vocabulary of 214 scientific terms spanning physics, chemistry, biology, 
computer science and astronomy. Scientific density is calculated as scientific 
terms per 100 words, normalised by episode length.

**Vocabulary Complexity:** Measured using type-token ratio (unique words divided 
by total words) per episode.

**Regression Models:**
- Scientific Density ~ Season Number (R² = 0.597, p = 0.009)
- Scientific Density ~ Google Trends Interest (R² = 0.327, p = 0.084)
- IMDb Rating ~ Scientific Density (R² = 0.021, p = 0.028)

**Character Analysis:** Scientific language density compared for Sheldon, Leonard, 
Howard, Raj and Penny between early seasons (S1–S3) and late seasons (S8–S10).

---

## Tools Used
- **Python** — pandas, numpy, matplotlib, seaborn, statsmodels, spacy
- **BeautifulSoup** — web scraping TBBT transcripts
- **Git/GitHub** — version control and project hosting
- **Jupyter Notebook** — blog presentation
- **HackMD** — formatted online article publication
- **Google Trends** — commercial success proxy data

