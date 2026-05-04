import pandas as pd
import numpy as np
import re

# ============================================================
# SCIENTIFIC LANGUAGE ANALYSIS
# Scores every episode of TBBT (S1-10) for scientific language
# density and vocabulary complexity, then aggregates by season
# and character for use in visualisations and regression
#
# Inputs:
#   - data/processed/tbbt_scripts_clean.csv
#   - data/processed/tbbt_master_by_season.csv
#   - data/processed/tbbt_master_by_episode.csv
#
# Outputs:
#   - data/processed/tbbt_scored_by_episode.csv
#   - data/processed/tbbt_scored_by_season.csv
#   - data/processed/tbbt_character_comparison.csv
#   - data/processed/tbbt_master_final.csv
# ============================================================

# ============================================================
# STEP 1: LOAD CLEANED SCRIPTS
# ============================================================

print("Loading cleaned scripts...")
tbbt_scripts = pd.read_csv("data/processed/tbbt_scripts_clean.csv")
tbbt_master_by_season = pd.read_csv("data/processed/tbbt_master_by_season.csv")
tbbt_master_by_episode = pd.read_csv("data/processed/tbbt_master_by_episode.csv")

print(f"Scripts loaded: {len(tbbt_scripts):,} lines of dialogue")

# ============================================================
# STEP 2: DEFINE SCIENTIFIC VOCABULARY
# A curated list of physics, chemistry, computer science and
# general academic terms relevant to TBBT's core characters
# ============================================================

tbbt_scientific_vocabulary = set([
    # Physics — Sheldon and Leonard's domain
    "quantum", "physics", "relativity", "entropy", "photon", "electron",
    "proton", "neutron", "atom", "atomic", "molecule", "molecular",
    "gravity", "gravitational", "velocity", "acceleration", "momentum",
    "trajectory", "string theory", "higgs", "boson", "quark", "lepton",
    "fermion", "dark matter", "dark energy", "supernova", "nebula",
    "galaxy", "cosmology", "astrophysics", "thermodynamics",
    "electromagnetic", "radiation", "wavelength", "frequency", "amplitude",
    "nuclear", "fusion", "fission", "schrodinger", "heisenberg",
    "uncertainty", "eigenvalue", "topology", "tensor", "laser",
    "neutrino", "antimatter", "spacetime", "wormhole", "singularity",
    "blackhole", "black hole", "superconductor", "semiconductor",
    "magnetic", "magnetism", "electromagnetism", "kinetic", "potential",
    "thermal", "quantum mechanics", "particle", "subatomic", "plasma",
    "doppler", "vacuum", "inertia", "centrifugal", "centripetal",
    "coefficient", "derivative", "integral", "calculus", "theorem",
    "equation", "formula", "variable", "constant", "infinity",
    "dimension", "matrix", "vector", "scalar", "tensor",

    # Chemistry
    "chemical", "chemistry", "element", "compound", "reaction",
    "catalyst", "periodic", "hydrogen", "carbon", "nitrogen", "oxygen",
    "polymer", "protein", "enzyme", "molecule", "isotope", "valence",
    "covalent", "ionic", "oxidation", "reduction", "titration",

    # Biology and neuroscience — Amy's domain
    "neuroscience", "neuron", "synapse", "cortex", "cerebral",
    "biology", "biological", "evolution", "genetics", "genome",
    "mutation", "chromosome", "dna", "rna", "species", "organism",
    "hippocampus", "amygdala", "prefrontal", "oxytocin", "serotonin",
    "dopamine", "primate", "mammal", "vertebrate", "cortical",

    # Engineering and computer science — Howard and Raj's domain
    "engineering", "mechanical", "aerospace", "algorithm", "binary",
    "software", "hardware", "processor", "bandwidth", "encryption",
    "neural", "simulation", "telescope", "satellite", "orbit",
    "trajectory", "thruster", "payload", "microwave", "antenna",
    "robot", "robotic", "circuit", "voltage", "current", "resistance",
    "transistor", "microchip", "bandwidth", "latency", "frequency",

    # Astronomy — Raj's domain
    "astronomy", "astrophysics", "telescope", "observatory", "cosmos",
    "celestial", "stellar", "solar", "lunar", "planetary", "asteroid",
    "comet", "nebula", "pulsar", "quasar", "redshift", "infrared",
    "ultraviolet", "spectrometer", "spectroscopy", "magnitude",

    # General academic and scientific language
    "hypothesis", "theory", "experiment", "laboratory", "research",
    "empirical", "analysis", "data", "statistical", "correlation",
    "peer review", "publication", "journal", "doctorate", "phd",
    "professor", "caltech", "nasa", "nobel", "scientific", "scientist",
    "theoretical", "theoretic", "methodology", "paradigm", "axiom",
    "postulate", "conjecture", "inference", "deduction", "induction",
    "peer reviewed", "citation", "abstract", "dissertation", "thesis"
])

print(f"Scientific vocabulary size: {len(tbbt_scientific_vocabulary)} terms")

# ============================================================
# STEP 3: DEFINE SCORING FUNCTIONS
# ============================================================

def count_scientific_terms(dialogue_text):
    """
    Count occurrences of scientific vocabulary terms in a line of dialogue.
    Uses word boundary matching to avoid partial matches.
    """
    dialogue_lower = str(dialogue_text).lower()
    term_count = 0
    for scientific_term in tbbt_scientific_vocabulary:
        # Use word boundary for single words, substring for phrases
        if " " in scientific_term:
            term_count += dialogue_lower.count(scientific_term)
        else:
            term_count += len(
                re.findall(r'\b' + re.escape(scientific_term) + r'\b',
                           dialogue_lower)
            )
    return term_count

def calculate_word_count(dialogue_text):
    """Count total words in a line of dialogue"""
    return len(str(dialogue_text).split())

def calculate_type_token_ratio(dialogue_text):
    """
    Calculate vocabulary complexity using type-token ratio.
    Type-token ratio = unique words / total words.
    Higher values indicate more varied and complex vocabulary.
    """
    dialogue_words = str(dialogue_text).lower().split()
    if len(dialogue_words) == 0:
        return 0
    return len(set(dialogue_words)) / len(dialogue_words)

# ============================================================
# STEP 4: SCORE EVERY LINE OF DIALOGUE
# ============================================================

print("\nScoring dialogue lines for scientific language...")
print("This may take 1-2 minutes...")

tbbt_scripts["sci_term_count"] = tbbt_scripts["dialogue"].apply(
    count_scientific_terms
)
tbbt_scripts["word_count"] = tbbt_scripts["dialogue"].apply(
    calculate_word_count
)
tbbt_scripts["type_token_ratio"] = tbbt_scripts["dialogue"].apply(
    calculate_type_token_ratio
)

# Scientific density = scientific terms per 100 words
# Adding 1 to word count avoids division by zero
tbbt_scripts["sci_density"] = (
    tbbt_scripts["sci_term_count"] /
    (tbbt_scripts["word_count"] + 1)
) * 100

print("Scoring complete!")
print(f"\nOverall scientific density statistics:")
print(tbbt_scripts["sci_density"].describe().round(3))

# ============================================================
# STEP 5: AGGREGATE SCORES BY EPISODE
# ============================================================

print("\nAggregating scores by episode...")

tbbt_scored_by_episode = tbbt_scripts.groupby(
    ["season", "episode"]
).agg(
    total_dialogue_lines=("dialogue", "count"),
    total_words=("word_count", "sum"),
    total_sci_terms=("sci_term_count", "sum"),
    avg_type_token_ratio=("type_token_ratio", "mean")
).reset_index()

# Calculate scientific density at episode level
tbbt_scored_by_episode["sci_density_per_100_words"] = (
    tbbt_scored_by_episode["total_sci_terms"] /
    (tbbt_scored_by_episode["total_words"] + 1)
) * 100

tbbt_scored_by_episode["sci_density_per_100_words"] = (
    tbbt_scored_by_episode["sci_density_per_100_words"].round(3)
)
tbbt_scored_by_episode["avg_type_token_ratio"] = (
    tbbt_scored_by_episode["avg_type_token_ratio"].round(3)
)

print(f"Episode-level scores shape: {tbbt_scored_by_episode.shape}")
print(f"\nScientific density by season (episode averages):")
print(
    tbbt_scored_by_episode.groupby("season")["sci_density_per_100_words"]
    .mean().round(3)
)

# ============================================================
# STEP 6: AGGREGATE SCORES BY SEASON
# ============================================================

print("\nAggregating scores by season...")

tbbt_scored_by_season = tbbt_scored_by_episode.groupby("season").agg(
    avg_sci_density=("sci_density_per_100_words", "mean"),
    avg_complexity=("avg_type_token_ratio", "mean"),
    total_words=("total_words", "sum"),
    total_sci_terms=("total_sci_terms", "sum")
).reset_index()

tbbt_scored_by_season["avg_sci_density"] = (
    tbbt_scored_by_season["avg_sci_density"].round(3)
)
tbbt_scored_by_season["avg_complexity"] = (
    tbbt_scored_by_season["avg_complexity"].round(3)
)

print(f"\nScientific density by season:")
print(tbbt_scored_by_season[["season", "avg_sci_density", "avg_complexity"]])

# ============================================================
# STEP 7: CHARACTER ANALYSIS
# Compare Sheldon, Leonard, Howard, Raj AND Penny
# between early seasons (S1-3) and late seasons (S8-10)
# ============================================================

print("\nRunning character analysis...")

# Filter to our five focal characters
focal_characters = ["SHELDON", "LEONARD", "HOWARD", "RAJ", "PENNY"]
tbbt_focal_chars = tbbt_scripts[
    tbbt_scripts["character"].isin(focal_characters)
].copy()

# Score by character and season
tbbt_char_by_season = tbbt_focal_chars.groupby(
    ["season", "character"]
).agg(
    total_words=("word_count", "sum"),
    total_sci_terms=("sci_term_count", "sum"),
    total_lines=("dialogue", "count")
).reset_index()

tbbt_char_by_season["sci_density"] = (
    tbbt_char_by_season["total_sci_terms"] /
    (tbbt_char_by_season["total_words"] + 1)
) * 100

tbbt_char_by_season["sci_density"] = (
    tbbt_char_by_season["sci_density"].round(3)
)

# Split into early (S1-3) and late (S8-10) seasons
tbbt_chars_early = tbbt_char_by_season[
    tbbt_char_by_season["season"] <= 3
].groupby("character")["sci_density"].mean().round(3)

tbbt_chars_late = tbbt_char_by_season[
    tbbt_char_by_season["season"] >= 8
].groupby("character")["sci_density"].mean().round(3)

# Combine into comparison dataframe
tbbt_character_comparison = pd.DataFrame({
    "early_seasons_s1_s3": tbbt_chars_early,
    "late_seasons_s8_s10": tbbt_chars_late
}).reset_index()

tbbt_character_comparison["change"] = (
    tbbt_character_comparison["late_seasons_s8_s10"] -
    tbbt_character_comparison["early_seasons_s1_s3"]
).round(3)

tbbt_character_comparison["pct_change"] = (
    (tbbt_character_comparison["change"] /
     tbbt_character_comparison["early_seasons_s1_s3"]) * 100
).round(1)

print(f"\nCharacter scientific density comparison:")
print(tbbt_character_comparison.to_string(index=False))

# ============================================================
# STEP 8: MERGE SCORES INTO MASTER DATASETS
# ============================================================

print("\nMerging scores into master datasets...")

# Merge season scores into master by season
tbbt_master_final = tbbt_master_by_season.merge(
    tbbt_scored_by_season[["season", "avg_sci_density", "avg_complexity"]],
    on="season",
    how="inner"
)

# Merge episode scores into master by episode
tbbt_master_episodes_final = tbbt_master_by_episode.merge(
    tbbt_scored_by_episode[[
        "season", "episode", "sci_density_per_100_words", "avg_type_token_ratio"
    ]],
    on=["season", "episode"],
    how="inner"
)

# Merge IMDb ratings into episode master
tbbt_imdb_clean = pd.read_csv("data/processed/tbbt_imdb_clean.csv")
tbbt_master_episodes_final = tbbt_master_episodes_final.merge(
    tbbt_imdb_clean[["season", "episode", "imdb_rating", "vote_count"]],
    on=["season", "episode"],
    how="left",
    suffixes=("", "_imdb")
)

# Drop duplicate rating column if it exists
if "imdb_rating_imdb" in tbbt_master_episodes_final.columns:
    tbbt_master_episodes_final = tbbt_master_episodes_final.drop(
        columns=["imdb_rating_imdb", "vote_count_imdb"]
    )

print(f"\nFinal master season dataset shape: {tbbt_master_final.shape}")
print(f"Final master episode dataset shape: {tbbt_master_episodes_final.shape}")
print(f"\nFinal master season dataset:")
print(tbbt_master_final[[
    "season", "avg_imdb_rating", "avg_search_interest",
    "avg_sci_density", "avg_complexity"
]])

# ============================================================
# STEP 9: SAVE ALL OUTPUTS
# ============================================================

print("\nSaving all outputs...")

tbbt_scored_by_episode.to_csv(
    "data/processed/tbbt_scored_by_episode.csv", index=False
)
tbbt_scored_by_season.to_csv(
    "data/processed/tbbt_scored_by_season.csv", index=False
)
tbbt_character_comparison.to_csv(
    "data/processed/tbbt_character_comparison.csv", index=False
)
tbbt_master_final.to_csv(
    "data/processed/tbbt_master_final.csv", index=False
)
tbbt_master_episodes_final.to_csv(
    "data/processed/tbbt_master_episodes_final.csv", index=False
)

print("All outputs saved to data/processed/")
print("\nFiles saved:")
print("  - tbbt_scored_by_episode.csv")
print("  - tbbt_scored_by_season.csv")
print("  - tbbt_character_comparison.csv")
print("  - tbbt_master_final.csv")
print("  - tbbt_master_episodes_final.csv")
