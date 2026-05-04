import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# VISUALISATIONS AND REGRESSION ANALYSIS
# for the TBBT Popularity blog post
#
# Inputs:
#   - data/processed/tbbt_master_final.csv
#   - data/processed/tbbt_master_episodes_final.csv
#   - data/processed/tbbt_character_comparison.csv
#   - data/processed/tbbt_char_by_season.csv
#
# Outputs:
#   - charts/01_summary_table.png
#   - charts/02_sci_density_by_season.png
#   - charts/03_sci_vs_trends.png
#   - charts/04_imdb_ratings.png
#   - charts/05_regression_scatter.png
#   - charts/06_character_breakdown.png
#   - data/processed/tbbt_regression_results.txt
# ============================================================

# ============================================================
# STEP 1: LOAD ALL PROCESSED DATASETS
# ============================================================

print("Loading processed datasets...")

tbbt_season_master = pd.read_csv("data/processed/tbbt_master_final.csv")
tbbt_episode_master = pd.read_csv("data/processed/tbbt_master_episodes_final.csv")
tbbt_char_comparison = pd.read_csv("data/processed/tbbt_character_comparison.csv")

print(f"Season master: {tbbt_season_master.shape}")
print(f"Episode master: {tbbt_episode_master.shape}")
print(f"Character comparison: {tbbt_char_comparison.shape}")

# Create charts directory if it doesn't exist
os.makedirs("charts", exist_ok=True)

# ============================================================
# STEP 2: SET CONSISTENT VISUAL STYLE
# ============================================================

# Colour palette — consistent across all charts
COLOUR_SCIENTIFIC = "#2166AC"   # deep blue for scientific density
COLOUR_MAINSTREAM = "#D6604D"   # warm red for google trends/mainstream
COLOUR_RATINGS = "#4DAC26"      # green for IMDb ratings
COLOUR_EARLY = "#2166AC"        # blue for early seasons in bar chart
COLOUR_LATE = "#D6604D"         # red for late seasons in bar chart

# Global style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150
})

print("Visual style configured")

# ============================================================
# CHART 1: SUMMARY STATISTICS TABLE
# ============================================================

print("\nGenerating Chart 1: Summary statistics table...")

summary_table_data = tbbt_season_master[[
    "season",
    "avg_sci_density",
    "avg_search_interest",
    "avg_imdb_rating"
]].copy()

summary_table_data.columns = [
    "Season",
    "Sci. Density\n(per 100 words)",
    "Google Trends\nInterest (0–100)",
    "Avg IMDb\nRating"
]
summary_table_data["Sci. Density\n(per 100 words)"] = (
    summary_table_data["Sci. Density\n(per 100 words)"].round(3)
)
summary_table_data["Google Trends\nInterest (0–100)"] = (
    summary_table_data["Google Trends\nInterest (0–100)"].round(1)
)
summary_table_data["Avg IMDb\nRating"] = (
    summary_table_data["Avg IMDb\nRating"].round(2)
)

fig_table, ax_table = plt.subplots(figsize=(10, 5))
ax_table.axis("off")

season_summary_table = ax_table.table(
    cellText=summary_table_data.values,
    colLabels=summary_table_data.columns,
    cellLoc="center",
    loc="center"
)

season_summary_table.auto_set_font_size(False)
season_summary_table.set_fontsize(10)
season_summary_table.scale(1.3, 2.0)

# Style header row
for col_idx in range(len(summary_table_data.columns)):
    season_summary_table[0, col_idx].set_facecolor(COLOUR_SCIENTIFIC)
    season_summary_table[0, col_idx].set_text_props(
        color="white", fontweight="bold"
    )

# Alternate row shading for readability
for row_idx in range(1, len(summary_table_data) + 1):
    for col_idx in range(len(summary_table_data.columns)):
        if row_idx % 2 == 0:
            season_summary_table[row_idx, col_idx].set_facecolor("#EBF2FB")

ax_table.set_title(
    "Table 1: Key Metrics by Season — The Big Bang Theory (S1–S10)",
    pad=20, fontsize=13, fontweight="bold"
)

plt.savefig("charts/01_summary_table.png")
plt.close()
print("  Chart 1 saved")

# ============================================================
# CHART 2: SCIENTIFIC DENSITY PER SEASON (HEADLINE FINDING)
# ============================================================

print("Generating Chart 2: Scientific density by season...")

fig_sci, ax_sci = plt.subplots(figsize=(11, 5))

ax_sci.plot(
    tbbt_season_master["season"],
    tbbt_season_master["avg_sci_density"],
    color=COLOUR_SCIENTIFIC,
    linewidth=2.5,
    marker="o",
    markersize=8,
    label="Scientific density"
)

# Add trend line
sci_trend_coeffs = np.polyfit(
    tbbt_season_master["season"],
    tbbt_season_master["avg_sci_density"], 1
)
sci_trend_line = np.poly1d(sci_trend_coeffs)
ax_sci.plot(
    tbbt_season_master["season"],
    sci_trend_line(tbbt_season_master["season"]),
    color=COLOUR_SCIENTIFIC,
    linestyle="--",
    linewidth=1.5,
    alpha=0.5,
    label="Overall trend"
)

# Shade early and late eras
ax_sci.axvspan(0.5, 3.5, alpha=0.06, color="green", label="Early era (S1–3)")
ax_sci.axvspan(7.5, 10.5, alpha=0.06, color="red", label="Late era (S8–10)")

# Annotate peak and lowest points
peak_season = tbbt_season_master.loc[
    tbbt_season_master["avg_sci_density"].idxmax(), "season"
]
peak_value = tbbt_season_master["avg_sci_density"].max()
lowest_season = tbbt_season_master.loc[
    tbbt_season_master["avg_sci_density"].idxmin(), "season"
]
lowest_value = tbbt_season_master["avg_sci_density"].min()

ax_sci.annotate(
    f"Peak: {peak_value:.3f}",
    xy=(peak_season, peak_value),
    xytext=(peak_season + 0.5, peak_value + 0.03),
    fontsize=9, color=COLOUR_SCIENTIFIC,
    arrowprops=dict(arrowstyle="->", color=COLOUR_SCIENTIFIC, lw=1)
)
ax_sci.annotate(
    f"Lowest: {lowest_value:.3f}",
    xy=(lowest_season, lowest_value),
    xytext=(lowest_season - 2, lowest_value + 0.05),
    fontsize=9, color=COLOUR_MAINSTREAM,
    arrowprops=dict(arrowstyle="->", color=COLOUR_MAINSTREAM, lw=1)
)

ax_sci.set_xlabel("Season")
ax_sci.set_ylabel("Scientific terms per 100 words")
ax_sci.set_title(
    "Figure 1: Scientific Language Density Across 10 Seasons\n"
    "The Big Bang Theory (2007–2017)"
)
ax_sci.set_xticks(range(1, 11))
ax_sci.legend(frameon=False, loc="upper right")
ax_sci.grid(axis="y", alpha=0.3, linestyle="--")
ax_sci.set_xlim(0.5, 10.5)

plt.savefig("charts/02_sci_density_by_season.png")
plt.close()
print("  Chart 2 saved")

# ============================================================
# CHART 3: SCIENTIFIC DENSITY VS GOOGLE TRENDS (DUAL AXIS)
# ============================================================

print("Generating Chart 3: Scientific density vs Google Trends...")

fig_dual, ax_sci_dual = plt.subplots(figsize=(11, 5))

# Scientific density on left axis
ax_sci_dual.plot(
    tbbt_season_master["season"],
    tbbt_season_master["avg_sci_density"],
    color=COLOUR_SCIENTIFIC,
    linewidth=2.5,
    marker="o",
    markersize=8,
    label="Scientific density"
)
ax_sci_dual.set_xlabel("Season")
ax_sci_dual.set_ylabel("Scientific terms per 100 words", color=COLOUR_SCIENTIFIC)
ax_sci_dual.tick_params(axis="y", labelcolor=COLOUR_SCIENTIFIC)

# Google Trends on right axis
ax_trends_dual = ax_sci_dual.twinx()
ax_trends_dual.plot(
    tbbt_season_master["season"],
    tbbt_season_master["avg_search_interest"],
    color=COLOUR_MAINSTREAM,
    linewidth=2.5,
    marker="s",
    markersize=8,
    linestyle="--",
    label="Google Trends interest"
)
ax_trends_dual.set_ylabel(
    "Google Trends interest (0–100)", color=COLOUR_MAINSTREAM
)
ax_trends_dual.tick_params(axis="y", labelcolor=COLOUR_MAINSTREAM)

ax_sci_dual.set_xticks(range(1, 11))
ax_sci_dual.set_xlim(0.5, 10.5)
ax_sci_dual.grid(axis="y", alpha=0.2, linestyle="--")

# Combined legend
sci_line = plt.Line2D([0], [0], color=COLOUR_SCIENTIFIC, linewidth=2,
                       marker="o", markersize=6, label="Scientific density")
trends_line = plt.Line2D([0], [0], color=COLOUR_MAINSTREAM, linewidth=2,
                          marker="s", markersize=6, linestyle="--",
                          label="Google Trends interest")
ax_sci_dual.legend(
    handles=[sci_line, trends_line], frameon=False, loc="center right"
)

ax_sci_dual.set_title(
    "Figure 2: As Mainstream Interest Rose, Scientific Language Fell\n"
    "Scientific Density vs Google Trends Interest by Season"
)

plt.savefig("charts/03_sci_vs_trends.png")
plt.close()
print("  Chart 3 saved")

# ============================================================
# CHART 4: IMDB RATINGS PER SEASON
# ============================================================

print("Generating Chart 4: IMDb ratings by season...")

fig_ratings, ax_ratings = plt.subplots(figsize=(11, 5))

ax_ratings.plot(
    tbbt_season_master["season"],
    tbbt_season_master["avg_imdb_rating"],
    color=COLOUR_RATINGS,
    linewidth=2.5,
    marker="o",
    markersize=8
)

ax_ratings.fill_between(
    tbbt_season_master["season"],
    tbbt_season_master["avg_imdb_rating"],
    tbbt_season_master["avg_imdb_rating"].min() - 0.1,
    alpha=0.08,
    color=COLOUR_RATINGS
)

# Annotate highest and lowest rated seasons
highest_rated_season = tbbt_season_master.loc[
    tbbt_season_master["avg_imdb_rating"].idxmax(), "season"
]
highest_rating = tbbt_season_master["avg_imdb_rating"].max()
lowest_rated_season = tbbt_season_master.loc[
    tbbt_season_master["avg_imdb_rating"].idxmin(), "season"
]
lowest_rating = tbbt_season_master["avg_imdb_rating"].min()

ax_ratings.annotate(
    f"Highest: {highest_rating:.2f}",
    xy=(highest_rated_season, highest_rating),
    xytext=(highest_rated_season + 0.5, highest_rating + 0.03),
    fontsize=9, color=COLOUR_RATINGS,
    arrowprops=dict(arrowstyle="->", color=COLOUR_RATINGS, lw=1)
)
ax_ratings.annotate(
    f"Lowest: {lowest_rating:.2f}",
    xy=(lowest_rated_season, lowest_rating),
    xytext=(lowest_rated_season - 2.5, lowest_rating + 0.05),
    fontsize=9, color=COLOUR_MAINSTREAM,
    arrowprops=dict(arrowstyle="->", color=COLOUR_MAINSTREAM, lw=1)
)

ax_ratings.set_xlabel("Season")
ax_ratings.set_ylabel("Average IMDb Rating")
ax_ratings.set_title(
    "Figure 3: Average IMDb Episode Rating by Season\n"
    "The Big Bang Theory (2007–2017)"
)
ax_ratings.set_xticks(range(1, 11))
ax_ratings.set_xlim(0.5, 10.5)
ax_ratings.set_ylim(
    tbbt_season_master["avg_imdb_rating"].min() - 0.2,
    tbbt_season_master["avg_imdb_rating"].max() + 0.2
)
ax_ratings.grid(axis="y", alpha=0.3, linestyle="--")

plt.savefig("charts/04_imdb_ratings.png")
plt.close()
print("  Chart 4 saved")

# ============================================================
# CHART 5: REGRESSION SCATTER PLOT
# Scientific density vs IMDb rating at episode level
# ============================================================

print("Generating Chart 5: Regression scatter plot...")

# Drop any missing values for regression
tbbt_regression_data = tbbt_episode_master[[
    "season", "episode", "sci_density_per_100_words", "imdb_rating"
]].dropna()

fig_scatter, ax_scatter = plt.subplots(figsize=(9, 6))

scatter_plot = ax_scatter.scatter(
    tbbt_regression_data["sci_density_per_100_words"],
    tbbt_regression_data["imdb_rating"],
    c=tbbt_regression_data["season"],
    cmap="RdYlBu_r",
    alpha=0.65,
    s=50,
    edgecolors="white",
    linewidth=0.4
)

# Add regression line
regression_coeffs = np.polyfit(
    tbbt_regression_data["sci_density_per_100_words"],
    tbbt_regression_data["imdb_rating"], 1
)
regression_line = np.poly1d(regression_coeffs)
x_range = np.linspace(
    tbbt_regression_data["sci_density_per_100_words"].min(),
    tbbt_regression_data["sci_density_per_100_words"].max(),
    100
)
ax_scatter.plot(
    x_range,
    regression_line(x_range),
    color="black",
    linewidth=1.8,
    linestyle="--",
    label="Regression line",
    zorder=5
)

colour_bar = plt.colorbar(scatter_plot, ax=ax_scatter)
colour_bar.set_label("Season", rotation=270, labelpad=15)

ax_scatter.set_xlabel("Scientific terms per 100 words")
ax_scatter.set_ylabel("IMDb Episode Rating")
ax_scatter.set_title(
    "Figure 4: Scientific Language Density vs IMDb Episode Rating\n"
    "Each point represents one episode, coloured by season"
)
ax_scatter.legend(frameon=False)
ax_scatter.grid(alpha=0.2, linestyle="--")

plt.savefig("charts/05_regression_scatter.png")
plt.close()
print("  Chart 5 saved")

# ============================================================
# CHART 6: CHARACTER BREAKDOWN BAR CHART
# ============================================================

print("Generating Chart 6: Character breakdown...")

# Sort characters by early season scientific density descending
tbbt_char_comparison_sorted = tbbt_char_comparison.sort_values(
    "early_seasons_s1_s3", ascending=False
).reset_index(drop=True)

character_positions = np.arange(len(tbbt_char_comparison_sorted))
bar_width = 0.35

fig_chars, ax_chars = plt.subplots(figsize=(10, 6))

early_bars = ax_chars.bar(
    character_positions - bar_width / 2,
    tbbt_char_comparison_sorted["early_seasons_s1_s3"],
    bar_width,
    label="Early seasons (S1–S3)",
    color=COLOUR_EARLY,
    alpha=0.85
)
late_bars = ax_chars.bar(
    character_positions + bar_width / 2,
    tbbt_char_comparison_sorted["late_seasons_s8_s10"],
    bar_width,
    label="Late seasons (S8–S10)",
    color=COLOUR_LATE,
    alpha=0.85
)

# Add percentage change labels above each pair of bars
for bar_idx, row in tbbt_char_comparison_sorted.iterrows():
    pct = row["pct_change"]
    max_bar_height = max(
        row["early_seasons_s1_s3"], row["late_seasons_s8_s10"]
    )
    ax_chars.text(
        bar_idx,
        max_bar_height + 0.015,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLOUR_MAINSTREAM,
        fontweight="bold"
    )

# Add value labels inside bars
for bar in early_bars:
    ax_chars.text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() / 2,
        f"{bar.get_height():.3f}",
        ha="center", va="center",
        fontsize=8, color="white", fontweight="bold"
    )
for bar in late_bars:
    ax_chars.text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() / 2,
        f"{bar.get_height():.3f}",
        ha="center", va="center",
        fontsize=8, color="white", fontweight="bold"
    )

ax_chars.set_xlabel("Character")
ax_chars.set_ylabel("Scientific terms per 100 words")
ax_chars.set_title(
    "Figure 5: Scientific Language by Character — Early vs Late Seasons\n"
    "Percentage change shown above each pair (S1–S3 vs S8–S10)"
)
ax_chars.set_xticks(character_positions)
ax_chars.set_xticklabels(
    tbbt_char_comparison_sorted["character"].str.title(),
    fontsize=11
)
ax_chars.legend(frameon=False)
ax_chars.grid(axis="y", alpha=0.3, linestyle="--")
ax_chars.set_ylim(0, tbbt_char_comparison_sorted["early_seasons_s1_s3"].max() + 0.12)

plt.savefig("charts/06_character_breakdown.png")
plt.close()
print("  Chart 6 saved")

# ============================================================
# STEP 3: RUN REGRESSION MODELS
# ============================================================

print("\nRunning regression models...")

regression_output_lines = []

# ---- Regression 1: Scientific density ~ Google Trends ----
regression_1 = smf.ols(
    "avg_sci_density ~ avg_search_interest",
    data=tbbt_season_master
).fit()

regression_output_lines.append("=" * 60)
regression_output_lines.append(
    "REGRESSION 1: Scientific Density ~ Google Trends Interest"
)
regression_output_lines.append(
    "Does mainstream interest predict less scientific content?"
)
regression_output_lines.append("=" * 60)
regression_output_lines.append(str(regression_1.summary()))

print("\nRegression 1: Scientific Density ~ Google Trends")
print(f"  R-squared: {regression_1.rsquared:.3f}")
print(f"  Coefficient (Google Trends): {regression_1.params['avg_search_interest']:.4f}")
print(f"  P-value: {regression_1.pvalues['avg_search_interest']:.4f}")

# ---- Regression 2: IMDb Rating ~ Scientific Density ----
regression_2 = smf.ols(
    "imdb_rating ~ sci_density_per_100_words",
    data=tbbt_regression_data
).fit()

regression_output_lines.append("\n" + "=" * 60)
regression_output_lines.append(
    "REGRESSION 2: IMDb Rating ~ Scientific Density"
)
regression_output_lines.append(
    "Did audiences reward or punish scientific content?"
)
regression_output_lines.append("=" * 60)
regression_output_lines.append(str(regression_2.summary()))

print("\nRegression 2: IMDb Rating ~ Scientific Density")
print(f"  R-squared: {regression_2.rsquared:.3f}")
print(f"  Coefficient (Sci Density): {regression_2.params['sci_density_per_100_words']:.4f}")
print(f"  P-value: {regression_2.pvalues['sci_density_per_100_words']:.4f}")

# ---- Regression 3: Scientific Density ~ Season Number ----
# Simple check of the linear trend over time
regression_3 = smf.ols(
    "avg_sci_density ~ season",
    data=tbbt_season_master
).fit()

regression_output_lines.append("\n" + "=" * 60)
regression_output_lines.append(
    "REGRESSION 3: Scientific Density ~ Season Number"
)
regression_output_lines.append(
    "Is the decline in scientific language statistically significant over time?"
)
regression_output_lines.append("=" * 60)
regression_output_lines.append(str(regression_3.summary()))

print("\nRegression 3: Scientific Density ~ Season Number")
print(f"  R-squared: {regression_3.rsquared:.3f}")
print(f"  Coefficient (Season): {regression_3.params['season']:.4f}")
print(f"  P-value: {regression_3.pvalues['season']:.4f}")

# Save regression results to text file
regression_output_path = "data/processed/tbbt_regression_results.txt"
with open(regression_output_path, "w") as regression_file:
    regression_file.write("\n".join(regression_output_lines))

print(f"\nRegression results saved to {regression_output_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*60}")
print("All visualisations and regressions complete!")
print(f"{'='*60}")
print("\nCharts saved:")
print("  charts/01_summary_table.png")
print("  charts/02_sci_density_by_season.png")
print("  charts/03_sci_vs_trends.png")
print("  charts/04_imdb_ratings.png")
print("  charts/05_regression_scatter.png")
print("  charts/06_character_breakdown.png")
print("\nRegression results saved:")
print("  data/processed/tbbt_regression_results.txt")