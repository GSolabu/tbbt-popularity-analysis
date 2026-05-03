import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ============================================================
# SCRAPE TBBT TRANSCRIPTS FROM BIGBANGTRANS.WORDPRESS.COM
# Collects all dialogue by character, season and episode
# for Series 1-10 of The Big Bang Theory
# ============================================================

BASE_URL = "https://bigbangtrans.wordpress.com"

request_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# STEP 1: GET ALL EPISODE URLS FROM THE MAIN PAGE

print("Fetching episode list from main page...")
main_page_response = requests.get(BASE_URL, headers=request_headers)
main_page_soup = BeautifulSoup(main_page_response.content, "html.parser")

all_page_links = main_page_soup.find_all("a", href=True)

episode_url_list = []
for link in all_page_links:
    href = link["href"]
    if "bigbangtrans.wordpress.com/series-" in href and "episode" in href:
        episode_url_list.append(href)

episode_url_list = sorted(list(set(episode_url_list)))

print(f"Found {len(episode_url_list)} episode URLs")
print("Sample URLs:")
for url in episode_url_list[:5]:
    print(f"  {url}")

# STEP 2: PARSE SEASON AND EPISODE NUMBER FROM URL

def parse_season_episode_from_url(url):
    """Extract season and episode number from episode URL"""
    match = re.search(r'series-(\d+)-episode-(\d+)', url)
    if match:
        season_num = int(match.group(1))
        episode_num = int(match.group(2))
        return season_num, episode_num
    return None, None

# STEP 3: SCRAPE DIALOGUE FROM A SINGLE EPISODE PAGE

def scrape_single_episode(url, season_num, episode_num):
    """
    Scrapes all character dialogue from a single episode page.
    Filters out scene directions, credits and stage directions.
    Returns a list of dicts with season, episode, character, dialogue.
    """
    try:
        episode_response = requests.get(url, headers=request_headers, timeout=10)
        episode_soup = BeautifulSoup(episode_response.content, "html.parser")

        # On this WordPress site the transcript lives inside the
        # main .entry or .post container — grab all paragraphs from it
        episode_content = (
            episode_soup.find("div", class_="entry-content")
            or episode_soup.find("div", class_="post-content")
            or episode_soup.find("div", class_="entry")
            or episode_soup.find("div", class_="post")
            or episode_soup.find("main")
            or episode_soup.find("article")
        )

        # If none of the specific containers found, fall back to body
        if not episode_content:
            episode_content = episode_soup.find("body")

        if not episode_content:
            print(f"  Could not find any content for {url}")
            return []

        episode_paragraphs = episode_content.find_all("p")
        episode_dialogue_lines = []

        for paragraph in episode_paragraphs:
            # Get raw text
            paragraph_text = paragraph.get_text(separator=" ").strip()

            # Skip empty paragraphs
            if not paragraph_text:
                continue

            # Skip scene descriptions
            if paragraph_text.lower().startswith("scene:"):
                continue

            # Skip credits and writer lines
            if "credits sequence" in paragraph_text.lower():
                continue
            if "written by" in paragraph_text.lower():
                continue

            # Skip share/social media links that appear at bottom of page
            if "share this" in paragraph_text.lower():
                continue

            # Skip paragraphs that are entirely italic
            # (scene directions and credits are italicised on this site)
            italic_content = "".join(
                element.get_text()
                for element in paragraph.find_all(["em", "i"])
            )
            if italic_content and len(italic_content) >= len(paragraph_text) * 0.8:
                continue

            # Skip pure stage directions enclosed in brackets
            cleaned_for_bracket_check = paragraph_text.strip()
            if cleaned_for_bracket_check.startswith("(") and cleaned_for_bracket_check.endswith(")"):
                continue

            # Match character name pattern: e.g. "Sheldon: dialogue here"
            character_line_match = re.match(
                r'^([A-Z][a-zA-Z\s\-]{1,30}):\s*(.+)', paragraph_text
            )

            if character_line_match:
                character_name = character_line_match.group(1).strip()
                dialogue_text = character_line_match.group(2).strip()

                # Remove inline stage directions e.g. (they all sit down)
                dialogue_text = re.sub(r'\(.*?\)', '', dialogue_text).strip()

                # Skip if nothing remains after removing stage directions
                if not dialogue_text:
                    continue

                # Skip if the character name looks like a sentence fragment
                # rather than an actual character name
                if len(character_name.split()) > 3:
                    continue

                episode_dialogue_lines.append({
                    "season": season_num,
                    "episode": episode_num,
                    "character": character_name.upper(),
                    "dialogue": dialogue_text
                })

        return episode_dialogue_lines

    except Exception as scrape_error:
        print(f"  Error scraping {url}: {scrape_error}")
        return []

# STEP 4: LOOP THROUGH ALL EPISODE URLS AND SCRAPE

all_dialogue_rows = []
failed_episode_urls = []

print(f"\nBeginning scrape of {len(episode_url_list)} episodes...")
print("Estimated time: 15-20 minutes")
print("A 2 second delay is used between requests\n")

for url_index, episode_url in enumerate(episode_url_list):
    season_num, episode_num = parse_season_episode_from_url(episode_url)

    # Skip URLs where season/episode could not be parsed
    if season_num is None:
        print(f"Skipping unrecognised URL: {episode_url}")
        continue

    # Only scrape seasons 1 to 10
    if season_num > 10:
        continue

    print(f"Scraping S{season_num:02d}E{episode_num:02d} "
          f"({url_index + 1}/{len(episode_url_list)}) — {episode_url}")

    episode_lines = scrape_single_episode(episode_url, season_num, episode_num)

    if episode_lines:
        all_dialogue_rows.extend(episode_lines)
        print(f"  Collected {len(episode_lines)} lines of dialogue")
    else:
        print(f"  WARNING: No dialogue collected for this episode")
        failed_episode_urls.append(episode_url)

    # Polite delay between requests
    time.sleep(2)

# STEP 5: SAVE RAW DIALOGUE TO CSV

tbbt_raw_dialogue = pd.DataFrame(all_dialogue_rows)

print(f"\n{'='*50}")
print(f"Scraping complete!")
print(f"Total lines of dialogue collected: {len(tbbt_raw_dialogue)}")
print(f"\nEpisodes per season:")
print(tbbt_raw_dialogue.groupby('season')['episode'].nunique())
print(f"\nTop 10 most frequent characters:")
print(tbbt_raw_dialogue['character'].value_counts().head(10))

if failed_episode_urls:
    print(f"\nFailed to scrape {len(failed_episode_urls)} episodes:")
    for failed_url in failed_episode_urls:
        print(f"  {failed_url}")

tbbt_raw_dialogue.to_csv("data/raw/tbbt_raw_dialogue.csv", index=False)
print(f"\nRaw dialogue saved to data/raw/tbbt_raw_dialogue.csv")
print("\nPreview of first 10 rows:")
print(tbbt_raw_dialogue.head(10)) 