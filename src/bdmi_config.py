"""Configuration file for BDMI project."""

BASE_URL = "https://www.imdb.com"

SERIES_ID = "0096697"

SERIES_EPISODES_URL = BASE_URL + f"/title/tt{SERIES_ID}/episodes"

SERIES_EPISODE_URL = BASE_URL + "/title/tt{}/"

SERIES_EPISODE_RATING_URL = BASE_URL + "/title/tt{}/ratings/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/118.0",
    "Accepted": "application/json",
    "Referer": BASE_URL + f"title/tt{SERIES_ID}/?ref_=ttep_ov_i",
}
