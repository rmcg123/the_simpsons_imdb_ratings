"""Function to facilitate the BDMI project."""
import os
import requests
import time
import random
import datetime as dt

import pandas as pd
import selenium.common.exceptions
from bs4 import BeautifulSoup
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

import bdmi_config as cfg


def get_basic_episodes_info(
    episodes_url,
    save_dir,
    retrieve_rating_distributions=False,
    verbose=True,
):
    """Function to retrieve overview information for all episodes from a
    specified TV series."""

    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print("Retrieving The Simpsons episodes data...")

    # Submit a GET request to retrieve the specified URL.
    req = requests.get(episodes_url, headers=cfg.HEADERS)

    # Use Beautiful Soup to parse the retrieved HTML.
    soup = BeautifulSoup(req.content, "html.parser")

    # Determine the number of seasons of the TV series from the tabs of the
    # table.
    season_links = soup.find(
        "ul", "ipc-tabs ipc-tabs--base ipc-tabs--align-left"
    ).find_all("li", attrs={"role": "tab"})

    # Get the names of the seasons.
    seasons = [x.get_text() for x in season_links]

    # The page displays the episode overviews for one season in the
    # "active tab", other seasons episode overviews need to be accessed through
    # query parameters. To do so we need to loop over the seasons.
    episodes_df = pd.DataFrame()
    episode_ratings_dict = {}
    for season in seasons:
        if verbose:
            print(f"Retrieving episodes from season {season}...")

        if retrieve_rating_distributions:
            episode_ratings_dict[f"season_{season}"] = {}

        # Get the page for a particular season as the active tab
        req = requests.get(
            url=episodes_url, headers=cfg.HEADERS, params={"season": season}
        )

        # Parse retrieved season episodes HTML.
        season_soup = BeautifulSoup(req.content, "html.parser")

        # Isolate section of HTML that contains episodes for the season.
        episodes = season_soup.find(
            "section", "sc-577bf7cb-0 jPRxOq"
        ).find_all("article")

        # Loop over each episode within specified season pulling out the
        # episode details.
        for episode in episodes:
            # Retrieve episode information.
            episode_df = _extract_episode_information_from_overview(episode)

            # If desired, can get the distribution of the ratings for the
            # episode. This is quite time intensive - probably around 5 seconds
            # per episode.
            if retrieve_rating_distributions:
                # Setup save directory.
                ratings_dir = save_dir + "episode_ratings/"
                os.makedirs(ratings_dir, exist_ok=True)

                # Extract episode number.
                episode_number = episode_df["number"].squeeze()

                if verbose:
                    print(
                        f"Retrieving ratings for episode {episode_number}..."
                    )

                # Get ratings distribution using the episode id.
                episode_ratings_df = _get_episode_rating_distributions(
                    episode_df["id"].squeeze()
                )

                # Add ratings for this episode to dictionary.
                episode_ratings_dict[f"season_{season}"][
                    f"episode_{episode}"
                ] = episode_ratings_df

                # Save episode ratings.
                episode_ratings_df.to_csv(
                    ratings_dir + f"s{season}e{episode_number}_ratings.csv"
                )
                time.sleep(random.random())

            # Add episode to accruing dataset.
            episodes_df = pd.concat(
                [episodes_df, episode_df], ignore_index=True
            )

        # Delay to next season tab to mimic organic browsing behaviour.
        time.sleep(5 * random.random())

    # Save retrieved episodes data.
    episodes_df.to_csv(save_dir + "simpsons_episodes.csv")

    return episodes_df, episode_ratings_dict


def _extract_episode_information_from_overview(episode):
    """Function to extract the information from a single episode
    section of the HTML and return as a DataFrame row."""

    try:
        # Isolate top line of overview and extract information.
        ep_top_line = episode.find("div", "sc-9115db22-5 gsvBvl")
        ep_top_line_start = ep_top_line.find("h4")
        ep_name = ep_top_line_start.text.split("∙")[-1].strip()
        ep_number = (
            ep_top_line_start.text.split("∙")[0].split(".")[-1][1:].strip()
        )
        ep_season = (
            ep_top_line_start.text.split("∙")[0].split(".")[0][1:].strip()
        )
        ep_url = cfg.BASE_URL + ep_top_line_start.find("a").get("href")
        ep_id = ep_top_line_start.find("a").get("href").split("/")[-2][2:]
        ep_air_date = ep_top_line.find("span").text

        # Get description of episode and summary rating info.
        ep_description = episode.find("div", "sc-9115db22-11 cnMwsE").text
        ep_rating_info = episode.find("div", "sc-9115db22-12 jHvXSo").text
        ep_av_rating = ep_rating_info.split("/")[0]
        ep_n_ratings = ep_rating_info[
            ep_rating_info.find("(") + 1 : ep_rating_info.find(")")
        ]
    except AttributeError:
        return pd.DataFrame()

    # Add episode information and continue to next episode.
    episode_df = pd.DataFrame(
        [
            [
                ep_name,
                ep_season,
                ep_number,
                ep_url,
                ep_id,
                ep_air_date,
                ep_description,
                ep_av_rating,
                ep_n_ratings,
            ]
        ],
        columns=[
            "name",
            "season",
            "number",
            "url",
            "id",
            "air_date",
            "description",
            "av_rating",
            "n_ratings",
        ],
    )

    return episode_df


def _get_episode_rating_distributions(episode_id):
    """Function to retrieve the ratings distribution for a selected episode."""

    # Setup episode url using the provided id.
    episode_ratings_url = cfg.SERIES_EPISODE_RATING_URL.format(episode_id)

    # To retrieve the data from the page we need to use the Selenium package
    # since the information is contained in dynamic content.
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(options=options)
    driver.get(episode_ratings_url)

    # If episodes haven't aired yet then they won't have the ratings content.
    try:
        ratings_div = driver.find_element(By.CLASS_NAME, "VictoryContainer")
    except selenium.common.exceptions.NoSuchElementException:
        driver.quit()
        return pd.DataFrame()

    # Extract the % shares and ratings counts from element.
    ratings_text = ratings_div.text.removeprefix("12345678910")
    ratings = list(range(1, 11))
    chart_data = ratings_text.split(")")[:-1]
    shares = [x.split("%")[0] for x in chart_data][::-1]
    counts = [x.split("%")[1][2:] for x in chart_data][::-1]

    driver.quit()

    # Create DataFrame with ratings data.
    rating_df = pd.DataFrame(
        data={"share": shares, "counts": counts}, index=ratings
    )

    return rating_df


def clean_episodes_df(episodes_df):
    """Takes the raw retrieved episodes data and cleans it."""

    # Some of the number of ratings have "K" suffix for thousands so need to
    # deal with that situation.
    episodes_df["units"] = np.where(
        episodes_df["n_ratings"].str.contains("K"), 1000, 1
    )
    episodes_df["n_ratings"] = episodes_df["n_ratings"].str.replace("K", "")

    # Change string columns to numeric.
    for numeric_columns in ["season", "number", "av_rating", "n_ratings"]:
        episodes_df[numeric_columns] = pd.to_numeric(
            episodes_df[numeric_columns], errors="coerce"
        )

    # Restore the magnitudes of the number of ratings where appropriate.
    episodes_df["n_ratings"] = episodes_df["n_ratings"] * episodes_df["units"]
    episodes_df.drop(columns=["units"], inplace=True)

    # Format date column.
    episodes_df["air_date"] = pd.to_datetime(
        episodes_df["air_date"], format="%a, %b %d, %Y", errors="coerce"
    )

    # Sort DataFrame.
    episodes_df = episodes_df.sort_values(
        by=["air_date", "season", "number"]
    ).reset_index(drop=True)

    # Create composite season episode number column.
    episodes_df["episode"] = (
        "S"
        + episodes_df["season"].astype(str)
        + "E"
        + episodes_df["number"].astype(str)
    )

    return episodes_df


def clean_episodes_ratings(episode_ratings_dict):
    """Takes the raw retrieved episodes ratings distributions and cleans them."""

    # For each episode within each season, if there is an episode rating
    # DataFrame then do the cleaning.
    for season, season_episodes in episode_ratings_dict.items():
        for episode_num, episode_rating_df in season_episodes.items():
            if not episode_rating_df.empty:
                # If the counts of the votes are in the 1000s then make
                # appropriate adjustments.
                if not episode_rating_df["counts"].dtype == "int64":
                    # Create units column.
                    episode_rating_df["units"] = np.where(
                        episode_rating_df["counts"].str.contains("K"), 1000, 1
                    )

                    # Remove K from strings and change data type.
                    episode_rating_df["counts"] = episode_rating_df[
                        "counts"
                    ].str.replace("K", "")
                    episode_rating_df["counts"] = pd.to_numeric(
                        episode_rating_df["counts"], errors="coerce"
                    )

                    # Restore units to counts, and drop units column.
                    episode_rating_df["counts"] = (
                        episode_rating_df["counts"]
                        * episode_rating_df["units"]
                    )
                    episode_rating_df.drop(columns=["units"], inplace=True)

                # Change share column, index to numeric.
                episode_rating_df["share"] = pd.to_numeric(
                    episode_rating_df["share"], errors="coerce"
                )
                episode_rating_df.index = pd.to_numeric(
                    episode_rating_df.index
                )
                episode_ratings_dict[season][episode_num] = episode_rating_df
            else:
                continue

    return episode_ratings_dict


def rolling_av_rating(
    episodes_df, rating_col, window, cis=True, ci_type="basic"
):
    """Create a rolling average of the ratings over a chosen window, also
    optionally create a"""

    episodes_df[f"rolling_mean_rating_last_{window}_episodes"] = (
        episodes_df[rating_col].rolling(window=window, min_periods=1).mean()
    )

    if cis:
        if ci_type == "basic":
            episodes_df[f"rolling_std_rating_last_{window}_episodes"] = (
                episodes_df[rating_col]
                .rolling(window=window, min_periods=1)
                .std()
            )
            upper = (
                episodes_df[f"rolling_mean_rating_last_{window}_episodes"]
                + episodes_df[f"rolling_std_rating_last_{window}_episodes"]
            )
            lower = (
                episodes_df[f"rolling_mean_rating_last_{window}_episodes"]
                - episodes_df[f"rolling_std_rating_last_{window}_episodes"]
            )
        elif ci_type == "episodic":
            pass

        episodes_df[
            f"rolling_mean_rating_last_{window}_episodes_upper"
        ] = np.where(upper.gt(10), 10, upper)
        episodes_df[
            f"rolling_mean_rating_last_{window}_episodes_lower"
        ] = np.where(lower.lt(1), 1, lower)

    return episodes_df


def longitudinal_ratings_plot(
    episodes_df,
    rolling_rating_col,
    save_dir="results/",
    save_name="simpsons_ratings.png",
    peak=True,
    golden_age=True,
):
    """Function to look at the longitudinal trend of ratings and draw attention
    to regions of interest."""

    fig, ax = plt.subplots()

    # Create lineplot of rolling average ratings.
    sns.lineplot(
        data=episodes_df, x="air_date", y=rolling_rating_col, zorder=1
    )

    # Set labels for axes and plot title.
    ax.set_xlabel("Date")
    ax.set_ylabel(rolling_rating_col.replace("_", " ").title())
    ax.set_title("The Simpsons IMDB Ratings over Time")

    # If we want to identify the window that had the largest rolling average
    # rating then do this:
    if peak:
        # Identify peak of TV show.
        peak_val = episodes_df[rolling_rating_col].max()
        peak_date = episodes_df.loc[
            episodes_df[rolling_rating_col].idxmax(), "air_date"
        ]
        peak_window_end = episodes_df.loc[
            episodes_df[rolling_rating_col].idxmax(), "episode"
        ]
        peak_window_start = episodes_df.loc[
            episodes_df[rolling_rating_col].idxmax() - 11, "episode"
        ]

        # Need to convert datetimes to numbers for adding elements to plot.
        peak_date_num = mdates.date2num(peak_date)
        peak_date_num_shift = mdates.date2num(
            peak_date + dt.timedelta(days=1000)
        )

        # Create a line to indicate and annotate the peak.
        line = Line2D(
            [peak_date_num, peak_date_num_shift],
            [peak_val, peak_val + 0.3],
            linestyle="-",
            linewidth=1,
            color="black",
            zorder=2,
        )
        ax.add_line(line)
        plt.text(
            peak_date_num_shift,
            peak_val + 0.3,
            f" Highest Rated Window:\n"
            f" {peak_window_start} to {peak_window_end}",
            zorder=2,
            va="center",
            ha="left",
        )

    # Change the coloring of the line plot and highlighting for episodes that
    # are included in windows of the rolling average that are within the
    # confidence interval of the peak rating point.
    if golden_age:
        # Split dataset into pre and post peak.
        peak_and_post = episodes_df[episodes_df[rolling_rating_col].idxmax() :]
        pre_peak = episodes_df[: episodes_df[rolling_rating_col].idxmax()]

        # Determine one standard deviation below peak and use this value as the
        # cut-off for the golden age.
        peak_lower_ci = episodes_df.loc[
            episodes_df[rolling_rating_col].idxmax(),
            rolling_rating_col + "_lower",
        ]

        # Get all episodes from the peak back until the first breach of the
        # threshold.
        first_index = episodes_df[rolling_rating_col].idxmax()
        for idx, episode_row in pre_peak[::-1].iterrows():
            if episode_row[rolling_rating_col] >= peak_lower_ci:
                first_index = idx
            else:
                break

        # Add all episodes from the peak forward until the first breach of the
        # threshold.
        last_index = episodes_df[rolling_rating_col].idxmax()
        for idx, episode_row in peak_and_post.iterrows():
            if episode_row[rolling_rating_col] >= peak_lower_ci:
                last_index = idx
            else:
                break

        # The episodes in the golden age are all of those within the determined
        # range including those contained in the earliest window.
        golden_age = list(range(first_index - 11, last_index + 1))
        golden_age_df = episodes_df.loc[episodes_df.index.isin(golden_age), :]
        pre_golden_age = episodes_df.loc[
            episodes_df.index.__lt__(golden_age[0]), :
        ]
        post_golden_age = episodes_df.loc[
            episodes_df.index.__gt__(golden_age[-1]), :
        ]

        # Fill the golden age region differently than pre and post era.
        ax.fill_between(
            golden_age_df["air_date"],
            golden_age_df[rolling_rating_col + "_lower"],
            golden_age_df[rolling_rating_col + "_upper"],
            alpha=0.4,
            color="gold",
            zorder=5,
        )
        mean_date = golden_age_df["air_date"].mean()
        mean_rating = golden_age_df[rolling_rating_col].mean()
        # Create a line to indicate and annotate the peak.
        line = Line2D(
            [mean_date, mean_date],
            [mean_rating - 0.5, mean_rating - 1.5],
            linestyle="-",
            linewidth=1,
            color="black",
            zorder=2,
        )
        ax.add_line(line)
        plt.text(
            mean_date,
            mean_rating - 1.5,
            f" Golden Age of The Simpsons:\n"
            f"{golden_age_df['episode'].head(1).squeeze()}"
            f" to {golden_age_df['episode'].tail(1).squeeze()}",
            zorder=2,
            va="top",
            ha="center",
        )
        ax.annotate(
            "Episodes in the Golden Age are those from contiguous windows that "
            "have an average rating within a (peak) standard deviation from the "
            "peak rating",
            xy=(0.05, 0.01),
            xycoords="figure fraction",
            fontsize=10,
        )
        for era in [pre_golden_age, post_golden_age]:
            ax.fill_between(
                era["air_date"],
                era[rolling_rating_col + "_lower"],
                era[rolling_rating_col + "_upper"],
                alpha=0.2,
                zorder=5,
                color="tab:blue",
            )

    else:
        ax.fill_between(
            episodes_df["air_date"],
            episodes_df[rolling_rating_col + "_lower"],
            episodes_df[rolling_rating_col + "_upper"],
            alpha=0.2,
            zorder=5,
        )

    fig.tight_layout()
    fig.savefig(save_dir + save_name)

    return fig, ax


def plot_rating_dist_barplot(
    episode_df, episode_rating_dict, stat, eps, save_dir
):
    """Function to plot the ratings distribution (counts and shares) of
    particular groups of episodes."""

    fig, ax = plt.subplots()

    # For each episode extract the ratings distribution from the episode rating
    # dictionary and add to accruing DataFrame.
    ratings_df = pd.DataFrame()
    for idx, row in episode_df.iterrows():
        season = row["season"]
        number = row["number"]
        episode = row["episode"]
        name = row["name"]

        rating_df = episode_rating_dict[f"season_{season}"][
            f"episode_{number}"
        ]
        rating_df["episode"] = episode
        rating_df["name"] = name
        rating_df["rating"] = rating_df.index
        ratings_df = pd.concat([ratings_df, rating_df], ignore_index=True)

    # Create composite season, episode number and episode name label for plots.
    ratings_df["episode_label"] = (
        ratings_df["episode"] + ": " + ratings_df["name"]
    )

    # Opposite order.
    ratings_df = ratings_df[::-1]

    # To create stacked bar chart need to increment each bar for each rating.
    bar_heights = pd.Series([0] * ratings_df["episode"].nunique())
    palette = sns.color_palette("RdBu", n_colors=10)

    # Loop over ratings create the section of the bar for each episode.
    for rating in list(range(1, 11, 1)):
        rating_df = ratings_df[ratings_df["rating"].eq(rating)]
        plt.barh(
            rating_df["episode_label"],
            rating_df[stat],
            height=0.8,
            left=bar_heights,
            color=palette[rating - 1],
        )
        bar_heights = bar_heights + rating_df.reset_index()[stat]

    # When peak then we need to reverse y-axis.
    if eps == "peak":
        plt.gca().invert_yaxis()

    # Create ratings legend.
    handles = [mpatches.Patch(color=x) for x in palette]
    labels = list(range(1, 11, 1))
    ax.legend(
        handles[::-1],
        labels[::-1],
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        title="Rating",
    )

    # Set labels and title.
    ax.set_ylabel("Episode")
    if stat == "share":
        ax.set_xlabel("% of Total Ratings")
    else:
        ax.set_xlabel("Count of Ratings")

    if eps == "peak":
        ax.set_title(
            "Distribution of IMDB Ratings for Each Episode Within Peak Window"
        )
    elif eps == "top":
        ax.set_title("Distribution of IMDB Ratings for Top 10 Episodes")

    # Save figure.
    fig.tight_layout()
    if eps == "peak":
        fig.savefig(
            save_dir + f"rating_dist_peak_window_episodes_{stat}.png", dpi=300
        )
    elif eps == "top":
        fig.savefig(save_dir + f"rating_dist_top_episodes_{stat}.png", dpi=300)

    return fig, ax
