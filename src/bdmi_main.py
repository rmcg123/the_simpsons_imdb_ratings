"""Main Script."""
import pandas as pd
from matplotlib import rcParams

import src.bdmi_functions as bf
import src.bdmi_config as cfg


rcParams["font.family"] = "Arial"
rcParams["figure.figsize"] = (16, 9)
rcParams["figure.dpi"] = 300
rcParams["axes.titlesize"] = 24
rcParams["axes.labelsize"] = 18
rcParams["font.size"] = 16
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["legend.fontsize"] = 16
rcParams["legend.title_fontsize"] = 18


def main():
    """Main function for BDMI."""

    # If data already retrieved then load directly, otherwise scrape it.
    try:

        # Episodes DataFrame would be here if it exists already.
        episodes_df = pd.read_csv(
            "data/simpsons_episodes.csv", index_col="Unnamed: 0"
        )

        # Loop over each season and all episodes within season adding ratings
        # to dictionary.
        episodes_ratings_dict = {}
        for season in list(range(1, 37, 1)):
            episodes_ratings_dict[f"season_{season}"] = {}
            for ep_num in list(range(1, 26, 1)):
                try:

                    # If it exists, rating data should be here.
                    ratings = pd.read_csv(
                        f"data/episode_ratings/s{season}e{ep_num}_ratings.csv",
                        index_col="Unnamed: 0",
                    )
                    episodes_ratings_dict[f"season_{season}"][
                        f"episode_{ep_num}"
                    ] = ratings
                except FileNotFoundError:
                    continue
    except FileNotFoundError:

        # Otherwise get the information from IMDB.
        episodes_df, episodes_ratings_dict = bf.get_basic_episodes_info(
            cfg.SERIES_EPISODES_URL,
            save_dir="data/",
            retrieve_rating_distributions=True,
        )

    # Clean the data and drop all episodes that have yet to air.
    episodes_df = bf.clean_episodes_df(episodes_df)
    episodes_df = episodes_df.loc[
        episodes_df["air_date"].le(pd.Timestamp.today()), :
    ]

    # Create rolling average IMDB rating from last 12 episodes.
    episodes_df = bf.rolling_av_rating(
        episodes_df,
        rating_col="av_rating",
        window=12,
    )

    # Clean the ratings distributions.
    episodes_ratings_dict = bf.clean_episodes_ratings(episodes_ratings_dict)

    # Create a longitudinal plot highlighting the rolling window that has the
    # highest average rating.
    _, _ = bf.longitudinal_ratings_plot(
        episodes_df,
        rolling_rating_col="rolling_mean_rating_last_12_episodes",
        save_name="simpsons_peak_window.png",
        golden_age=False,
    )

    # Create a longitudinal plot highlighting the golden age of The Simpsons.
    _, _ = bf.longitudinal_ratings_plot(
        episodes_df,
        rolling_rating_col="rolling_mean_rating_last_12_episodes",
        save_name="simpsons_golden_age.png",
        peak=False,
    )

    # Take slice of episodes corresponding to peak window.
    peak_window = episodes_df[
        episodes_df[episodes_df["episode"].eq("S6E22")]
        .index.values[0] : episodes_df[episodes_df["episode"].eq("S7E8")]
        .index.values[0]
        + 1
    ]

    # Produce bar plots showing the rating distributions both as absolute
    # counts and as shares.
    for stat in ["counts", "share"]:
        bf.plot_rating_dist_barplot(
            peak_window,
            episodes_ratings_dict,
            stat=stat,
            eps="peak",
            save_dir="results/",
        )

    # Do the same but for all episodes that are rated 9 or above.
    top_eps = episodes_df.loc[episodes_df["av_rating"].ge(9.0), :].sort_values(
        by="av_rating", ascending=False
    )
    for stat in ["counts", "share"]:
        bf.plot_rating_dist_barplot(
            top_eps,
            episodes_ratings_dict,
            stat=stat,
            eps="top",
            save_dir="results/",
        )


if __name__ == "__main__":
    main()
