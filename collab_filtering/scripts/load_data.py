import pandas as pd
import numpy as np

# Location of files
artist_path = "../data/ydata-ymusic-artist-names-v1_0.txt"
user_path = "../data/ydata-ymusic-user-artist-ratings-v1_0.txt"
encoding = "Windows-1252"

artist_df = pd.read_table(artist_path, encoding=encoding, names=["code", "artist"], skiprows=2)

def user_artist_df(nrows=None):

    user_df = pd.read_table(user_path, encoding=encoding, nrows=nrows, names=["user", "artist", "rating"])

    # Create dataframe where the rows are each artist and the rows are each
    # of the users that rated the artists
    df = pd.crosstab(index=user_df.artist, columns=user_df.user, values=user_df.rating, aggfunc=np.sum)

    # Remove 255 tag: TODO: Rationalize the way this tag is used
    df = df[df != 255]

    # Remove false: 24538 tag in index
    df = df.drop(24538)

    return df


def artist_mean_rate(nrows=None):
    df = user_artist_df(nrows=nrows)

    # a) Get mean rate for each artist and count the number of users
    # that rated them. Compute a weighted value to allocate their
    # 'positive popularity'

    # sum over all users and count how many rated each artist; their mean rate
    artist_rating_count = df.count(axis=1) 
    artist_rating_mean = df.mean(axis=1)
    total_count_ratings = sum(artist_rating_count)

    artist_mean_a = pd.DataFrame({"mean_rating": artist_rating_mean,
                                  "count_ratings": artist_rating_count})
    artist_mean_a["weighted_mean_rate"] = artist_mean_a.mean_rating * artist_mean_a.count_ratings / total_count_ratings


    # b) Get the name of each artist in the dataset given its respective code
    artist_mean_b = artist_df[artist_df["code"].isin(df.axes[0])]
    artist_mean_b.index = artist_mean_b.code
    artist_mean_b.drop("code", axis=1)
    artist_mean_b = artist_mean_b.drop("code", axis=1)

    # Merge a) and b)
    artist_mean = pd.concat([artist_mean_a, artist_mean_b], axis=1)

    # Remove NaNs in artist_mean
#    keep_rows = ~np.array([str(artist) == "nan" for artist in artist_mean.artist.values])
#    artist_mean = artist_mean.ix[keep_rows]

    return artist_mean


