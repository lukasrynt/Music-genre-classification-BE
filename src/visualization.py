import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
from tslearn.barycenters import dtw_barycenter_averaging


def calculate_barycenter(mfcc: pd.DataFrame):
    return dtw_barycenter_averaging(mfcc)


def plot_barycenter(song):
    barycenter = song.get_barycenter()
    for series in song.mfcc:
        plt.plot(series.ravel(), "k-", alpha=.2)
    plt.plot(barycenter.ravel(), "r-", linewidth=2)
    plt.show()


def time_warp_dist(x, y) -> float:
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance


def tsne_reduce(df: pd.DataFrame, perplexity: int = 30) -> pd.DataFrame:
    """
    Reduces the dataset into 2 dimensions using t-SNE dimensionality reduction
    :param perplexity: Perplexity of t-SNE dimensionality reduction
    :param df: Dataset of learned representations
    :return: Learned representations reduced to 2 dimensions
    """
    feature_cols = [x for x in df.columns if isinstance(x, int)]
    x = df.loc[:, feature_cols]
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, square_distances=True,
                init='random', metric=lambda x, y: time_warp_dist(x, y))
    tsne_results = tsne.fit_transform(x)
    cols = [f"t-SNE axe {col + 1}" for col in range(2)]
    tsne_df = pd.DataFrame(data=tsne_results, columns=cols)
    return pd.concat([tsne_df, df[['token', 'language']]], axis=1)
