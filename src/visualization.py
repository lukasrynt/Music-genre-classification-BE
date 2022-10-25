import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from src.database import Database


def plot_barycenter(song):
    barycenter = song.get_barycenter()
    for series in song.mfcc:
        plt.plot(series.ravel(), "k-", alpha=.2)
    plt.plot(barycenter.ravel(), "r-", linewidth=2)
    plt.show()


class Visualizer:
    def __init__(self, db: Database):
        self.db = db
        self.genres = None

    def visualize_barycenters(self):
        barycenters = self.__get_barycenters()
        reduced = self.__tsne_reduce(barycenters, perplexity=10)
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(reduced[1], reduced[2], c=reduced.genre, cmap='jet')
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, loc="lower right", title="Genre", labels=self.genres.tolist())
        plt.show()

    def __get_barycenters(self):
        arr = [song for songs in self.db.songs.values() for song in songs.values()]
        minimum = min(map(lambda x: x.mfcc.shape[1], arr))
        barycenters = pd.DataFrame(map(lambda x: [x.genre] + x.get_barycenter().ravel().tolist()[:minimum], arr))
        barycenters.rename(columns={0: 'genre'}, inplace=True)
        barycenters = self.__remap_genres(barycenters)
        return barycenters

    def __remap_genres(self, barycenters):
        self.genres = barycenters.genre.unique()
        genres_mapping = {self.genres[i]: i for i in range(len(self.genres))}
        barycenters.genre = barycenters.apply(lambda x: genres_mapping[x.genre], axis=1)
        return barycenters

    @staticmethod
    def __tsne_reduce(df: pd.DataFrame, perplexity: int = 30) -> pd.DataFrame:
        """
        Reduces the dataset into 2 dimensions using t-SNE dimensionality reduction
        :param perplexity: Perplexity of t-SNE dimensionality reduction
        :param df: Dataset of learned representations
        :return: Learned representations reduced to 2 dimensions
        """
        feature_cols = [x for x in df.columns if isinstance(x, int)]
        x = df.loc[:, feature_cols]
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init='random')
        tsne_results = tsne.fit_transform(x)
        cols = [col + 1 for col in range(2)]
        tsne_df = pd.DataFrame(data=tsne_results, columns=cols)
        return pd.concat([tsne_df, df[['genre']]], axis=1)
