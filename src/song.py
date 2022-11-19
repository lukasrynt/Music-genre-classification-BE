import os

import librosa
import numpy as np

from .autoencoder.conv_autoencoder import ConvAutoencoder
from .autoencoder.song import WeightedSongEmbedding
from .dynamic_time_warp import dtw
from tslearn.barycenters import dtw_barycenter_averaging

REQUEST_SONG_NAME = 'req'


class UnsupportedStrategyError(Exception):
    def __init__(self, strategy):
        self.message = f'{strategy} is not among allowed strategies - only mfcc and autoencoder is allowed'
        super().__init__(self.message)


class Song:
    def __init__(self, path: str, genre: str = 'unknown', media_format: str = 'wav'):
        self.media_format = media_format
        self.genre = genre
        self.path = path
        self.y, self.samplerate = librosa.load(path)
        self.__set_mfcc()
        self.embedding = None
        if not genre == 'unknown':
            self.name = self.__get_song_name(path)

    def delete_file(self):
        os.remove(self.path)

    def get_barycenter(self):
        return dtw_barycenter_averaging(self.mfcc)

    def distance_from(self, other, strategy: str) -> float:
        if strategy == 'mfcc':
            return dtw(self.mfcc.T, other.mfcc.T)
        elif strategy == 'autoencoder':
            return np.linalg.norm(self.embedding - other.embedding)
        else:
            raise UnsupportedStrategyError(strategy)

    def precalculate_embedding(self, autoencoder: ConvAutoencoder):
        if not self.embedding:
            self.embedding = WeightedSongEmbedding(self).calculate_embedding(autoencoder)

    @staticmethod
    def save_bytes(song: bytes, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f'{REQUEST_SONG_NAME}.mp3'), 'wb') as f:
            f.write(song)

    @staticmethod
    def load_request_song(path: str):
        return Song(os.path.join(path, f'{REQUEST_SONG_NAME}.mp3'))

    def __get_song_name(self, path: str):
        return f"{self.genre}-{int(path.split('/')[-1].split('.')[1])}"

    def __set_mfcc(self):
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.samplerate, n_mfcc=5)
