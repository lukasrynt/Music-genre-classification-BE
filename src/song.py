import os

import librosa

from src.visualization import time_warp_dist

REQUEST_SONG_NAME = 'req'


class Song:
    def __init__(self, path: str, genre: str = 'unknown', media_format: str = 'wav'):
        self.media_format = media_format
        self.genre = genre
        self.path = path
        self.y, self.samplerate = librosa.load(path)
        self.__set_mfcc()
        if not genre == 'unknown':
            self.name = self.__get_song_name(path)

    def delete_file(self):
        os.remove(self.path)

    def distance_from(self, other, min_len: int) -> float:
        return time_warp_dist(self.mfcc[:, :min_len], other.mfcc[:, :min_len])

    @staticmethod
    def save_bytes(song: bytes, path: str, media_format: str):
        with open(os.path.join(path, f'{REQUEST_SONG_NAME}.{media_format}'), 'wb') as f:
            f.write(song)

    @staticmethod
    def load_request_song(path: str, media_format: str):
        return Song(os.path.join(path, f'{REQUEST_SONG_NAME}.{media_format}'))

    def __get_song_name(self, path: str):
        return f"{self.genre}-{int(path.split('/')[-1].split('.')[1])}"

    def __set_mfcc(self):
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.samplerate)
