import os
from typing import List

from src.song import Song


class Database:
    def __init__(self, path: str, genres: List[str], songs_per_genre: int = 10):
        self.songs_per_genre = songs_per_genre
        self.path = path
        self.genres = genres
        self.songs = {}

    def relevant_genres(self, song_bytes: bytes, media_format: str):
        folder_path = os.path.join(self.path, 'requests')
        Song.save_bytes(song_bytes, folder_path, media_format)
        song = Song.load_request_song(folder_path, media_format)
        return song.mfcc

    def calculate_index(self):
        for song_path, genre in self.__iterate_songs():
            song = Song(song_path, genre)
            self.songs[song.name] = song

    def __iterate_songs(self):
        for genre in self.genres:
            arr = [self.__path_for_song(genre, idx) for idx in range(self.songs_per_genre)]
            for item in arr:
                yield item, genre

    def __path_for_song(self, genre: str, idx: int):
        return os.path.join(self.path, genre, f"{genre}.{'{:05d}'.format(idx)}.wav")

