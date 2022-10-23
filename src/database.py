import os
from typing import List

from src.song import Song


class Database:
    def __init__(self, path: str, genres: List[str], songs_per_genre: int = 10):
        self.songs_per_genre = songs_per_genre
        self.path = path
        self.genres = genres
        self.songs = {}

    def relevant_genres(self, music):
        pass

    def calculate_index(self):
        for song_path, genre in self.__iterate_songs():
            song = Song(genre, song_path)
            self.songs[song.name] = song

    def __iterate_songs(self):
        for genre in self.genres:
            arr = [self.__path_for_song(genre, idx) for idx in range(self.songs_per_genre)]
            for item in arr:
                yield item, genre

    def __path_for_song(self, genre: str, idx: int):
        return os.path.join(self.path, genre, f"{genre}.{'{:05d}'.format(idx)}.wav")

