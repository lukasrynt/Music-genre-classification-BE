import os
from typing import List, Dict

from src.song import Song


class Database:
    def __init__(self, path: str, genres: List[str], songs_per_genre: int = 10):
        self.songs_per_genre = songs_per_genre
        self.path = path
        self.genres = genres
        self.songs = {key: {} for key in genres}

    def relevant_genres(self, song_bytes: bytes, media_format: str) -> Dict[str, float]:
        folder_path = os.path.join(self.path, 'requests')
        Song.save_bytes(song_bytes, folder_path, media_format)
        song = Song.load_request_song(folder_path, media_format)
        distances = self.calculate_distances(song)
        percentages = self.__normalize_distances(distances)
        song.delete_file()
        return percentages

    def calculate_index(self):
        for song_path, genre in self.__iterate_songs():
            song = Song(song_path, genre)
            self.songs[genre][song.name] = song

    def calculate_distances(self, other_song: Song) -> Dict[str, float]:
        res = {}
        min_dimension = min(self.__get_min_mfcc_dim(), other_song.mfcc.shape[1])
        map(lambda x: x.mfcc.shape[1], self.songs.values())
        for genre, songs in self.songs.items():
            res[genre] = sum([other_song.distance_from(song, min_dimension) for (song_name, song) in songs.items()])
        return res

    def __get_min_mfcc_dim(self):
        arr = [songs.values() for songs in self.songs.values()]
        return min([min(map(lambda x: x.mfcc.shape[1], arr)) for arr in arr])

    @staticmethod
    def __normalize_distances(distances: Dict[str, float]) -> Dict[str, float]:
        distance_sum = sum(distances.values())
        return {genre: ((distance / distance_sum) * 100) for (genre, distance)
                in sorted(distances.items(), key=lambda item: item[1])}

    def __iterate_songs(self):
        for genre in self.genres:
            arr = [self.__path_for_song(genre, idx) for idx in range(self.songs_per_genre)]
            for item in arr:
                yield item, genre

    def __path_for_song(self, genre: str, idx: int):
        return os.path.join(self.path, genre, f"{genre}.{'{:05d}'.format(idx)}.wav")
