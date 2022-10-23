import librosa


class Song:
    def __init__(self, genre: str, path: str):
        self.genre = genre
        self.y, self.sr = librosa.load(path)
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        self.name = self.__get_song_name(path)

    def __get_song_name(self, path: str):
        return f"{self.genre}-{int(path.split('/')[-1].split('.')[1])}"
