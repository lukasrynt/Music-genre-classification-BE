import librosa
import numpy as np
import torch

from src.autoencoder.conv_autoencoder import ConvAutoencoder


class SegmentedSong:
    def __init__(self, song, window: int = 400, shift: int = 1):
        self.window = window
        self.shift = shift
        self.song = song
        self.spec_data = None
        try:
            self.spec_data = self.__get_spec_data()
        except:
            pass

    def __len__(self):
        if self.spec_data is None:
            return 0
        else:
            return int(self.spec_data.shape[1] - self.window) // self.shift

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.song.y is None:
            raise StopIteration

        if self.n < int(self.spec_data.shape[1] - self.window) // self.shift:
            self.n += 1
            spec_segment = self.at(self.n)
            return spec_segment
        else:
            raise StopIteration

    def at(self, index):
        return self.spec_data[:, index * self.shift:(index * self.shift + self.window)]

    @staticmethod
    def __convert_to_librosa(segment):
        samples = segment.get_array_of_samples()
        y = np.array(samples).astype(np.float32) / 32768  # 16 bit
        return y, segment.frame_rate

    def __get_spec_data(self):
        # this is the number of samples in a window per fft
        n_fft = 2048
        # The amount of samples we are shifting after each fft
        hop_length = 512

        mel_signal = librosa.feature.melspectrogram(y=self.song.y, sr=self.song.samplerate,
                                                    hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(mel_signal)
        power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
        return power_to_db


class WeightedSongEmbedding:
    def __init__(self, song):
        self.segments = SegmentedSong(song, shift=20)

    def calculate_embedding(self, autoencoder: ConvAutoencoder):
        embeddings = self.__calculate_embeddings(autoencoder)
        return np.mean(embeddings, axis=0)

    def __calculate_embeddings(self, autoencoder):
        embeddings = np.ones(300)
        for segment in iter(self.segments):
            in_seg = torch.Tensor(np.array([segment])).view(1, 1, 128, 400)
            _, embedding = autoencoder(in_seg)
            detached = embedding.detach().numpy()
            embeddings = np.vstack([embeddings, detached])
        return embeddings[1:]
