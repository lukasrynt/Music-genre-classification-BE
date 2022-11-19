import os

import torch
import uvicorn
from fastapi import FastAPI, File, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.autoencoder.conv_autoencoder import ConvAutoencoder
from src.database import Database
from src.song import UnsupportedStrategyError

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

autoencoder = ConvAutoencoder()
if os.path.isfile('./autoencoder.pth'):
    autoencoder.load_state_dict(torch.load('./autoencoder.pth', map_location=torch.device('cpu')))
db = Database('./data', ['classical', 'jazz', 'rock', 'hiphop', 'reggae', 'country', 'metal'],
              autoencoder, songs_per_genre=10)
db.calculate_index()


def log_request(song: bytes):
    print(f'Received bytes of length: {len(song)}')


@app.get("/ping/")
def ping():
    return {"pong"}


@app.post("/relevant_genres/")
def relevant_genres(strategy: str, song: bytes = File()):
    """
    Calculates the most relevant genres for the song
    :param strategy: either 'mfcc' or 'autoencoder' strategy
    :param song: the bytes of song
    :return: dictionary of genres mapped to their percentage values
    """
    log_request(song)
    try:
        return db.relevant_genres(song, strategy)
    except UnsupportedStrategyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
