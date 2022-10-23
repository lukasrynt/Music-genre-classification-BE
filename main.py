import uvicorn
from fastapi import FastAPI, File
from src.database import Database

app = FastAPI()
db = Database('./data', ['classical', 'jazz', 'rock', 'hiphop', 'reggae', 'country', 'metal'])
db.calculate_index()


def log_request(song: bytes, media_format: str):
    print(f'Received bytes of length: {len(song)} with format: {media_format}')


@app.get("/ping/")
def ping():
    return {"pong"}


@app.post("/relevant_genres/")
async def relevant_genres(media_format: str, song: bytes = File()):
    log_request(song, media_format)
    return db.relevant_genres(song, media_format)[0].tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
