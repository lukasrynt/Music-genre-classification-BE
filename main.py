import uvicorn
from fastapi import FastAPI, File
from src.database import Database

app = FastAPI()
db = Database('./data', ['classical', 'jazz', 'rock', 'hiphop', 'reggae', 'country', 'metal'])
db.calculate_index()


@app.get("/ping/")
def ping():
    return {"pong"}


@app.post("/relevant_genres/")
async def relevant_genres(song: bytes = File()):
    return {"file_size": len(song)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
