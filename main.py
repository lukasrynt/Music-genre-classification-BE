from fastapi import FastAPI
from src.database import Database

app = FastAPI()
db = Database('./data', ['classical', 'jazz', 'rock', 'hiphop', 'reggae', 'country', 'metal'])
db.calculate_index()


@app.get("/my-first-api")
def hello():
    return {"Hello world!"}
