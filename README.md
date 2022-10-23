# Music genres classification 

## Installation

To setup environment run (assuming you have conda installed):
1. `conda env create -f environment.yml`
2. `conda activate vmm`

## Running server

To run the server locally simply run `uvicorn main:app --reload` or `python main.py`. The server will then be running on [localhost](http://127.0.0.1:8000).

## Data
[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) was used to learn. Songs are expected in data folder. For classification, the following genres were chosen:
- classical
- jazz
- rock
- hiphop
- reggae
- country
- metal