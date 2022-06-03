# Importing Required Libraries
import os
import sys
from fastapi import FastAPI,Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import json
import joblib
from sqlalchemy import create_engine


# load config.json
with open("./config.json") as json_file:
    config = json.load(json_file)

# fetching assets path to use tokenizer and model
dir_name, file_name = os.path.split(os.path.abspath(__file__))
assets_path = os.path.join(dir_name, "assets")

# instantiate an app
app = FastAPI()

# Create a templates object that you can re-use later
templates = Jinja2Templates(directory="templates")

# Mount a StaticFiles() instance in a specific path.
app.mount("/static", StaticFiles(directory="static"), name="static")

# load data from database
database_filepath = config["DATABASE"]
engine = create_engine(f"sqlite:///{database_filepath}")
df = pd.read_sql_table("messages", engine)

# load model from pickle file
sys.path.append(assets_path)
model = joblib.load(config["TRAINED_MODEL"])


# index webpage displays cool visuals and receives user input text for model
@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )

@app.post("/predict")
async def predict(request: Request,text: str = Form(...)):
    # use model to predict classification for message
    classification_labels = model.predict([text])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return templates.TemplateResponse("index.html", {"request": request, "text" : text,"result": classification_results})

    

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)


