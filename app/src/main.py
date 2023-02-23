import base64

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.neural_network import load_training_data, categorize_image, train_model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    image: str


class ImageLabel(BaseModel):
    label: int


@app.get("/train")
async def train():
    X_train, y_train = load_training_data()
    train_model(X_train, y_train)
    return "Model Trained"


@app.post("/identify_image")
async def identify_image(image_data: ImageData):
    base64_data = image_data.image.split(",")[1]
    decoded_data = base64.b64decode(base64_data)
    result = categorize_image(decoded_data)

    return ImageLabel(label=result)
