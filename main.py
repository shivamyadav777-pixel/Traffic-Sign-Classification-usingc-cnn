from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import pickle
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open(r"C:\Users\shiva\traffic\model_trained.p", "rb")
MODEL = pickle.load(pickle_in)

CLASS_NAMES = [
'Speed Limit 20 km/h',
    'Speed Limit 30 km/h',
    'Speed Limit 50 km/h',
    'Speed Limit 60 km/h',
    'Speed Limit 70 km/h',
    'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h',
    'Speed Limit 100 km/h',
    'Speed Limit 120 km/h',
    'No passing',
    'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Vehicles over 3.5 metric tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve to the left',
    'Dangerous curve to the right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End of all speed and passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Resize the image to (32, 32)
    image = tf.image.resize(image, [32, 32])

    # Convert the image to grayscale if needed
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)

    # Normalize pixel values
    image = image / 255.0

    return image.numpy()


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    preprocessed_image = preprocess_image(image)
    img_batch = np.expand_dims(preprocessed_image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
