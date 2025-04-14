from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("model.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 classes

# Image preprocessing
def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).resize((32, 32))
    return np.array(image) / 255.0

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = read_imagefile(image_bytes)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return JSONResponse(content={
        "class": predicted_class,
        "confidence": round(confidence, 2)
    })
