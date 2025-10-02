from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf


model = tf.keras.models.load_model('models/emotion_recognition_model_full.h5')

classes =["angry", "disgust", "happy", "neutral", "sad", "surprise" ,"fear"]

app = FastAPI(title="Facial Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin; you can specify your frontend URL instead
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    predictions = model.predict(image)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)

    return JSONResponse(content={"class": predicted_class, "confidence": confidence})