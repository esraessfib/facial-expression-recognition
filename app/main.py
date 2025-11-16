from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import tensorflow as tf
from fastapi.responses import JSONResponse, HTMLResponse


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
# Serve static files (HTML, JS)
app.mount("/static", StaticFiles(directory="app"), name="static")
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("app/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        #  Load and preprocess image
        image = Image.open(file.file).convert("L")     # Grayscale
        image = image.resize((48, 48))                 # Same as training
        img_array = np.array(image, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape (1, 48, 48, 1)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions))         # convert float32 → float

        print("Predictions:", predictions)
        print("Predicted:", predicted_class, confidence)

        #  Return JSON response
        return JSONResponse(content={
            "class": predicted_class,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)