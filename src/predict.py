import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load the trained model"""
    model = tf.keras.models.load_model(model_path)
    return model

def load_and_preprocess_image(image_path, target_size=(48, 48)):
    """Load and preprocess image for prediction"""
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_emotion(model, image_array, class_names):
    """Predict emotion from image array"""
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    emotion = class_names[predicted_class]
    
    return emotion, confidence, predictions

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from facial expression')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='../models/model_weights.h5', help='Path to the trained model')
    args = parser.parse_args()
    
    # Define class names
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load model
    model = load_model(args.model)
    
    # Load and preprocess image
    image_array = load_and_preprocess_image(args.image)
    
    # Predict emotion
    emotion, confidence, predictions = predict_emotion(model, image_array, class_names)
    
    # Display results
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}")
    print("\nAll predictions:")
    for i, (cls, prob) in enumerate(zip(class_names, predictions[0])):
        print(f"{cls}: {prob:.4f}")
    
    # Display image
    img = load_img(args.image, target_size=(200, 200))
    plt.imshow(img.convert('RGB'))
    plt.title(f"Prediction: {emotion} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()