import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img

def visualize_training_history(history):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.savefig('../results/training_history.png')
    plt.show()

def visualize_sample_images(base_path):
    """Display sample images from each category"""
    plt.figure(0, figsize=(12, 20))
    cpt = 0
    
    for expression in os.listdir(base_path + "train/"):
        expression_path = os.path.join(base_path + "train/", expression)
        if os.path.isdir(expression_path):
            image_files = os.listdir(expression_path)
            if len(image_files) >= 5:  # Ensure there are at least 5 images
                for i in range(5):
                    cpt += 1
                    plt.subplot(7, 5, cpt)
                    img = load_img(os.path.join(expression_path, image_files[i]), 
                                  target_size=(48, 48), 
                                  color_mode="grayscale")
                    plt.imshow(img, cmap="gray")
                    plt.title(expression)
                    plt.axis('off')

    plt.tight_layout()
    plt.savefig('../results/sample_images.png')
    plt.show()