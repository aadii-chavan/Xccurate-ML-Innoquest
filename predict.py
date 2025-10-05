#!/usr/bin/env python
import os
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Configuration ---
IMG_SIZE = (150, 150)
CLASS_LABELS = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

# --- Prediction ---
def predict_image(model_path, img_path):
    """
    Loads a model and an image, preprocesses the image, and predicts the class.
    """
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at '{model_path}'"}
    if not os.path.exists(img_path):
        return {"error": f"Image file not found at '{img_path}'"}

    try:
        model = load_model(model_path)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))

        pred_label = CLASS_LABELS.get(pred_class, "Unknown")

        return {"prediction": pred_label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict brain tumor from MRI image.')
    parser.add_argument('--model', type=str, required=True, help='Path to the .keras model file.')
    parser.add_argument('--image', type=str, required=True, help='Path to the MRI image file.')
    args = parser.parse_args()

    prediction_result = predict_image(args.model, args.image)
    print(prediction_result)