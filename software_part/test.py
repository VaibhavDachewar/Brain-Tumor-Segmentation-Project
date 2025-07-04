import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from metrics import dice_loss, dice_coef
from train import load_dataset
from unet import build_unet

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

""" Global parameters """
H = 256
W = 256

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, y_pred_prob, save_image_path):
    """
    Generate a heatmap overlay of the predicted tumor region on the input image.

    Args:
        image: Input MRI image (H, W, 3), already normalized to [0,1].
        y_pred_prob: Raw prediction probabilities from the model (H, W), in [0,1].
        save_image_path: Path to save the output image.
    """
    # Ensure image is in uint8 format for visualization (scale back to [0, 255])
    image_uint8 = (image * 255).astype(np.uint8)

    # Normalize the prediction probabilities to [0, 255] for heatmap
    y_pred_prob = np.clip(y_pred_prob, 0, 1)  # Ensure values are in [0, 1]
    heatmap = (y_pred_prob * 255).astype(np.uint8)

    # Apply a colormap to the prediction probabilities (e.g., JET for blue-to-red gradient)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original image
    alpha = 0.5  # Transparency factor for heatmap
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    # Optionally, find contours of the predicted region and draw them
    y_pred_binary = (y_pred_prob >= 0.5).astype(np.uint8)  # Binarize for contour
    contours, _ = cv2.findContours(y_pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 165, 0), 2)  # Orange contours

    # Save the result
    cv2.imwrite(save_image_path, overlay)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    # Create the results directory on Google Drive
    results_dir = "/content/drive/MyDrive/results"
    create_dir(results_dir)

    # Also create the 'files' directory on Google Drive for score.csv
    create_dir("/content/drive/MyDrive/files")

    """ Load the model """
    # Assuming the model is also stored on Google Drive
    model_path = "/content/files/model.h5"
    try:
        with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
            model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the model has been trained and saved, or upload the model.h5 file.")
        raise

    """ Dataset """
    dataset_path = "/content/extracted_files/archive (2)"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    """ Prediction and Evaluation """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = x.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  ## [H, W, 3]
        if image is None:
            print(f"Failed to load image: {x}")
            continue
        image = cv2.resize(image, (W, H))        ## [H, W, 3]
        x_normalized = image / 255.0             ## [H, W, 3]
        x = np.expand_dims(x_normalized, axis=0) ## [1, H, W, 3]

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {y}")
            continue
        mask = cv2.resize(mask, (W, H))

        """ Prediction """
        y_pred = model.predict(x, verbose=0)[0]  # [H, W, 1]
        y_pred_prob = np.squeeze(y_pred, axis=-1)  # [H, W], raw probabilities
        y_pred_binary = (y_pred_prob >= 0.5).astype(np.int32)  # Binarized for metrics

        """ Saving the prediction with heatmap overlay to Google Drive """
        save_image_path = os.path.join(results_dir, name)
        save_results(x_normalized, y_pred_prob, save_image_path)

        """ Flatten the array for metrics """
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred_binary = y_pred_binary.flatten()

        """ Calculating the metrics values """
        f1_value = f1_score(mask, y_pred_binary, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, y_pred_binary, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred_binary, labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(mask, y_pred_binary, labels=[0, 1], average="binary", zero_division=0)
        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"F1: {score[0]:0.5f}")
    print(f"Jaccard: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("/content/drive/MyDrive/results/score.csv")
