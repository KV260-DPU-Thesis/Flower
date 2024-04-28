import numpy as np
import cv2
import os
import time
import tensorflow as tf

# Constants
MODEL_PATH = "flower_model.h5"
IMAGES_FOLDER = "images"

flower_labels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# List images in the directory
image_filenames = [f for f in os.listdir(IMAGES_FOLDER) if os.path.isfile(os.path.join(IMAGES_FOLDER, f))]
num_images = len(image_filenames)

all_process_start_time = time.time()
total_inference_duration_ms = 0

# Process each image
for filename in image_filenames:
    IMG_PATH = os.path.join(IMAGES_FOLDER, filename)
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # normalization

    # Start timing for inference
    inference_start_time = time.time()

    # Predict
    predictions = model.predict(image, verbose=0)
    idx = np.argmax(predictions[0][0])

    print(f"{filename} - Prediction - {flower_labels[idx]} with accuracy {predictions[0][idx]:.5f}")

    # Inference duration
    inference_end_time = time.time()
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    total_inference_duration_ms += inference_duration_ms

# Summarize the process
average_inference_time_ms = total_inference_duration_ms / num_images

print(f"Average inference duration: {average_inference_time_ms:.5f} ms")
print(f"Total inference duration: {total_inference_duration_ms:.5f} ms")

all_process_end_time = time.time()
all_process_duration_ms = (all_process_end_time - all_process_start_time) * 1000
print(f"All process duration: {all_process_duration_ms:.5f} ms")
