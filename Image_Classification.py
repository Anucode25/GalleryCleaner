import os
import shutil
import cv2
import numpy as np
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Step 1: Load and Preprocess Images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize to match model input size
            images.append(img)
            labels.append(label)  # Assign numerical label
    return np.array(images), np.array(labels)

# Paths to your data folders
meme_folder = r"D:\memes"
genuine_folder = r"D:\Genuine"

# Load data from folders
meme_images, meme_labels = load_images_from_folder(meme_folder, 0)  # Label 0 for memes
genuine_images, genuine_labels = load_images_from_folder(genuine_folder, 1)  # Label 1 for genuine

# Combine and shuffle data
images = np.concatenate((meme_images, genuine_images))
labels = np.concatenate((meme_labels, genuine_labels))
images, labels = shuffle(images, labels, random_state=42)

# Normalize image data
images = images / 255.0

# Step 2: Create a Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Freeze base model layers
base_model.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Split Data into Training and Validation Sets
split_idx = int(0.8 * len(images))  # 80% training, 20% validation
train_images, val_images = images[:split_idx], images[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

# Step 4: Train the Model and Track Training History
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=32)

# Step 5: Plot Training and Validation Accuracy Graph
def plot_training_history(history):
    # Plotting training & validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_training_history(history)

# Step 6: Evaluate Model Accuracy on the Validation Set
val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Step 7: Classify New Images and Move Meme Images to Trash
def classify_and_move_images(model, image_folder, trash_folder):
    # Create the trash folder if it doesn't exist
    if not os.path.exists(trash_folder):
        os.makedirs(trash_folder)

    total_images = 0
    memes_moved = 0
    start_time = time.time()

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            total_images += 1
            img = cv2.resize(img, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            prediction = model.predict(img)[0][0]
            label = "genuine" if prediction > 0.5 else "meme"

            # If it's a meme, move it to the trash folder
            if label == "meme":
                memes_moved += 1
                trash_path = os.path.join(trash_folder, filename)
                shutil.move(img_path, trash_path)

    end_time = time.time()
    total_time = end_time - start_time

    return total_images, memes_moved, total_time

# Path to the folder with new images
new_images_folder = r"D:\input_images"
trash_folder = r"D:\trash"  # Folder to store meme images

# Ensure folder exists and is not empty
if not os.path.exists(new_images_folder):
    raise ValueError(f"The folder '{new_images_folder}' does not exist.")
if len(os.listdir(new_images_folder)) == 0:
    raise ValueError(f"The folder '{new_images_folder}' is empty.")

# Classify images and move meme images to the trash folder
classified_count, memes_count, processing_time = classify_and_move_images(model, new_images_folder, trash_folder)

# Output the results
print(f"\nClassification Results:")
print(f"Total Images Classified: {classified_count}")
print(f"Memes Moved to Trash: {memes_count}")
print(f"Time Taken for Processing: {processing_time:.2f} seconds")
