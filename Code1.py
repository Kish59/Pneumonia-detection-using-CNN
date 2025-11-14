# STEP 1: Install Required Libraries
!pip install kagglehub

# STEP 2: Import Libraries
import kagglehub
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 3: Download Dataset from KaggleHub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("✅ Dataset Path:", path)

# STEP 4: Define Directory Paths
train_dir = os.path.join(path, "chest_xray/train")
val_dir = os.path.join(path, "chest_xray/val")
test_dir = os.path.join(path, "chest_xray/test")

# STEP 5: Image Preprocessing (Rescale + Resize)
IMG_SIZE = (150, 150)  # You can change to (224, 224) for higher quality
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

val_gen = datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

test_gen = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

print("✅ Image preprocessing completed and generators are ready!")
