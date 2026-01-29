import os
import shutil
import numpy as np
import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import ast

dataset_dir = "grocery_data"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

wanted_categories = ["cucumber", "banana", "tomato", "milk", "cheese", "butter", "chicken", "bacon"]

path = kagglehub.dataset_download("liamboyd1/multi-class-food-image-dataset")
source_root = os.path.join(path, "train")
if not os.path.exists(source_root):
    source_root = path 

for category in wanted_categories:
    dest_path = os.path.join(dataset_dir, category)
    found_folder = None
    for f in os.listdir(source_root):
        if f.lower() == category.lower():
            found_folder = f
            break
            

    shutil.copytree(os.path.join(source_root, found_folder), dest_path)
    print(f" -> Downloaded & Added: {category}")





batch_size = 32
img_height = 180 
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


print("\nBuilding Model")

num_classes = len(class_names)

model = Sequential([
  # Fix: Explicit Input layer
  tf.keras.Input(shape=(img_height, img_width, 3)),
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print("\nTraining")

epochs = 10 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


print("\nSavingModel to grocery_model")
model.save('grocery_model.h5')

class_indices = {i: name for i, name in enumerate(class_names)}
with open('class_indices.txt', 'w') as f:
    f.write(str(class_indices))
print("END, Run TASK 3 And hope it works ")
