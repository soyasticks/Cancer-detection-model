import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load CSV metadata
data = pd.read_csv('dataset.csv')
print(f"Total images in dataset: {len(data)}")

# Create image paths
data['path'] = data['image_id'].apply(lambda x: os.path.join('images', f'{x}.jpg'))

# Verify images exist
existing_images = []
for path in data['path']:
    if os.path.exists(path):
        existing_images.append(True)
    else:
        print(f"Warning: Image not found: {path}")
        existing_images.append(False)

data = data[existing_images]
print(f"Images found: {len(data)}")

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

# 2. Load and preprocess images
IMG_SIZE = 224  # Increased size for better feature detection
images = []
labels = []

for i, row in data.iterrows():
    try:
        img = cv2.imread(row['path'])
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(row['label_encoded'])
    except Exception as e:
        print(f"Error processing image {row['path']}: {str(e)}")
        continue

X = np.array(images) / 255.0
y = np.array(labels)

# 3. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 4. Define CNN with improved architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes: melanoma and squamous
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train with data augmentation
history = model.fit(
    datagen.flow(X, y, batch_size=2),
    steps_per_epoch=len(X) // 2,
    epochs=10
)

# 7. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 8. Save model
model.save("skin_cancer_model.h5")
print("Model saved as skin_cancer_model.h5")

# Print class mapping
print("\nClass mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")
