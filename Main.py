import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

# Download latest dataset version
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Dataset path:", path)

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 43
EPOCHS = 10

# Helper: crop to ROI then resize
def preprocess_image(row, base_dir, target_size):
    img_path = os.path.join(base_dir, row.Path)
    img = cv2.imread(img_path)
    # Extract ROI coords
    x1, y1, x2, y2 = map(int, [row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]])
    crop = img[y1:y2, x1:x2]
    resized = cv2.resize(crop, target_size)
    return resized

# Generator yielding batches of (images, labels)
def make_dataset(df, base_dir, batch_size, subset, datagen):
    df = df.copy().reset_index(drop=True)
    num_samples = len(df)
    while True:
        idxs = np.random.choice(num_samples, batch_size)
        images, labels = [], []
        for i in idxs:
            row = df.loc[i]
            img = preprocess_image(row, base_dir, IMAGE_SIZE)
            if subset == "training":
                img = datagen.random_transform(img)
            img = img.astype("float32") / 255.0
            images.append(img)
            labels.append(int(row.ClassId))
        X = np.stack(images)
        y = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
        yield X, y

# Load and split data
df = pd.read_csv(os.path.join(path, "Train.csv"))
df.ClassId = df.ClassId.astype(int)
train_df = df.sample(frac=0.8, random_state=42)
val_df   = df.drop(train_df.index)

# DataGenerators for augmentation and normalization
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator()

steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df)   // BATCH_SIZE
train_gen = make_dataset(train_df, path, BATCH_SIZE, "training", train_datagen)
val_gen   = make_dataset(val_df,   path, BATCH_SIZE, "validation", val_datagen)

# Build model (Stage A: head only)
def build_model():
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Main training routine
if __name__ == "__main__":
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Stage A: train head
    print("--- Stage A: Training head only ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stop]
    )

    # Stage B: fine-tune last conv block
    for layer in model.layers:
        if layer.name.startswith("block5_"):
            layer.trainable = True
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("--- Stage B: Fine-tuning last conv block ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stop]
    )

    # Final evaluation & saving
    print("Evaluating on validation set...")
    loss, acc = model.evaluate(val_gen, steps=validation_steps)
    print(f"Val Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
    model.save("gtsrb_finetuned_model.keras")
    print("Model saved as gtsrb_finetuned_model.keras")
