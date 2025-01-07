
# import required packages
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Step 2: Preprocess training and validation images
train_generator = train_data_gen.flow_from_directory(
    'archive/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change to "rgb" for ResNet50
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'archive/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # Change to "rgb" for ResNet50
    class_mode='categorical'
)

# Step 3: Load the ResNet50 model pre-trained on ImageNet, without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Step 4: Freeze the base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Step 5: Add custom layers on top of ResNet50 base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout to avoid overfitting
x = Dense(512, activation='relu')(x)  # Another fully connected layer
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # Output layer with 7 classes (emotions)

# Step 6: Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Step 7: Compile the model with Adam optimizer and categorical crossentropy
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 8: Train only the custom layers (freeze base ResNet50 layers)
history_phase1=model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,  # Adjust according to dataset size
    epochs=10,  # Train for a few epochs
    validation_data=validation_generator,
    validation_steps=7178 // 64
)

print("Training accuracy after each epoch in Phase 1:")
for i, acc in enumerate(history_phase1.history['accuracy'], 1):
    print(f"Epoch {i}: Training Accuracy = {acc:.4f}, Validation Accuracy = {history_phase1.history['val_accuracy'][i-1]:.4f}")

# Step 9: Unfreeze the last few layers of ResNet50 for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Step 10: Re-compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 11: Continue training with fine-tuning
history_phase2=model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,  # Continue training with more epochs for fine-tuning
    validation_data=validation_generator,
    validation_steps=7178 // 64
)
 #Print training and validation accuracy for each epoch in Phase 2
print("\nTraining accuracy after each epoch in Phase 2:")
for i, acc in enumerate(history_phase2.history['accuracy'], 1):
    print(f"Epoch {i}: Training Accuracy = {acc:.4f}, Validation Accuracy = {history_phase2.history['val_accuracy'][i-1]:.4f}")

import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.plot(history_phase1.history['accuracy'] + history_phase2.history['accuracy'], label='Training Accuracy')
plt.plot(history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 12: Save the fine-tuned model structure to JSON
model_json = model.to_json()
with open("fine_tuned_resnet50_emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Step 13: Save the fine-tuned model weights to an .h5 file
#model.save_weights("fine_tuned_resnet50_emotion_model.weights.h5")
model.save("my_model.h5")

print("Model fine-tuning complete and saved.")