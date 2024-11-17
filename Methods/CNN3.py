import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_dir = '/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_GS4/Train'
test_dir = '/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_GS4/Test'

# Image parameters
#img_height, img_width = 128, 128
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    #target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    #target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu'),#, input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
epochs = 10
history = model.fit(
    train_generator,
    #steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    #validation_steps=test_generator.samples // batch_size
)

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Predict probabilities on the test set
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

# Calculate evaluation metrics
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
precision.update_state(y_true, y_pred)
recall.update_state(y_true, y_pred)
f1_score = 2 * (precision.result().numpy() * recall.result().numpy()) / (precision.result().numpy() + recall.result().numpy())
auc_roc = roc_auc_score(y_true, y_pred_prob)

print(f"Test accuracy: {test_accuracy:.10f}")
print(f"Precision: {precision.result().numpy():.10f}")
print(f"Recall: {recall.result().numpy():.10f}")
print(f"F1 Score: {f1_score:.10f}")
print(f"AUC-ROC: {auc_roc:.10f}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load a single image from your test data
img_path = '/Users/philipnegrin/Downloads/AICodeDetection/Code_Net/Data_GS4/Test/AI/AI_image10.png'
img = tf.keras.preprocessing.image.load_img(img_path)#, target_size=(img_height, img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

# Get the class index
predicted_class = np.argmax(model.predict(img_array), axis=-1)[0]

# Compute gradients
with tf.GradientTape() as tape:
    tape.watch(img_array)
    predictions = model(img_array)
    loss = predictions[0][predicted_class]  # Focus on the predicted class

grads = tape.gradient(loss, img_array)

# Compute saliency map
saliency = np.max(np.abs(grads), axis=-1)[0]  # Aggregate across color channels

# Normalize for visualization
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

# Plot the original image and saliency map
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(tf.keras.preprocessing.image.array_to_img(img_array[0]))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency, cmap='hot')
plt.title("Saliency Map")
plt.axis('off')

plt.tight_layout()
plt.show()



"""
# Plotting loss and accuracy over epochs
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Time')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Time')

plt.tight_layout()
plt.show()

"""
