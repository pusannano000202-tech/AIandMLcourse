import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load a sample image (using MNIST for simplicity, but treating it as an image)
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
image = x_train[0]
image = np.expand_dims(image, axis=-1) # (28, 28, 1)
image = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image)) # Convert to 3 channels for standard augmentation layers if needed, though grayscale works too.
# Let's resize it to be a bit bigger to see effects better
image = tf.image.resize(image, [100, 100])
image = tf.expand_dims(image, 0) / 255.0 # (1, 100, 100, 3)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

print("Generating augmented images...")
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")

plt.suptitle('Data Augmentation Examples')
plt.savefig(os.path.join(output_dir, '03_augmentation_examples.png'))
print(f"Result saved to {os.path.join(output_dir, '03_augmentation_examples.png')}")
