import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading pre-trained MobileNetV2 model...")
# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model
base_model.trainable = False

print("Building new model on top...")
model = models.Sequential([
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(1) # Binary classification example
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Save summary to file
summary_file = os.path.join(output_dir, '04_transfer_learning_summary.txt')
with open(summary_file, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"Model summary saved to {summary_file}")
print("Note: Actual training is skipped in this example to save time and resources, but the model structure is ready for Transfer Learning.")
