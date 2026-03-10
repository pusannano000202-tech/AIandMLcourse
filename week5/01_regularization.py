import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_model(regularization_type='none'):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
    
    if regularization_type == 'l2':
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    elif regularization_type == 'dropout':
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
    elif regularization_type == 'batch_norm':
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation='relu'))
    else:
        model.add(layers.Dense(128, activation='relu'))
        
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Generate dummy data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
# Use a small subset to make training fast for demonstration
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:200]
y_test = y_test[:200]

# Train models
configs = ['none', 'l2', 'dropout', 'batch_norm']
history_dict = {}

print("Training models with different regularization techniques...")
for config in configs:
    print(f"Training with {config}...")
    model = create_model(config)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
    history_dict[config] = history.history['val_loss']

# Plot results
plt.figure(figsize=(10, 6))
for config, val_loss in history_dict.items():
    plt.plot(val_loss, label=f'{config}')

plt.title('Validation Loss with Different Regularization')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, '01_regularization_plot.png'))
print(f"Result saved to {os.path.join(output_dir, '01_regularization_plot.png')}")
