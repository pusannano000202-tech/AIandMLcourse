import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate synthetic data
def generate_data(n_samples):
    X = np.linspace(-3, 3, n_samples)
    y = np.sin(X) + np.random.normal(0, 0.1, n_samples)
    return X, y

X_train, y_train = generate_data(20) # Small data for overfitting
X_test, y_test = generate_data(100)

def create_model(complexity):
    model = models.Sequential()
    if complexity == 'underfit':
        model.add(layers.Dense(1, input_shape=(1,))) # Too simple
    elif complexity == 'overfit':
        model.add(layers.Dense(128, activation='relu', input_shape=(1,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1))
    else: # Balanced
        model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
    return model

models_dict = {}
histories = {}

print("Training models to demonstrate overfitting vs underfitting...")
for complexity in ['underfit', 'balanced', 'overfit']:
    print(f"Training {complexity} model...")
    model = create_model(complexity)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test))
    models_dict[complexity] = model
    histories[complexity] = history.history

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Predictions
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Training Data', color='red')
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
for complexity, model in models_dict.items():
    y_pred = model.predict(X_plot, verbose=0)
    plt.plot(X_plot, y_pred, label=f'{complexity} prediction')
plt.title('Model Predictions')
plt.legend()

# Plot 2: Loss Curves
plt.subplot(1, 2, 2)
for complexity, history in histories.items():
    plt.plot(history['val_loss'], label=f'{complexity} Val Loss')
plt.title('Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.ylim(0, 0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_overfitting_underfitting.png'))
print(f"Result saved to {os.path.join(output_dir, '02_overfitting_underfitting.png')}")
