import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("TensorFlow Version:", tf.__version__)

# 1. 데이터 준비 (Data Preparation)
# 학습할 관계: y = 2x - 1
# 입력(x)와 정답(y) 데이터 생성
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_clean = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Add random noise
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.0, size=len(X))
y = y_clean + noise

print("\nTraining Data:")
print("X:", X)
print("y (clean):", y_clean)
print("y (noisy):", y)

# 2. 모델 구성 (Model Architecture)
# 가장 간단한 신경망: 1개의 뉴런, 1개의 입력
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 모델 컴파일 (Compilation)
# Optimizer: SGD (Stochastic Gradient Descent, 확률적 경사 하강법)
# Loss function: MSE (Mean Squared Error, 평균 제곱 오차)
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. 모델 학습 (Training)
print("\nStarting training...")
history = model.fit(X, y, epochs=500, verbose=0)
print("Training finished!")

# 5. 예측 (Prediction)
# 학습된 모델로 새로운 데이터(x=10.0)에 대한 예측 수행
# 정답은 2*10 - 1 = 19.0 이어야 함
new_x = 10.0
prediction = model.predict(np.array([[new_x]]))
print(f"\nPrediction for x={new_x}: {prediction[0][0]:.4f}")
print(f"Expected value: {2 * new_x - 1}")

# 6. 학습 과정 시각화 (Visualization)
# 6-1. Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_loss.png'))
print(f"\nLoss plot saved to {os.path.join(output_dir, 'training_loss.png')}")

# 6-2. Model Fit Graph
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Noisy Data', s=100)
plt.plot(X, y_clean, 'k:', label='True Function (y=2x-1)', alpha=0.5)

# Predict for plotting line
x_range = np.linspace(-2, 5, 100)
y_pred = model.predict(x_range.reshape(-1, 1), verbose=0)
plt.plot(x_range, y_pred, label='Neural Network Fit', color='blue')

plt.title('Neural Network Regression with Noisy Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'model_fit.png'))
print(f"Model fit plot saved to {os.path.join(output_dir, 'model_fit.png')}")

# 7. 모델 가중치 확인
weights = model.get_weights()
w = weights[0][0][0]
b = weights[1][0]
print(f"\nLearned Parameters:")
print(f"Weight (w): {w:.4f} (Expected: 2.0)")
print(f"Bias (b): {b:.4f} (Expected: -1.0)")
print(f"Formula: y = {w:.4f}x + {b:.4f}")
