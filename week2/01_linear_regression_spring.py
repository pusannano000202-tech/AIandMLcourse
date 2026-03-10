import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정 (Environment Setup)
# 결과를 저장할 폴더 생성
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"TensorFlow Version: {tf.__version__}")

# 1. 데이터 준비 (Data Preparation)
# 훅의 법칙 (Hooke's Law): F = kx (힘 = 용수철 상수 * 늘어난 길이)
# 여기서는 무게(Weight, kg)에 따른 용수철의 전체 길이(Length, cm)를 예측해봅니다.
# 식: Length = (Initial Length) + (Stretch per kg) * Weight
# 가정: 초기 길이 = 10cm, 1kg당 2cm 늘어남 -> Length = 2 * Weight + 10

# 입력 데이터: 추의 무게 (kg)
weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

# 정답 데이터 (Clean): 이상적인 용수철 길이
true_lengths = 2 * weights + 10

# 실제 데이터 (Noisy): 측정 오차 추가 (랜덤 노이즈)
np.random.seed(42) # 결과를 똑같이 만들기 위해 랜덤 시드 고정
noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights)) # 표준편차 1.5cm의 오차
measured_lengths = true_lengths + noise

print("\n[데이터 확인]")
print("무게(kg):", weights)
print("측정된 길이(cm):", np.round(measured_lengths, 2))

# 2. 모델 구성 (Model Architecture)
# 입력(무게) 1개 -> 출력(길이) 1개인 가장 단순한 선형 모델
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 모델 컴파일 (Compilation)
# 최적화 도구(Optimizer): SGD (확률적 경사 하강법) - 학습률(learning_rate) 0.01
# 손실 함수(Loss): MSE (평균 제곱 오차) - 예측값과 실제값의 차이를 제곱해서 평균 냄
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), 
              loss='mean_squared_error')

# 4. 모델 학습 (Training)
print("\n[학습 시작]")
# epochs: 전체 데이터를 몇 번 반복해서 공부할 것인가
history = model.fit(weights, measured_lengths, epochs=500, verbose=0)
print("학습 완료!")

# 5. 결과 확인 및 예측 (Prediction)
# 학습된 파라미터(가중치와 편향) 확인
learned_w = float(model.layers[0].get_weights()[0][0]) # 기울기 (1kg당 늘어나는 길이)
learned_b = float(model.layers[0].get_weights()[1][0]) # 절편 (초기 길이)

print(f"\n[학습 결과]")
print(f"예측된 식: 길이 = {learned_w:.2f} * 무게 + {learned_b:.2f}")
print(f"실제 식  : 길이 = 2.00 * 무게 + 10.00")

# 새로운 무게에 대한 예측
new_weight = 15.0
predicted_length = float(model.predict(np.array([[new_weight]]), verbose=0)[0][0])
print(f"\n[예측 테스트]")
print(f"15kg 추를 매달았을 때 예측 길이: {predicted_length:.2f} cm")
print(f"이론상 실제 길이: {2 * new_weight + 10:.2f} cm")

# 6. 시각화 (Visualization)
plt.figure(figsize=(10, 6))

# 1. 실제 측정 데이터 (점)
plt.scatter(weights, measured_lengths, color='blue', label='Measured Data (Noisy)')

# 2. 진짜 법칙 (점선)
plt.plot(weights, true_lengths, 'g--', label='True Law (y=2x+10)')

# 3. AI가 예측한 선 (빨간 실선)
# 그래프를 그리기 위한 예측값 생성
plot_weights = np.linspace(0, 15, 100) # 0부터 15까지 100개의 점
plot_lengths = model.predict(plot_weights.reshape(-1, 1), verbose=0)
plt.plot(plot_weights, plot_lengths, 'r-', label='AI Prediction')

plt.title('Hooke\'s Law Regression (Spring Experiment)')
plt.xlabel('Weight (kg)')
plt.ylabel('Spring Length (cm)')
plt.legend()
plt.grid(True)

# 그래프 저장
save_path = os.path.join(output_dir, 'spring_fitting.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
