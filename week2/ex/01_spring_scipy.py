import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 0. 환경 설정
# 상위 폴더의 outputs에 저장
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== SciPy를 이용한 선형 회귀 (Linear Regression with SciPy) ===")

# 1. 데이터 준비 (Data Preparation)
# week2/01_linear_regression_spring.py와 동일한 데이터 생성
weights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
true_lengths = 2 * weights + 10

np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=1.5, size=len(weights))
measured_lengths = true_lengths + noise

print("\n[데이터 확인]")
print("무게(kg):", weights)
print("측정된 길이(cm):", np.round(measured_lengths, 2))

# 2. 모델 정의 (Model Definition)
# 우리가 찾고 싶은 함수 형태: y = ax + b
def linear_func(x, a, b):
    return a * x + b

# 3. 최적화 (Optimization using curve_fit)
# curve_fit은 데이터와 가장 잘 맞는 파라미터(a, b)를 찾아줍니다.
# popt: 최적화된 파라미터 (Optimal parameters)
# pcov: 공분산 행렬 (Covariance matrix) - 추정의 불확실성을 알 수 있음
popt, pcov = curve_fit(linear_func, weights, measured_lengths)

learned_a, learned_b = popt

print(f"\n[학습 결과 (SciPy)]")
print(f"예측된 식: 길이 = {learned_a:.2f} * 무게 + {learned_b:.2f}")
print(f"실제 식  : 길이 = 2.00 * 무게 + 10.00")

# 4. 예측 (Prediction)
new_weight = 15.0
predicted_length = linear_func(new_weight, learned_a, learned_b)
print(f"\n[예측 테스트]")
print(f"15kg 추를 매달았을 때 예측 길이: {predicted_length:.2f} cm")

# 5. 시각화 (Visualization)
plt.figure(figsize=(10, 6))

# 실제 데이터
plt.scatter(weights, measured_lengths, color='blue', label='Measured Data (Noisy)')

# 진짜 법칙
plt.plot(weights, true_lengths, 'g--', label='True Law (y=2x+10)')

# SciPy 예측 결과
x_range = np.linspace(0, 15, 100)
y_pred = linear_func(x_range, learned_a, learned_b)
plt.plot(x_range, y_pred, 'r-', label='SciPy Fit')

plt.title('Hooke\'s Law Regression (SciPy curve_fit)')
plt.xlabel('Weight (kg)')
plt.ylabel('Spring Length (cm)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, 'ex_01_spring_scipy.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
