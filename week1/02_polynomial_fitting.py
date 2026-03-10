import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Ensure outputs directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*50)
print("Numerical Methods vs Neural Networks")
print("="*50)

# 1. 데이터 준비 (Data Preparation)
# 01_hello_nn.py와 동일한 데이터
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_clean = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Add random noise (scale=1.0)
np.random.seed(42) # For reproducibility
noise = np.random.normal(loc=0.0, scale=1.0, size=len(X))
y = y_clean + noise

print("Data:")
print(f"X: {X}")
print(f"y (clean): {y_clean}")
print(f"y (noisy): {y}")
print("-" * 50)

# ---------------------------------------------------------
# Method 1: Polynomial Fitting (NumPy)
# 다항식 근사: 최소자승법(Least Squares Method)을 이용한 해석적 해법
# ---------------------------------------------------------
print("\n[Method 1] NumPy Polyfit (Least Squares)")

# 1차 다항식 (y = ax + b)으로 피팅
# deg=1 : 1차식
coefficients = np.polyfit(X, y, deg=1)
slope_poly = coefficients[0]
intercept_poly = coefficients[1]

print(f"Result: y = {slope_poly:.4f}x + {intercept_poly:.4f}")
print(f"Expected: y = 2.0000x + -1.0000")

# 예측
new_x = 10.0
pred_poly = slope_poly * new_x + intercept_poly
print(f"Prediction for x={new_x}: {pred_poly:.4f}")


# ---------------------------------------------------------
# Method 2: Numerical Optimization (SciPy)
# 수치 최적화: curve_fit (Non-linear Least Squares)
# 신경망 학습과 유사하게 에러를 최소화하는 파라미터를 찾음
# ---------------------------------------------------------
print("\n[Method 2] SciPy Curve Fit (Optimization)")

def linear_function(x, w, b):
    return w * x + b

# 초기값 설정 (p0)
popt, pcov = curve_fit(linear_function, X, y, p0=[0.5, 0.5])
w_opt, b_opt = popt

print(f"Result: y = {w_opt:.4f}x + {b_opt:.4f}")

# 예측
pred_scipy = linear_function(new_x, w_opt, b_opt)
print(f"Prediction for x={new_x}: {pred_scipy:.4f}")


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# 원본 데이터 (Noisy)
plt.scatter(X, y, color='red', label='Noisy Data', s=100, zorder=5)
# 정답 데이터 (Clean)
plt.plot(X, y_clean, 'k:', label='True Function (y=2x-1)', alpha=0.5)

# 근사한 직선 그리기
x_range = np.linspace(-2, 11, 100)
y_poly = slope_poly * x_range + intercept_poly

plt.plot(x_range, y_poly, label=f'Polyfit: y={slope_poly:.2f}x{intercept_poly:.2f}', color='blue', linestyle='--')

# 예측 지점 표시
plt.scatter([new_x], [pred_poly], color='green', marker='*', s=200, label=f'Prediction (x={new_x})', zorder=5)

plt.title('Polynomial Fitting vs Neural Network Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, '02_numerical_fitting.png')
plt.savefig(save_path)
print(f"\nPlot saved to {save_path}")

# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------
print("-" * 50)
print("Summary:")
print("Neural Network (Previous): Iterative learning (Gradient Descent)")
print("NumPy Polyfit: Analytical solution (Linear Algebra)")
print("SciPy Curve Fit: Numerical optimization (Levenberg-Marquardt)")
print("-" * 50)
