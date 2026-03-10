import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# 0. 환경 설정
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== SciPy를 이용한 최적화 (Optimization with SciPy) ===")

# 1. 목적 함수 정의 (Objective Function)
# 조금 더 재미있는 함수: y = (x-2)^2 + 1
# 최소값은 x=2일 때 y=1
def objective_func(x):
    return (x - 2)**2 + 1

# 2. 최적화 수행 (Optimization)
# 시작점
x0 = -3.0
print(f"시작 위치: x = {x0}")

# 과정을 기록하기 위한 콜백 함수
history = []
def callback(x):
    history.append(x[0])

# minimize 함수 사용
# method='BFGS': 널리 쓰이는 준-뉴턴(Quasi-Newton) 방법 중 하나
result = minimize(objective_func, x0, method='BFGS', callback=callback)

print("\n[최적화 결과]")
print(f"성공 여부: {result.success}")
print(f"메시지: {result.message}")
print(f"최적의 x: {result.x[0]:.4f}")
print(f"최소값 y: {result.fun:.4f}")
print(f"반복 횟수: {result.nit}")

# 3. 시각화 (Visualization)
plt.figure(figsize=(10, 6))

# 함수 그래프
x_range = np.linspace(-4, 6, 100)
y_range = objective_func(x_range)
plt.plot(x_range, y_range, 'k-', label='Objective Function y=(x-2)^2 + 1')

# 이동 경로
# 시작점 추가
path_x = [x0] + history
path_y = [objective_func(x) for x in path_x]

path_x = np.array(path_x)
path_y = np.array(path_y)

plt.scatter(path_x, path_y, color='red', s=100, zorder=5)
plt.plot(path_x, path_y, 'r--', label='Optimization Path (BFGS)')

# 시작과 끝 표시
plt.text(path_x[0], path_y[0] + 1, 'Start', ha='center', color='red', fontweight='bold')
plt.text(path_x[-1], path_y[-1] + 1, 'End', ha='center', color='red', fontweight='bold')

plt.title('Optimization using SciPy minimize (BFGS)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, 'ex_04_optimization_scipy.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
