import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 최적화 (Optimization) 예제: 경사 하강법 (Gradient Descent) ===")

# 1. 손실 함수 정의 (Loss Function)
# 가장 간단한 2차 함수: y = x^2
# 목표: y가 가장 작아지는 x를 찾아라! (정답은 x=0)
def loss_function(x):
    return x**2

# 미분 함수 (기울기)
# y = x^2 의 미분은 2x
def gradient(x):
    return 2 * x

# 2. 경사 하강법 시뮬레이션
# 산 꼭대기에서 공을 굴리는 것과 같습니다.

# 시작 위치 (랜덤하게 -4에서 시작)
current_x = -4.0
learning_rate = 0.1 # 한 번에 이동하는 보폭 (너무 크면 튕겨나가고, 너무 작으면 느림)
steps = [] # 이동 경로를 기록할 리스트

print(f"시작 위치: x = {current_x}")

for i in range(20): # 20번 이동
    # 현재 위치 기록
    current_loss = loss_function(current_x)
    steps.append((current_x, current_loss))
    
    # 기울기 계산 (어느 쪽이 내리막길인가?)
    grad = gradient(current_x)
    
    # 이동 (기울기 반대 방향으로 보폭만큼)
    # x_new = x_old - (learning_rate * gradient)
    current_x = current_x - learning_rate * grad
    
    print(f"Step {i+1}: x = {current_x:.4f}, Loss = {current_loss:.4f}")

print(f"\n최종 위치: x = {current_x:.4f} (목표값 0.0에 매우 가까움)")

# 3. 시각화 (Visualization)
plt.figure(figsize=(10, 6))

# (1) 손실 함수 그래프 그리기 (배경)
x_range = np.linspace(-5, 5, 100)
y_range = loss_function(x_range)
plt.plot(x_range, y_range, 'k-', label='Loss Function (y=x^2)')

# (2) 공이 굴러가는 경로 그리기
steps = np.array(steps)
plt.scatter(steps[:, 0], steps[:, 1], color='red', s=100, zorder=5) # 점
plt.plot(steps[:, 0], steps[:, 1], 'r--', label='Gradient Descent Path') # 점선 연결

# 시작점과 끝점 표시
plt.text(steps[0, 0], steps[0, 1] + 1, 'Start', ha='center', color='red', fontweight='bold')
plt.text(steps[-1, 0], steps[-1, 1] + 1, 'End', ha='center', color='red', fontweight='bold')

plt.title('Optimization: Gradient Descent (Ball rolling down)')
plt.xlabel('Parameter x')
plt.ylabel('Loss y')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, '04_gradient_descent.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
print("설명: 빨간 점들이 경사를 타고 점점 가장 낮은 곳(0)으로 내려가는 것을 볼 수 있습니다.")
