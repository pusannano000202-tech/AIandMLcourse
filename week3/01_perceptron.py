import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.font_manager as fm

# 한글 폰트 설정 (Robust)
def set_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in font_list:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'Gulim' in font_list:
        plt.rcParams['font.family'] = 'Gulim'
    elif 'Batang' in font_list:
        plt.rcParams['font.family'] = 'Batang'
    elif 'AppleGothic' in font_list:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 퍼셉트론 (Perceptron) 예제 ===")

# 1. 퍼셉트론 정의 (Perceptron)
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # 가중치와 편향 랜덤 초기화
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate
    
    def activation(self, x):
        # 계단 함수 (Step Function)
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        # y = step(w·x + b)
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation(summation)
    
    def train(self, training_inputs, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # 오차 계산
                error = label - prediction
                # 가중치 업데이트
                self.weights += self.lr * error * inputs
                self.bias += self.lr * error

# 2. AND 게이트 학습
print("\n[AND 게이트 학습]")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X_and, y_and, epochs=100)

print("입력 | 예측 | 정답")
for inputs, label in zip(X_and, y_and):
    pred = perceptron_and.predict(inputs)
    print(f"{inputs} | {pred}  | {label}")

# 3. OR 게이트 학습
print("\n[OR 게이트 학습]")
y_or = np.array([0, 1, 1, 1])

perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_and, y_or, epochs=100)

print("입력 | 예측 | 정답")
for inputs, label in zip(X_and, y_or):
    pred = perceptron_or.predict(inputs)
    print(f"{inputs} | {pred}  | {label}")

# 4. XOR 게이트 시도 (실패할 것!)
print("\n[XOR 게이트 학습 시도]")
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X_and, y_xor, epochs=1000)

print("입력 | 예측 | 정답")
errors = 0
for inputs, label in zip(X_and, y_xor):
    pred = perceptron_xor.predict(inputs)
    print(f"{inputs} | {pred}  | {label}")
    if pred != label:
        errors += 1

print(f"\n오류 개수: {errors}/4")
print("→ XOR은 단일 퍼셉트론으로 해결 불가능!")

# 5. 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def plot_decision_boundary(ax, perceptron, X, y, title):
    # 결정 경계 그리기
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    xx,yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = np.array([perceptron.predict(np.array([xi, yi])) 
                  for xi, yi in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
    
    # 데이터 포인트
    for i, (point, label) in enumerate(zip(X, y)):
        color = 'red' if label == 1 else 'blue'
        marker = 'o' if label == 1 else 'x'
        ax.scatter(point[0], point[1], c=color, marker=marker, s=200, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# AND, OR, XOR 시각화
plot_decision_boundary(axes[0], perceptron_and, X_and, y_and, 'AND Gate (선형 분리 가능)')
plot_decision_boundary(axes[1], perceptron_or, X_and, y_or, 'OR Gate (선형 분리 가능)')
plot_decision_boundary(axes[2], perceptron_xor, X_and, y_xor, 'XOR Gate (선형 분리 불가능!)')

plt.tight_layout()
save_path = os.path.join(output_dir, '01_perceptron.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
