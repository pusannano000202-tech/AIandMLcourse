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

print("=== Numpy로 구현하는 MLP (XOR 문제 해결) ===")

# 1. 활성화 함수와 미분
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 오버플로우 방지

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 2. MLP 클래스 (순수 Numpy 구현)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # 가중치 초기화 (Xavier Initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.lr = learning_rate
        self.loss_history = []
    
    def forward(self, X):
        """순전파 (Forward Propagation)"""
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """역전파 (Backpropagation)"""
        m = X.shape[0]  # 샘플 개수
        
        # Output layer gradients
        dz2 = output - y  # MSE 미분 * Sigmoid 미분 (간소화)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 가중치 업데이트
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs, verbose=True):
        """학습"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Loss 계산 (MSE)
            loss = np.mean((output - y) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, output)
            
            # 진행 상황 출력
            if verbose and (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if verbose:
            print(f"\n최종 Loss: {self.loss_history[-1]:.6f}")
    
    def predict(self, X):
        """예측"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# 3. XOR 데이터
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)

y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=float)

print("\n[학습 데이터]")
print("입력 | 출력")
for inputs, label in zip(X_xor, y_xor):
    print(f"{inputs} | {label[0]}")

# 4. 모델 학습
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
print("\n[학습 시작]")
mlp.train(X_xor, y_xor, epochs=10000, verbose=True)

# 5. 결과 확인
print("\n[학습 결과]")
predictions = mlp.forward(X_xor)
print("입력 | 예측 | 정답")
for inputs, pred, label in zip(X_xor, predictions, y_xor):
    print(f"{inputs} | {pred[0]:.4f} | {int(label[0])}")

# 정확도
pred_labels = mlp.predict(X_xor)
accuracy = np.mean(pred_labels == y_xor.astype(int)) * 100
print(f"\n정확도: {accuracy:.1f}%")
print("→ XOR 문제 해결 성공! Multi-Layer Perceptron의 힘!")

# 6. 시각화
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 6-1. Loss 그래프
ax = axes[0]
ax.plot(mlp.loss_history, linewidth=2)
ax.set_title('Training Loss (MSE)', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# 6-2. 결정 경계
ax = axes[1]
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
plt.colorbar(contour, ax=ax, label='Output Probability')

# 데이터 포인트
for i, (point, label) in enumerate(zip(X_xor, y_xor)):
    color = 'red' if label[0] == 1 else 'blue'
    marker = 'o' if label[0] == 1 else 'x'
    
    if marker == 'x':
        ax.scatter(point[0], point[1], c=color, marker=marker, s=300, linewidth=3, zorder=5)
    else:
        ax.scatter(point[0], point[1], c=color, marker=marker, s=300, edgecolors='black', linewidth=3, zorder=5)
    
    ax.text(point[0], point[1]-0.15, f'({int(point[0])},{int(point[1])})', 
            ha='center', fontsize=10, fontweight='bold')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.3)

# 6-3. 은닉층 활성화 시각화
ax = axes[2]
hidden_activations = mlp.a1  # (4, 4) - 4개 샘플, 4개 은닉 뉴런

im = ax.imshow(hidden_activations.T, cmap='viridis', aspect='auto')
ax.set_yticks(range(4))
ax.set_yticklabels([f'Hidden {i+1}' for i in range(4)])
ax.set_xticks(range(4))
ax.set_xticklabels(['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
ax.set_title('Hidden Layer Activations', fontsize=14, fontweight='bold')
ax.set_xlabel('Input Pattern')
plt.colorbar(im, ax=ax, label='Activation')

# 각 셀에 값 표시
for i in range(4):
    for j in range(4):
        text = ax.text(j, i, f'{hidden_activations[j, i]:.2f}',
                      ha="center", va="center", color="white", fontweight='bold')

plt.tight_layout()
save_path = os.path.join(output_dir, '04_mlp_training.png')
plt.savefig(save_path, dpi=100)
print(f"\n그래프가 저장되었습니다: {save_path}")

print("\n=== Backpropagation 핵심 공식 ===")
print("1. Forward Pass:")
print("   z₁ = X @ W₁ + b₁")
print("   a₁ = σ(z₁)")
print("   z₂ = a₁ @ W₂ + b₂")
print("   a₂ = σ(z₂)")
print("\n2. Backward Pass (Chain Rule):")
print("   δ₂ = (a₂ - y) ⊙ σ'(z₂)")
print("   dW₂ = a₁ᵀ @ δ₂")
print("   δ₁ = (δ₂ @ W₂ᵀ) ⊙ σ'(z₁)")
print("   dW₁ = Xᵀ @ δ₁")
print("\n3. Update:")
print("   W ← W - α·dW")
