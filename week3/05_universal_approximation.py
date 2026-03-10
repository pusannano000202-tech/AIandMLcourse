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

print("=== Universal Approximation Theorem 시연 ===")

# 1. 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# 2. 간단한 신경망 (1개 은닉층)
class UniversalApproximator:
    def __init__(self, n_hidden, activation='tanh'):
        self.n_hidden = n_hidden
        self.activation = activation
        
        # 가중치 초기화
        # 가중치 초기화 (Xavier Initialization)
        # Tanh 활성화 함수에 적합한 초기화
        limit = np.sqrt(6 / (1 + n_hidden))
        self.W1 = np.random.uniform(-limit, limit, (1, n_hidden))
        self.b1 = np.zeros(n_hidden)
        
        limit = np.sqrt(6 / (n_hidden + 1))
        self.W2 = np.random.uniform(-limit, limit, (n_hidden, 1))
        self.b2 = np.zeros(1)
    
    def get_param_count(self):
        """총 파라미터 수 계산"""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def activate(self, x):
        if self.activation == 'tanh':
            return tanh(x)
        elif self.activation == 'relu':
            return relu(x)
        else:
            return sigmoid(x)
    
    def forward(self, x):
        # 은닉층
        z1 = x @ self.W1 + self.b1
        a1 = self.activate(z1)
        
        # 출력층
        z2 = a1 @ self.W2 + self.b2
        
        return z2
    
    def train(self, X, y, epochs=5000, lr=0.01):
        for epoch in range(epochs):
            # Forward
            z1 = X @ self.W1 + self.b1
            a1 = self.activate(z1)
            output = a1 @ self.W2 + self.b2
            
            # Loss
            loss = np.mean((output - y)**2)
            
            # Backward (간단한 업데이트)
            dL_doutput = 2 * (output - y) / len(X)
            dL_dW2 = a1.T @ dL_doutput
            dL_db2 = np.sum(dL_doutput, axis=0)
            
            dL_da1 = dL_doutput @ self.W2.T
            
            if self.activation == 'tanh':
                dL_dz1 = dL_da1 * (1 - a1**2)
            elif self.activation == 'relu':
                dL_dz1 = dL_da1 * (z1 > 0)
            else:
                dL_dz1 = dL_da1 * a1 * (1 - a1)
            
            dL_dW1 = X.T @ dL_dz1
            dL_db1 = np.sum(dL_dz1, axis=0)
            
            # Update
            self.W2 -= lr * dL_dW2
            self.b2 -= lr * dL_db2
            self.W1 -= lr * dL_dW1
            self.b1 -= lr * dL_db1
            
            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

# 3. 근사할 함수들
def target_sin(x):
    """사인 함수"""
    return np.sin(2 * np.pi * x)

def target_step(x):
    """계단 함수"""
    return np.where(x < 0.5, 0, 1)

def target_complex(x):
    """복잡한 함수"""
    return np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.3*np.cos(6*np.pi*x)

# 4. 데이터 생성
x_train = np.linspace(0, 1, 100).reshape(-1, 1)
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

# 5. 시각화
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# 함수 목록
targets = [
    ('Sine Wave', target_sin),
    ('Step Function', target_step),
    ('Complex Function', target_complex)
]

# 뉴런 수 비교
neuron_counts = [3, 10, 50]

print("\n=== 학습 시작 ===")

for col, (title, target_func) in enumerate(targets):
    print(f"\n{title}:")
    y_train = target_func(x_train)
    y_test_true = target_func(x_test)
    
    for row, n_neurons in enumerate(neuron_counts):
        print(f"\n  {n_neurons}개 뉴런 사용:")
        ax = axes[row, col]
        
        # 모델 학습
        # 모델 학습 (학습률 조정)
        model = UniversalApproximator(n_hidden=n_neurons, activation='tanh')
        # 뉴런이 많을수록 학습률을 낮춰야 안정적임
        current_lr = 0.05 if n_neurons < 20 else 0.01
        print(f"  (Total Parameters: {model.get_param_count()})")
        model.train(x_train, y_train, epochs=5000, lr=current_lr)
        
        # 예측
        y_pred = model.forward(x_test)
        
        # 플롯
        ax.plot(x_test, y_test_true, 'b-', linewidth=2, label='True Function', alpha=0.7)
        ax.plot(x_test, y_pred, 'r--', linewidth=2, label=f'NN ({n_neurons} neurons)')
        ax.scatter(x_train[::10], y_train[::10], c='green', s=30, alpha=0.5, label='Training Data')
        
        # MSE 계산
        mse = np.mean((y_pred - y_test_true)**2)
        
        if row == 0:
            ax.set_title(f'{title}\n{n_neurons} Neurons (MSE: {mse:.4f})', 
                        fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{n_neurons} Neurons (MSE: {mse:.4f})', fontsize=11)
        
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

plt.tight_layout()
save_path = os.path.join(output_dir, '05_universal_approximation.png')
plt.savefig(save_path, dpi=100)
print(f"\n\n그래프가 저장되었습니다: {save_path}")

print("\n=== Universal Approximation Theorem ===")
print("\n정리 (Cybenko, 1989):")
print("하나의 은닉층을 가진 신경망은 충분한 수의 뉴런이 있다면")
print("어떤 연속 함수도 임의의 정확도로 근사할 수 있다.")
print("\n공식적으로:")
print("f: [0,1]ⁿ → ℝ가 연속 함수일 때,")
print("∀ε > 0, ∃N, W, b 다음을 만족:")
print("  |f(x) - Σᵢ wᵢ·σ(vᵢᵀx + bᵢ)| < ε,  ∀x ∈ [0,1]ⁿ")
print("\n핵심 통찰:")
print("1. 깊이(depth)보다 폭(width)이 이론적으로 충분")
print("2. 하지만 실제로는 깊은 네트워크가 더 효율적")
print("3. 뉴런이 많을수록 더 정확한 근사")
print("\n관찰 결과:")
print("- 3개 뉴런: 매우 거친 근사")
print("- 10개 뉴런: 대략적인 형태 잡힘")
print("- 50개 뉴런: 거의 완벽한 근사!")
