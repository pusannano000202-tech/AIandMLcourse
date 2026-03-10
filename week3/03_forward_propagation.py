import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

print("=== Forward Propagation (순전파) 시각화 ===")

# 1. 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 2. Forward Propagation 구현
class SimpleNetwork:
    def __init__(self):
        # 2-3-1 네트워크 (입력 2개, 은닉층 3개, 출력 1개)
        np.random.seed(42)
        
        # Layer 1: Input -> Hidden
        self.W1 = np.random.randn(2, 3) * 0.5  # (2, 3)
        self.b1 = np.random.randn(3) * 0.1     # (3,)
        
        # Layer 2: Hidden -> Output
        self.W2 = np.random.randn(3, 1) * 0.5  # (3, 1)
        self.b2 = np.random.randn(1) * 0.1     # (1,)
    
    def forward(self, X, verbose=True):
        """순전파 수행"""
        if verbose:
            print("\n[Step 1] 입력층 → 은닉층")
            print(f"입력: {X}")
            print(f"가중치 W1:\n{self.W1}")
            print(f"편향 b1: {self.b1}")
        
        # 은닉층 계산
        self.z1 = np.dot(X, self.W1) + self.b1  # 선형 결합
        if verbose:
            print(f"z1 = X @ W1 + b1 = {self.z1}")
        
        self.a1 = relu(self.z1)  # 활성화
        if verbose:
            print(f"a1 = ReLU(z1) = {self.a1}")
        
        if verbose:
            print("\n[Step 2] 은닉층 → 출력층")
            print(f"가중치 W2:\n{self.W2}")
            print(f"편향 b2: {self.b2}")
        
        # 출력층 계산
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 선형 결합
        if verbose:
            print(f"z2 = a1 @ W2 + b2 = {self.z2}")
        
        self.a2 = sigmoid(self.z2)  # 활성화
        if verbose:
            print(f"a2 = Sigmoid(z2) = {self.a2}")
        
        return self.a2

# 3. 예제 실행
network = SimpleNetwork()
X = np.array([0.5, 0.8])
output = network.forward(X, verbose=True)

print(f"\n최종 출력: {output[0]:.4f}")

# 4. 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4-1. 네트워크 구조 다이어그램
ax = axes[0, 0]
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_title('Neural Network Architecture\n(2-3-1)', fontsize=14, fontweight='bold')

# 뉴런 위치
input_y = [1, 3]
hidden_y = [0.5, 2, 3.5]
output_y = [2]

# 입력층
for i, y in enumerate(input_y):
    circle = mpatches.Circle((0.5, y), 0.2, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(0.5, y, f'x{i+1}', ha='center', va='center', fontweight='bold')

# 은닉층
for i, y in enumerate(hidden_y):
    circle = mpatches.Circle((2, y), 0.2, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(2, y, f'h{i+1}', ha='center', va='center', fontweight='bold')

# 출력층
circle = mpatches.Circle((3.5, 2), 0.2, color='lightcoral', ec='black', linewidth=2)
ax.add_patch(circle)
ax.text(3.5, 2, 'y', ha='center', va='center', fontweight='bold')

# 연결선
for iy in input_y:
    for hy in hidden_y:
        ax.plot([0.7, 1.8], [iy, hy], 'k-', alpha=0.3, linewidth=1)

for hy in hidden_y:
    ax.plot([2.2, 3.3], [hy, 2], 'k-', alpha=0.3, linewidth=1)

# 레이블
ax.text(0.5, -0.3, 'Input\nLayer', ha='center', fontsize=10, fontweight='bold')
ax.text(2, -0.3, 'Hidden\nLayer\n(ReLU)', ha='center', fontsize=10, fontweight='bold')
ax.text(3.5, -0.3, 'Output\nLayer\n(Sigmoid)', ha='center', fontsize=10, fontweight='bold')

# 4-2. 선형 결합 시각화 (Layer 1)
ax = axes[0, 1]

# 각 층의 값을 개별적으로 그리기
x_labels = [f'Neuron {i}' for i in range(len(network.z1))]
x_positions = np.arange(len(x_labels))
width = 0.25

# Input 값 (2개)
input_extended = np.zeros(len(network.z1))
input_extended[:2] = X
ax.bar(x_positions - width, input_extended, width, label='Input', color='blue', alpha=0.7)

# z1 값
ax.bar(x_positions, network.z1, width, label='z1 (before ReLU)', color='orange', alpha=0.7)

# a1 값
ax.bar(x_positions + width, network.a1, width, label='a1 (after ReLU)', color='green', alpha=0.7)

ax.set_title('Layer 1: Input → Hidden (ReLU)', fontsize=12, fontweight='bold')
ax.set_ylabel('Value')
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)
ax.legend()
ax.grid(True, alpha=0.3)

# 4-3. 선형 결합 시각화 (Layer 2)
ax = axes[1, 0]
layers = ['a1 (input)', 'z2 (before Sigmoid)', 'a2 (after Sigmoid)']
values = [network.a1[:1], network.z2, network.a2]  # 첫 번째만
colors = ['green', 'orange', 'red']

for val, layer, color in zip(values, layers, colors):
    ax.barh(layer, val[0], color=color, alpha=0.7)
    ax.text(val[0] + 0.05, layer, f'{val[0]:.3f}', va='center')

ax.set_title('Layer 2: Hidden → Output (Sigmoid)', fontsize=12, fontweight='bold')
ax.set_xlabel('Value')
ax.grid(True, alpha=0.3)

# 4-4. 행렬 연산 시각화
ax = axes[1, 1]
ax.axis('off')
ax.set_title('Matrix Operations', fontsize=14, fontweight='bold')

# 수식 표시
equations = [
    "Forward Propagation 수식:",
    "",
    "Layer 1 (Input → Hidden):",
    "z₁ = X @ W₁ + b₁",
    f"   = [{X[0]:.2f}, {X[1]:.2f}] @ W₁ + b₁",
    f"   = {network.z1}",
    "",
    "a₁ = ReLU(z₁)",
    f"   = {network.a1}",
    "",
    "Layer 2 (Hidden → Output):",
    "z₂ = a₁ @ W₂ + b₂",
    f"   = {network.a1} @ W₂ + b₂",
    f"   = {network.z2[0]:.4f}",
    "",
    "a₂ = Sigmoid(z₂)",
    f"   = {network.a2[0]:.4f}",
]

y_pos = 0.95
for eq in equations:
    if eq == "":
        y_pos -= 0.04
    else:
        fontsize = 11 if "Layer" in eq or "Forward" in eq else 9
        fontweight = 'bold' if "Layer" in eq or "Forward" in eq else 'normal'
        ax.text(0.05, y_pos, eq, fontsize=fontsize, fontweight=fontweight, 
                family='monospace', transform=ax.transAxes)
        y_pos -= 0.055

plt.tight_layout()
save_path = os.path.join(output_dir, '03_forward_propagation.png')
plt.savefig(save_path, dpi=100)
print(f"\n그래프가 저장되었습니다: {save_path}")
