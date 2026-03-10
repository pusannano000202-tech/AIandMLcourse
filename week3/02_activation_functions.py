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

print("=== 활성화 함수 (Activation Functions) 비교 ===")

# 1. 활성화 함수 정의
def sigmoid(x):
    """시그모이드: S자 곡선, 출력 범위 (0, 1)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """시그모이드 미분"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """하이퍼볼릭 탄젠트: S자 곡선, 출력 범위 (-1, 1)"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh 미분"""
    return 1 - np.tanh(x)**2

def relu(x):
    """ReLU: x > 0이면 x, 아니면 0"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 미분"""
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: x > 0이면 x, 아니면 alpha*x"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU 미분"""
    return np.where(x > 0, 1, alpha)

# 2. 데이터 생성
x = np.linspace(-5, 5, 200)

# 3. 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3-1. 활성화 함수 비교
ax = axes[0, 0]
ax.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
ax.plot(x, tanh(x), label='Tanh', linewidth=2)
ax.plot(x, relu(x), label='ReLU', linewidth=2)
ax.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2, linestyle='--')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.set_title('Activation Functions', fontsize=14, fontweight='bold')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3-2. 미분 (Gradient) 비교
ax = axes[0, 1]
ax.plot(x, sigmoid_derivative(x), label="Sigmoid'", linewidth=2)
ax.plot(x, tanh_derivative(x), label="Tanh'", linewidth=2)
ax.plot(x, relu_derivative(x), label="ReLU'", linewidth=2)
ax.plot(x, leaky_relu_derivative(x), label="Leaky ReLU'", linewidth=2, linestyle='--')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.set_title('Derivatives (Gradients)', fontsize=14, fontweight='bold')
ax.set_xlabel('Input (x)')
ax.set_ylabel("f'(x)")
ax.legend()
ax.grid(True, alpha=0.3)

# 3-3. Sigmoid vs Tanh 비교
ax = axes[1, 0]
ax.plot(x, sigmoid(x), label='Sigmoid: (0, 1)', linewidth=3)
ax.plot(x, tanh(x), label='Tanh: (-1, 1)', linewidth=3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3, label='Sigmoid center')
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.set_title('Sigmoid vs Tanh (중심이 다름!)', fontsize=14, fontweight='bold')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output')
ax.legend()
ax.grid(True, alpha=0.3)

# 3-4. ReLU vs Leaky ReLU 비교
ax = axes[1, 1]
ax.plot(x, relu(x), label='ReLU (x < 0: 죽음)', linewidth=3)
ax.plot(x, leaky_relu(x), label='Leaky ReLU (x < 0: 약간 살아있음)', linewidth=3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='ReLU 경계')
ax.set_title('ReLU vs Leaky ReLU (Dying ReLU 문제)', fontsize=14, fontweight='bold')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(output_dir, '02_activation_functions.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")

# 4. 특성 비교 출력
print("\n=== 활성화 함수 특성 비교 ===")
print("\n1. Sigmoid σ(x) = 1/(1+e^-x)")
print("   - 범위: (0, 1)")
print("   - 장점: 확률 해석 가능, 부드러운 곡선")
print("   - 단점: Vanishing Gradient (기울기 소실), 출력이 0 중심이 아님")
print("   - 용도: 이진 분류 출력층")

print("\n2. Tanh tanh(x) = (e^x - e^-x)/(e^x + e^-x)")
print("   - 범위: (-1, 1)")
print("   - 장점: 0 중심, Sigmoid보다 기울기 큼")
print("   - 단점: Vanishing Gradient 여전히 존재")
print("   - 용도: 은닉층 (Sigmoid보다 선호됨)")

print("\n3. ReLU f(x) = max(0, x)")
print("   - 범위: [0, ∞)")
print("   - 장점: 계산 빠름, Vanishing Gradient 없음, 희소성")
print("   - 단점: Dying ReLU (음수 영역에서 뉴런 죽음)")
print("   - 용도: 은닉층 (현대 신경망의 표준)")

print("\n4. Leaky ReLU f(x) = max(αx, x)")
print("   - 범위: (-∞, ∞)")
print("   - 장점: Dying ReLU 문제 해결")
print("   - 단점: α 값 선택 필요")
print("   - 용도: ReLU의 대안")

print("\n=== 권장 사용법 ===")
print("- 은닉층: ReLU (또는 Leaky ReLU)")
print("- 이진 분류 출력층: Sigmoid")
print("- 다중 분류 출력층: Softmax (다음 주 배움)")
print("- 회귀 출력층: 활성화 함수 없음 (선형)")
