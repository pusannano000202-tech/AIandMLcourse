# Week 4: 물리 데이터로 학습하기

## 개요

이번 주차에서는 Neural Network를 사용하여 물리 데이터를 학습하고 예측하는 방법을 실습합니다. TensorFlow/Keras를 사용하여 1D 함수 근사, 2D 회귀, 과적합/과소적합 현상, 그리고 진자 운동 주기 예측을 구현합니다.

## 프로그램 구조

```
week4/
├── 01perfect1d.py             # 1D 함수 근사 (단독 실행)
├── 02projectile.py            # 포물선 운동 회귀 (단독 실행)
├── 03overfitting.py           # 과적합 vs 과소적합 (단독 실행)
├── 04pendulum.py              # 진자 주기 예측 (단독 실행)
└── outputs/                   # 생성된 그래프 저장
    ├── perfect_1d_approximation.png
    ├── network_size_comparison.png
    ├── extreme_function_test.png
    ├── 02_projectile_*.png
    ├── 03_overfitting_*.png
    └── 04_pendulum_*.png
```

## 실행 방법

### 1. 환경 설정

프로젝트 루트 디렉토리에서 필요한 패키지를 설치합니다:

```bash
# uv를 사용하는 경우
uv sync

# 또는 pip를 사용하는 경우
pip install tensorflow numpy matplotlib
```

### 2. 프로그램 실행

```bash
cd week4

# Lab 1: 1D 함수 근사
python 01perfect1d.py

# Lab 2: 포물선 운동 회귀
python 02projectile.py

# Lab 3: 과적합 vs 과소적합
python 03overfitting.py

# Lab 4: 진자 주기 예측
python 04pendulum.py
```

각 스크립트는 독립적으로 실행되며, `outputs/` 디렉토리에 고품질 PNG 그래프를 자동 생성합니다.

---

## Lab 1: 01perfect1d.py - 완벽한 1D 함수 근사

### 목적
다양한 1D 함수를 Neural Network로 근사하고 네트워크 크기의 영향을 분석

### 이론적 배경

Universal Approximation Theorem에 의하면, 충분히 넓은 hidden layer를 가진 Neural Network는 임의의 연속 함수를 원하는 정확도로 근사할 수 있습니다.

### 실험 내용

#### 1. 기본 함수 근사
- `sin(x)`
- `cos(x) + 0.5sin(2x)`
- `x·sin(x)`
- 각 함수마다 [128, 128, 64] 네트워크 사용
- 3000 epochs 학습

#### 2. 네트워크 크기 비교
- **Small [32]**: 단순한 네트워크
- **Medium [64, 64]**: 중간 복잡도
- **Large [128, 128]**: 큰 네트워크
- **Very Large [128, 128, 64]**: 매우 깊은 네트워크

#### 3. 극한 복잡도 테스트
- 함수: `sin(x) + 0.5sin(2x) + 0.3cos(3x) + 0.2sin(5x) + 0.1x·cos(x)`
- 네트워크: [256, 256, 128, 64]
- 5000 epochs 학습
- MSE < 0.001 달성

### Neural Network 구조
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(128, activation='tanh'),
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(1, activation='linear')
])
```

### 생성 파일
- `outputs/perfect_1d_approximation.png`: 3개 함수 근사 결과
- `outputs/network_size_comparison.png`: 네트워크 크기 비교
- `outputs/extreme_function_test.png`: 극한 복잡도 테스트

### 실행 시간
약 3-5분

### 주의사항: Validation Split 문제

**중요**: 데이터를 랜덤하게 섞지 않으면 오른쪽 끝부분이 잘 학습되지 않습니다!

- TensorFlow의 `validation_split=0.2`는 **데이터의 마지막 20%**를 validation으로 사용
- 데이터가 순차적으로 정렬되어 있으면 → 오른쪽 끝부분이 학습에서 제외됨
- **해결책**: `np.random.permutation`으로 데이터를 섞어서 전 범위가 골고루 학습되도록 함

---

## Lab 2: 02projectile.py - 포물선 운동 회귀

### 목적
포물선 운동의 궤적을 Neural Network로 예측

### 물리 법칙
```
x(t) = v₀·cos(θ)·t
y(t) = v₀·sin(θ)·t - 0.5·g·t²
```

Neural Network는 이러한 물리 법칙을 데이터로부터 학습합니다.

### 실험 내용

#### 1. 데이터 생성
- 2000개 학습 샘플 (다양한 v₀, θ, t)
- 입력: (초기속력, 발사각도, 시간)
- 출력: (x 좌표, y 좌표)
- 노이즈 추가로 현실적인 데이터 시뮬레이션

#### 2. 모델 구조
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(2, activation='linear')  # x, y
])
```

#### 3. 다양한 조건 테스트
- v₀=20m/s, θ=30°
- v₀=30m/s, θ=45° (최대 사거리)
- v₀=40m/s, θ=60°

#### 4. 오차 분석
- 각도에 따른 오차 변화
- 속도에 따른 오차 변화
- 최대 높이, 최대 사거리 비교

### 물리적 의미

- **작은 각도**: 사거리 증가, 높이 감소
- **45도**: 최대 사거리 (공기 저항 무시)
- **큰 각도**: 높이 증가, 사거리 감소

### 생성 파일
- `outputs/02_projectile_trajectories.png`: 3가지 조건 궤적 비교
- `outputs/02_projectile_analysis.png`: 학습 곡선 및 오차 분석

### 실행 시간
약 2-3분

---

## Lab 3: 03overfitting.py - 과적합 vs 과소적합 데모

### 목적
모델 복잡도가 성능에 미치는 영향 시연

### 이론적 배경

**Underfitting (과소적합)**:
- 모델이 너무 단순하여 데이터의 패턴을 학습하지 못함
- 높은 training loss, 높은 validation loss

**Good Fit (적절한 학습)**:
- 모델이 데이터의 패턴을 잘 학습
- 낮은 training loss, 낮은 validation loss

**Overfitting (과적합)**:
- 모델이 너무 복잡하여 학습 데이터의 노이즈까지 학습
- 낮은 training loss, 높은 validation loss

### 테스트 함수
`y = sin(2x) + 0.5x + noise`

### 3가지 모델 비교

#### 1. Underfit Model (과소적합)
- 구조: [4] - 너무 단순
- 결과: 높은 train/val loss
- 문제: 데이터 패턴을 학습하지 못함

#### 2. Good Fit Model (적절)
- 구조: [32, 16] with Dropout(0.2)
- 결과: 낮은 train/val loss (비슷함)
- 특징: 최고의 일반화 성능

#### 3. Overfit Model (과적합)
- 구조: [256, 128, 64, 32] - 너무 복잡
- 결과: 낮은 train loss, 높은 val loss
- 문제: 노이즈까지 학습

### 실험 내용
- 200 epochs 학습
- Train vs Validation loss 비교
- 예측 곡선 비교
- 오차 분석
- 성능 비교 테이블

### 과적합 방지 방법

1. **Dropout**: 학습 중 일부 뉴런을 랜덤하게 비활성화
2. **Early Stopping**: Validation loss가 증가하면 학습 중단
3. **L1/L2 Regularization**: 가중치에 페널티 부여
4. **더 많은 데이터**: 학습 데이터 증가
5. **모델 단순화**: Layer 수 또는 뉴런 수 감소

### 생성 파일
- `outputs/03_overfitting_comparison.png`: 3개 모델 예측 비교
- `outputs/03_training_curves.png`: 학습 곡선
- `outputs/03_error_analysis.png`: 오차 분석
- `outputs/03_comparison_table.png`: 성능 비교표

### 실행 시간
약 2-3분

---

## Lab 4: 04pendulum.py - 진자 주기 예측

### 목적
진자의 주기를 예측하고 운동을 시뮬레이션

### 물리 법칙

**작은 각도 근사** (θ₀ < 15°):
```
T = 2π√(L/g)
```

**큰 각도** (θ₀ > 15°):
타원 적분을 사용한 근사식:
```
T ≈ T₀ [1 + (1/16)θ₀² + (11/3072)θ₀⁴ + ...]
```

**운동 방정식**:
```
d²θ/dt² = -(g/L)sin(θ)
```

### 실험 내용

#### 1. 주기 예측
- 입력: (진자 길이 L, 초기 각도 θ₀)
- 출력: 주기 T
- 2000개 학습 샘플
- MAPE < 1%

#### 2. 다양한 길이 테스트
- L = 0.5m, 1.0m, 2.0m
- 각도 범위: 5° ~ 80°
- 비선형 관계 학습

#### 3. RK4 시뮬레이션
- Runge-Kutta 4차 방법으로 운동 방정식 수치 적분
- 진자 운동 시뮬레이션
- 위상 공간 플롯 (각도 vs 각속도)

#### 4. 성능 분석
- 길이에 따른 오차
- 각도에 따른 오차
- 학습 곡선

### 모델 구조
```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='linear')  # T
])
```

### 물리적 통찰

1. **길이의 영향**:
   - L이 4배 증가 → T가 2배 증가 (제곱근 관계)

2. **각도의 영향**:
   - 작은 각도: 각도와 무관 (등시성)
   - 큰 각도: 각도가 클수록 주기 증가

3. **비선형성**:
   - `sin(θ) ≈ θ` (작은 각도)
   - `sin(θ) < θ` (큰 각도) → 복원력 감소 → 주기 증가

### 생성 파일
- `outputs/04_pendulum_prediction.png`: 3가지 길이 주기 예측
- `outputs/04_pendulum_simulation.png`: RK4 시뮬레이션
- `outputs/04_pendulum_analysis.png`: 학습 곡선 및 오차 분석

### 실행 시간
약 2-3분

---

## TensorFlow/Keras 핵심 개념

### Sequential Model

순차적으로 층을 쌓는 가장 간단한 모델:

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
```

### Dense Layer

완전 연결 층 (Fully Connected Layer):
- 모든 입력이 모든 출력에 연결
- `y = activation(W·x + b)`

### Activation Functions

**ReLU** (Rectified Linear Unit):
- `f(x) = max(0, x)`
- 기울기 소실 문제 해결
- 대부분의 hidden layer에 사용

**Tanh**:
- `f(x) = (e^x - e^(-x))/(e^x + e^(-x))`
- -1 ~ 1 범위
- 함수 근사에 효과적

**Linear**:
- `f(x) = x`
- 회귀 문제의 output layer

### Dropout

과적합 방지를 위한 정규화 기법:
- 학습 중 랜덤하게 뉴런을 비활성화
- 일반적으로 0.1 ~ 0.5

```python
keras.layers.Dropout(0.2)  # 20% 뉴런 비활성화
```

### Optimizer

**Adam** (Adaptive Moment Estimation):
- 가장 많이 사용되는 optimizer
- Learning rate 자동 조정
- Momentum + RMSprop 결합

```python
optimizer=keras.optimizers.Adam(learning_rate=0.001)
```

### Loss Functions

**MSE** (Mean Squared Error):
- 회귀 문제에 사용
- `L = (1/n)Σ(y_true - y_pred)²`

**MAE** (Mean Absolute Error):
- 이상치에 덜 민감
- `L = (1/n)Σ|y_true - y_pred|`

**MAPE** (Mean Absolute Percentage Error):
- 상대 오차 (%)
- `L = (100/n)Σ|((y_true - y_pred) / y_true|`

---

## Neural Network 학습 팁

### 1. 하이퍼파라미터 튜닝

**Learning Rate**:
- 너무 크면: 발산 또는 진동
- 너무 작으면: 학습 속도 느림
- 권장값: 0.001 ~ 0.0001

**Batch Size**:
- 작은 값(16-32): 노이즈가 많지만 일반화 좋음
- 큰 값(128-256): 안정적이지만 과적합 위험

**Epochs**:
- 너무 적으면: Underfitting
- 너무 많으면: Overfitting
- Validation loss 관찰하며 조정

### 2. 데이터 전처리

**정규화**:
```python
# Min-Max Scaling
x_normalized = (x - x_min) / (x_max - x_min)

# Standardization
x_standardized = (x - mean) / std
```

**데이터 셔플링**:
- `validation_split` 사용 시 반드시 데이터를 섞어야 함
- 순차 데이터는 일부 영역이 학습되지 않을 수 있음

### 3. 모델 평가 메트릭

**MSE (Mean Squared Error)**:
- 큰 오차에 더 민감
- 물리량 단위의 제곱

**MAE (Mean Absolute Error)**:
- 평균 절대 오차
- 직관적인 해석

**MAPE (Mean Absolute Percentage Error)**:
- 상대 오차 (%)
- 스케일에 무관한 비교

---

## 문제 해결 가이드

### 학습이 되지 않는 경우

1. **Loss가 감소하지 않음**:
   - Learning rate 조정
   - 데이터 정규화 확인
   - 모델 구조 단순화

2. **Loss가 NaN**:
   - Learning rate 감소
   - Gradient clipping 적용
   - 데이터 범위 확인

3. **학습 속도가 너무 느림**:
   - Batch size 증가
   - Learning rate 증가
   - GPU 사용 고려

### 과적합 발생 시

1. **Dropout 추가**
2. **Early stopping 구현**
3. **학습 데이터 증가**
4. **모델 단순화**

### 메모리 부족 시

1. **Batch size 감소**
2. **모델 크기 축소**
3. **데이터 샘플 수 감소**

---

## 추가 학습 자료

### TensorFlow 공식 문서
- https://www.tensorflow.org/tutorials
- https://www.tensorflow.org/api_docs/python/tf/keras

### 물리 시뮬레이션
- Numerical Recipes (Press et al.)
- Computational Physics (Giordano & Nakanishi)

### Neural Networks
- Deep Learning (Goodfellow et al.)
- Neural Networks and Deep Learning (Nielsen)

---

## 과제 및 도전 과제

### 기본 과제

1. 각 Lab의 파라미터를 변경하며 결과 관찰
2. 학습 곡선 분석 및 최적 epoch 찾기
3. 다양한 조건에서 예측 정확도 비교

### 심화 과제

1. **Lab 1**: `tanh(x)`, `x³` 등 다른 함수 추가
2. **Lab 2**: 공기 저항 추가 모델
3. **Lab 3**: L1/L2 regularization 구현
4. **Lab 4**: 감쇠 진자 (damped pendulum) 구현

### 프로젝트 아이디어

1. 스프링-질량 시스템의 고유 진동수 예측
2. 2D Ising Model 상전이 예측
3. 단순 조화 진동자의 에너지 준위 예측
4. Kepler 문제의 궤도 예측

---

## 결론

이번 주차에서는 Neural Network를 사용하여 물리 데이터를 학습하고 예측하는 방법을 실습했습니다.

### 주요 학습 내용

1. **1D 함수 근사**: Universal Approximation Theorem 실제 적용
2. **2D 회귀**: 다차원 입출력 처리
3. **과적합/과소적합**: 모델 복잡도의 중요성
4. **진자 주기 예측**: 비선형 물리 법칙 학습

### 실습 특징

- **독립 실행**: 각 스크립트는 독립적으로 실행
- **자동 저장**: 고품질 그래프 자동 생성
- **재현 가능**: 일관된 결과 보장
- **보고서 작성**: 출력 파일을 바로 활용 가능

### 생성 파일 요약

총 **10개**의 PNG 파일 생성:
- Lab 1: 3개 (함수 근사, 네트워크 비교, 극한 테스트)
- Lab 2: 2개 (궤적, 분석)
- Lab 3: 4개 (비교, 학습 곡선, 오차, 표)
- Lab 4: 3개 (예측, 시뮬레이션, 분석)

다음 주차에서는 Physics-Informed Neural Networks (PINNs)를 통해 물리 법칙을 직접 손실 함수에 삽입하는 방법을 학습합니다.

---

**작성자**: AI Physics Course  
**버전**: 3.0  
**최종 수정**: 2024  
**업데이트**: PySide6 제거, 단독 실행 스크립트만 유지
