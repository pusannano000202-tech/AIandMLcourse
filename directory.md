# AI and ML Coursework - Project Directory Structure

## 📁 전체 프로젝트 개요

**부산대학교 물리학과 - 전산물리 (2학년 1학기)**

이 저장소는 AI 및 머신러닝 과목의 실습 코드와 문서를 포함합니다.

### 👥 팀 프로젝트 정보
- **팀 구성**: 4명 1팀
- **협업 방식**: GitHub를 통한 코드 공유
- **발표**: 학기말 팀별 프로젝트 발표 예정

### 주요 내용 (주차별 순서)

**Part I: Neural Networks & Deep Learning (Weeks 1-7)**
- Week 1: 강의 소개 및 환경 설정
- Week 2: 머신러닝 기초
- Week 3: 신경망 기초
- Week 4: 물리 데이터로 학습하기
- Week 5: 딥러닝 핵심 개념
- Week 6: Transformer와 Attention Mechanism
- Week 7: Large Language Models (LLM) 개론

**Part II: LLM Vibe Coding for Physics (Weeks 9-12)**
- Week 9: 고전 역학 시뮬레이션
- Week 10: 전자기학 시뮬레이션
- Week 11: 양자역학 시뮬레이션
- Week 12: 통계물리 및 Monte Carlo 시뮬레이션

**Part III: Physics-Informed Neural Networks (Weeks 13-14)**
- Week 13: PINN 기초 이론 (ODE 편)
- Week 14: PINN 응용 - 편미분방정식

---

## 📂 Week 1: 강의 소개 및 환경 설정

**주제:** 개발 환경 구축 및 첫 번째 신경망

### 파일 구조

```
week1/
├── 00_hello_world.py               # Python 환경 테스트
├── 01_hello_nn.py                  # 첫 번째 Neural Network
├── 02_polynomial_fitting.py        # 다항식 피팅
├── outputs/                        # 생성된 그래프와 결과
└── week1.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- Python 환경 설정 (uv, Git, VS Code)
- AI 코딩 어시스턴트 (Claude) 사용법
- 첫 번째 Neural Network 구현
- 다항식 피팅을 통한 ML 기초 이해

### 핵심 개념
- Development Environment
- Neural Network 기초
- Overfitting의 개념

---

## 📂 Week 2: 머신러닝 기초

**주제:** 지도/비지도 학습과 데이터 전처리

### 파일 구조

```
week2/
├── 01_linear_regression_spring.py  # 선형 회귀 (후크 법칙)
├── 02_unsupervised_clustering.py   # 비지도 학습 (클러스터링)
├── 03_data_preprocessing.py        # 데이터 전처리
├── 04_gradient_descent_vis.py      # 경사하강법 시각화
├── outputs/                        # 생성된 그래프와 결과
└── week2.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- 지도/비지도/강화 학습의 차이
- 손실 함수 (Loss Function)
- Gradient Descent 알고리즘
- 데이터 정규화 및 전처리

### 핵심 개념
- Supervised/Unsupervised Learning
- Loss Functions
- Optimization
- Data Normalization

---

## 📂 Week 3: 신경망 기초

**주제:** Perceptron부터 MLP까지

### 파일 구조

```
week3/
├── 01_perceptron.py                # Perceptron과 XOR 문제
├── 02_activation_functions.py      # 활성화 함수 비교
├── 03_forward_propagation.py       # Forward Pass
├── 04_mlp_numpy.py                 # Multi-Layer Perceptron
├── 05_universal_approximation.py   # Universal Approximation Theorem
├── check_fonts.py                  # 한글 폰트 확인
├── outputs/                        # 생성된 그래프와 결과
└── week3.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- Perceptron의 원리와 한계 (XOR 문제)
- Activation Functions: ReLU, Sigmoid, Tanh
- Forward Propagation의 수학적 구조
- Multi-Layer Perceptron 구현
- Universal Approximation Theorem 시연

### 핵심 개념
- Perceptron
- Activation Functions
- Forward Pass
- MLP Architecture

---

## 📂 Week 4: 물리 데이터로 학습하기

**주제:** Neural Network를 사용한 물리 데이터 학습

### 파일 구조

```
week4/
├── 01perfect1d.py                  # 1D 함수 근사
├── 02projectile.py                 # 포물선 운동 회귀
├── 03overfitting.py                # 과적합 vs 과소적합
├── 04pendulum.py                   # 진자 주기 예측
├── outputs/                        # 생성된 그래프와 결과
└── week4.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- TensorFlow/Keras를 이용한 1D/2D 회귀
- 과적합(Overfitting)과 과소적합(Underfitting)
- 모델 복잡도와 성능의 관계
- 물리 법칙 학습 (진자 주기)

### 핵심 개념
- Regression with Neural Networks
- Overfitting/Underfitting
- Model Complexity
- Physics Data Learning

---

## 📂 Week 5: 딥러닝 핵심 개념

**주제:** Regularization, Augmentation, Transfer Learning

### 파일 구조

```
week5/
├── 01_regularization.py            # L1/L2, Dropout, BatchNorm
├── 02_overfitting_underfitting.py  # 모델 복잡도 분석
├── 03_data_augmentation.py         # 데이터 증강 기법
├── 04_transfer_learning.py         # 전이 학습
├── 05_mnist_cnn.py                 # MNIST CNN 실습
├── outputs/                        # 생성된 그래프와 결과
└── week5.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- Regularization 기법: L1/L2, Dropout, Batch Normalization
- Data Augmentation으로 데이터 부족 해결
- Transfer Learning 개념과 활용
- CNN을 이용한 MNIST 손글씨 인식

### 핵심 개념
- Regularization
- Data Augmentation
- Transfer Learning
- CNN Basics

---

## 📂 Week 6: Transformer와 Attention Mechanism

**주제:** RNN의 한계를 극복하는 Transformer 아키텍처

### 파일 구조

```
week6/
├── 01_attention_basics.py          # Attention 메커니즘 기초
├── 02_self_attention.py            # Self-Attention과 Multi-Head
├── 03_positional_encoding.py       # 위치 인코딩
├── 04_transformer_block.py         # 완전한 Transformer Block
├── 05_sequence_modeling.py         # 실전 시퀀스 모델링
├── outputs/                        # 생성된 그래프와 결과
├── week6.md                        # 학생용 상세 문서
└── run.bat                         # 자동 실행 스크립트
```

### 주요 학습 내용
- Attention 메커니즘의 원리 (Query, Key, Value)
- Self-Attention과 Multi-Head Attention
- Positional Encoding의 필요성
- 완전한 Transformer Encoder Block 구현
- 실제 시퀀스 예측 문제 적용

### 핵심 개념
- Attention Mechanism
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Architecture

**참고 논문:** "Attention Is All You Need" (Vaswani et al., 2017)

---

## 📂 Week 7: Large Language Models (LLM) 개론

**주제:** GPT, BERT, Claude - LLM의 이해와 활용

### 파일 구조

```
week7/
├── 01_tokens_and_embeddings.py     # Token과 Embedding 기초
├── 02_gpt_bert_architectures.py    # GPT vs BERT 아키텍처
├── 03_pretraining_finetuning.py    # Pre-training과 Fine-tuning
├── 04_claude_api_simple.py         # Claude API 개념 (시뮬레이션)
├── outputs/                        # 생성된 그래프와 결과
├── week7.md                        # 학생용 상세 문서
└── run.bat                         # 자동 실행 스크립트
```

### 주요 학습 내용
- Tokenization: Character, Word, BPE
- Token Embedding과 Context Window
- GPT (Decoder-only) vs BERT (Encoder-only)
- Pre-training, Fine-tuning, RLHF
- LLM API 사용법 (Claude API 시뮬레이션)

### 핵심 개념
- Tokenization
- Embeddings
- GPT vs BERT
- Pre-training/Fine-tuning
- RLHF
- Prompt Engineering

**참고 논문:**
- "BERT" (Devlin et al., 2018)
- "GPT-3" (Brown et al., 2020)

---

## 📂 Week 9: 고전 역학 시뮬레이션

**주제:** 수치 적분과 혼돈 시스템

### 파일 구조

```
week9/
├── 01euler_rk4.py                  # Euler vs RK4 비교
├── 02planetary.py                  # 행성 운동 시뮬레이션
├── 03chaotic_pendulum.py           # 혼돈 진자
├── 04lagrangian_hamiltonian.py     # 라그랑지안/해밀토니안
├── outputs/                        # 생성된 그래프와 결과
└── week9.md                        # 학생용 상세 문서
```

### 주요 학습 내용
- 수치 적분 방법 (Euler, RK4)
- 행성 운동과 케플러 법칙
- 혼돈 시스템 (이중 진자)
- 라그랑지안과 해밀토니안 역학

### 핵심 개념
- Numerical Integration
- ODEs
- Chaotic Systems
- Lagrangian/Hamiltonian Mechanics

---

## 📂 Week 10: 전자기학 시뮬레이션

**주제:** Maxwell 방정식과 전자기파

### 파일 구조

```
week10/
├── 01_electric_field_basics.py     # 전기장 기초
├── 02_electric_potential.py        # 전위 계산
├── 03_electric_field_lines.py      # 전기력선
├── 04_magnetic_field_basics.py     # 자기장 기초
├── 05_lorentz_force.py             # 로렌츠 힘
├── 06_maxwell_1d.py                # Maxwell 방정식 1D
├── 07_maxwell_2d.py                # Maxwell 방정식 2D
├── 08_multiple_charges.py          # 다중 전하
├── 09_em_wave_animation.py         # 전자기파 애니메이션
├── 10_conductor_potential.py       # 도체 전위
├── outputs/                        # 생성된 그래프와 결과
└── week10.md                       # 학생용 상세 문서
```

### 주요 학습 내용
- 전기장과 자기장 계산 및 시각화
- Maxwell 방정식의 수치 해법 (FDTD)
- 라플라스 방정식과 정전기 문제
- 전자기파 전파 시뮬레이션

### 핵심 개념
- Electric/Magnetic Fields
- Maxwell Equations
- FDTD (Finite Difference Time Domain)
- Electromagnetic Waves

---

## 📂 Week 11: 양자역학 시뮬레이션

**주제:** Schrödinger 방정식과 양자 현상

### 파일 구조

```
week11/
├── 01schrodinger.py                # Schrödinger 방정식 기초
├── 02wavefunction.py               # 파동함수 시각화
├── 03tunneling.py                  # 터널링 효과
├── 04wells_oscillator.py           # 양자 우물과 조화 진동자
├── outputs/                        # 생성된 그래프와 결과
└── week11.md                       # 학생용 상세 문서
```

### 주요 학습 내용
- Schrödinger 방정식의 수치 해법
- 파동함수와 확률 해석
- 터널링 효과 시뮬레이션
- Finite Well, Harmonic Oscillator

### 핵심 개념
- Schrödinger Equation
- Wave Functions
- Quantum Tunneling
- Potential Wells
- Quantum Harmonic Oscillator

---

## 📂 Week 12: 통계물리 및 Monte Carlo 시뮬레이션

**주제:** Monte Carlo 방법과 2D Ising Model

### 파일 구조

```
week12/
├── 01_random_walk.py               # Random Walk 시뮬레이션
├── 02_pi_estimation.py             # Monte Carlo로 π 추정
├── 03_ising_1d.py                  # 1D Ising Model
├── 04_metropolis.py                # Metropolis 알고리즘
├── 05_ising_2d_basic.py            # 2D Ising Model 기초
├── 06_phase_transition.py          # 상전이 분석
├── 07_thermodynamics.py            # 열역학적 성질
├── 08_ising_2d_advanced.py         # 고급 분석
├── outputs/                        # 생성된 그래프와 결과
└── week12.md                       # 학생용 상세 문서
```

### 주요 학습 내용
- Monte Carlo 방법론
- Random Walk와 통계적 성질
- Ising 모델과 상전이
- Metropolis-Hastings 알고리즘
- 열역학적 성질 계산 (에너지, 비열, 자화율)

### 핵심 개념
- Monte Carlo Methods
- Random Walk
- Ising Model
- Metropolis Algorithm
- Phase Transitions
- Statistical Physics

**참고 교재:**
- "Statistical Mechanics" (Pathria)
- "Monte Carlo Methods" (Landau & Binder)

---

## 📂 Week 13: PINN 기초 이론 (ODE 편)

**주제:** Physics-Informed Neural Networks로 ODE 풀기

### 파일 구조

```
week13/
├── 01_simple_ode.py                # 단순 ODE (TensorFlow)
├── 02_harmonic_oscillator.py       # 조화 진동자 (PyTorch)
├── 03_damped_oscillator.py         # 감쇠 진동자 (TensorFlow)
├── 04_boundary_value_problem.py    # 경계값 문제 (PyTorch)
├── 05_lorenz_system.py             # 로렌츠 시스템 (혼돈 역학)
├── 06_comparison_frameworks.py     # TensorFlow vs PyTorch 비교
├── outputs/                        # 생성된 그래프와 결과
├── week13.md                       # 학생용 상세 문서
└── README.md                       # 실행 가이드
```

### 주요 학습 내용
- PINN의 기본 개념과 작동 원리
- 물리 법칙을 Loss Function에 포함하기
- Automatic Differentiation
- TensorFlow와 PyTorch로 PINN 구현
- ODE 문제에 PINN 적용
- 전통적인 수치 해법(RK4)과 비교

### 핵심 개념
- Physics-Informed Neural Networks
- Physics Loss
- Automatic Differentiation
- Boundary Conditions
- TensorFlow vs PyTorch

**참고 논문:**
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)

---

## 📂 Week 14: PINN 응용 - 편미분방정식

**주제:** PINN으로 PDE 풀기

### 파일 구조

```
week14/
├── 01_basic_pinn.py                # PINN 기본 구조
├── 02_heat_equation_1d.py          # 1D 열전도 방정식
├── 03_wave_equation_1d.py          # 1D 파동 방정식
├── 04_heat_equation_2d.py          # 2D 열전도 방정식
├── 05_burgers_equation.py          # Burgers 방정식 (비선형)
├── 06_wave_equation_2d.py          # 2D 파동 방정식
├── 07_complex_boundary.py          # 복잡한 경계조건
├── run_all.py                      # 모든 스크립트 자동 실행
├── outputs/                        # 생성된 그래프와 결과
├── week14.md                       # 학생용 상세 문서
└── RUN_ALL.md                      # 실행 가이드
```

### 주요 학습 내용
- 1D/2D Heat Equation (열전도 방정식)
- 1D/2D Wave Equation (파동 방정식)
- Burgers Equation (비선형 PDE)
- 복잡한 경계조건 처리
- PINN의 장단점 분석

### 핵심 개념
- Partial Differential Equations (PDEs)
- Heat Equation
- Wave Equation
- Burgers Equation
- Boundary Conditions
- PINN for Spatial-Temporal Problems

**참고 논문:**
- Raissi et al., "Physics-informed neural networks" (2019)
- Cuomo et al., "Scientific Machine Learning" (2022)

---

## 🔧 공통 설정 및 규칙

### .cursorrules 파일

모든 Python 코드는 다음 규칙을 따릅니다:

**1. 한글 폰트 설정:**
```python
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
```

**2. Weight 초기화 (신경망):**
```python
# Xavier/Glorot for Sigmoid/Tanh
limit = np.sqrt(6 / (n_in + n_out))
W = np.random.uniform(-limit, limit, (n_in, n_out))

# He for ReLU
std = np.sqrt(2 / n_in)
W = np.random.randn(n_in, n_out) * std
```

**3. Scatter Plot:**
```python
# Unfilled markers (x, +): edgecolors 사용 X
ax.scatter(..., marker='x', c='red')

# Filled markers (o, s): edgecolors 사용 가능
ax.scatter(..., marker='o', c='red', edgecolors='black')
```

---

## 📊 프로젝트 비교

### 주차별 난이도 및 특징

| Week | 주제 | 난이도 | 핵심 기술 | 응용 분야 |
|------|------|--------|----------|----------|
| 1 | 환경 설정 | ⭐ | Python, Git | 기초 |
| 2 | ML 기초 | ⭐⭐ | Regression, Clustering | 데이터 분석 |
| 3 | 신경망 기초 | ⭐⭐ | Perceptron, MLP | 분류 |
| 4 | 물리 학습 | ⭐⭐⭐ | TensorFlow/Keras | 회귀 |
| 5 | 딥러닝 기법 | ⭐⭐⭐ | Regularization, CNN | 이미지 |
| 6 | Transformer | ⭐⭐⭐⭐ | Attention, Self-Attention | NLP |
| 7 | LLM | ⭐⭐⭐ | GPT, BERT | 언어 모델 |
| 9 | 고전 역학 | ⭐⭐⭐ | ODE, RK4 | 시뮬레이션 |
| 10 | 전자기학 | ⭐⭐⭐⭐ | PDE, FDTD | 파동 |
| 11 | 양자역학 | ⭐⭐⭐⭐ | Schrödinger | 양자 |
| 12 | 통계물리 | ⭐⭐⭐⭐ | Monte Carlo, Ising | 상전이 |
| 13 | PINN ODE | ⭐⭐⭐⭐⭐ | PINN, ODE | AI+물리 |
| 14 | PINN PDE | ⭐⭐⭐⭐⭐ | PINN, PDE | AI+물리 |

### 프레임워크 사용 현황

- **Pure Numpy**: Week 1-3, 6 (기초 이론)
- **TensorFlow/Keras**: Week 4-5 (딥러닝 기초)
- **Matplotlib**: Week 1-14 (시각화)
- **SciPy**: Week 9-11 (수치 해법)
- **TensorFlow/PyTorch**: Week 13-14 (PINN)

---

## 💻 실행 방법

### 기본 실행 (모든 주차 공통)

```bash
# 해당 주차 디렉토리로 이동
cd week1

# 순서대로 실행
uv run python 01_파일명.py
uv run python 02_파일명.py
...

# 결과 확인
ls outputs/
```

### 자동 실행 스크립트 (Week 6, 7)

```bash
# Week 6 또는 Week 7 디렉토리로 이동
cd week6

# run.bat 실행 (Windows)
./run.bat

# 모든 Python 파일이 순차적으로 실행되고 에러 시 자동 중단
```

### Week 14 전체 실행

```bash
cd week14

# 모든 PINN 실습 한번에 실행
uv run python run_all.py
```

---

## 📚 학습 순서 추천

### 초급 과정 (Week 1-5)

1. **Week 1**: 환경 설정 및 첫 NN
2. **Week 2**: ML 기본 개념
3. **Week 3**: 신경망 이론
4. **Week 4**: 물리 데이터 학습
5. **Week 5**: 딥러닝 기법

👉 **목표**: Neural Network의 기본 원리 이해

### 중급 과정 (Week 6-7, 9-11)

1. **Week 6**: Transformer 아키텍처
2. **Week 7**: LLM 개론
3. **Week 9**: 고전 역학 시뮬레이션
4. **Week 10**: 전자기학 시뮬레이션
5. **Week 11**: 양자역학 시뮬레이션

👉 **목표**: 딥러닝과 물리 시뮬레이션 능력

### 고급 과정 (Week 12-14)

1. **Week 12**: Monte Carlo와 통계물리
2. **Week 13**: PINN 기초 (ODE)
3. **Week 14**: PINN 응용 (PDE)

👉 **목표**: AI와 물리의 융합 (PINN)

---

## 🎯 학습 목표 달성 체크리스트

### Part I: Neural Networks & Deep Learning

**기본:**
- [ ] Python 환경 설정 완료
- [ ] Neural Network 작동 원리 이해
- [ ] Loss Function과 Optimization 이해
- [ ] Activation Functions의 역할 설명 가능

**중급:**
- [ ] Backpropagation 알고리즘 이해
- [ ] Overfitting 방지 기법 적용 가능
- [ ] CNN으로 이미지 분류 가능
- [ ] Transformer의 Attention 메커니즘 이해

**고급:**
- [ ] LLM의 아키텍처 비교 가능
- [ ] Prompt Engineering 활용 가능
- [ ] Transfer Learning 적용 가능

### Part II: LLM Vibe Coding for Physics

**기본:**
- [ ] ODE 수치 해법 (Euler, RK4) 구현
- [ ] 전기장/자기장 계산 가능
- [ ] Schrödinger 방정식 기초 이해
- [ ] Monte Carlo 원리 이해

**중급:**
- [ ] 혼돈 시스템 시뮬레이션
- [ ] Maxwell 방정식 수치 해법
- [ ] 양자 터널링 시뮬레이션
- [ ] Metropolis 알고리즘 구현

**고급:**
- [ ] 라그랑지안/해밀토니안 역학 활용
- [ ] FDTD로 전자기파 시뮬레이션
- [ ] 양자 조화 진동자 분석
- [ ] 2D Ising 모델 상전이 분석

### Part III: Physics-Informed Neural Networks

**기본:**
- [ ] PINN의 기본 개념 이해
- [ ] Physics Loss 구성 방법 이해
- [ ] Automatic Differentiation 활용

**중급:**
- [ ] ODE 문제를 PINN으로 해결
- [ ] TensorFlow와 PyTorch로 PINN 구현
- [ ] 경계조건 처리 방법 이해

**고급:**
- [ ] PDE 문제를 PINN으로 해결
- [ ] 복잡한 경계조건 처리
- [ ] PINN과 전통적 방법 비교 분석
- [ ] 실제 물리 문제에 PINN 적용

---

## 🔗 관련 자료

### 필수 논문

**Transformers & LLM:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT" (Devlin et al., 2018)
- "GPT-3" (Brown et al., 2020)

**PINN:**
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)
- Cuomo et al., "Scientific Machine Learning" (2022)

### 코드 저장소

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Physics-Informed Neural Networks](https://github.com/maziarraissi/PINNs)
- [DeepXDE Library](https://deepxde.readthedocs.io/)

### 교재

**Neural Networks:**
- *Deep Learning* by Goodfellow, Bengio, and Courville
- MIT 6.S191: Introduction to Deep Learning

**Computational Physics:**
- *Computational Physics* by Mark Newman
- *Statistical Mechanics* (Pathria)
- *Monte Carlo Methods* (Landau & Binder)

---

## 🆘 문제 해결

### 일반적인 이슈

**1. Import 오류:**
```bash
uv pip install numpy matplotlib scipy tensorflow torch
```

**2. 한글 폰트 깨짐:**
- Windows: 'Malgun Gothic' 설치 확인
- Mac: 'AppleGothic' 기본 제공
- Linux: 'Nanum Gothic' 설치

**3. Out of Memory:**
- 시퀀스 길이 줄이기
- Batch size 감소
- 격자 크기 축소 (Ising Model)
- PINN 네트워크 레이어 감소

**4. 실행 시간 오래 걸림:**
- Monte Carlo sweeps 수 줄이기
- PINN epoch 수 감소
- 샘플 수 감소
- GPU 사용 고려

**5. GPU 설정 (선택사항):**
```python
# TensorFlow
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# PyTorch
import torch
print(torch.cuda.is_available())
```

---

## 📝 라이센스 및 인용

이 코드는 교육 목적으로 작성되었습니다.

**사용 시 참조:**
- Week 1-7: Based on MIT 6.S191 materials
- Week 6: Based on "Attention Is All You Need" (Vaswani et al., 2017)
- Week 9-12: Based on computational physics textbooks
- Week 13-14: Based on PINN research papers

**GitHub 저장소:**
```
https://github.com/BogKim2/AIandML
```

---

## 👨‍🏫 강의자 노트

### 강의 진행 팁

**Part I (Week 1-7):**
- 이론과 코드 구현을 균형있게 다룰 것
- 학생들이 직접 코드를 수정해보도록 유도
- Attention 메커니즘은 시각화를 통해 설명

**Part II (Week 9-12):**
- LLM을 활용한 "vibe coding" 실습 강조
- 물리적 직관과 수치 결과 비교
- 시각화를 통한 이해 강화

**Part III (Week 13-14):**
- PINN은 어려운 주제이므로 충분한 시간 배정
- TensorFlow와 PyTorch 모두 다루되, 학생이 선택 가능하도록
- 전통적 방법과의 비교를 통해 PINN의 장점 강조

---

## 📊 프로젝트 통계

### 전체 코드 구성
- **총 주차**: 13주 (Week 8 제외)
- **총 Python 파일**: 약 60개
- **총 시각화 출력**: 200개 이상
- **코드 라인 수**: 약 15,000줄

### 다루는 물리 분야
- 고전 역학 (Week 4, 9)
- 전자기학 (Week 10)
- 양자역학 (Week 11)
- 통계물리 (Week 12)
- PINN 응용 (Week 13-14)

---

*마지막 업데이트: 2025-01-21*
*버전: 2.0 (주차별 순차 정리)*
