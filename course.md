# 전산물리: AI와 물리학의 만남
## Computational Physics: From Neural Networks to Physics-Informed AI

**부산대학교 물리학과**
**학년**: 2학년
**학기**: 2026년 1학기
**강의시간**: 주 3시간 (강의 2시간 + 실습 1시간)
**사용 도구**: Python, Claude AI / Cursor, uv

---

## 📚 강의 개요

본 강의는 현대 물리학 연구에서 필수적인 인공지능 기술을 학습하고, 이를 실제 물리 문제 해결에 적용하는 능력을 배양합니다. Neural Network의 기초부터 최신 Large Language Model(LLM), 그리고 물리학 특화 AI인 Physics-Informed Neural Networks(PINN)까지 다룹니다.

**핵심 특징**:
- MIT 강의 자료 기반의 체계적인 이론 학습
- Claude AI / Cursor를 활용한 "vibe coding" 실습
- 역학, 전자기학, 양자역학, 통계물리 문제의 AI 기반 수치 해법
- 물리 법칙을 학습에 직접 포함하는 PINN 기법
- `uv` 패키지 매니저를 이용한 최신 Python 개발 환경

---

## 🎯 학습 목표

1. Neural Network의 기본 원리와 작동 방식 이해
2. Deep Learning과 Transformer 아키텍처 학습
3. LLM을 활용한 효율적인 코딩 능력 배양 (Vibe Coding)
4. 물리 문제를 AI로 해결하는 실전 경험
5. PINN을 이용한 미분방정식 해법 습득

---

## 📖 교재 및 참고자료

### 주교재
- **MIT 6.S191**: Introduction to Deep Learning (강의 노트 및 영상)
- 강의자 제공 자료: Python 코드 예제, 주차별 가이드 (week*.md)

### 참고서적
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Physics-Informed Neural Networks* (관련 논문 모음)
- *Computational Physics* by Mark Newman

### 온라인 자료
- Claude AI / Cursor Documentation
- PyTorch / TensorFlow Tutorials
- ArXiv papers on PINN applications

---

## 📅 강의 일정 (16주)

### 🔷 Part I: Neural Networks & Deep Learning (Weeks 1-7)

#### **Week 1: 강의 소개 및 환경 설정**
- 강의 목표 및 평가 방식 소개
- 개발 환경 설정: **Git, Cursor (AI 기반 IDE), uv (Python 패키지 매니저)**
- Claude AI 계정 생성 및 기본 사용법
- **실습**:
  - "Hello, Neural Network!" - TensorFlow로 첫 번째 신경망 구현 (`y = 2x - 1` SGD, 500 에폭)
  - 수치 해법 비교: NumPy `polyfit` (최소자승법) vs SciPy `curve_fit` (비선형 최적화) vs 신경망

**주요 개념**: Development Environment, uv, Cursor, AI-Assisted Coding, Numerical Methods vs Neural Networks

---

#### **Week 2: 머신러닝 기초**
- 머신러닝의 세 가지 유형: 지도/비지도/강화 학습
- 데이터 전처리와 정규화 (Min-Max Normalization)
- 손실 함수(Loss Function)와 최적화 (Gradient Descent)
- **실습**:
  - 선형 회귀: 훅의 법칙(Hooke's Law) 데이터 피팅 (TensorFlow vs SciPy)
  - K-Means 군집화 (비지도 학습)
  - Gradient Descent 시각화

**주요 개념**: Supervised Learning, Loss Functions, Gradient Descent, Normalization

**과제 1**: 실험 데이터를 이용한 선형/비선형 회귀 분석

---

#### **Week 3: Neural Network 기초 이론**
- Perceptron과 Multi-Layer Perceptron (MLP)
- Activation Functions: ReLU, Sigmoid, Tanh
- Forward Propagation의 수학적 구조
- **Backpropagation** 알고리즘 (Chain Rule)
- Universal Approximation Theorem
- **실습**:
  - Perceptron으로 논리 게이트 구현 (AND, OR, XOR의 한계)
  - Pure Numpy로 MLP 구현 → XOR 문제 해결
  - Universal Approximation 시각화

**주요 개념**: Neurons, Activation, Forward/Backward Pass, Chain Rule

**참고**: MIT 6.S191 Lecture 1

---

#### **Week 4: 물리 데이터로 학습하기**
- Neural Network로 물리 법칙 학습 (TensorFlow/Keras)
- Universal Approximation Theorem 실전 적용
- 과적합(Overfitting) vs 과소적합(Underfitting) 개념 도입
- RK4를 이용한 수치 적분과 Neural Network 연동
- **실습**:
  - 1D 함수 근사: `sin(x)`, `cos(x) + 0.5sin(2x)`, `x·sin(x)` 등 (네트워크 크기 비교)
  - 포물선 운동 궤적 예측: `(초기속력, 발사각도, 시간) → (x, y)`
  - 과적합/과소적합 비교 실험 (Dropout, 모델 복잡도 조절)
  - 진자 주기 예측: 비선형 물리 법칙 학습 (RK4 시뮬레이션 포함)

**주요 개념**: Function Approximation, Overfitting, Dropout, Adam Optimizer, RK4

**과제 2**: 다양한 물리 데이터에 대한 Neural Network 예측 실험 및 분석

---

#### **Week 5: Deep Learning의 핵심 기법**
- Regularization: L1/L2, Dropout, Batch Normalization
- Overfitting vs. Underfitting (심화)
- Data Augmentation
- Transfer Learning의 개념 (MobileNetV2)
- **실습**: MNIST 손글씨 인식 (CNN: Conv2D, MaxPooling, Flatten)

**주요 개념**: Regularization, Generalization, CNN Basics, Transfer Learning

**참고**: MIT 6.S191 Lecture 2

---

#### **Week 6: Transformer와 Attention Mechanism**
- RNN의 한계와 Attention의 등장
- Scaled Dot-Product Attention: Query, Key, Value
- Self-Attention과 Multi-Head Attention
- Positional Encoding (Sinusoidal)
- Residual Connection & Layer Normalization
- **실습**: 시계열 예측 (Transformer vs RNN 성능 비교)

**주요 개념**: Attention, Transformers, Sequence Modeling, Positional Encoding

**참고**: "Attention Is All You Need" - Vaswani et al. (2017)

---

#### **Week 7: Large Language Models (LLM) 개론**
- GPT (Decoder-only) vs BERT (Encoder-only) 아키텍처 비교
- Tokenization (BPE, Subword), Embedding, Context Window
- Pre-training, Fine-tuning, RLHF
- Zero-shot, Few-shot Learning
- LLM의 물리학 응용 가능성
- **실습**: Claude API 개념 이해, Prompt Engineering 실습

**주요 개념**: LLM Architecture, Tokenization, Prompting, RLHF

**프로젝트 중간 발표**: Part I 학습 내용 요약 및 미니 프로젝트

---

### 🔷 Part II: LLM Vibe Coding for Physics (Weeks 8-12)

#### **Week 8: 중간고사 / LLM 기반 코딩 입문**
- "Vibe Coding"이란? - 자연어로 코드 생성
- Prompt Engineering 심화 기법
- Claude를 이용한 효율적인 디버깅 및 코드 리뷰
- **실습**: 자연어로 물리 시뮬레이션 코드 생성 및 검증

**주요 개념**: Prompt Engineering, AI-Assisted Programming

---

#### **Week 9: 고전 역학 문제 해결**
- 뉴턴 방정식의 수치 해법: Euler Method vs **RK4** (정확도 비교)
- 행성 운동 시뮬레이션 및 케플러 법칙 검증 (T² ∝ a³)
- 혼돈 시스템 (이중 진자, 나비 효과, 리아푸노프 지수)
- 라그랑지안과 해밀토니안 역학 (세 가지 정식화 비교)
- **실습**:
  - Euler vs RK4 정확도/안정성 비교
  - 태양계 행성 궤도 시뮬레이션
  - 이중 진자의 혼돈 시뮬레이션
  - 라그랑지안/해밀토니안 방법으로 단순 진자 풀기

**주요 개념**: ODEs, Numerical Integration, Chaotic Dynamics, Kepler's Laws

**과제 3**: LLM을 활용한 역학 문제 풀이 및 시각화

---

#### **Week 10: 전자기학 시뮬레이션**
- 맥스웰 방정식의 수치 해법 (FDTD: Finite Difference Time Domain)
- 전기장/전위 계산 및 시각화 (쿨롱 법칙, 가우스 법칙)
- 자기장 계산 (비오-사바르 법칙) 및 로렌츠 힘
- 전자기파 전파 시뮬레이션 (1D/2D CFL 조건)
- 라플라스 방정식 해법 (가우스-자이델 반복법)
- **실습**:
  - 점전하/다중 전하 전기장 벡터 시각화
  - 전자기파 애니메이션 (E ⊥ B ⊥ k)
  - 도체 내부 전위 분포 (라플라스 방정식)

**주요 개념**: Maxwell Equations, FDTD, CFL Condition, Vector Field Visualization

**과제 4**: 복잡한 전하 배치의 전기장 분석 및 시각화

---

#### **Week 11: 양자역학 시뮬레이션**
- 슈뢰딩거 방정식의 수치 해법 (유한 차분법 → 행렬 고유값 문제)
- 파동함수 시각화: 가우시안 파동 패킷, 중첩 상태, 수소 원자 오비탈
- 터널링 효과 (사각 장벽, 공명 터널링)
- 유한 우물(깊이/너비 의존성), 조화 진동자 (에르미트 다항식)
- **실습**:
  - 무한/유한 사각 우물, 조화 진동자 에너지 준위 계산
  - 파동 패킷 시각화 및 불확정성 원리 확인
  - 양자 터널링 투과 확률 계산
  - 고전 vs 양자 비교 (보어 대응 원리)

**주요 개념**: Quantum Mechanics, Wave Functions, Tunneling, Numerical Eigenvalue Problems

**과제 5**: 다양한 포텐셜에서의 양자 상태 분석

---

#### **Week 12: 통계물리 및 Monte Carlo 시뮬레이션**
- Monte Carlo 방법론 (랜덤 워크, π 추정)
- **Metropolis-Hastings 알고리즘** (Detailed Balance, 볼츠만 분포)
- 이징 모델 (1D → 2D): 강자성 상전이
- 임계 현상: 자화율, 비열, 임계 지수 (β, γ, ν)
- 열역학적 성질 계산: 분배 함수, 자유 에너지, 엔트로피
- **실습**:
  - 2D Ising 모델 시뮬레이션 (Tc ≈ 2.269 J/kB)
  - 상전이 온도 계산 및 임계 지수 추정
  - 히스테리시스 루프, 클러스터 분석, 자기상관 시간

**주요 개념**: Statistical Physics, Monte Carlo, Phase Transitions, Critical Exponents

**프로젝트 중간 점검**: Part II 실습 결과 공유

---

### 🔷 Part III: Physics-Informed Neural Networks (Weeks 13-14)

#### **Week 13: PINN 기초 이론 - 상미분방정식(ODE)**
- Physics-Informed Neural Networks 개념과 원리
- **자동 미분(Automatic Differentiation)**: TensorFlow `GradientTape` (중첩으로 2차 미분)
- PINN Loss 함수 설계: Physics Loss + IC Loss + BC Loss
- Collocation Points, 가중치 조정 (IC/BC에 100배 가중치)
- PINN vs. 전통적 수치 해법 (RK4) 비교
- **실습** (전체 TensorFlow 구현):
  - 단순 ODE (`dy/dx = -y`, `y(0)=1`) → `y=exp(-x)` (`01_simple_ode.py`)
  - 단순 조화 진동자 (`d²y/dt²+ω²y=0`, ω=2, 에너지 보존 검증) (`02_harmonic_oscillator.py`)
  - 감쇠 진동자 (under/critically/over-damped 세 가지 regime) (`03_damped_oscillator.py`)
  - 경계값 문제 (`d²y/dx²=-x`, BVP vs FDM 비교) (`04_boundary_value_problem.py`)
  - 로렌츠 시스템 (σ=10, ρ=28, β=8/3, 혼돈 역학, 3D attractor) (`05_lorenz_system.py`)
  - PINN vs RK4 비교: Van der Pol 진동자 (μ=1, 성능 지표) (`06_comparison_frameworks.py`)

**주요 개념**: PINN, Physics Loss, Boundary Conditions, Automatic Differentiation, Energy Conservation

**참고 논문**:
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)

---

#### **Week 14: PINN 응용 I - 편미분방정식(PDE)**
- PINN을 PDE에 적용: 2차 공간/시간 미분 처리
- 고차원 문제에서 PINN의 강점 (격자 불필요)
- Dirichlet BC vs Neumann BC 처리
- **실습**:
  - PINN 기초: ODE 복습 (TensorFlow)
  - **1D Heat Equation**: 열전도 (`∂u/∂t = α ∂²u/∂x²`)
  - **1D Wave Equation**: 파동 (`∂²u/∂t² = c² ∂²u/∂x²`, 2차 시간 미분)
  - **2D Heat Equation**: 2D 열전도 (3차원 입력 `(x, y, t)`)
  - **Burgers Equation**: 비선형 PDE (`∂u/∂t + u∂u/∂x = ν∂²u/∂x²`)
  - **2D Wave Equation**: 2D 막 진동
  - **복잡한 경계조건**: L자 영역 + 혼합 BC (Dirichlet + Neumann)

**주요 개념**: PDEs, PINN for Spatial-Temporal Problems, Neumann BC

**과제 6**: PINN을 이용한 편미분방정식 해법 프로젝트 (전통 방법과 비교)

---

### 🔷 Part IV: 최종 프로젝트 (Weeks 15-16)

#### **Week 15: PINN 응용 II - 고급 주제 & 최종 프로젝트 시작**
- Inverse Problems with PINN (데이터 → 물리 파라미터 추정)
- Multi-fidelity Learning
- Transfer Learning in PINN
- 현대 물리학 연구에서의 AI 활용 사례
- **최종 프로젝트 시작**: 팀별 주제 선정 및 계획 수립

**프로젝트 주제 예시**:
1. PINN을 이용한 Schrödinger 방정식 풀이
2. 복잡한 경계조건의 전자기 문제 해결
3. 상전이 시뮬레이션에 AI 적용
4. 실험 데이터 기반 물리량 예측
5. 역문제(Inverse Problem) 해결 (파라미터 추정)

---

#### **Week 16: 최종 프로젝트 발표 및 기말고사**
- 팀별 프로젝트 결과 발표 (20분)
- 코드 리뷰 및 피드백
- 기말 시험 (이론 + 실습 문제)
- 강의 총평 및 향후 학습 방향 제시

---

## 📊 평가 방식

| 항목 | 비율 | 세부 내용 |
|------|------|----------|
| **과제** | 30% | 주간 과제 6회 (각 5%) |
| **중간 프로젝트** | 15% | Part I 요약 발표 |
| **최종 프로젝트** | 25% | 팀 프로젝트 (코드 + 보고서 + 발표) |
| **중간고사** | 10% | 이론 및 기본 실습 |
| **기말고사** | 15% | 종합 평가 |
| **출석 및 참여** | 5% | 수업 참여도 및 토론 |

### 과제 제출 방식
- GitHub Repository를 통한 코드 제출
- Python 스크립트 형식 (`.py`)
- README.md에 실행 방법 및 결과 분석 포함

### 최종 프로젝트 요구사항
- 2-3명 팀 구성
- GitHub을 통한 협업
- 10-15페이지 보고서 (LaTeX 권장)
- 20분 발표 + 10분 질의응답
- 재현 가능한 코드 및 데이터

---

## 💻 실습 환경

### 필수 소프트웨어

**1. Git 설치 (버전 관리)**
```bash
# 설치 확인
git --version
```

**2. Cursor 설치 (AI 기반 IDE)**
- 다운로드: [cursor.sh](https://cursor.sh/)
- VS Code 기반 + 내장 AI (Claude/GPT-4)
- `Ctrl+K`, `Ctrl+L`로 AI와 대화

**3. uv 설치 (초고속 Python 패키지 매니저)**
```powershell
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**4. 프로젝트 환경 설정**
```bash
# 가상환경 생성
uv venv

# 패키지 설치 (pyproject.toml 기반)
uv sync

# 개별 패키지 설치
uv pip install tensorflow numpy matplotlib scipy torch
```

**5. 패키지 실행**
```bash
# 각 주차 코드 실행
cd week1
uv run python 01_hello_nn.py

# 또는 직접 python
python 01_hello_nn.py
```

### 필수 패키지
```toml
# pyproject.toml
[project]
dependencies = [
    "numpy",
    "scipy",
    "matplotlib",
    "tensorflow",
    "torch",
    "anthropic",      # Claude API
]
```

### Claude AI 설정
- Claude.ai 계정 생성
- API Key 발급 (실습용)
- Cursor 에디터와 연동

---

## 📚 주차별 읽기 자료

### Week 1-2
- Newman, "Computational Physics", Chapter 1-2
- Python for Physics 튜토리얼

### Week 3-5
- MIT 6.S191 Lecture Notes 1-3
- Goodfellow et al., "Deep Learning", Chapter 6

### Week 6-7
- "Attention Is All You Need" - Vaswani et al. (2017)
- GPT-3 Paper (Brown et al., 2020)
- InstructGPT Paper (Ouyang et al., 2022)

### Week 9-12
- Newman, "Computational Physics", Chapter 8 (ODEs)
- Landau & Binder, "Monte Carlo Simulations in Statistical Physics"
- Griffiths, "Introduction to Quantum Mechanics"

### Week 13-14
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)
- Cuomo et al., "Scientific Machine Learning" (2022)
- DeepXDE Documentation: [deepxde.readthedocs.io](https://deepxde.readthedocs.io/)

---

## 🎓 선수 과목

- **필수**:
  - 일반물리학 I, II
  - 역학
  - 전자기학 I
  - Python 프로그래밍 기초

- **권장**:
  - 양자역학 I
  - 수리물리학
  - 통계물리

---

## 🔗 유용한 링크

### 강의 자료
- [MIT 6.S191 Course Website](http://introtodeeplearning.com/)
- [Physics-Informed Neural Networks GitHub](https://github.com/maziarraissi/PINNs)

### 개발 도구
- [Cursor IDE](https://cursor.sh/)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [Claude AI](https://claude.ai/)

### 온라인 리소스
- [DeepXDE Library](https://deepxde.readthedocs.io/)
- [PhET Physics Simulations](https://phet.colorado.edu/)
- [Harvard Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

### 논문 저장소
- [ArXiv: Physics + AI](https://arxiv.org/list/physics.comp-ph/recent)
- [ArXiv: Machine Learning](https://arxiv.org/list/cs.LG/recent)

---

## 📝 과제 상세 정보

### 과제 1: 데이터 피팅 (Week 2)
실험 데이터를 제공하고, 다양한 회귀 모델(TensorFlow vs SciPy)로 피팅 후 최적 모델 선택

### 과제 2: 물리 데이터 Neural Network 예측 (Week 4)
다양한 물리 시스템(포물선 운동, 진자 등)을 Neural Network로 예측하고 모델 복잡도 분석

### 과제 3: 역학 시뮬레이션 (Week 9)
LLM을 활용하여 복잡한 역학 문제 시뮬레이션 코드 작성 (RK4 구현 포함)

### 과제 4: 전자기 시각화 (Week 10)
주어진 전하 배치의 전기장을 계산하고 FDTD로 전자기파 시뮬레이션 및 시각화

### 과제 5: 양자 시뮬레이션 (Week 11)
다양한 포텐셜에서의 양자 상태 계산 및 분석 (터널링 확률 포함)

### 과제 6: PINN 프로젝트 (Week 14)
선택한 편미분방정식을 PINN으로 풀고 전통 방법(FDM)과 비교 분석

---

## 🌟 학습 성과

본 강의를 성공적으로 이수한 학생은:

1. ✅ Neural Network의 작동 원리를 수학적으로 이해
2. ✅ LLM을 활용한 효율적인 코딩 능력 (Vibe Coding)
3. ✅ 역학, 전자기학, 양자역학, 통계물리 문제의 수치 해법
4. ✅ PINN을 이용한 미분방정식 해법 습득 (ODE + PDE)
5. ✅ 최신 AI 기술의 물리학 응용 능력
6. ✅ GitHub을 통한 협업 및 코드 관리 능력
7. ✅ 연구 논문 작성 및 발표 능력

---

## 💡 강의 철학

> "물리학자는 자연을 이해하는 사람이며,
> 계산물리학자는 자연을 시뮬레이션하는 사람이고,
> AI 시대의 물리학자는 자연을 학습하고 예측하는 사람입니다."

본 강의는 단순히 AI 도구 사용법을 가르치는 것이 아니라,
**물리적 직관과 AI 기술을 융합하여 새로운 문제 해결 방식**을
배우는 것을 목표로 합니다.

---

## 📧 연락처 및 Office Hours

**강의실**: TBA
**실습실**: TBA
**Office Hours**: 수요일 14:00-16:00 (사전 예약 권장)
**이메일**: TBA
**강의 GitHub**: TBA
**Q&A**: Slack 채널 운영

---

## 📌 주의사항

1. **학술 윤리**: AI 도구 사용 시 출처를 명확히 밝혀야 합니다
2. **코드 공유**: 과제는 개인 작업이며, 코드 복사는 금지됩니다
3. **출석**: 실습 중심 강의이므로 출석이 매우 중요합니다
4. **준비물**: 노트북 필수 (실습용)
5. **AI 검증**: AI가 생성한 코드와 결과는 반드시 물리적으로 검증할 것

---

## 🎯 학기말 기대 성과물

### 개인
- 6개의 과제 포트폴리오
- 개인 GitHub Repository
- 물리 시뮬레이션 코드 모음 (Week 1-14)

### 팀
- 최종 프로젝트 보고서
- 발표 자료
- 오픈소스 코드 (GitHub)

---

## 📚 추가 학습 자료 (선택)

### 온라인 강좌
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS231n (Convolutional Neural Networks)
- Stanford CS224N (NLP with Deep Learning)

### 유튜브 채널
- 3Blue1Brown (Neural Networks Series)
- Two Minute Papers
- Arxiv Insights

### 책
- *Neural Networks and Deep Learning* by Michael Nielsen (무료)
- *Dive into Deep Learning* by Zhang et al. (무료)
- *Computational Physics* by Giordano & Nakanishi
- *Nonlinear Dynamics and Chaos* by Strogatz

---

## 🔄 강의 계획 변경

- 본 계획서는 학습 진도에 따라 조정될 수 있습니다
- 중요한 변경사항은 최소 1주 전에 공지됩니다
- 학생 피드백을 반영하여 실습 내용을 조정할 수 있습니다

---

## 🎉 마치며

이 강의를 통해 여러분은 **21세기 물리학자에게 필수적인 AI 도구**를
마스터하게 될 것입니다. 단순히 코드를 실행하는 것을 넘어,
**물리적 직관과 AI 기술을 결합하여 창의적인 문제 해결**을
할 수 있기를 기대합니다.

함께 흥미로운 학기를 만들어갑시다! 🚀

---

**Last Updated**: 2026-02-28
**Version**: 2.1
**Instructor**: TBA
