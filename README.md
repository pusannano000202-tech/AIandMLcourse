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

## 📅 강의 일정 (16주)

### 🔷 Part I: Neural Networks & Deep Learning (Weeks 1-7)

#### **Week 1: 강의 소개 및 환경 설정** ✅
- 개발 환경 설정: Git, Cursor (AI 기반 IDE), uv (Python 패키지 매니저)
- Claude AI 계정 생성 및 기본 사용법
- **실습**: "Hello, Neural Network!" - TensorFlow로 첫 번째 신경망 구현, 수치 해법 비교

**Python 파일**: `00_hello_world.py`, `01_hello_nn.py`, `02_polynomial_fitting.py`

---

#### **Week 2: 머신러닝 기초** ✅
- 지도/비지도/강화 학습, 데이터 전처리, 손실 함수와 최적화
- **실습**: 훅의 법칙 선형 회귀, K-Means 군집화, Gradient Descent 시각화

**Python 파일**: `01_linear_regression_spring.py`, `02_unsupervised_clustering.py`, `03_data_preprocessing.py`, `04_gradient_descent_vis.py`

---

#### **Week 3: Neural Network 기초 이론** ✅
- Perceptron, MLP, Activation Functions, Forward/Backward Propagation
- Universal Approximation Theorem
- **실습**: Perceptron 논리 게이트, Numpy MLP로 XOR 해결

**Python 파일**: `01_perceptron.py`, `02_activation_functions.py`, `03_forward_propagation.py`, `04_mlp_numpy.py`, `05_universal_approximation.py`

---

#### **Week 4: 물리 데이터로 학습하기** ✅
- Neural Network로 물리 법칙 학습, 과적합/과소적합, RK4 수치 적분
- **실습**: 1D 함수 근사, 포물선 운동 예측, 과적합 실험, 진자 주기 예측

**Python 파일**: `01perfect1d.py`, `02projectile.py`, `03overfitting.py`, `04pendulum.py`

---

#### **Week 5: Deep Learning의 핵심 기법** ✅
- Regularization (L1/L2, Dropout, Batch Normalization), Data Augmentation, Transfer Learning
- **실습**: MNIST 손글씨 인식 (CNN)

**Python 파일**: `01_regularization.py`, `02_overfitting_underfitting.py`, `03_data_augmentation.py`, `04_transfer_learning.py`, `05_mnist_cnn.py`

---

#### **Week 6: Transformer와 Attention Mechanism** ✅
- Self-Attention, Multi-Head Attention, Positional Encoding
- Residual Connection & Layer Normalization
- **실습**: 시계열 예측 (Transformer vs RNN 비교)

**Python 파일**: `01_attention_basics.py`, `02_self_attention.py`, `03_positional_encoding.py`, `04_transformer_block.py`, `05_sequence_modeling.py`

---

#### **Week 7: Large Language Models (LLM) 개론** ✅
- GPT vs BERT 아키텍처 비교, Tokenization, RLHF
- **실습**: Claude API 개념 이해, Prompt Engineering

**Python 파일**: `01_tokens_and_embeddings.py`, `02_gpt_bert_architectures.py`, `03_pretraining_finetuning.py`, `04_claude_api_simple.py`

---

### 🔷 Part II: LLM Vibe Coding for Physics (Weeks 8-12)

#### **Week 8: 중간고사 / LLM 기반 코딩 입문**
- Vibe Coding, Prompt Engineering 심화, AI 디버깅

#### **Week 9: 고전 역학 문제 해결**
- Euler vs RK4, 행성 운동, 이중 진자 혼돈, 라그랑지안/해밀토니안 역학

#### **Week 10: 전자기학 시뮬레이션**
- Maxwell 방정식, FDTD, 전기장/자기장 시각화, 전자기파 전파

#### **Week 11: 양자역학 시뮬레이션**
- Schrödinger 방정식, 파동함수 시각화, 터널링 효과, 조화 진동자

#### **Week 12: 통계물리 및 Monte Carlo 시뮬레이션**
- Metropolis-Hastings, Ising 모델, 상전이, 임계 현상

---

### 🔷 Part III: Physics-Informed Neural Networks (Weeks 13-14)

#### **Week 13: PINN 기초 이론 - ODE**
- PINN 개념, 자동 미분, Physics Loss 설계, PINN vs RK4 비교

#### **Week 14: PINN 응용 - PDE**
- 1D/2D Heat Equation, Wave Equation, Burgers Equation, 복잡한 경계조건

---

### 🔷 Part IV: 최종 프로젝트 (Weeks 15-16)

#### **Week 15: PINN 응용 II & 최종 프로젝트 시작**
#### **Week 16: 최종 프로젝트 발표 및 기말고사**

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

---

## 💻 실습 환경

### 환경 설정

```bash
# 1. Git 설치 확인
git --version

# 2. Cursor 설치: https://cursor.sh/

# 3. uv 설치 (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 4. 프로젝트 환경 설정
uv venv
uv sync

# 5. 코드 실행
cd week1
uv run python 01_hello_nn.py
```

### 필수 패키지
- numpy, scipy, matplotlib
- tensorflow, torch
- anthropic (Claude API)

---

## 📖 교재 및 참고자료

### 주교재
- **MIT 6.S191**: Introduction to Deep Learning
- 강의자 제공 자료: Python 코드 예제, 주차별 가이드 (`week*.md`)

### 참고서적
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Computational Physics* by Mark Newman
- *Physics-Informed Neural Networks* (관련 논문 모음)

### 온라인 자료
- [MIT 6.S191](http://introtodeeplearning.com/)
- [Cursor IDE](https://cursor.sh/)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [DeepXDE Library](https://deepxde.readthedocs.io/)

---

## 🎓 선수 과목

- **필수**: 일반물리학 I/II, 역학, 전자기학 I, Python 기초
- **권장**: 양자역학 I, 수리물리학, 통계물리

---

## 📌 주의사항

1. **학술 윤리**: AI 도구 사용 시 출처를 명확히 밝혀야 합니다
2. **코드 공유**: 과제는 개인 작업이며, 코드 복사는 금지됩니다
3. **출석**: 실습 중심 강의이므로 출석이 매우 중요합니다
4. **준비물**: 노트북 필수 (실습용)
5. **AI 검증**: AI가 생성한 코드와 결과는 반드시 물리적으로 검증할 것

---

**Last Updated**: 2026-03-04
**Version**: 2.2
