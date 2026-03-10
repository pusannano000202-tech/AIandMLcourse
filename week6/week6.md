# Week 6: Transformer와 Attention Mechanism

## 📚 학습 목표

이번 주차에서는 현대 딥러닝의 핵심인 Transformer 아키텍처를 학습합니다.

**배울 내용:**
1. RNN의 한계와 Attention의 등장 배경
2. Self-Attention 메커니즘의 원리
3. Positional Encoding의 필요성과 구현
4. 완전한 Transformer Block 구조
5. 실제 Sequence Modeling 적용

**왜 중요한가?**
- Transformer는 GPT, BERT, ChatGPT의 기반
- 자연어 처리(NLP)의 혁명을 일으킴
- 병렬 처리로 학습 속도 대폭 향상
- Attention으로 모델 해석 가능
- Computer Vision, 음성 인식 등 다양한 분야로 확장

---

## 🎯 들어가기: RNN의 한계

### RNN이 가진 문제들

**1. 순차 처리 (Sequential Processing)**
```
t=1 → t=2 → t=3 → ... → t=n
```
- 한 번에 한 time step씩만 처리
- 병렬화 불가능 → GPU 활용 제한
- 긴 문장 처리 시 매우 느림

**2. 긴 거리 의존성 (Long-Range Dependencies)**
```
"The cat, which we found yesterday, is very cute."
 ↑                                           ↑
 주어와 동사 사이가 멀면 연결이 어려움
```
- 멀리 떨어진 단어 간 관계 학습 어려움
- 그래디언트 소실/폭발 문제
- LSTM/GRU로 개선했지만 근본적 한계

**3. 고정된 Hidden State**
- 모든 정보를 하나의 벡터에 압축
- 정보 병목 현상 (Information Bottleneck)
- 긴 문장일수록 정보 손실

### Attention의 등장

**핵심 아이디어:**
> "모든 위치에서 모든 위치를 직접 볼 수 있다면?"

**장점:**
- ✅ 병렬 처리 가능
- ✅ 거리 무관하게 O(1) 연결
- ✅ 어디를 보는지 해석 가능
- ✅ 동적으로 중요한 정보 선택

---

## 🔬 Lab 1: Attention의 기초 (01_attention_basics.py)

### 목적
Attention 메커니즘의 가장 기본적인 형태인 Scaled Dot-Product Attention을 이해합니다.

### 프로그램 실행
```bash
cd week6
python 01_attention_basics.py
```

### 핵심 개념: Query, Key, Value

**일상적 비유:**
```
도서관에서 책 찾기:
- Query (질문): "머신러닝 책 찾아줘"
- Key (색인): 각 책의 제목/키워드
- Value (내용): 실제 책의 내용

Attention = 질문과 가장 관련 있는 책들을 가중평균
```

**수학적 정의:**
```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

여기서:
- Q: Query 행렬 (n_queries × d_k)
- K: Key 행렬 (n_keys × d_k)
- V: Value 행렬 (n_values × d_v)
- d_k: Key의 차원
```

### 단계별 계산

**Step 1: 유사도 점수 계산**
```
Scores = Q · K^T
```
- 각 Query와 각 Key의 내적
- 내적이 크면 유사도 높음

**Step 2: Scaling**
```
Scaled_Scores = Scores / sqrt(d_k)
```
- 왜 나누나? → Softmax의 그래디언트 안정화
- d_k가 크면 내적 값도 커짐
- 너무 큰 값은 softmax를 saturate 시킴

**Step 3: Softmax로 확률화**
```
Attention_Weights = softmax(Scaled_Scores)
```
- 각 행의 합 = 1.0 (확률 분포)
- 높은 점수 → 높은 가중치

**Step 4: Value의 가중 평균**
```
Output = Attention_Weights · V
```
- 중요한 Value에 높은 가중치
- 최종 출력은 관련 정보의 조합

### 주요 결과

**Attention Weights의 의미:**
- 각 단어가 다른 단어들에 얼마나 주목하는지
- Heatmap으로 시각화 가능
- 모델이 "어디를 보는지" 알 수 있음

**Scaling의 효과:**
- Scaling 없으면: 극단적 확률 (0.9, 0.05, 0.05)
- Scaling 있으면: 부드러운 분포 (0.5, 0.3, 0.2)
- 학습 안정성 향상

### 출력 파일
- `outputs/01_attention_weights.png`: Attention 가중치 행렬
- `outputs/01_scaling_effect.png`: Scaling 효과 비교
- `outputs/01_longer_sequence.png`: 긴 문장에서의 Attention

---

## 🔬 Lab 2: Self-Attention과 Multi-Head (02_self_attention.py)

### 목적
Self-Attention의 원리와 Multi-Head Attention이 왜 필요한지 이해합니다.

### Self-Attention이란?

**일반 Attention:**
```
Encoder → Decoder 간 attention
Query: Decoder의 현재 상태
Key, Value: Encoder의 모든 출력
```

**Self-Attention:**
```
Query, Key, Value 모두 같은 입력에서!
자기 자신 내부의 관계를 학습
```

**예시:**
```
문장: "The cat sat on the mat"

"cat"의 Self-Attention:
- "The"와의 관계: 0.15 (관사)
- "cat"와의 관계: 0.30 (자기 자신)
- "sat"와의 관계: 0.35 (동사 - 중요!)
- "on"와의 관계: 0.05
- "the"와의 관계: 0.05
- "mat"와의 관계: 0.10 (목적어)

→ "cat"은 주로 "sat"에 주목!
```

### Multi-Head Attention

**왜 여러 개의 Head?**

**단일 Attention의 한계:**
- 한 번에 하나의 관점만 학습
- "cat sat" (주어-동사) 관계 학습 중이면
- "cat on mat" (위치 관계)는 놓칠 수 있음

**Multi-Head의 장점:**
```
Head 1: 주어-동사 관계
Head 2: 수식어 관계
Head 3: 위치 관계
Head 4: 의미적 유사성

→ 병렬로 다양한 패턴 학습!
```

**구현:**
```
각 Head마다 독립적인 Q, K, V 변환
Head_i = Attention(Q_i, K_i, V_i)

MultiHead = Concat(Head_1, ..., Head_h) · W_O
```

### RNN vs Self-Attention 비교

**계산 복잡도:**
```
RNN:
- 시간 복잡도: O(n) steps (sequential)
- 한 스텝당: O(d²) 연산
- 병렬화: 불가능

Self-Attention:
- 시간 복잡도: O(1) steps (parallel)
- 총 연산: O(n²·d)
- 병렬화: 완전 가능
```

**Path Length (정보 전달 경로):**
```
RNN: 위치 i → j까지 |i-j| 스텝
Self-Attention: 모든 위치 간 1 스텝!

"I love you very very much"
 ↑                        ↑
RNN: 5 스텝 거쳐야 연결
Attention: 직접 연결!
```

**Trade-off:**
```
n < d (짧은 시퀀스, 큰 차원):
  → Attention이 효율적

n >> d (매우 긴 시퀀스):
  → RNN이 메모리 효율적
  → Sparse Attention 등 개선 방법 필요
```

### 주요 관찰

**Head의 전문화:**
- 각 Head가 다른 패턴 학습
- 어떤 Head는 인접 단어 집중
- 어떤 Head는 먼 거리 의존성 포착

**Attention Diversity:**
- Head 간 분산이 클수록 좋음
- 다양한 정보 추출

### 출력 파일
- `outputs/02_self_attention_components.png`: Self-Attention 구성 요소
- `outputs/02_multi_head_attention.png`: Multi-Head 패턴 비교
- `outputs/02_rnn_vs_attention.png`: RNN과 복잡도 비교

---

## 🔬 Lab 3: Positional Encoding (03_positional_encoding.py)

### 목적
Self-Attention은 순서를 모름 - Positional Encoding으로 위치 정보 추가

### 문제: Permutation Invariance

**Self-Attention의 맹점:**
```
Input 1: "I love you"
Input 2: "You love I"

Self-Attention만 사용하면:
→ 같은 출력! (순서 무시)
```

**증명:**
```
Attention(Q, K, V)는 집합 연산
단어 순서를 바꿔도 결과 동일
→ 위치 정보가 없음!
```

### 해결: Positional Encoding

**기본 아이디어:**
```
Word_Embedding + Positional_Encoding = Final_Input

예:
"cat" 임베딩: [0.2, 0.5, -0.1, ...]
Position 3 인코딩: [0.1, -0.2, 0.3, ...]
→ Final: [0.3, 0.3, 0.2, ...]
```

### Sinusoidal Positional Encoding

**수식 (Vaswani et al., 2017):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

여기서:
- pos: 위치 (0, 1, 2, ...)
- i: 차원 인덱스
- d_model: 임베딩 차원
```

**직관적 이해:**
```
짝수 차원: sin 함수
홀수 차원: cos 함수

낮은 차원 (i가 작음):
  → 높은 주파수 → 빠르게 변함
  → 세밀한 위치 구분

높은 차원 (i가 큼):
  → 낮은 주파수 → 천천히 변함
  → 큰 범위의 패턴
```

**왜 Sine/Cosine?**

1. **주기 함수** → 연속적이고 부드러움
2. **다양한 주파수** → 여러 스케일 표현
3. **상대 위치 표현:**
   ```
   PE(pos+k)는 PE(pos)의 선형 함수
   → 모델이 상대 거리 학습 가능
   ```
4. **학습 불필요** → 파라미터 0개
5. **외삽 가능** → 학습 시보다 긴 시퀀스도 OK

### 다른 방법들과 비교

**1. Learned Positional Embeddings:**
```python
pos_embedding = nn.Embedding(max_length, d_model)
```
- 장점: 유연함, 학습 가능
- 단점: max_length보다 긴 시퀀스 불가
- 사용: BERT, GPT 초기 버전

**2. Linear Encoding:**
```
PE(pos) = pos / max_length
```
- 간단하지만 정보량 부족

**3. Relative Positional Encoding:**
- 절대 위치 대신 상대 거리
- Transformer-XL, T5에서 사용

### 효과 분석

**위치 정보 없으면:**
```
"I love you" = "you love I"
문법적으로 완전히 다른 의미인데 구분 못함
```

**위치 정보 추가 후:**
```
각 단어가 위치별 고유한 표현
순서가 바뀌면 다른 벡터
→ 문장 구조 학습 가능
```

**Similarity Pattern:**
```
인접한 위치: 높은 유사도
먼 위치: 낮은 유사도
→ 모델이 거리 감각 가짐
```

### 출력 파일
- `outputs/03_positional_encoding_sinusoidal.png`: Sinusoidal 인코딩 패턴
- `outputs/03_encoding_comparison.png`: 다양한 인코딩 방법 비교
- `outputs/03_relative_position.png`: 위치 간 유사도
- `outputs/03_position_effect.png`: Attention에 미치는 영향

---

## 🔬 Lab 4: 완전한 Transformer Block (04_transformer_block.py)

### 목적
모든 요소를 결합한 완전한 Transformer Encoder Block을 구현하고 이해합니다.

### Transformer Block 구조

```
Input
  ↓
[Multi-Head Self-Attention]
  ↓
[Add & Normalize] ← Residual Connection
  ↓
[Feed-Forward Network]
  ↓
[Add & Normalize] ← Residual Connection
  ↓
Output
```

### 각 컴포넌트 상세

**1. Multi-Head Self-Attention**
```
이미 배운 내용:
- 여러 Head로 다양한 패턴 학습
- 병렬 처리
- 관계 포착
```

**2. Residual Connection (Add)**
```
Output = X + Sublayer(X)

왜 필요한가?
```

**그래디언트 소실 문제:**
```
깊은 네트워크:
Layer 1 → Layer 2 → ... → Layer N

Backpropagation 시:
∂Loss/∂Layer1 = ∂Loss/∂LayerN × ∂LayerN/∂Layer(N-1) × ...

각 곱셈마다 값이 작아지면
→ 그래디언트 소실 (Vanishing Gradient)
```

**Residual의 해결책:**
```
y = x + F(x)

∂y/∂x = 1 + ∂F(x)/∂x

항상 최소 1 이상!
→ 그래디언트가 직접 흐를 수 있는 경로 (Highway)
→ 매우 깊은 네트워크 학습 가능
```

**정보 보존:**
```
입력 정보가 출력까지 직접 전달
→ Identity mapping
→ 학습 초기에도 안정적
```

**3. Layer Normalization**
```
LN(x) = γ · (x - μ) / σ + β

여기서:
- μ: 평균 (across features)
- σ: 표준편차 (across features)
- γ, β: 학습 가능한 파라미터
```

**Batch Norm vs Layer Norm:**
```
Batch Normalization:
- 배치 내 샘플들 간 정규화
- 배치 크기에 의존
- RNN에 부적합

Layer Normalization:
- 각 샘플의 feature 간 정규화
- 배치 독립적
- RNN, Transformer에 적합
```

**효과:**
```
1. Internal Covariate Shift 감소
2. 학습 안정화
3. 더 큰 learning rate 사용 가능
4. 빠른 수렴
```

**4. Feed-Forward Network (FFN)**
```
FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂

구조:
d_model → d_ff → d_model
보통 d_ff = 4 × d_model

예: 512 → 2048 → 512
```

**왜 필요한가?**

**비선형성 추가:**
```
Attention은 선형 변환들의 조합
→ 표현력 한계

FFN의 비선형 활성화:
→ 복잡한 패턴 학습 가능
```

**Position-wise:**
```
각 위치에 독립적으로 적용
같은 변환, 하지만 병렬 처리
```

**Capacity 증가:**
```
d_ff가 크면 (4x):
→ 더 많은 패턴 학습
→ 모델 용량 증가
```

**GELU vs ReLU:**
```
ReLU: max(0, x)
  - 간단, 빠름
  - 음수 정보 완전 손실

GELU: x·Φ(x) (Gaussian Error Linear Unit)
  - 부드러운 곡선
  - 음수도 작은 값으로 전달
  - Transformer에서 성능 우수
```

### 데이터 흐름 분석

**각 단계에서 변화:**
```
1. Input: 원본 임베딩
2. Attention: 문맥 정보 추가
3. Residual 1: 원본 + 문맥
4. Norm 1: 정규화
5. FFN: 비선형 변환
6. Residual 2: 정보 보존
7. Norm 2: 최종 정규화
```

**통계적 변화:**
```
평균 (μ):
- LayerNorm 후 ≈ 0

분산 (σ²):
- LayerNorm 후 ≈ 1

→ 학습 안정성 향상
```

### Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer):**
```
X → Sublayer → Add → Norm
```

**Pre-Norm (현대적):**
```
X → Norm → Sublayer → Add
```

**Pre-Norm 장점:**
```
- 학습 더 안정적
- 큰 모델에서 유리
- Warm-up 덜 필요
- GPT-3, T5 등에서 사용
```

### 출력 파일
- `outputs/04_transformer_dataflow.png`: 데이터 흐름 단계별 시각화
- `outputs/04_attention_patterns.png`: Multi-Head Attention 패턴
- `outputs/04_residual_effect.png`: Residual Connection 효과

---

## 🔬 Lab 5: 실전 Sequence Modeling (05_sequence_modeling.py)

### 목적
Transformer를 실제 시퀀스 예측 문제에 적용하고 RNN과 비교합니다.

### 실험 설정

**Task: 시계열 예측**
```
Input: [y₀, y₁, ..., y₁₉]  (20 time steps)
Output: y₂₀                 (다음 값 예측)

Data: 사인파
y(t) = A·sin(f·t + φ)
- A: 진폭 (random)
- f: 주파수 (random)
- φ: 위상 (random)
```

### 모델 구성

**Transformer:**
```
1. Input Projection: 1D → 32D
2. Positional Encoding 추가
3. Multi-Head Attention (4 heads)
4. Feed-Forward (32 → 128 → 32)
5. Pooling (마지막 위치)
6. Output Projection: 32D → 1D
```

**RNN (비교군):**
```
1. Hidden state: 64D
2. Sequential processing
3. Output: 최종 hidden state → 1D
```

### Attention 분석

**무엇을 학습했나?**

**위치별 중요도:**
```
마지막 위치에서 각 과거 위치로의 attention:

가까운 과거 (t-1, t-2, t-3):
  → 높은 attention (0.3~0.5)
  → 최근 추세 중요

중간 과거 (t-10):
  → 중간 attention (0.1~0.2)
  → 주기 파악

먼 과거 (t-20):
  → 낮은 attention (0.05)
  → 덜 중요
```

**Multi-Head 전문화:**
```
Head 1: 직전 값에 집중 (local trend)
Head 2: 주기적 패턴 포착 (periodicity)
Head 3: 전체 범위 고려 (global context)
Head 4: 특정 위치 조합 (specific patterns)
```

### 성능 비교

**정량적 결과:**
```
MSE Loss (예시):
- Transformer: 0.0234 ± 0.0156
- RNN: 0.0289 ± 0.0198

→ Transformer가 약간 우수
→ 분산도 더 작음 (안정적)
```

**왜 Transformer가 좋은가?**

1. **병렬 처리:**
   ```
   전체 시퀀스를 한 번에 처리
   → 더 풍부한 문맥 정보
   ```

2. **직접 연결:**
   ```
   t=0과 t=20이 직접 연결
   → RNN은 20 스텝 거쳐야 함
   → 장거리 의존성 학습 유리
   ```

3. **해석 가능성:**
   ```
   Attention weights 시각화
   → 어느 시점이 중요한지 알 수 있음
   → RNN의 hidden state는 불투명
   ```

### 실전 적용 시 고려사항

**언제 Transformer?**
```
✅ 문맥이 중요한 작업 (번역, 요약)
✅ 병렬 처리 가능한 환경 (GPU)
✅ 해석이 필요한 경우
✅ 중간 길이 시퀀스 (수백~수천)
```

**언제 RNN?**
```
✅ 매우 긴 시퀀스 (메모리 제약)
✅ 온라인/스트리밍 처리
✅ 순차적 의존성이 강한 경우
✅ 작은 모델 필요 (모바일)
```

**Hybrid Approaches:**
```
1. Conformer (Speech):
   Convolution + Transformer

2. Longformer:
   Local + Global Attention

3. Linformer:
   Linear complexity Attention
```

### 출력 파일
- `outputs/05_sample_sequences.png`: 입력 데이터 예시
- `outputs/05_attention_patterns.png`: Attention 패턴 분석
- `outputs/05_performance_comparison.png`: Transformer vs RNN 성능
- `outputs/05_multihead_analysis.png`: Multi-Head 상세 분석
- `outputs/05_position_importance.png`: 위치별 중요도

---

## 📊 주요 결과 요약

### Transformer의 핵심 요소

**1. Attention Mechanism:**
```
- Query, Key, Value로 관련성 계산
- Softmax로 가중치 결정
- 동적이고 해석 가능
```

**2. Self-Attention:**
```
- 입력 내부의 관계 학습
- 모든 위치 간 O(1) 연결
- 병렬 처리 가능
```

**3. Multi-Head:**
```
- 다양한 패턴 동시 학습
- Head별 전문화
- 표현력 증가
```

**4. Positional Encoding:**
```
- 순서 정보 추가
- Sinusoidal: 학습 불필요
- 상대 위치 표현 가능
```

**5. Residual + LayerNorm:**
```
- 깊은 네트워크 학습 가능
- 그래디언트 흐름 개선
- 학습 안정화
```

**6. Feed-Forward:**
```
- 비선형성 추가
- 용량 증가 (4x expansion)
- Position-wise 적용
```

### 복잡도 분석

**Time Complexity:**
```
Self-Attention: O(n²·d)
  - n: 시퀀스 길이
  - d: 임베딩 차원

RNN: O(n·d²)
  - 순차 처리 필요

n < d: Attention이 유리
n > d: RNN이 유리 (보통 d가 더 큼)
```

**Space Complexity:**
```
Attention: O(n²) (attention matrix)
RNN: O(1) (hidden state만)

→ 매우 긴 시퀀스: 메모리 부담
```

**Path Length:**
```
Attention: O(1) (직접 연결)
RNN: O(n) (순차 전달)

→ Attention이 장거리 의존성 학습 유리
```

### 실전 성능

**NLP Tasks:**
```
기계 번역: BLEU 점수 향상
감성 분석: 정확도 향상
질의 응답: F1 score 향상

→ 대부분의 NLP에서 SOTA
```

**장점:**
```
1. 병렬 처리 → 빠른 학습
2. 장거리 의존성 포착
3. 해석 가능 (attention weights)
4. 전이 학습 (Pre-training)
```

**단점:**
```
1. 메모리 사용량 큼 (O(n²))
2. 매우 긴 시퀀스 어려움
3. 파라미터 수 많음
4. 작은 데이터셋에서 overfitting
```

---

## 💡 실전 팁

### 프로그램 실행 순서

1. **01_attention_basics.py** (~1분)
   - Attention 메커니즘 기초
   - Query-Key-Value 이해
   - Scaling 효과

2. **02_self_attention.py** (~2분)
   - Self-Attention
   - Multi-Head
   - RNN 비교

3. **03_positional_encoding.py** (~1분)
   - 위치 인코딩 필요성
   - Sinusoidal 패턴
   - 효과 분석

4. **04_transformer_block.py** (~2분)
   - 완전한 블록 구조
   - Residual 효과
   - LayerNorm 역할

5. **05_sequence_modeling.py** (~2분)
   - 실제 적용
   - 성능 비교
   - Attention 해석

### 하이퍼파라미터 조정

**Model Dimension (d_model):**
```
작게 (32-64): 빠르지만 용량 부족
중간 (128-256): 일반적 작업
크게 (512-1024): 복잡한 작업, 큰 데이터

권장: 64배수 (GPU 최적화)
```

**Number of Heads:**
```
보통: 4, 8, 16
d_model % n_heads == 0 필수

많을수록: 다양한 패턴, 하지만 head당 차원 감소
```

**FFN Dimension:**
```
일반적: d_ff = 4 × d_model
크게: 6x, 8x (용량 증가)
```

**Layer 개수:**
```
작은 작업: 2-4 layers
중간 작업: 6-12 layers (BERT Base)
큰 작업: 24+ layers (GPT-3)
```

### 학습 팁

**1. Warm-up Learning Rate:**
```
처음에는 작은 lr로 시작
점진적으로 증가
이후 감소

→ 초기 불안정성 방지
```

**2. Label Smoothing:**
```
Hard target: [0, 0, 1, 0]
Soft target: [0.025, 0.025, 0.9, 0.025]

→ Overconfidence 방지
```

**3. Dropout:**
```
Attention weights에 dropout
FFN에 dropout

→ Overfitting 방지
```

### 디버깅

**자주 하는 실수:**

1. **Positional Encoding 빠뜨림:**
   ```
   증상: 순서 무시
   해결: 임베딩에 pos_enc 추가 확인
   ```

2. **Dimension 불일치:**
   ```
   증상: Shape error
   해결: d_model % n_heads == 0 확인
   ```

3. **Scaling 누락:**
   ```
   증상: 학습 불안정
   해결: scores / sqrt(d_k) 확인
   ```

4. **Softmax axis 잘못:**
   ```
   증상: 이상한 attention
   해결: axis=-1 (마지막 차원)
   ```

---

## 📖 더 공부하려면

### 필수 논문

1. **"Attention Is All You Need" (2017)**
   - Vaswani et al.
   - 원조 Transformer 논문
   - 반드시 읽어야 함!

2. **"BERT" (2018)**
   - Bidirectional Encoder
   - Pre-training + Fine-tuning
   - NLP의 혁명

3. **"GPT-3" (2020)**
   - 175B 파라미터
   - Few-shot learning
   - 거대 모델의 가능성

### 발전된 아키텍처

**Efficient Transformers:**
- Linformer: Linear complexity
- Reformer: LSH attention
- Longformer: Local + Global
- Performer: FAVOR+ mechanism

**Vision Transformers:**
- ViT: Image classification
- DETR: Object detection
- Swin: Hierarchical structure

**Multi-modal:**
- CLIP: Image + Text
- DALL-E: Text → Image
- Flamingo: Vision-Language

### 실습 자료

**코드:**
- [Harvard NLP's Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

**강의:**
- Stanford CS224N (NLP with Deep Learning)
- CMU 11-747 (Neural Networks for NLP)
- Fast.ai NLP Course

**책:**
- "Natural Language Processing with Transformers" (HuggingFace)
- "Speech and Language Processing" (Jurafsky & Martin)

---

## 🔧 문제 해결

### 흔한 오류

**1. Out of Memory (OOM)**
```
원인: Attention matrix O(n²)
해결:
  - Batch size 줄이기
  - Gradient checkpointing
  - Sequence length 제한
```

**2. NaN Loss**
```
원인:
  - Learning rate 너무 큼
  - Gradient explosion
  - Numerical instability

해결:
  - Learning rate warm-up
  - Gradient clipping
  - Mixed precision training
```

**3. Underfitting**
```
원인: 모델 용량 부족
해결:
  - d_model 증가
  - Layer 추가
  - d_ff 증가
```

**4. Overfitting**
```
원인: 모델이 너무 큼 or 데이터 부족
해결:
  - Dropout 증가
  - 데이터 augmentation
  - 정규화 강화
```

### 성능 최적화

**빠른 실험:**
```
- 작은 d_model (32-64)
- 적은 heads (2-4)
- 짧은 sequence
- 작은 batch
```

**최종 모델:**
```
- 큰 d_model (256-512)
- 많은 heads (8-16)
- 긴 sequence
- 큰 batch (GPU 메모리 허용 범위)
```

---

## 🎓 학습 점검

### 기본 개념

- [ ] Attention이 왜 필요한지 설명할 수 있나?
- [ ] Query, Key, Value의 역할을 이해하나?
- [ ] Scaled Dot-Product Attention 식을 쓸 수 있나?
- [ ] Self-Attention과 일반 Attention의 차이는?

### 중급 개념

- [ ] Multi-Head Attention이 왜 필요한지 아는가?
- [ ] Positional Encoding이 없으면 어떻게 되나?
- [ ] Sinusoidal encoding의 장점은?
- [ ] Residual Connection의 역할은?

### 고급 개념

- [ ] Layer Normalization vs Batch Normalization?
- [ ] Pre-Norm vs Post-Norm 차이는?
- [ ] Transformer의 복잡도 O(n²)를 줄이는 방법은?
- [ ] Attention weights를 어떻게 해석하나?

### 실습 과제

1. **Attention 분석:**
   - 다양한 문장에서 attention pattern 관찰
   - 어떤 단어가 어디에 주목하는지 분석

2. **Position 실험:**
   - Positional encoding 없이 학습
   - 성능 차이 측정

3. **Head 비교:**
   - Head 수 변화시키며 성능 비교
   - 1, 2, 4, 8 heads

4. **RNN 비교:**
   - 다양한 시퀀스 길이에서 성능 비교
   - 계산 시간 측정

---

## ✨ 결론

**배운 것:**
- Attention: 동적으로 중요한 정보 선택
- Self-Attention: 입력 내부의 관계 학습
- Multi-Head: 다양한 패턴 병렬 학습
- Positional Encoding: 순서 정보 추가
- Transformer Block: 모든 요소의 조화

**핵심 통찰:**
```
"Attention Is All You Need"

순차 처리 (Sequential) → 병렬 처리 (Parallel)
고정 문맥 (Fixed) → 동적 문맥 (Dynamic)
불투명 (Opaque) → 해석 가능 (Interpretable)
```

**영향:**
- NLP 전 분야에서 SOTA
- Computer Vision으로 확장
- Multi-modal AI의 기반
- ChatGPT, GPT-4의 핵심
- AI 연구의 새로운 패러다임

**다음 단계:**
- Pre-training (BERT, GPT 스타일)
- Fine-tuning 기법
- Prompt Engineering
- Vision Transformer (ViT)
- Efficient Transformers

---

*"Attention은 단순한 메커니즘이지만, 현대 AI의 핵심입니다!"*

Transformer는 Deep Learning 역사의 전환점입니다. 이 주차에서 배운 내용은 ChatGPT, DALL-E, Stable Diffusion 등 최신 AI 시스템을 이해하는 기초가 됩니다!
