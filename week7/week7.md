# Week 7: Large Language Models (LLM) 개론

## 📚 학습 목표

이번 주차에서는 GPT, BERT, Claude 등 현대 LLM의 기초를 학습합니다.

**배울 내용:**
1. Tokenization과 Embedding의 원리
2. GPT vs BERT 아키텍처 비교
3. Pre-training과 Fine-tuning 과정
4. Claude API 사용법 (개념)
5. 물리학 연구에서의 LLM 활용

**왜 중요한가?**
- LLM은 AI의 혁명적 발전
- 코드 생성, 문제 해결, 연구 보조 가능
- 물리학 교육 및 연구에 적용 가능
- ChatGPT, Claude의 작동 원리 이해

---

## 🎯 들어가기: LLM이란?

### Large Language Model의 정의

**LLM = Large Language Model**
- Large: 수억~수천억 개의 파라미터
- Language: 자연어 이해 및 생성
- Model: Transformer 기반 신경망

**발전 과정:**
```
2017: Transformer 등장 (Attention Is All You Need)
2018: BERT, GPT-1 (언어 이해의 혁신)
2019: GPT-2 (더 큰 모델)
2020: GPT-3 (175B params, Few-shot learning)
2022: ChatGPT (RLHF로 대중화)
2023: GPT-4, Claude 3 (Multimodal)
2024: 더 긴 context, 더 나은 추론
```

---

## 🔬 Lab 1: Tokens and Embeddings (01_tokens_and_embeddings.py)

### 목적
텍스트를 LLM이 이해할 수 있는 형태로 변환하는 과정을 학습합니다.

### Tokenization이란?

**문제:**
```
컴퓨터는 숫자만 이해
텍스트 → 숫자 변환 필요
```

**해결책: Tokenization**
```
Text: "The cat sat"
→ Tokens: ["The", "cat", "sat"]
→ IDs: [45, 123, 89]
→ Embeddings: 벡터로 변환
```

### Tokenization 방법 비교

**1. Character-level:**
```
Input: "Hello"
Tokens: ['H', 'e', 'l', 'l', 'o']

장점: 간단, OOV 없음
단점: 시퀀스 너무 길음
```

**2. Word-level:**
```
Input: "Hello world"
Tokens: ['hello', 'world']

장점: 직관적
단점: 어휘 크기 거대, OOV 문제
```

**3. Subword (BPE):**
```
Input: "unhappiness"
Tokens: ['un', 'happiness']

장점: 균형 잡힘, OOV 해결
단점: 약간 복잡
```

### Token Embedding

**토큰 → 벡터:**
```
Token "cat" → [0.2, -0.5, 0.8, ..., 0.1]  (768 dim)
Token "dog" → [0.3, -0.4, 0.7, ..., 0.2]  (768 dim)

유사한 의미 → 유사한 벡터
```

**학습 방법:**
- Pre-training 중 자동 학습
- 문맥 속에서 의미 파악
- 가까운 단어는 비슷한 벡터

### Context Window

**정의:**
```
모델이 한 번에 처리할 수 있는 최대 토큰 수
```

**모델별 비교:**
```
GPT-2: 1,024 tokens
GPT-3: 2,048 tokens
GPT-4 (8K): 8,192 tokens
GPT-4 (32K): 32,768 tokens
Claude 3: 200,000 tokens
```

**왜 중요한가?**
- 긴 문서 처리 가능 여부
- 문맥 이해 범위
- 메모리 및 비용과 직결

---

## 🔬 Lab 2: GPT vs BERT Architectures (02_gpt_bert_architectures.py)

### 목적
Transformer 기반의 두 가지 주요 아키텍처를 비교합니다.

### GPT (Generative Pre-trained Transformer)

**구조: Decoder-only**
```
[Token 1] → [Token 2] → [Token 3] → ...

각 토큰은 이전 토큰만 볼 수 있음 (Causal Masking)
```

**학습 방법: Autoregressive**
```
주어진 문장: "The cat sat on the"
예측: "mat"

P(mat | The cat sat on the) 최대화
```

**강점:**
- 텍스트 생성 탁월
- 창의적 작문
- 대화 시스템
- Few-shot learning

**약점:**
- 양방향 문맥 부족
- 분류 작업에서 약함

### BERT (Bidirectional Encoder Representations from Transformers)

**구조: Encoder-only**
```
모든 토큰이 모든 토큰을 볼 수 있음 (Full Attention)
```

**학습 방법: Masked Language Model**
```
입력: "The [MASK] sat on the mat"
예측: "cat"

빈칸 채우기
```

**강점:**
- 양방향 문맥 이해
- 분류 작업 우수
- 질의응답
- 개체명 인식

**약점:**
- 텍스트 생성 불가
- Fine-tuning 필수

### Attention Masking 차이

**GPT (Causal Mask):**
```
Position:  0  1  2  3
    0:    ✓  ✗  ✗  ✗
    1:    ✓  ✓  ✗  ✗
    2:    ✓  ✓  ✓  ✗
    3:    ✓  ✓  ✓  ✓

하삼각 행렬 (Lower Triangular)
```

**BERT (No Mask):**
```
Position:  0  1  2  3
    0:    ✓  ✓  ✓  ✓
    1:    ✓  ✓  ✓  ✓
    2:    ✓  ✓  ✓  ✓
    3:    ✓  ✓  ✓  ✓

전체 attention
```

### 현대 트렌드 (2024)

**Decoder-only 모델 지배:**
- GPT-3, GPT-4
- Claude (Anthropic)
- Llama (Meta)
- PaLM (Google)

**이유:**
- 생성 능력이 더 중요
- Fine-tuning으로 분류도 가능
- Scaling이 더 효과적

---

## 🔬 Lab 3: Pre-training and Fine-tuning (03_pretraining_finetuning.py)

### 목적
LLM 학습의 두 단계를 이해합니다.

### Pre-training (사전 학습)

**목적:**
```
일반적인 언어 이해 능력 획득
```

**데이터:**
- 웹 크롤링 (CommonCrawl)
- 위키백과
- 책 (Books3)
- 코드 (GitHub)
- 논문 (arXiv)

**규모:**
```
GPT-3: 300B tokens
Claude: 수조 tokens (추정)

1 token ≈ 0.75 words
300B tokens ≈ 225B words
```

**학습 방법 (GPT):**
```
입력: "The cat sat on"
예측: "the"

입력: "The cat sat on the"
예측: "mat"

다음 토큰 예측 (Next Token Prediction)
```

**비용:**
```
수천만~수억 달러
수천 개의 GPU
몇 달간 학습
```

**결과:**
```
Foundation Model (기반 모델)
- 언어 이해
- 세상 지식
- 추론 능력 (일부)
```

### Fine-tuning (미세 조정)

**목적:**
```
특정 작업에 맞게 모델 조정
```

**데이터:**
- 작업별 labeled data
- 1,000 ~ 1,000,000 examples
- 고품질 데이터 중요

**방법:**
1. **Supervised Fine-tuning (SFT)**
   ```
   Input: Question
   Output: Correct answer

   예시 학습
   ```

2. **Instruction Tuning**
   ```
   다양한 지시사항 따르기 학습
   "Translate", "Summarize", "Explain" 등
   ```

3. **Task-specific Fine-tuning**
   ```
   감성 분석, QA, NER 등
   특정 도메인 적응
   ```

**비용:**
```
수천~수만 달러
몇 시간~며칠
상대적으로 저렴
```

### RLHF (Reinforcement Learning from Human Feedback)

**ChatGPT의 비밀:**
```
Step 1: SFT
  고품질 대화 예시로 학습

Step 2: Reward Model
  인간이 선호하는 답변 학습
  A vs B 비교 → 더 좋은 것 선택

Step 3: PPO (Proximal Policy Optimization)
  Reward 최대화하는 방향으로 학습
  너무 멀리 가지 않도록 제약 (KL divergence)
```

**결과:**
- Helpful: 도움이 되는 답변
- Harmless: 해롭지 않은 답변
- Honest: 정직한 답변 (모를 땐 모른다고)

### Learning Paradigms

**Zero-shot:**
```
학습 데이터 0개
프롬프트만으로 작업 수행

예: "Translate to French: Hello"
```

**Few-shot:**
```
학습 데이터 5-10개 (프롬프트에 포함)

예:
"Translate:
Hello → Bonjour
Goodbye → Au revoir
Thank you → [model completes]"
```

**Fine-tuning:**
```
학습 데이터 1000+ 개
모델 파라미터 업데이트
```

---

## 🔬 Lab 4: Claude API 개념 (04_claude_api_simple.py)

### 목적
LLM API 사용 방법을 이해합니다 (실제 API 키 없이 개념 학습).

### API 구조

**요청 (Request):**
```python
{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1024,
  "temperature": 0.7,
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum mechanics"
    }
  ]
}
```

**응답 (Response):**
```python
{
  "content": [{"text": "Quantum mechanics is..."}],
  "usage": {
    "input_tokens": 5,
    "output_tokens": 150
  }
}
```

### 주요 파라미터

**Temperature:**
```
0.0: 완전 결정적 (가장 확률 높은 토큰)
0.7: 균형 잡힌 (기본값)
1.0: 창의적
2.0: 매우 무작위

용도:
- 0.0-0.3: 사실 기반 작업 (수학, 코드)
- 0.7-1.0: 일반 대화
- 1.0-2.0: 창의적 글쓰기
```

**Max Tokens:**
```
응답 최대 길이 제한

GPT-4: 최대 4096 output tokens
Claude 3: 최대 4096 output tokens

비용과 직결
```

### Prompt Engineering

**나쁜 예:**
```
"물리학 설명해"
```

**좋은 예:**
```
"고등학생을 위해 뉴턴의 제2법칙 (F=ma)을
실생활 예시와 함께 설명해주세요."
```

**더 나은 예:**
```
당신은 물리학 선생님입니다.

고등학생을 대상으로 뉴턴의 제2법칙을 설명해주세요.

포함할 내용:
1. 공식과 각 변수의 의미
2. 실생활 예시 2가지
3. 흔한 오개념
4. 간단한 연습 문제

쉬운 언어와 비유를 사용하세요.
```

### 물리학 응용 사례

**1. 문제 풀이:**
```
프롬프트:
"다음 물리 문제를 단계별로 풀어주세요:
20 m/s로 던진 공의 최고 높이는? (g=9.8)"

응답:
1. 주어진 값 정리
2. 적절한 공식 선택
3. 계산 과정
4. 답과 단위
```

**2. 코드 생성:**
```
프롬프트:
"발사체 운동을 시뮬레이션하는 Python 함수를
작성해주세요. 궤적을 그래프로 그리는 코드 포함."

응답:
완전한 Python 코드 + docstring + 예시
```

**3. 개념 설명:**
```
프롬프트:
"양자 터널링을 비전공자도 이해할 수 있게
비유를 사용해 설명해주세요."

응답:
직관적인 비유 + 정확한 설명
```

### 주의사항

**항상 검증 필요:**
- 수식 계산 확인
- 물리 법칙 확인
- 단위 확인
- 코드 테스트

**LLM의 한계:**
- 때때로 자신있게 틀림
- 최신 정보 부족 (학습 cutoff)
- 복잡한 수치 계산 약함
- 도구로 사용, 맹신하지 말 것

---

## 📊 주요 결과 요약

### LLM의 핵심 요소

**1. Tokenization:**
- 텍스트 → 숫자 변환
- Subword (BPE) 방식 주로 사용
- Vocabulary: 30K-100K tokens

**2. Architecture:**
- GPT: Decoder-only, 생성 특화
- BERT: Encoder-only, 이해 특화
- 현대: Decoder-only 지배

**3. Training:**
- Pre-training: 일반 언어 능력
- Fine-tuning: 작업 특화
- RLHF: 인간 선호 정렬

**4. API 사용:**
- Temperature로 창의성 조절
- Prompt engineering 중요
- 항상 검증 필요

### 물리학에서의 활용

**유용한 응용:**
- 문제 풀이 도우미
- 코드 생성
- 개념 설명
- 논문 요약
- LaTeX 작성

**주의할 점:**
- 수식 계산 검증
- 물리 법칙 확인
- 보조 도구로만 사용

---

## 💡 실전 팁

### 프로그램 실행 순서

```bash
cd week7

# 순서대로 실행
python 01_tokens_and_embeddings.py      # ~1분
python 02_gpt_bert_architectures.py    # ~1분
python 03_pretraining_finetuning.py    # ~1분
python 04_claude_api_simple.py         # ~1분

# 또는 한 번에 실행
./run.bat
```

### Prompt 작성 팁

**명확성:**
- 구체적인 요구사항
- 예시 제공
- 형식 지정

**구조화:**
- 역할 부여
- 단계별 요청
- 제약 조건 명시

**반복 개선:**
- 첫 시도에 완벽하지 않음
- 응답 보고 프롬프트 수정
- 여러 번 실험

---

## 📖 더 공부하려면

### 필수 논문

1. **"Attention Is All You Need" (2017)**
   - Transformer 원조

2. **"BERT" (2018)**
   - Bidirectional pre-training

3. **"Language Models are Few-Shot Learners" (GPT-3, 2020)**
   - Scaling laws
   - Few-shot learning

4. **"Training language models to follow instructions" (InstructGPT, 2022)**
   - RLHF 방법론

### 온라인 자료

**공식 문서:**
- OpenAI API Docs
- Anthropic Claude Docs
- HuggingFace Transformers

**강의:**
- Stanford CS224N
- DeepLearning.AI LLM courses
- Fast.ai Practical Deep Learning

**실습:**
- HuggingFace Hub
- Google Colab tutorials
- LangChain documentation

---

## 🎓 학습 점검

### 기본 개념
- [ ] Tokenization이 왜 필요한지
- [ ] GPT와 BERT의 차이
- [ ] Pre-training vs Fine-tuning
- [ ] Temperature 파라미터의 역할

### 중급 개념
- [ ] BPE 알고리즘
- [ ] Causal masking vs Full attention
- [ ] RLHF 과정
- [ ] Context window의 중요성

### 고급 개념
- [ ] Prompt engineering 전략
- [ ] Few-shot learning 원리
- [ ] LLM의 한계와 대처법
- [ ] 물리학 연구에 적용 방법

---

## ✨ 결론

**배운 것:**
- LLM은 Transformer 기반
- Tokenization → Embedding → Attention
- GPT(생성) vs BERT(이해)
- Pre-training + Fine-tuning + RLHF
- API 사용법과 Prompt Engineering

**핵심 통찰:**
```
"Language Models are Universal Interfaces"

LLM은 단순한 챗봇이 아니라,
인간과 컴퓨터 간 소통의 새로운 방식
```

**물리학자에게:**
- 연구 도우미로 활용
- 코드 생성 자동화
- 교육 자료 작성
- 하지만 항상 검증!

**다음 단계:**
- 실제 API 사용 (API 키 발급)
- LangChain, LlamaIndex 등 프레임워크
- RAG (Retrieval-Augmented Generation)
- Fine-tuning 실습

---

*"LLMs are powerful tools, but tools nonetheless. Use them wisely!"*

LLM은 물리학 연구와 교육을 혁신할 잠재력이 있지만,
여전히 인간의 전문성과 판단이 핵심입니다!
