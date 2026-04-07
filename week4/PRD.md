# PRD — 물리 데이터 Neural Network 인터랙티브 학습 앱
**Product Requirements Document**

| 항목 | 내용 |
|------|------|
| 과목 | AI와 머신러닝 (PH2002141-033) |
| 주차 | Week 4 — 물리 데이터로 학습하기 |
| 제출자 | 김충현 |
| 제출일 | 2026-04-07 |
| 버전 | v1.0 |

---

## 1. 배경 및 목적

Week 4 수업에서 학습한 **TensorFlow/Keras 기반 물리 데이터 학습** 개념 (1D 함수 근사, 포물선 운동, 과적합/과소적합, 진자 주기 예측)을 **PySide6 GUI 앱**으로 통합하여, 파라미터 조작·실시간 학습 시각화·물리식과 NN 예측 비교를 한 화면에서 체험할 수 있도록 한다.

### Week 3 앱과의 차별점 (Superpowers)

| 기능 | Week 3 (Numpy) | Week 4 (TF/Keras) |
|------|---------------|-------------------|
| 학습 엔진 | Pure NumPy | TensorFlow 2.21 |
| 스레드 | QApplication.processEvents | QThread 완전 분리 |
| 실시간 Loss | 없음 | 매 N epoch 스트리밍 |
| 학습 중단 | 없음 | Stop 버튼 (즉시 중단) |
| 물리식 패널 | 없음 | 각 탭 상단 수식 표시 |
| 인터랙티브 예측 | 없음 | 학습 후 슬라이더로 실시간 예측 |

---

## 2. 사용자 및 실행 환경

| 항목 | 내용 |
|------|------|
| 주 사용자 | 물리 + AI를 함께 배우는 수강생 |
| 실행 환경 | Windows 11, Python 3.10+, PySide6 6.x, TensorFlow 2.21, NumPy, Matplotlib |
| 실행 명령 | `.venv/Scripts/python.exe week4/hw4_pyside6_app.py` |

---

## 3. 기능 요구사항

### FR-01: 탭 기반 멀티 Lab UI

앱은 4개 탭으로 구성되며 각 탭은 독립적으로 학습·시각화·예측한다.

| 탭 | 제목 | 대응 파일 |
|----|------|----------|
| 1 | 1D 함수 근사 | `01perfect1d.py` |
| 2 | 포물선 운동 회귀 | `02projectile.py` |
| 3 | 과적합/과소적합 | `03overfitting.py` |
| 4 | 진자 주기 예측 | `04pendulum.py` |

---

### FR-02: Lab 1 — 1D 함수 근사

**조절 파라미터:**
- 근사할 함수: sin(x) / cos(x)+0.5sin(2x) / x·sin(x) / Extreme (multi-freq)
- 네트워크 아키텍처: Small[32] / Medium[64,64] / Large[128,128] / XL / XXL
- 에폭 수 (100~10,000)
- 학습률 (0.0001~0.1)

**출력:**
- 실시간 Loss 스트리밍 그래프 (학습 중)
- 학습 완료 후: 함수 근사 / Loss 곡선 / 절대 오차 3-panel 시각화
- MSE, Max Error 수치 출력

**수용 기준:** Large 이상 아키텍처로 sin(x) MSE < 0.0001

---

### FR-03: Lab 2 — 포물선 운동 회귀

**물리 배경:** x(t) = v₀cos(θ)t, y(t) = v₀sin(θ)t − ½gt²

**조절 파라미터:**
- 샘플 수 (500~10,000)
- 에폭 수
- 학습률

**출력:**
- 실시간 Train/Val Loss 비교 (학습 중)
- 완료 후: x 예측 scatter / y 예측 scatter / Loss 곡선
- **인터랙티브 예측 패널:** v₀, θ, t 입력 → x, y 실시간 출력

**수용 기준:** Test MSE < 5 m² (노이즈 0.5m 수준)

---

### FR-04: Lab 3 — 과적합/과소적합

**3개 모델 동시 학습:**
- Underfit: Dense(4) + Dense(1)
- Good Fit: Dense(32)+Dropout(0.1)+Dense(16)+Dense(1)
- Overfit: Dense(256)+Dense(128)+Dense(64)+Dense(32)+Dense(1)

**조절 파라미터:**
- 학습 데이터 수 (30~500)
- 에폭 수
- 노이즈 수준

**출력:**
- 실시간 Val Loss 3-line 비교 (학습 중)
- 완료 후: 함수 근사 비교 / Train vs Val Loss / 성능 비교 테이블

**수용 기준:**
- Underfit: Train/Val 모두 높음 (확인 가능)
- Good Fit: Train ≈ Val (간극 최소)
- Overfit: Train << Val (과적합 증거)

---

### FR-05: Lab 4 — 진자 주기 예측

**물리 배경:**
- 소각도: T = 2π√(L/g)
- 대각도 보정: T ≈ T₀[1 + θ²/16 + 11θ⁴/3072 + ...]

**조절 파라미터:**
- 샘플 수 (500~5,000)
- 에폭 수
- 학습률

**출력:**
- 실시간 Loss 스트리밍
- 완료 후: T 예측 scatter / 길이·각도별 MAPE / Loss 곡선
- **인터랙티브 예측 패널:** L, θ₀ → NN 예측 T vs 물리식 T, 오차 % 실시간 표시

**수용 기준:** MAPE < 1%

---

## 4. Superpower 기능 요구사항

| ID | 기능 | 설명 |
|----|------|------|
| SP-01 | 실시간 Loss 스트리밍 | QThread→Signal으로 매 N epoch 실시간 그래프 갱신 |
| SP-02 | 학습 중단 버튼 | Keras EarlyStopping 커스텀 CB로 즉시 중단 |
| SP-03 | 인터랙티브 예측 | 학습 후 슬라이더로 입력 변경 → 실시간 예측 출력 |
| SP-04 | NN vs 물리식 비교 | Lab 4에서 NN 결과와 해석적 물리 공식을 동시 출력 |
| SP-05 | 물리 수식 패널 | 각 탭 상단에 관련 물리 방정식 HTML 표시 |
| SP-06 | 진행률 바 | 학습 진행률 0~100% 실시간 업데이트 |

---

## 5. 비기능 요구사항

| ID | 요구사항 |
|----|----------|
| NFR-01 | UI 스레드와 학습 스레드 완전 분리 (QThread) — 학습 중 UI 응답 보장 |
| NFR-02 | 창 종료 시 실행 중 Worker 안전 종료 (closeEvent) |
| NFR-03 | 기본 창 크기 1300×880, 리사이즈 지원 |
| NFR-04 | 한글 폰트 자동 감지 (Malgun Gothic fallback) |
| NFR-05 | TF 로그 억제 (TF_CPP_MIN_LOG_LEVEL=2) |

---

## 6. 범위 외 (Out of Scope)

- GPU 가속 설정 UI (자동 감지에 맡김)
- 모델 저장/불러오기 기능
- MNIST 등 실제 데이터셋
- 하이퍼파라미터 자동 최적화 (AutoML)

---

## 7. 성공 기준

1. 4개 탭 모두 오류 없이 학습·시각화 완료
2. Lab 1: Large 아키텍처로 sin(x) MSE < 0.0001
3. Lab 3: Overfit 모델의 Val Loss > Train Loss 시각적으로 확인
4. Lab 4: MAPE < 1%, 인터랙티브 예측 오차 < 2%
5. 학습 중 UI가 멈추지 않음 (QThread 분리 확인)
