# TRD — 물리 데이터 Neural Network 인터랙티브 학습 앱
**Technical Requirements Document**

| 항목 | 내용 |
|------|------|
| 과목 | AI와 머신러닝 (PH2002141-033) |
| 주차 | Week 4 — 물리 데이터로 학습하기 |
| 제출자 | 김충현 |
| 제출일 | 2026-04-07 |
| 버전 | v1.0 |

---

## 1. 기술 스택

| 구분 | 기술 | 버전 |
|------|------|------|
| GUI 프레임워크 | PySide6 | 6.x |
| 딥러닝 | TensorFlow / Keras | 2.21 |
| 수치 연산 | NumPy | ≥1.24 |
| 시각화 | Matplotlib (QtAgg 백엔드) | ≥3.7 |
| 언어 | Python | ≥3.10 |
| 패키지 관리 | uv / pip | — |

---

## 2. 아키텍처 개요

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Lab1Tab  ─── Lab1Worker (QThread)
    ├── Lab2Tab  ─── Lab2Worker (QThread)
    ├── Lab3Tab  ─── Lab3Worker (QThread)
    └── Lab4Tab  ─── Lab4Worker (QThread)
```

### 스레드 통신 패턴

```
QThread (Worker)                   Qt Main Thread (UI)
─────────────────────────────────────────────────────
StreamCallback.on_epoch_end()
  → Signal.emit(epoch, loss, val)  → Tab._on_epoch()
                                      → canvas.draw()
Worker.run() 완료
  → finished.emit(model, data)     → Tab._on_done()
                                      → 최종 시각화
```

모든 `canvas.draw()` 호출은 Main Thread에서만 발생하여 스레드 안전성을 보장한다.

---

## 3. 핵심 클래스 설계

### 3-1. StreamCallback (Keras Callback)

```python
class StreamCallback(keras.callbacks.Callback):
    """학습 중 N epoch마다 loss를 Signal로 전달"""
    def __init__(self, signal, total_epochs, freq=20):
        ...
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._freq == 0:
            self._sig.emit(epoch, loss, val_loss)
```

`freq = max(1, total_epochs // 100)` — 에폭 수에 비례해 업데이트 빈도 자동 조정  
UI가 매 epoch 갱신되지 않도록 throttling.

---

### 3-2. Lab1Worker — 1D Function Approximation

```
입력: func_name, arch_name, epochs, lr
모델: Input(1) → [Dense(units, tanh)] × N → Dense(1, linear)
콜백: StreamCallback + StopCheck + ReduceLROnPlateau
출력 Signal: epoch_update(int, float, float)
완료 Signal: finished(x_test, y_true, y_pred, history)
```

**함수 목록:**
| 키 | 수식 |
|----|------|
| sin(x) | sin(x) |
| cos(x)+0.5sin(2x) | cos(x) + 0.5·sin(2x) |
| x·sin(x) | x·sin(x) |
| Extreme | sin(x)+0.5sin(2x)+0.3cos(3x)+0.2sin(5x) |

**아키텍처 목록:**
| 키 | 레이어 |
|----|--------|
| Small [32] | [32] |
| Medium [64,64] | [64, 64] |
| Large [128,128] | [128, 128] |
| XL [128,128,64] | [128, 128, 64] |
| XXL [256,256,128,64] | [256, 256, 128, 64] |

---

### 3-3. Lab2Worker — Projectile Motion

```
데이터 생성:
  v0    ~ Uniform(10, 50)  [m/s]
  theta ~ Uniform(20, 70)  [deg]
  t     ~ Uniform(0, t_max * 0.9)
  x = v0·cos(θ)·t  (+ Gaussian noise 0.5m)
  y = v0·sin(θ)·t − 0.5·g·t²  (y ≥ 0만 유효)

전처리: Min-Max Standardization (mean/std)
입력 차원: 3 (v0, theta, t)
출력 차원: 2 (x, y)

모델: Input(3) → Dense(128,relu) → Dense(128,relu) → Dense(64,relu) → Dense(2,linear)
Loss: MSE | Optimizer: Adam
```

인터랙티브 예측 역정규화:
```python
Xn = (X - X_mean) / X_std          # 정규화
out = model.predict(Xn) * Y_std + Y_mean  # 역정규화
```

---

### 3-4. Lab3Worker — Overfitting / Underfitting

세 모델을 **순차적으로** 학습 (각각 동일 데이터, 동일 에폭):

| 모델명 | 구조 | 파라미터 수 (approx) |
|--------|------|---------------------|
| Underfit | Dense(4)→Dense(1) | ~9 |
| Good Fit | Dense(32)+Dropout(0.1)→Dense(16)→Dense(1) | ~577 |
| Overfit | Dense(256)→Dense(128)→Dense(64)→Dense(32)→Dense(1) | ~51,297 |

`TaggedCB`: 각 모델마다 이름을 포함한 Signal emit  
→ UI에서 3색 실시간 그래프 구분

---

### 3-5. Lab4Worker — Pendulum Period

```
물리 공식 (ground truth 생성):
  T_small = 2π√(L/g)
  correction = 1 + θ²/16 + 11θ⁴/3072
  T_true = T_small × correction

데이터:
  L     ~ Uniform(0.5, 3.0) [m]
  theta ~ Uniform(5, 80)    [deg]
  T_noisy = T_true × (1 + N(0, 0.01))

입력 차원: 2 (L, theta0)
출력 차원: 1 (T)

모델: Input(2) → Dense(64,relu)+Dropout(0.1) → Dense(32,relu)+Dropout(0.1)
       → Dense(16,relu) → Dense(1,linear)
```

인터랙티브 예측에서 NN과 물리식을 동시 계산하여 오차 % 실시간 표시.

---

## 4. UI 컴포넌트 명세

### 공통 레이아웃 패턴

```
QVBoxLayout (Tab root)
├── QLabel (탭 제목, 파란색 bold)
├── QTextEdit (물리 수식 HTML 패널, read-only, 파란 배경)
├── QGroupBox (학습 설정 컨트롤)
│   └── QHBoxLayout [QSpinBox/QDoubleSpinBox/QComboBox + Run/Stop btn]
├── QProgressBar (0~100%)
├── FigureCanvas (matplotlib, Expanding)
├── [QGroupBox 인터랙티브 예측] — Lab 2, 4만
└── QTextEdit (결과 로그, read-only, max 80~90px)
```

### 컨트롤 위젯 목록

| Lab | 파라미터 | 위젯 | 범위 |
|-----|----------|------|------|
| 1 | 함수 선택 | QComboBox | 4가지 |
| 1 | 아키텍처 | QComboBox | 5가지 |
| 1 | 에폭 | QSpinBox | 100~10,000 |
| 1 | LR | QDoubleSpinBox | 1e-4~0.1 |
| 2 | 샘플 수 | QSpinBox | 500~10,000 |
| 2 | 에폭 | QSpinBox | 100~5,000 |
| 2 | LR | QDoubleSpinBox | 1e-4~0.01 |
| 2 | v₀ (예측) | QDoubleSpinBox | 10~50 |
| 2 | θ (예측) | QDoubleSpinBox | 20~70 |
| 2 | t (예측) | QDoubleSpinBox | 0.1~6 |
| 3 | 학습 데이터 수 | QSpinBox | 30~500 |
| 3 | 에폭 | QSpinBox | 100~3,000 |
| 3 | 노이즈 | QDoubleSpinBox | 0.05~1.0 |
| 4 | 샘플 수 | QSpinBox | 500~5,000 |
| 4 | 에폭 | QSpinBox | 100~5,000 |
| 4 | LR | QDoubleSpinBox | 1e-4~0.01 |
| 4 | L (예측) | QDoubleSpinBox | 0.5~3.0 |
| 4 | θ₀ (예측) | QDoubleSpinBox | 5~80 |

---

## 5. 시각화 명세

| Lab | 학습 중 | 완료 후 |
|-----|---------|---------|
| 1 | 실시간 Loss (1×1) | 함수근사/Loss/절대오차 (1×3) |
| 2 | Train+Val Loss (1×1) | x예측scatter / y예측scatter / Loss (1×3) |
| 3 | 3색 Val Loss (1×1) | 모델비교 / Train-Val Loss / 성능표 (1×3) |
| 4 | Train+Val Loss (1×1) | 예측scatter / 길이-각도 MAPE / Loss (1×3) |

모든 캔버스: `Figure(figsize=(12~13, 4.5))`, `tight_layout=True`

---

## 6. 학습 중단 메커니즘

```python
class StopCheck(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if outer_self._stop:
            self.model.stop_training = True
```

`worker.stop()` 호출 → `_stop = True` → 다음 epoch 종료 시 Keras가 루프 탈출  
최대 1 epoch 지연 후 안전 종료.

---

## 7. 창 종료 안전 처리

```python
def closeEvent(self, event):
    for i in range(tabs.count()):
        tab = tabs.widget(i)
        if hasattr(tab, "_worker") and tab._worker and tab._worker.isRunning():
            tab._worker.stop()
            tab._worker.wait(2000)   # 최대 2초 대기
    event.accept()
```

---

## 8. 파일 구조

```
week4/
├── hw4_pyside6_app.py     ← 메인 앱 (본 제출물)
├── PRD.md                 ← 제품 요구사항 문서
├── TRD.md                 ← 기술 요구사항 문서 (이 파일)
├── 01perfect1d.py         ← 원본 Lab 1 (콘솔 실행)
├── 02projectile.py        ← 원본 Lab 2
├── 03overfitting.py       ← 원본 Lab 3
├── 04pendulum.py          ← 원본 Lab 4
└── outputs/               ← 원본 스크립트 출력 이미지
```

---

## 9. 실행 방법

```bash
# PySide6 설치 (최초 1회)
.venv/Scripts/pip install pyside6

# 앱 실행
.venv/Scripts/python.exe week4/hw4_pyside6_app.py
```

---

## 10. 알려진 제약

| 제약 | 내용 |
|------|------|
| Lab 3 소요 시간 | 3개 모델 × 에폭 → 기본 설정 약 30~60초 |
| GPU 지원 | TF가 자동 감지. CPU만 있어도 동작 |
| 모델 비저장 | 학습 완료 모델은 앱 종료 시 소멸 |
| 한글 폰트 | Malgun Gothic 없으면 영문 폰트로 fallback (경고 없음) |
| TF 초기화 | 첫 import에 약 2~3초 소요 (정상) |
