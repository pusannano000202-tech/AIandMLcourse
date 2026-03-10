# Week 5: 딥러닝 핵심 개념 및 실습

## 📚 학습 목표

이번 주차에서는 딥러닝 모델 성능을 향상시키기 위한 핵심 기법들을 배우고, TensorFlow/Keras를 사용하여 직접 구현해 봅니다. 이론적인 내용과 함께 코드를 실행하여 결과를 시각적으로 확인합니다.

**배울 내용:**
1. **Regularization**: 과적합을 막기 위한 규제 기법 (L1/L2, Dropout, Batch Normalization)
2. **Overfitting vs Underfitting**: 모델의 복잡도와 성능의 관계
3. **Data Augmentation**: 데이터 부족 문제를 해결하는 증강 기법
4. **Transfer Learning**: 사전 학습된 모델을 활용하는 전이 학습
5. **CNN 실습**: MNIST 손글씨 인식 모델 구현

---

## 🛡️ 1. Regularization (규제) (01_regularization.py)

### 개념
모델이 훈련 데이터에 너무 과도하게 맞춰져서(Overfitting) 새로운 데이터에 대한 성능이 떨어지는 것을 막기 위한 기법들입니다.

- **L1/L2 Regularization**: 가중치(Weight)의 크기에 페널티를 부여하여 모델을 단순하게 만듭니다.
  - L1: 가중치의 절대값 합을 손실 함수에 추가 (일부 가중치를 0으로 만듦 -> 희소성)
  - L2: 가중치의 제곱 합을 손실 함수에 추가 (가중치를 작게 만듦)
- **Dropout**: 학습 시 무작위로 일부 뉴런을 꺼버려서(0으로 만듦) 특정 뉴런에 의존하는 것을 방지합니다.
- **Batch Normalization**: 각 층의 입력을 정규화(평균 0, 분산 1)하여 학습을 안정화하고 속도를 높입니다.

### 결과 해석
`01_regularization.py`를 실행하면 다양한 규제 기법을 적용했을 때의 검증 손실(Validation Loss) 변화를 볼 수 있습니다.

![Regularization Result](outputs/01_regularization_plot.png)

**그림 읽기:**
- **None**: 규제가 없을 때, 훈련이 진행될수록 검증 손실이 다시 증가할 수 있습니다 (과적합).
- **Dropout/L2**: 과적합을 억제하여 검증 손실이 안정적으로 유지되거나 낮아지는 것을 확인할 수 있습니다.

---

## ⚖️ 2. Overfitting vs Underfitting (02_overfitting_underfitting.py)

### 개념
- **Underfitting (과소적합)**: 모델이 너무 단순해서 데이터의 패턴을 제대로 학습하지 못한 상태입니다.
- **Overfitting (과적합)**: 모델이 너무 복잡해서 훈련 데이터의 노이즈까지 학습해버린 상태입니다.
- **Balanced (적절함)**: 모델의 복잡도가 데이터에 적절하여 일반화 성능이 좋은 상태입니다.

### 결과 해석
`02_overfitting_underfitting.py`는 데이터 포인트가 적을 때 모델 복잡도에 따른 차이를 보여줍니다.

![Overfitting vs Underfitting](outputs/02_overfitting_underfitting.png)

**그림 읽기:**
- **Left (Predictions)**:
  - `Underfit`: 데이터의 경향을 전혀 따라가지 못함 (직선 등).
  - `Overfit`: 훈련 데이터 점들을 지나치게 구불구불하게 따라감.
  - `Balanced`: 전체적인 사인 파형(Sine wave)을 잘 근사함.
- **Right (Loss Curves)**:
  - `Overfit` 모델은 훈련 손실은 낮지만 검증 손실(Val Loss)이 매우 높게 치솟습니다.

---

## 🖼️ 3. Data Augmentation (데이터 증강) (03_data_augmentation.py)

### 개념
데이터가 부족할 때, 기존 이미지를 변형(회전, 뒤집기, 확대/축소 등)하여 데이터의 다양성을 늘리는 기법입니다. 이는 모델이 위치나 각도 변화에 강건해지도록 돕습니다.

### 코드 예시
```python
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])
```

### 결과 해석
`03_data_augmentation.py`는 하나의 이미지를 다양하게 변형한 예시를 보여줍니다.

![Data Augmentation](outputs/03_augmentation_examples.png)

---

## 🧠 4. Transfer Learning (전이 학습) (04_transfer_learning.py)

### 개념
이미 대량의 데이터(예: ImageNet)로 학습된 모델(Pre-trained Model)의 지식을 가져와서, 내가 가진 적은 데이터의 문제 해결에 활용하는 방법입니다.
- **Feature Extraction**: 사전 학습된 모델의 합성곱 층(Convolutional Base)을 고정(Freeze)하고, 분류기(Classifier)만 새로 학습합니다.
- **Fine-tuning**: 사전 학습된 모델의 일부 상위 층도 미세하게 같이 학습합니다.

### 실습 내용
`04_transfer_learning.py`에서는 `MobileNetV2` 모델을 불러와서 기본 층을 얼리고(Freeze), 새로운 분류 층을 추가하는 구조를 만듭니다. (시간 관계상 실제 학습은 생략하고 모델 구조를 확인합니다.)

---

## ✍️ 5. 실습: MNIST 손글씨 인식 (CNN) (05_mnist_cnn.py)

### 개념
**CNN (Convolutional Neural Network)**은 이미지 처리에 특화된 딥러닝 구조입니다.
- **Conv2D**: 이미지의 특징(Feature)을 추출합니다.
- **MaxPooling2D**: 이미지 크기를 줄이면서 중요한 특징만 남깁니다.
- **Flatten & Dense**: 추출된 특징을 바탕으로 최종 분류를 수행합니다.

### 모델 구조
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

### 결과 해석
`05_mnist_cnn.py`를 실행하면 CNN 모델이 MNIST 데이터를 학습하는 과정과 정확도를 확인할 수 있습니다.

![MNIST CNN Result](outputs/05_mnist_cnn_result.png)

---

## 🚀 실행 방법

각 스크립트는 `week5` 디렉토리에서 다음과 같이 실행할 수 있습니다.

```bash
# week5 디렉토리로 이동
cd week5

# 각 실습 파일 실행
uv run python 01_regularization.py
uv run python 02_overfitting_underfitting.py
uv run python 03_data_augmentation.py
uv run python 04_transfer_learning.py
uv run python 05_mnist_cnn.py
```

실행 후 `week5/outputs` 디렉토리에서 생성된 이미지 파일들을 확인해보세요!
