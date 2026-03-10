"""
03. Overfitting vs Underfitting Demonstration
과적합과 과소적합을 TensorFlow/Keras로 시연

개념:
- Underfitting: 모델이 너무 단순하여 패턴을 학습하지 못함
- Good Fit: 적절한 복잡도로 패턴을 잘 학습
- Overfitting: 모델이 너무 복잡하여 노이즈까지 학습
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow import keras
import os

# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Overfitting vs Underfitting Demonstration")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print("="*70)

def true_function(x):
    """실제 함수: y = sin(2x) + 0.5x"""
    return np.sin(2 * x) + 0.5 * x

def generate_data(n_train=100, n_val=50, n_test=200, noise_level=0.3):
    """데이터 생성"""
    # 학습 데이터
    x_train = np.random.uniform(-2, 2, n_train).reshape(-1, 1)
    y_train = true_function(x_train) + np.random.normal(0, noise_level, (n_train, 1))
    
    # 검증 데이터
    x_val = np.random.uniform(-2, 2, n_val).reshape(-1, 1)
    y_val = true_function(x_val) + np.random.normal(0, noise_level, (n_val, 1))
    
    # 테스트 데이터 (노이즈 없음, 촘촘하게)
    x_test = np.linspace(-2, 2, n_test).reshape(-1, 1)
    y_test = true_function(x_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def create_underfit_model():
    """과소적합 모델: 너무 단순"""
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_good_model():
    """적절한 모델"""
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_overfit_model():
    """과적합 모델: 너무 복잡"""
    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ============================================================================
# 1. 데이터 생성 및 모델 학습
# ============================================================================

print("\n" + "="*70)
print("1. 데이터 생성 및 모델 학습")
print("="*70)

# 데이터 생성
x_train, y_train, x_val, y_val, x_test, y_test = generate_data(
    n_train=100, n_val=50, n_test=200, noise_level=0.3
)

print(f"학습 데이터: {x_train.shape[0]} 샘플")
print(f"검증 데이터: {x_val.shape[0]} 샘플")
print(f"테스트 데이터: {x_test.shape[0]} 샘플")

# 모델 생성 및 학습
models = {
    'underfit': create_underfit_model(),
    'good': create_good_model(),
    'overfit': create_overfit_model()
}

histories = {}
predictions = {}

epochs = 200

for name, model in models.items():
    print(f"\n{name} 모델 학습 중...")
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=16,
        verbose=0
    )
    
    histories[name] = history
    predictions[name] = model.predict(x_test, verbose=0)
    
    # 평가
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")

# ============================================================================
# 2. 모델 예측 비교
# ============================================================================

print("\n" + "="*70)
print("2. 모델 예측 비교")
print("="*70)

fig = plt.figure(figsize=(18, 6))

colors = {'underfit': 'blue', 'good': 'green', 'overfit': 'red'}
titles = {
    'underfit': 'Underfitting (Too Simple)',
    'good': 'Good Fit (Just Right)',
    'overfit': 'Overfitting (Too Complex)'
}

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = fig.add_subplot(1, 3, idx+1)
    
    # 학습 데이터
    ax.scatter(x_train, y_train, alpha=0.5, s=30, label='Training data', color='gray')
    
    # 실제 함수
    ax.plot(x_test, y_test, 'k-', linewidth=2.5, label='True function', alpha=0.7)
    
    # 예측
    ax.plot(x_test, y_pred, color=colors[name], linestyle='--', linewidth=2, 
            label=f'{name.capitalize()} prediction')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(titles[name], fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Model Predictions Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_overfitting_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_overfitting_comparison.png")
plt.close()

# ============================================================================
# 3. 학습 곡선 분석
# ============================================================================

print("\n" + "="*70)
print("3. 학습 곡선 분석")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, name in enumerate(['underfit', 'good', 'overfit']):
    ax = axes[idx]
    history = histories[name]
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    ax.plot(epochs_range, history.history['loss'], 
            color=colors[name], linestyle='-', linewidth=2, label='Train loss')
    ax.plot(epochs_range, history.history['val_loss'], 
            color=colors[name], linestyle='--', linewidth=2, label='Val loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(titles[name], fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.suptitle('Training vs Validation Loss', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_training_curves.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_training_curves.png")
plt.close()

# ============================================================================
# 4. 상세 분석
# ============================================================================

print("\n" + "="*70)
print("4. 상세 분석")
print("="*70)

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig)

# 모든 모델을 한 그래프에
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(x_train, y_train, alpha=0.5, s=30, label='Training data', color='gray')
ax1.plot(x_test, y_test, 'k-', linewidth=2.5, label='True function', alpha=0.7)

for name, y_pred in predictions.items():
    ax1.plot(x_test, y_pred, color=colors[name], linestyle='--', linewidth=2, 
            label=f'{name.capitalize()}')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('All Models Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 오차 분석
for idx, name in enumerate(['underfit', 'good', 'overfit']):
    ax = fig.add_subplot(gs[1, idx])
    
    y_pred = predictions[name]
    error = np.abs(y_pred - y_test)
    
    ax.plot(x_test, error, color=colors[name], linewidth=2)
    ax.fill_between(x_test.flatten(), 0, error.flatten(), 
                     color=colors[name], alpha=0.3)
    
    mean_error = np.mean(error)
    max_error = np.max(error)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title(f'{titles[name]}\nMean: {mean_error:.4f}, Max: {max_error:.4f}', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_error_analysis.png")
plt.close()

# ============================================================================
# 5. 최종 비교 테이블
# ============================================================================

print("\n" + "="*70)
print("최종 성능 비교")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# 데이터 수집
table_data = [['Model', 'Final Train Loss', 'Final Val Loss', 'Test MSE', 'Test MAE']]

for name in ['underfit', 'good', 'overfit']:
    history = histories[name]
    final_train = history.history['loss'][-1]
    final_val = history.history['val_loss'][-1]
    
    model = models[name]
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    
    table_data.append([
        titles[name],
        f'{final_train:.6f}',
        f'{final_val:.6f}',
        f'{test_loss:.6f}',
        f'{test_mae:.6f}'
    ])

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더 스타일
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 행 색상
colors_row = {'Underfitting (Too Simple)': '#BBDEFB', 
              'Good Fit (Just Right)': '#C8E6C9', 
              'Overfitting (Too Complex)': '#FFCDD2'}

for i, row in enumerate(table_data[1:], 1):
    model_name = row[0]
    for j in range(5):
        table[(i, j)].set_facecolor(colors_row.get(model_name, 'white'))

plt.title('Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig(f'{output_dir}/03_comparison_table.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/03_comparison_table.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print("""
과적합 vs 과소적합 실험 결과:

Underfitting (과소적합):
- 너무 단순한 모델 (4개 뉴런)
- Train/Val loss 모두 높음
- 데이터의 패턴을 제대로 학습하지 못함

Good Fit (적절한 학습):
- 적절한 복잡도 (32-16 뉴런 + Dropout)
- Train/Val loss가 비슷하게 낮음
- 일반화 성능이 가장 좋음

Overfitting (과적합):
- 너무 복잡한 모델 (256-128-64-32 뉴런)
- Train loss는 낮지만 Val loss는 높음
- 학습 데이터의 노이즈까지 학습

생성된 파일:
1. outputs/03_overfitting_comparison.png
2. outputs/03_training_curves.png
3. outputs/03_error_analysis.png
4. outputs/03_comparison_table.png
""")

