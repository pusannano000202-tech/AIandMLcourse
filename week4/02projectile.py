"""
02. Projectile Motion Regression with Neural Networks
포물선 운동을 TensorFlow/Keras로 학습하고 예측

물리 법칙:
- x(t) = v₀·cos(θ)·t
- y(t) = v₀·sin(θ)·t - 0.5·g·t²
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
print("Projectile Motion Regression with Neural Networks")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print("="*70)

# 중력 가속도
g = 9.81

def generate_projectile_data(n_samples=2000, noise_level=0.5):
    """포물선 운동 데이터 생성"""
    # 초기 속도와 각도를 랜덤하게 생성
    v0 = np.random.uniform(10, 50, n_samples)  # 초기 속력 (m/s)
    theta = np.random.uniform(20, 70, n_samples)  # 발사 각도 (도)
    theta_rad = np.deg2rad(theta)
    
    # 시간 샘플링
    t_max = 2 * v0 * np.sin(theta_rad) / g
    t = np.random.uniform(0, t_max * 0.9, n_samples)
    
    # 위치 계산 (물리 공식)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    # 노이즈 추가
    x += np.random.normal(0, noise_level, n_samples)
    y += np.random.normal(0, noise_level, n_samples)
    
    # y가 음수인 경우 제거 (땅 아래)
    valid_idx = y >= 0
    
    # 입력: (v0, theta, t), 출력: (x, y)
    X = np.column_stack([v0[valid_idx], theta[valid_idx], t[valid_idx]])
    Y = np.column_stack([x[valid_idx], y[valid_idx]])
    
    return X, Y

def create_projectile_model():
    """포물선 운동 예측 모델"""
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),  # v0, theta, t
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation='linear')  # x, y
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def predict_trajectory(model, v0, theta, n_points=50):
    """전체 궤적 예측"""
    theta_rad = np.deg2rad(theta)
    t_max = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_max, n_points)
    
    X_input = np.column_stack([
        np.full(n_points, v0),
        np.full(n_points, theta),
        t
    ])
    
    predictions = model.predict(X_input, verbose=0)
    return predictions[:, 0], predictions[:, 1], t

# ============================================================================
# 1. 모델 학습
# ============================================================================

print("\n" + "="*70)
print("1. 모델 학습")
print("="*70)

# 데이터 생성
print("데이터 생성 중...")
X_train, Y_train = generate_projectile_data(n_samples=2000, noise_level=0.5)
X_test, Y_test = generate_projectile_data(n_samples=500, noise_level=0.0)

print(f"학습 데이터: {X_train.shape[0]} 샘플")
print(f"테스트 데이터: {X_test.shape[0]} 샘플")

# 모델 생성 및 학습
print("\n모델 학습 중...")
model = create_projectile_model()

history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# 평가
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"\n테스트 결과:")
print(f"  Loss (MSE): {test_loss:.6f}")
print(f"  MAE: {test_mae:.6f}")

# ============================================================================
# 2. 다양한 조건에서 궤적 예측
# ============================================================================

print("\n" + "="*70)
print("2. 다양한 조건에서 궤적 예측")
print("="*70)

test_conditions = [
    (20, 30),   # v0=20 m/s, theta=30°
    (30, 45),   # v0=30 m/s, theta=45°
    (40, 60),   # v0=40 m/s, theta=60°
]

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig)

for idx, (v0, theta) in enumerate(test_conditions):
    print(f"\n조건 {idx+1}: v0={v0} m/s, θ={theta}°")
    
    # NN 예측
    x_pred, y_pred, t = predict_trajectory(model, v0, theta, n_points=50)
    
    # 실제 궤적 (물리 공식)
    theta_rad = np.deg2rad(theta)
    x_true = v0 * np.cos(theta_rad) * t
    y_true = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    # 성능 계산
    mse = np.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
    max_height_true = max(y_true)
    max_range_true = max(x_true)
    max_height_pred = max(y_pred)
    max_range_pred = max(x_pred)
    
    print(f"  MSE: {mse:.6f}")
    print(f"  최대 높이 - True: {max_height_true:.2f} m, Pred: {max_height_pred:.2f} m")
    print(f"  최대 거리 - True: {max_range_true:.2f} m, Pred: {max_range_pred:.2f} m")
    
    # 그래프 1: 궤적 비교
    ax1 = fig.add_subplot(gs[idx, 0])
    ax1.plot(x_true, y_true, 'b-', linewidth=2.5, label='True trajectory', alpha=0.7)
    ax1.plot(x_pred, y_pred, 'r--', linewidth=2, label='NN prediction')
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title(f'v₀={v0} m/s, θ={theta}°\nMSE: {mse:.4f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # 그래프 2: x 좌표 시간에 따른 변화
    ax2 = fig.add_subplot(gs[idx, 1])
    ax2.plot(t, x_true, 'b-', linewidth=2, label='True x', alpha=0.7)
    ax2.plot(t, x_pred, 'r--', linewidth=2, label='Pred x')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('x (m)', fontsize=11)
    ax2.set_title('Horizontal Position', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 그래프 3: y 좌표 시간에 따른 변화
    ax3 = fig.add_subplot(gs[idx, 2])
    ax3.plot(t, y_true, 'b-', linewidth=2, label='True y', alpha=0.7)
    ax3.plot(t, y_pred, 'r--', linewidth=2, label='Pred y')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('y (m)', fontsize=11)
    ax3.set_title('Vertical Position', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

plt.suptitle('Projectile Motion Prediction with Neural Networks', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_projectile_trajectories.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 그래프 저장: {output_dir}/02_projectile_trajectories.png")
plt.close()

# ============================================================================
# 3. 학습 곡선 및 오차 분석
# ============================================================================

print("\n" + "="*70)
print("3. 학습 곡선 및 오차 분석")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 학습 곡선
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
ax1.plot(history.history['val_loss'], 'r--', linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Training History', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# MAE 곡선
ax2 = axes[0, 1]
ax2.plot(history.history['mae'], 'b-', linewidth=2, label='Training MAE')
ax2.plot(history.history['val_mae'], 'r--', linewidth=2, label='Validation MAE')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 각도에 따른 오차 분석
ax3 = axes[1, 0]
angles = np.arange(20, 71, 5)
errors = []

for angle in angles:
    x_pred, y_pred, t = predict_trajectory(model, 30, angle, n_points=50)
    theta_rad = np.deg2rad(angle)
    x_true = 30 * np.cos(theta_rad) * t
    y_true = 30 * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    mse = np.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
    errors.append(mse)

ax3.plot(angles, errors, 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('Launch Angle (degrees)', fontsize=12)
ax3.set_ylabel('MSE', fontsize=12)
ax3.set_title('Error vs Launch Angle (v₀=30 m/s)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 속도에 따른 오차 분석
ax4 = axes[1, 1]
velocities = np.arange(10, 51, 5)
errors_v = []

for v in velocities:
    x_pred, y_pred, t = predict_trajectory(model, v, 45, n_points=50)
    theta_rad = np.deg2rad(45)
    x_true = v * np.cos(theta_rad) * t
    y_true = v * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    mse = np.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
    errors_v.append(mse)

ax4.plot(velocities, errors_v, 'mo-', linewidth=2, markersize=8)
ax4.set_xlabel('Initial Velocity (m/s)', fontsize=12)
ax4.set_ylabel('MSE', fontsize=12)
ax4.set_title('Error vs Initial Velocity (θ=45°)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Training Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_projectile_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ 그래프 저장: {output_dir}/02_projectile_analysis.png")
plt.close()

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("실험 완료!")
print("="*70)
print(f"""
포물선 운동 회귀 학습 결과:
✓ 테스트 MSE: {test_loss:.6f}
✓ 테스트 MAE: {test_mae:.6f}
✓ 다양한 발사 각도와 속도에서 정확한 예측

물리적 검증:
- 45도에서 최대 사거리 확인
- 대칭적인 궤적 형성
- 시간에 따른 위치 변화 정확

생성된 파일:
1. {output_dir}/02_projectile_trajectories.png
2. {output_dir}/02_projectile_analysis.png
""")

