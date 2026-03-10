import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 데이터 전처리 (Data Preprocessing) 예제: 정규화 (Normalization) ===")

# 1. 데이터 생성 (Data Generation)
# 단위가 매우 다른 두 가지 데이터를 만듭니다.
# 예: 키(cm)와 몸무게(kg) -> 키는 160~190, 몸무게는 50~100 정도지만,
# 더 극단적인 예로 '연봉(원)'과 '나이(세)'를 들어봅시다.
# 연봉: 30,000,000 ~ 100,000,000 (단위가 큼)
# 나이: 20 ~ 60 (단위가 작음)

np.random.seed(42)
n_samples = 50

# 연봉 (3천만원 ~ 1억원)
salary = np.random.uniform(30000000, 100000000, n_samples)
# 나이 (20세 ~ 60세)
age = np.random.uniform(20, 60, n_samples)

# 2. 정규화 (Normalization) - Min-Max Scaling
# 모든 데이터를 0과 1 사이로 압축합니다.
# 공식: (값 - 최소값) / (최대값 - 최소값)

salary_min, salary_max = salary.min(), salary.max()
age_min, age_max = age.min(), age.max()

salary_normalized = (salary - salary_min) / (salary_max - salary_min)
age_normalized = (age - age_min) / (age_max - age_min)

print("\n[데이터 비교]")
print(f"연봉(원본): 최소 {salary_min:,.0f}, 최대 {salary_max:,.0f}")
print(f"연봉(변환): 최소 {salary_normalized.min():.1f}, 최대 {salary_normalized.max():.1f}")
print("-" * 30)
print(f"나이(원본): 최소 {age_min:.0f}, 최대 {age_max:.0f}")
print(f"나이(변환): 최소 {age_normalized.min():.1f}, 최대 {age_normalized.max():.1f}")

# 3. 시각화 (Visualization)
plt.figure(figsize=(12, 5))

# (1) 원본 데이터 그래프
plt.subplot(1, 2, 1)
plt.scatter(age, salary, color='orange')
plt.title('Before Scaling (Raw Data)')
plt.xlabel('Age (Years)')
plt.ylabel('Salary (Won)')
plt.grid(True)
# 축의 비율을 똑같이 맞추면 데이터가 얼마나 납작한지 보입니다.
# 하지만 여기서는 값의 범위가 너무 달라서 그냥 둡니다.

# (2) 정규화된 데이터 그래프
plt.subplot(1, 2, 2)
plt.scatter(age_normalized, salary_normalized, color='blue')
plt.title('After Scaling (Normalized)')
plt.xlabel('Age (0~1)')
plt.ylabel('Salary (0~1)')
plt.grid(True)
# 정사각형 비율로 설정하여 데이터가 고르게 퍼진 것을 확인
plt.axis('square') 

save_path = os.path.join(output_dir, '03_preprocessing.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
print("설명: 왼쪽은 단위가 달라서 다루기 힘들지만, 오른쪽은 0~1 사이로 예쁘게 정렬되었습니다.")
