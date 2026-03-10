import numpy as np
import matplotlib.pyplot as plt
import os

# 0. 환경 설정
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=== 비지도 학습 (Unsupervised Learning) 예제: 군집화 (Clustering) ===")

# 1. 데이터 생성 (Data Generation)
# 정답(Label)이 없는 데이터를 만듭니다.
# 여기서는 가상의 고객 데이터(예: 구매 금액, 방문 횟수)라고 가정해봅시다.
np.random.seed(42)

# 3개의 그룹(Cluster)을 가진 데이터를 임의로 생성합니다.
# 그룹 1: (2, 2) 근처
group1 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
# 그룹 2: (8, 3) 근처
group2 = np.random.normal(loc=[8, 3], scale=0.5, size=(30, 2))
# 그룹 3: (5, 8) 근처
group3 = np.random.normal(loc=[5, 8], scale=0.5, size=(30, 2))

# 모든 데이터를 하나로 합칩니다. (AI에게는 이 데이터만 줍니다. 누가 어느 그룹인지 안 알려줌!)
X = np.vstack([group1, group2, group3])

print(f"데이터 개수: {len(X)}개")
print("데이터 예시 (앞 5개):")
print(X[:5])

# 2. K-Means 알고리즘 구현 (간단 버전)
# 목표: 데이터들이 뭉쳐있는 3개의 중심점(Center)을 찾아라!

# (1) 초기화: 랜덤한 점 3개를 중심점으로 찍습니다.
k = 3
centers = X[np.random.choice(len(X), k, replace=False)]

print("\n[초기 중심점]")
print(centers)

# (2) 반복 학습: 중심점을 조금씩 이동시킵니다.
for i in range(10): # 10번 반복
    # 각 데이터에서 가장 가까운 중심점을 찾습니다.
    # 거리 계산: (x1-x2)^2 + (y1-y2)^2
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    closest_cluster = np.argmin(distances, axis=0)
    
    # 새로운 중심점 계산: 각 그룹에 속한 점들의 평균 위치로 이동
    new_centers = np.array([X[closest_cluster == j].mean(axis=0) for j in range(k)])
    
    # 중심점이 더 이상 안 움직이면 멈춤
    if np.all(centers == new_centers):
        break
    centers = new_centers

print("\n[학습 완료된 중심점]")
print(centers)

# 3. 시각화 (Visualization)
plt.figure(figsize=(8, 6))

# 각 그룹별로 다른 색으로 점 찍기
colors = ['red', 'green', 'blue']
for j in range(k):
    cluster_data = X[closest_cluster == j]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[j], label=f'Group {j+1}', alpha=0.6)

# 중심점 표시 (별 모양)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=300, label='Centroids')

plt.title('Unsupervised Learning: K-Means Clustering')
plt.xlabel('Feature 1 (e.g., Purchase Amount)')
plt.ylabel('Feature 2 (e.g., Visit Count)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_dir, '02_clustering.png')
plt.savefig(save_path)
print(f"\n그래프가 저장되었습니다: {save_path}")
print("설명: 색깔은 AI가 스스로 분류한 그룹이고, 검은 별은 각 그룹의 중심입니다.")
