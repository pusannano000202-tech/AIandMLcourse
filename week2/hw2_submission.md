# Week 2 과제 제출 기록

- GitHub 계정: `https://github.com/pusannano000202-tech`
- 저장소: `https://github.com/pusannano000202-tech/AIandMLcourse`
- 제출자: `202110114 김충현`
- 실행일: `2026-03-22`

## 실행한 프로그램

1. `week2/01_linear_regression_spring.py`
2. `week2/02_unsupervised_clustering.py`
3. `week2/03_data_preprocessing.py`
4. `week2/04_gradient_descent_vis.py`
5. `week2/ex/01_spring_scipy.py`
6. `week2/ex/04_optimization_scipy.py`

## 생성된 결과 파일 (이미지)

1. `week2/outputs/spring_fitting.png`
2. `week2/outputs/02_clustering.png`
3. `week2/outputs/03_preprocessing.png`
4. `week2/outputs/04_gradient_descent.png`
5. `week2/outputs/ex_01_spring_scipy.png`
6. `week2/outputs/ex_04_optimization_scipy.png`

## 실행 로그 파일

1. `week2/outputs/01_linear_regression_spring_output.txt`
2. `week2/outputs/02_unsupervised_clustering_output.txt`
3. `week2/outputs/03_data_preprocessing_output.txt`
4. `week2/outputs/04_gradient_descent_vis_output.txt`
5. `week2/outputs/ex_01_spring_scipy_output.txt`
6. `week2/outputs/ex_04_optimization_scipy_output.txt`

## 핵심 결과 요약

- TensorFlow 선형 회귀:
  - 학습식: `길이 = 2.02 * 무게 + 10.27`
  - 15kg 예측: `40.58 cm`
- SciPy curve_fit:
  - 학습식: `길이 = 1.93 * 무게 + 10.90`
  - 15kg 예측: `39.85 cm`
- Gradient Descent:
  - 시작값 `x = -4.0`에서 20 step 후 `x = -0.0461`
- SciPy minimize (BFGS):
  - 최적해 `x = 2.0000`, 최소값 `y = 1.0000`, 반복 `3회`

## Week 2 체크리스트

- [x] K-Means 군집화를 실행하고 결과를 해석했다
- [x] 데이터 정규화 전후를 비교했다
- [x] Gradient Descent 시각화를 관찰했다
- [x] TensorFlow로 선형 회귀를 구현했다
- [x] SciPy 방법과 TensorFlow를 비교했다

## 비고

- `week2` 실습 코드는 2026-03-22에 전부 재실행했으며, 결과 파일 타임스탬프가 갱신됨.
- 실행 로그 파일(`*_output.txt`)은 UTF-8로 저장하여 GitHub에서 한글이 정상 표시됨.
