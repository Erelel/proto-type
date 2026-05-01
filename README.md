# Clustering Pipeline + Stateless Inference

모바일 앱 사용 로그에서 파생 변수(derived features)를 만들고, KMeans 클러스터링 결과를 시각화/평가하는 파이프라인과
Firebase Cloud Functions 환경에서 **모델 파일 없이 파라미터만으로 예측**하는 stateless 추론 코드를 함께 정리합니다.

- 주요 파생 변수
   - `focus_intensity`
   - `switch_frequency`
- 비교 대상
   - `MinMaxScaler + KMeans`

## 0. 팀 전달용 핵심 요약

### 0.1 반드시 전달해야 하는 코드

- 오프라인 재학습 파이프라인
   - `clustering_pipeline/preprocessing.py`: 입력 검증, log1p 판단, 파생 변수 생성
   - `clustering_pipeline/modeling.py`: KMeans 학습 및 지표 계산
   - `clustering_pipeline/main.py`: 단일 k 실행 엔트리
   - `clustering_pipeline/k_range_evaluation.py`: k 범위 평가
- 온라인 추론/매핑 함수 (모델 파일 없이 사용)
   - `modeling_test.py`: `predict_cluster()`, `align_clusters()`

### 0.2 파이프라인에서 꼭 필요한 부분

- Day 0 실시간 추론: 저장된 `scaler_params`, `centroids`, `log_transform_applied`만 필요
- Day 7+ 재학습: 새 KMeans 결과를 `align_clusters()`로 매핑해 군집 의미 보존
- 결과 저장: 아래 파라미터는 반드시 저장/배포
   - `scaler_params` (각 파생 변수 min/max)
   - `centroids` (KMeans 중심점)
   - `log_transform_applied` (bool)

## 1. 폴더 구성

- `main.py`: 단일 k에 대해 전처리, 클러스터링, 시각화, 메트릭 저장까지 한 번에 실행
- `k_range_evaluation.py`: k 범위를 바꿔가며 성능(실루엣, DBI, inertia) 비교
- `preprocessing.py`: 입력 검증, 결측 제거, 왜도(skewness) 기반 log1p 적용 판단, 파생 변수 생성
- `modeling.py`: KMeans 학습 및 지표 계산
- `visualization.py`: 히스토그램/산점도/비교 시각화 생성
- `constants.py`: 공통 상수 정의
- `result/`: 실행 결과 이미지/CSV 저장 폴더(기본)

## 2. Stateless 추론/매핑 사용법 (Cloud Functions)

### 2.1 저장 파라미터 스키마

```json
{
   "scaler_params": {
      "focus_intensity": {"min": 0.0, "max": 4.5},
      "switch_frequency": {"min": 0.0, "max": 3.0}
   },
   "centroids": [[0.1, 0.2], [0.3, 0.7], [0.8, 0.4], [0.9, 0.9]],
   "log_transform_applied": true
}
```

### 2.2 추론/매핑 함수 위치

- `modeling_test.py`
   - `predict_cluster(raw_data, saved_params)`
   - `align_clusters(old_centroids, new_centroids)`

### 2.3 동작 요약

- `predict_cluster()`
   - 파생 변수 생성 -> 필요 시 log1p -> 수동 MinMax -> 유클리디안 거리로 군집 결정
- `align_clusters()`
   - `cdist`로 비용 행렬 계산 -> 헝가리안 알고리즘으로 1:1 매핑

### 2.4 Mock 테스트

`modeling_test.py` 하단의 `__main__` 블록에 Cloud Function 호출을 가정한 Mock 테스트가 포함되어 있습니다.
실서비스에서는 `raw_data`, `saved_params`를 JSON으로 받아 그대로 전달하면 됩니다.

## 3. 입력 데이터 요구사항

CSV에 아래 컬럼이 반드시 있어야 합니다.

- `foreground_app_duration_sum`
- `foreground_app_switch_per_hour`
- `concentration_ratio`

`preprocessing.py`에서 위 3개 컬럼만 사용하며, 결측값(`NaN`)이 있는 행은 제거합니다.

## 4. 파이프라인 동작 요약

1. CSV 로드 및 필수 컬럼 검증
2. 결측값 제거
3. source 변수 2개의 왜도 계산
   - `foreground_app_duration_sum`
   - `foreground_app_switch_per_hour`
4. 두 변수 모두 강한 우측 치우침(`skew > 1.0`)이고 음수가 없으면 `log1p` 적용
5. 파생 변수 생성
   - `focus_intensity = duration_base * concentration_ratio`
   - `switch_frequency = switch_base * (1 - concentration_ratio)`
6. MinMaxScaler 적용
7. KMeans 학습 및 지표 계산
   - Silhouette (높을수록 좋음)
   - Davies-Bouldin Index (낮을수록 좋음)
8. 시각화 및 결과 저장

출력 경로는 실행 위치와 무관하게 프로젝트 루트의 `result/` 하위로 정규화됩니다.

## 5. 설치

Python 3.9+ 권장

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## 6. 실행 방법

### 6.1 단일 k 실행 (`main.py`)

프로젝트 루트에서 실행 예시:

```bash
python clustering_pipeline/main.py --csv baseline_train.csv --k 4 --out-dir result/0402_image
```

옵션:

- `--csv`: 입력 CSV 경로 (기본: `baseline_train.csv`)
- `--collected-csv`: 선택 입력. `--csv`로 학습한 KMeans를 동결(frozen)한 채 수집 데이터에 적용하여 평가
- `--k`: 클러스터 개수 (기본: `4`)
- `--out-dir`: 출력 폴더 (기본: 프로젝트 루트 `result/0402_image`)

생성 파일:

- `derived_feature_histograms.png`
- `derived_feature_histograms_minmax.png`
- `kmeans_focus_switch_clusters.png`
- `kmeans_focus_switch_clusters_collected.png` (`--collected-csv` 사용 시)
- `kmeans_metrics.csv`
- `kmeans_cluster_distribution.csv`

`--collected-csv`를 사용하면 `kmeans_metrics.csv`와 `kmeans_cluster_distribution.csv`에
학습 데이터(`baseline_train`)와 수집 데이터(`collected_eval`) 결과가 함께 저장됩니다.

### 6.2 k 범위 평가 (`k_range_evaluation.py`)

```bash
python clustering_pipeline/k_range_evaluation.py --csv baseline_train.csv --k-min 3 --k-max 8 --csv-dir result/0402_csv --image-dir result/0402_image
```

옵션:

- `--k-min`: 최소 k (2 이상)
- `--k-max`: 최대 k (`k-min` 이상)
- `--csv-dir`: k 범위 메트릭 CSV 출력 폴더
- `--image-dir`: 엘보우 유사 진단 플롯 출력 폴더

생성 파일 예시:

- `result/0402_csv/kmeans_k_3_8_metrics.csv`
- `result/0402_image/kmeans_elbow_like_k_3_8.png`

## 7. (옵션) 현재 결과 이미지 예시

선택 사항입니다. 시각화 결과가 필요할 때만 확인하세요.
아래 이미지는 `result/` 하위에 저장된 결과를 그대로 참조합니다.

![Derived feature histograms](result/0402_image/derived_feature_histograms.png)

![Derived feature histograms after MinMax](result/0402_image/derived_feature_histograms_minmax.png)

![KMeans clusters on derived features](result/0402_image/kmeans_focus_switch_clusters.png)

## 8. 해석 팁

- 실루엣 점수는 군집 분리/응집이 좋을수록 상승합니다.
- DBI는 군집 간 분리가 좋고 군집 내부 응집이 높을수록 감소합니다.
- 두 지표를 함께 보고 k를 선택하는 것이 안정적입니다.
- MinMax 적용 후 분포 압축 정도를 히스토그램으로 함께 확인하면 설명력 확보에 도움이 됩니다.
