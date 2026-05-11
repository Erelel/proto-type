# 실제 데이터 K=4 검증

이 디렉토리는 `merged.csv` 실데이터를 사용해 `K=4` 군집 결과를 검증하는 아티팩트를 담고 있습니다. 목적은 `CH Index`가 계속 상승하는 현실을 숨기지 않으면서도, `K=4`가 서비스 관점에서 충분히 해석 가능하고 경계가 명확하다는 점을 함께 보여주는 것입니다.

## 구성 파일

- `real_data_validation.py`: 실제 데이터 전처리, `K=2..8` CH Index 계산, `K=4` 군집 시각화, 의사결정나무 경계 시각화
- `real_data_validation.png`: 1x3 검증 Figure
- `real_data_ch_scores.csv`: `K=2..8`의 CH Index 결과
- `real_data_summary.txt`: 실행 요약

## 실행 방법

```bash
python test/another_valid/real_data_validation.py
```

## Figure 구성

1. `실제 데이터 CH Index 변화`
   - `K=2..8`에 대해 Calinski-Harabasz Index를 계산합니다.
   - 실데이터에서는 CH가 `K=4`에서 멈추지 않고 더 증가할 수 있음을 그대로 보여줍니다.

2. `실제 데이터 K=4 군집 결과`
   - `clustering_pipeline.preprocessing`의 실제 파생 변수 `focus_intensity`, `switch_frequency` 위에 KMeans 결과를 그대로 표시합니다.
   - 중심점은 스케일된 공간의 centroid를 원래 파생 특성 축으로 역변환해 겹쳐 그립니다.

3. `의사결정나무 결정 경계`
   - `K=4` KMeans 레이블을 target으로 하여 `DecisionTreeClassifier(max_depth=3)`를 학습합니다.
   - Test F1-Score는 군집 경계를 단순 규칙으로 얼마나 잘 설명할 수 있는지 보여주는 보조 지표입니다.

## 해석 포인트

- CH Index만 보면 `K=4`가 유일한 수학적 최적점이라고 말하기 어렵습니다.
- 그러나 `K=4`는 실제 서비스에서 해석 가능한 구획을 제공하고, 얕은 의사결정나무로도 높은 F1을 얻을 수 있습니다.
- 따라서 이 검증은 `K=4`를 절대적 최적해라고 주장하기보다, `지표의 한계를 인정한 상태에서 실무적으로 방어 가능한 선택`이라는 점을 보여주는 데 초점을 둡니다.
