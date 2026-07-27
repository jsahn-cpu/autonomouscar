# SAM3 오프라인 자동 라벨링 + 경량 실시간 세그멘테이션 학습

`autodrive_perception/core/lane_detector.py`의 `mask_white()`(adaptiveThreshold
기반)를 대체하기 위한 준비 단계. SAM3는 라이브 파이프라인이 아니라 **오프라인
자동 라벨링 도구**로만 쓰고, 그 라벨로 경량 실시간 세그멘테이션 모델을
학습시킨다. 후보 아키텍처는 두 개 (`ml/training/model.py`의 TinyUNet과
`ml/training/pidnet.py`의 PIDNetLite) -- 같은 데이터/파이프라인으로 둘 다
학습시켜서 `compare_with_lane_detector.py`로 비교한 뒤 하나만 고른다. 학습된
모델을 실제 ROS 노드에 통합하는 것은 이 계획의 범위 밖이다 (다음 단계).

이 디렉터리는 **colcon이 인식하는 `src/` 밖**에 있는 순수 파이썬 트리다 (colcon
빌드/`ament_python` 패키지가 아님) -- 이유는 아래 "환경" 참고.

## 환경

두 개의 완전히 분리된 파이썬 환경을 오간다:

| 언제 | 환경 | 이유 |
|---|---|---|
| ① 프레임 수집 | ROS 시스템 Python 3.10 (`source install/setup.bash`) | rclpy가 필요한 유일한 단계 |
| ②③④ + 검증 | conda `sam3` env (`/home/rozen/miniconda3/envs/sam3`, Python 3.12) | SAM3/PyTorch(cu128) 등 pip 전용 GPU 의존성 -- ROS의 apt/rosdep 관례로는 설치 불가. **이 env를 시스템 python3.10과 절대 섞지 말 것** (ROS Humble이 거기 물려있음). |

```bash
conda activate sam3
pip install -r requirements-sam3-env.txt   # opencv-python/pyyaml/tqdm -- torch/sam3는 이미 설치돼 있음
```

## 단계별 실행

### ① 프레임 수집

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash          # repo root에서
python3 ml/labeling/collect_frames.py --session-name <조명/시간대 태그>
```

라이브 카메라든 `ros2 bag play`로 재생하는 녹화본이든 동일하게 동작한다
(단순 subscriber). 녹화는 표준 CLI만 사용:
```bash
ros2 bag record -o my_bag /camera/front/image_mono/compressed
```

**다른 프로젝트에서 녹화한 bag (토픽명이 다른 경우)**: 실차로 찍은 `data/rosbag2_*`
9세트는 토픽명이 `/camera/front`/`/camera/back` (컬러 JPEG, 1280x720)로 이 프로젝트의
`/camera/front/image_mono/compressed`와 다르다 -- `--topic`을 여러 개 넘기면 한 번의
실행으로 두 카메라를 동시에 받고, 세션명에 자동으로 `_front`/`_back`이 붙어 구분된다:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

for bag in data/rosbag2_2026_07_23-*/; do
  session=$(basename "$bag")
  python3 ml/labeling/collect_frames.py --session-name "$session" --topic /camera/front /camera/back &
  COLLECTOR_PID=$!
  sleep 1   # subscriber가 뜰 때까지 잠깐 대기 후 재생 시작
  ros2 bag play "$bag"
  kill $COLLECTOR_PID
  wait $COLLECTOR_PID 2>/dev/null
done
```
해상도(1280x720 vs 1920x1080)는 둘 다 16:9라 학습 파이프라인에서 왜곡 없이 섞어 써도
된다 -- SAM3는 어차피 1008x1008로, 학습은 어차피 `train.yaml`의 고정 크기로 리사이즈함.

**세션 이름을 신중히 고를 것** -- 나중에 train/val 분할이 세션 단위로 이뤄지고,
데이터 다양성(조명 조건)을 추적하는 유일한 단서다. 여러 세션(다른 시간대/조명,
주차 마킹 유무 등)을 최소 2개 이상 모아야 정상적인 세션 단위 split이 된다.

### ② SAM3 자동 라벨링

새 세션이면 먼저 점수 분포부터 확인 (조명이 다르면 threshold/프롬프트가 안 맞을 수 있음):
```bash
conda activate sam3
python ml/labeling/score_probe.py --session <세션명>
```
출력된 top score들이 `labeling.yaml`의 `confidence_threshold`(기본 0.3)보다
전반적으로 한참 낮으면, 라벨링 전에 threshold를 낮추거나 프롬프트를 보강할 것.

먼저 소표본으로 소요시간 가늠 (항상 먼저 할 것 -- 전체 배치는 수시간 걸릴 수 있음):
```bash
python ml/labeling/label_with_sam3.py --session <세션명> --limit 20
```
출력된 예상 전체 소요시간을 보고 괜찮으면 `--limit` 없이 (혹은 세션 전체를
아우르는 큰 값으로) 재실행. `--skip-existing`이 기본값이라 중단돼도 안전하게
재실행 가능.

### ③ 마스크 → 학습 라벨

```bash
python ml/labeling/masks_to_labels.py
```
SAM3를 다시 돌리지 않으므로 `labeling.yaml`의 필터 값을 바꿔가며 몇 번이고
재실행해도 비용이 없다. "빈 라벨 비율"이 20~30%를 넘으면 threshold/프롬프트를
다시 봐야 한다는 신호로 콘솔에 출력됨.

육안 검수 (선택, 권장):
```bash
python ml/labeling/review_labels_interactive.py --every-n 20
# 창에서 한 장씩: y=검증(verified_frames.txt에 기록), n/s=스킵, b=뒤로, q=중단
# 검증(통과)한 프레임만 verified_frames.txt(화이트리스트)에 쌓임. 파일은 아무것도
# 지우지 않음. 학습에서 이 화이트리스트만 쓰려면 train.yaml의 require_verified: true
```

### ④ 경량 모델 학습

```bash
python ml/training/train.py --exp-name <실험명>   # PIDNetLite (configs/train.yaml)
```
- Best checkpoint: `ml/runs/<실험명>/best.pt` (어떤 아키텍처로 학습했는지도 체크포인트 안에 같이 저장됨 -- 로드하는 쪽에서 따로 지정할 필요 없음)
- 정성적 확인용 스냅샷(입력|SAM3 라벨|모델 예측 나란히): `ml/runs/<실험명>/val_samples/epoch_*.png`
- `num_workers`가 너무 크면(코어 수 대비) 오히려 느려질 수 있음 -- `train.yaml`/`train_pidnet.yaml`의 주석 참고 (특히 train/val 로더에 동시에 많은 워커를 주면 서로 자원을 다퉈서 역효과가 났던 사례가 있었음).

**노트북/저전력 GPU에서 유난히 느리다면**: GPU 전력 제한(power limit)이 기본값으로
낮게 잡혀있는지 먼저 확인할 것 (특히 ThinkPad류 슬림 비즈니스 노트북 -- 게이밍
노트북과 달리 칩 스펙상 최대치보다 훨씬 낮은 TGP로 공장 출하되는 경우가 흔함):
```bash
nvidia-smi -q -d PERFORMANCE | grep -A3 "Clocks Event Reasons"
nvidia-smi -q -d POWER | grep -E "Current Power Limit|Default Power Limit|Max Power Limit"
```
`SW Power Cap: Active`가 뜨고 클럭이 최대치의 절반 이하라면 전력 제한이 병목이다.
직접 올릴 수 있는 만큼만 올려짐 (하드웨어/펌웨어가 실제로 버틸 수 있는 선에서
드라이버가 알아서 clamp함, 그 이상은 거부됨):
```bash
sudo nvidia-smi -pl <원하는 W>   # 라이브에서 즉시 적용, 재부팅하면 초기화됨
```

### 검증: 기존 파이프라인과 비교

```bash
python ml/training/compare_with_lane_detector.py --checkpoint ml/runs/<실험명>/best.pt --val-only
```
`LaneDetector`를 서브클래싱해서 `mask_white()`만 학습된 모델로 교체하고, 나머지
(Hough/클러스터링/트래킹)는 그대로 재사용 -- **`lane_detector_node.py`나
`lane_detector.yaml`은 건드리지 않는 순수 비교 도구**다. 체크포인트에 저장된
`model` 값(tinyunet/pidnet)을 보고 알아서 맞는 아키텍처로 로드하므로 두 후보
아무 체크포인트나 그대로 넘기면 됨. 결과는 `ml/runs/<실험명>/compare/`에:
- `<frame_id>.png`: baseline(adaptiveThreshold) vs learned(TinyUNet/PIDNet) 나란히 비교
- `tracking_stability.csv`: 프레임별 confirmed 라인 개수 및 x_near/x_far (프레임 간 안정성 육안+수치 비교용)

## 다른 머신에서 이어하기 (예: 더 강력한 GPU 데스크탑)

`ml/data/`와 `ml/runs/`는 `.gitignore`에 걸려있어 **git으로는 안 넘어간다** --
코드(`git pull`)만 받아지고, 아래는 USB/rsync 등 별도 방법으로 옮겨야 한다:

- `ml/data/raw_frames/`, `ml/data/labels/`, `ml/data/splits.json` -- ④ 학습에 필요한 전부.
  `sam3_raw/`는 이미 ③을 거쳤다면 굳이 안 옮겨도 됨 (필터 값 재조정하고 싶을 때만 필요).
- SAM3(②)는 원래 머신에서 이미 다 돌렸다면 새 머신에 SAM3/conda env를 새로
  설치할 필요 없음 -- ④(`train.py`)와 검증(`compare_with_lane_detector.py`)만
  돌리면 되고, 이건 PyTorch + opencv-python만 있으면 됨 (`requirements-sam3-env.txt`
  참고, SAM3 자체 설치는 불필요).

## 데이터 규모 가이드

- **1차 smoke test**: 단일 세션 150~300장 (1fps 기준 5~10분) -- 파이프라인
  전체가 기계적으로 이어지는지부터 확인.
- **본 학습**: 여러 세션(다른 시간대/조명, 주차 마킹 유무 포함) 1500~2000장 이상.
  **프레임 수보다 조명 조건 다양성 확보가 우선** -- adaptiveThreshold를
  대체하려는 핵심 동기와 직결된다.

## 범위 밖 (다음 단계)

- 학습된 모델을 실제 `lane_detector_node.py`/`lane_detector.py`에 통합/교체
- ROS 노드 실행 환경(시스템 python3.10)에 PyTorch 등을 설치하는 배포 준비
