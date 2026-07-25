# autodrive_ws

## 1. 프로젝트 목적

전방 카메라 한 대만을 이용하는 자율주행 차량을 위한 ROS2 Humble 워크스페이스입니다.
차량 모델은 kinematic bicycle model을 사용하며, 최종 저수준 액추에이터(조향 모터 등)
제어는 Arduino Mega가 담당합니다. 이 저장소는 현재 알고리즘 구현이 아니라
**패키지 구조와 최소 실행 가능한 노드 골격**을 제공하는 것을 목표로 합니다.

장애물 회피, 신호등 인식, 주차 미션은 아직 구현하지 않으며, `autodrive_missions`
패키지 아래에 향후 구현을 위한 빈 폴더만 마련해 두었습니다.

## 2. ROS2 패키지별 역할

| 패키지 | 빌드 타입 | 역할 |
|---|---|---|
| `autodrive_description` | ament_cmake | 차량 URDF/Xacro, TF 구조 (`base_link`, `camera_front_link`, `camera_front_optical_frame`) |
| `autodrive_sensors` | ament_python | 카메라/조향 피드백/Arduino 센서 등 하드웨어 입력 전용 (인지·위치추정 로직 없음) |
| `autodrive_perception` | ament_python | 전방 카메라 영상에서 차선/마킹 인식 + 추종할 목표 경로(레퍼런스 경로, 이미지 좌표계) 생성 — `lane_detector_node`가 실제로 동작하는 부분(장애물/신호등/주차 인식 없음). `bev_node`/`local_map_node`(BEV 변환, Local Map 생성)는 아직 미구현 스켈레톤으로 남아있고, 현재는 이 방향 대신 카메라→차선 인식→목표 경로의 직접 경로로 방향을 잡음 (아래 3절 참고) |
| `autodrive_localization` | ament_python | Global Map Matching + EKF 기반 위치추정 ([X, Y, yaw]) |
| `autodrive_planning` | ament_python | 사전 저장된 Reference Path 발행 (A*, Hybrid A*, RRT, Mission Planner 없음) |
| `autodrive_control` | ament_python | Tracking Error 계산 + LQR 조향 제어 (Pure Pursuit/RL로 확장 가능한 구조) |
| `autodrive_vehicle` | ament_python | ROS 명령 → Arduino Mega 전달, 저수준 조향 PID |
| `autodrive_safety` | ament_python | 최종 명령 중재(arbitration) 및 watchdog. Arduino로 가는 유일한 통로 |
| `autodrive_bringup` | ament_python | launch 파일과 YAML 파라미터 관리 |
| `autodrive_tools` | ament_python | 개발/테스트용 도구 (더미 퍼블리셔, 모니터, 캘리브레이션) |
| `autodrive_missions` | ament_python | 장애물 회피 / 신호등 / 주차 — 현재는 완전히 빈 폴더 (Future implementation) |

## 3. 전체 데이터 흐름

**현재 실제로 동작하는 경로 (1차 목표: 카메라 기반 차선 인식/추종)**

```
Camera
  -> Lane Detection + Tracking (lane_detector_node: LaneDetector, LaneTracker)
  -> Reference Path (image-space, /perception/lane_reference — ReferenceLaneBuilder)
  -> [아직 미연결] 저수준 조향
```

`/perception/lane_reference`는 아직 아무 노드도 구독하지 않는다 — 저수준 조향(`autodrive_vehicle`)과
연결하는 것이 다음 단계다. 지금은 카메라 → 차선 인식 → 목표 경로 생성까지만 실제로 동작한다.

**장기 비전 (아래 대부분 미구현 스켈레톤 — 현재는 이 방향 대신 위 단순 경로로 진행 중)**

```
Camera
  -> BEV
  -> Local Map
  -> Global Map Matching
  -> Camera Pose Measurement
  -> Motion Model + EKF
  -> Estimated Pose
  -> Reference Path
  -> Tracking Error
  -> LQR
  -> Safety Command Arbiter
  -> Vehicle Interface
  -> Arduino Mega
```

두 경로 다 최종적으로는 `/safety/command`를 거쳐 Arduino로 가는 구조는 같으나,
전자는 EKF/Global Map/LQR 같은 전역 위치추정·최적제어 없이 카메라 영상에서 곧바로
목표 경로를 뽑아 따라가는 훨씬 단순한 구조다.

## 4. 토픽 목록

| 토픽 | 메시지 타입 |
|---|---|
| `/camera/front/image/compressed` | `sensor_msgs/msg/CompressedImage` (컬러 JPEG) |
| `/camera/front/image_mono/compressed` | `sensor_msgs/msg/CompressedImage` (그레이스케일 JPEG, `publish_mono: true`일 때만. `lane_detector_node`가 구독하는 토픽 — 색상이 필요 없는 소비자는 컬러 디코드+BGR2GRAY를 건너뛸 수 있음) |
| `/camera/front/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/camera/rear/image/compressed` | `sensor_msgs/msg/CompressedImage` (컬러 JPEG) |
| `/camera/rear/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/perception/bev/image` | `sensor_msgs/msg/Image` |
| `/perception/local_map` | `nav_msgs/msg/OccupancyGrid` |
| `/perception/lane_debug/image/compressed` | `sensor_msgs/msg/CompressedImage` (JPEG. 추적된 차선을 원본 프레임 위에 그린 디버그 뷰 — 주행 판단에는 쓰이지 않음) |
| `/perception/lane_mask/compressed` | `sensor_msgs/msg/CompressedImage` (PNG. `LaneDetector.mask_white()`의 흑백 이진 마스크 그대로) |
| `/perception/lane_pipeline_debug/image/compressed` | `sensor_msgs/msg/CompressedImage` (JPEG. `publish_pipeline_debug: true`일 때만 발행. 마스크/raw Hough segment/클러스터(색상별)/최종결과 2x2 타일 뷰 — 재튜닝용, 평소엔 꺼둠) |
| `/perception/lane_lines` | `std_msgs/msg/Float32MultiArray` (아래 필드 규약 참고) |
| `/localization/camera_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| `/localization/pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` |
| `/planning/reference_path` | `nav_msgs/msg/Path` |
| `/control/tracking_error` | `geometry_msgs/msg/Vector3Stamped` |
| `/control/command` | `ackermann_msgs/msg/AckermannDriveStamped` |
| `/safety/command` | `ackermann_msgs/msg/AckermannDriveStamped` |
| `/vehicle/steering_feedback` | `std_msgs/msg/Float32` (표준 메시지 목록에 없어 임시로 채택, 필요 시 재검토) |
| `/vehicle/status` | `std_msgs/msg/String` (표준 메시지 목록에 없어 임시로 채택, 필요 시 재검토) |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` |

**`/control/tracking_error` (`geometry_msgs/msg/Vector3Stamped`) 필드 규약:**

- `x`: lateral error `e_y` [m]
- `y`: heading error `e_psi` [rad]
- `z`: reserved (미사용)

**`/perception/lane_lines` (`std_msgs/msg/Float32MultiArray`) 필드 규약:**

- `data`: 확정(confirmed)된 선마다 `[x_near_0, x_far_0, x_near_1, x_far_1, ...]` 순서로 이어붙인 flat 배열
  (픽셀 단위, `x_near`가 작은 순 = 왼쪽부터 정렬)
- `x_near`/`x_far`: 그 선이 ROI의 근거리/원거리 기준행과 만나는 x좌표 (칼만 필터로 평활화된 값,
  `autodrive_perception/core/lane_tracker.py` 참고)
- **횡방향 오차(e_y)나 차선 중앙 추정치가 아닙니다.** 이 중 어떤 선이 실제 주행 차선의 좌우
  경계인지 고르는 로직은 아직 없습니다 (주차장 마킹 등과 동점이 나는 문제가 남아있음,
  `lane_tracker.py`의 `_select_evenly_spaced` 주석 참고) — 소비하는 쪽에서 직접 선택 로직을
  구현해야 합니다.

**`/safety/command`는 `autodrive_vehicle`이 구동 명령으로 구독할 수 있는 유일한 토픽입니다.**
컨트롤러(`autodrive_control`)는 Arduino로 직접 명령을 보낼 수 없습니다.

## 5. 좌표계 규약

- `base_link`: 차량 기준 좌표계 (원점: 후륜 축 중심 부근)
- `camera_front_link`: 전방 카메라 마운트 좌표계 (base_link 기준 고정 변환)
- `camera_front_optical_frame`: REP 103을 따르는 카메라 광학 좌표계 (x-right, y-down, z-forward)
- `camera_rear_link`: 후방 카메라 마운트 좌표계 (base_link 기준 고정 변환, yaw=pi로 뒤를 향함)
- `camera_rear_optical_frame`: REP 103을 따르는 카메라 광학 좌표계 (x-right, y-down, z-forward)

카메라 extrinsic(설치 위치/자세)은 아직 실측/보정되지 않았으며,
`autodrive_description/urdf/autodrive.urdf.xacro`에 TODO로 표시되어 있습니다.

차량 제원 초기값:

- wheelbase: `0.545 m`
- track width: `0.430 m`
- camera height: `0.68 m`

## 6. 빌드 방법

```bash
# 필요한 경우 ROS2 표준 메시지 패키지를 설치 (아래 "사전 준비" 참고)
cd ~/autodrive_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### 사전 준비

`autodrive_control`, `autodrive_vehicle`, `autodrive_safety`, `autodrive_tools`는
`ackermann_msgs/msg/AckermannDriveStamped`를 사용합니다. 이 패키지가 시스템에
설치되어 있지 않다면 아래 명령이 필요합니다 (sudo 필요, 직접 실행하지 않았습니다):

```bash
sudo apt update
sudo apt install ros-humble-ackermann-msgs
```

`rosdep`이 설치되어 있다면 의존성 확인에 사용할 수 있습니다:

```bash
sudo apt install python3-rosdep
sudo rosdep init   # 최초 1회
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

## 7. 실행 방법

```bash
source ~/autodrive_ws/install/setup.bash

# 차량 description만 확인
ros2 launch autodrive_description description.launch.py

# 미션을 제외한 전체 스택 (description, sensors, perception, localization,
# planning, control, safety, vehicle)
ros2 launch autodrive_bringup core.launch.py
```

개별 노드도 `ros2 run <package> <executable>` 형태로 단독 실행할 수 있습니다.
예: `ros2 run autodrive_sensors camera_node`

## 8. 현재 구현 범위

- 카메라(전/후방)와 차선 인식/추종 목표 경로 생성(`autodrive_perception`의
  `lane_detector_node`)은 실제로 동작합니다 — `bev_node`/`local_map_node`는
  아직 미구현 스켈레톤입니다 (3절 참고). 생성된 목표 경로(`/perception/lane_reference`)는
  아직 저수준 조향에 연결되지 않았습니다 (다음 단계).
- 차량은 **Ackermann 조향**(전륜 조향 모터 1개로 회전)이며 구동은 좌/우 바퀴
  각각 독립된 모터/PWM 핀으로 배선되어 있습니다 (differential-drive처럼 좌우를
  다른 속도로 돌리진 않고, 지금은 동일한 throttle 값을 좌우에 동시에 적용 —
  회전은 조향 모터가 전담). `autodrive_vehicle`이 Arduino Mega와 텍스트 기반
  시리얼 프로토콜로 통신합니다 (`M <throttle_pwm>`, `ST <steer_pwm> <duration_ms>`,
  `SC`, `STOP` — 상세는 `autodrive_vehicle/core/serial_protocol.py`와
  `autodrive_vehicle/arduino/mega_motor_controller/mega_motor_controller.ino` 참고).
  단위는 **PWM**이며 토크가 아닙니다.
- **조향 각도 피드백 센서는 아직 설치 전**이라 `steering_pid_node`는 지금
  닫힌루프 PID(`SteeringPID`) 대신 **오픈루프 매핑**(`SteeringOpenLoop`:
  desired_angle → steer_pwm, 게인/한계값 전부 0으로 미확정)으로 동작합니다.
  센서가 설치되면 같은 노드/토픽 구조를 그대로 두고 `SteeringPID`로 교체하면
  됩니다.
- 나머지 알고리즘(BEV 변환, Global Map Matching, EKF, LQR 게인 계산 등)은
  여전히 TODO로 남아 있는 골격(skeleton) 수준입니다.
- `core/` 하위 모듈은 rclpy에 의존하지 않는 순수 로직 클래스이며, `nodes/`
  하위 모듈만 ROS 메시지 변환과 pub/sub을 담당합니다.
- 장애물 회피, 신호등, 주차 관련 코드는 전혀 작성하지 않았습니다.
- 실제 카메라 장치 번호(후방), Arduino 시리얼 포트/baudrate, 조향/스로틀
  PWM 게인, 카메라 extrinsic 등 확인되지 않은 값은 전부 빈 문자열/0/TODO로
  남겨져 있습니다 — 절대 추측해서 채우지 않았습니다.
- **주의**: 아두이노와 시리얼로 통신하는 노드가 3개(`arduino_bridge_node`,
  `arduino_sensor_node`, `steering_feedback_node`)인데, 각자 독립적으로
  시리얼 포트를 여는 구조라 **같은 물리 포트를 가리키면 서로 충돌**할 수
  있습니다. 실제로 몇 개의 Arduino/포트가 있는지에 따라 통합이 필요할 수
  있습니다 (아직 미해결).
- `ml/`(이 저장소의 colcon 워크스페이스 밖, 별도 파이썬 트리)에 `LaneDetector.mask_white()`의
  adaptiveThreshold를 대체하기 위한 준비 작업이 진행 중입니다: SAM3를 오프라인
  자동 라벨링 도구로 써서 차선 마스크 학습 데이터를 만들고, 그 데이터로 경량
  실시간 세그멘테이션 모델(TinyUNet/PIDNetLite 후보)을 학습시킵니다. 아직 실제
  ROS 노드에는 통합되지 않은 별도 실험 단계이며, 상세 실행 가이드는
  `ml/README.md` 참고.

## 9. 향후 미션 구현 계획

`autodrive_missions` 패키지 아래에 다음 3개 폴더만 준비되어 있으며,
각 폴더의 `README.md`에는 `Future implementation`이라고만 적혀 있습니다.

- `obstacle_avoidance/`
- `traffic_light/`
- `parking/`

이 패키지는 `autodrive_bringup/launch/core.launch.py`에서 절대 실행되지
않으며, 위 미션들의 실제 구현은 이후 별도 작업으로 진행할 예정입니다.
