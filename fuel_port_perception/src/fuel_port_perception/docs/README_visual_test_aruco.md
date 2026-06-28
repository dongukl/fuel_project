# visual_test.usda용 ArUco 인식 패키지

이 패키지는 `visual_test.usda`의 실제 차량/주유소 USD 구도에 맞춰 다시 만든 ArUco 버전입니다.
기존 색상 detector(`multi_color_detector.py`)는 더 이상 사용하지 않으며, 토픽/모드 이름도
ArUco 전용으로 `/aruco_detector/*`, `door`/`cap`/`hole`을 씁니다.

## 파일 구조

ROS2 ament_python 패키지(`fuel_port_perception`) 안에 다음과 같이 배치되어 있습니다.

```text
fuel_port_perception/src/fuel_port_perception/
├── config/
│   └── aruco_detector_params_visual_test.yaml   # detector params (colcon install 시 share/로 설치됨)
├── docs/
│   ├── README_visual_test_aruco.md              # 이 문서
│   ├── visual_test_aruco_values.md              # marker 위치/offset 계산값 요약
│   └── aruco_4x4_50_id0_visual_test.png         # 참고용 marker 이미지
├── fuel_port_perception/
│   ├── multi_color_detector.py                  # 기존 색상 detector
│   └── aruco_marker_detector.py                  # ArUco detector (entry point 등록됨)
└── setup.py                                      # entry_point + config data_files 등록

cobot3_ws/isaacpjt/M0609/
└── create_aruco_marker_grid_visual_test.py       # Isaac Sim Script Editor 전용 (ROS 노드 아님)
```

- `aruco_marker_detector.py`
  - ArUco marker pose 기반 detector
  - `/rgb`, `/camera_info`, `/aruco_detector/mode_switch` 구독
  - `/aruco_detector/pose`, `/aruco_detector/target_locked`, `/aruco_detector/current_mode`, `/aruco_detector/debug_image` 발행

- `create_aruco_marker_grid_visual_test.py`
  - Isaac Sim Script Editor에서 실행
  - `/World/aruco_vehicle_marker`에 4x4_50 id=0 marker를 geometry cell로 생성
  - texture 파일이 필요 없음

- `aruco_detector_params_visual_test.yaml`
  - visual_test.usda 기준 offset과 marker size가 들어간 params

- `visual_test_aruco_values.md`
  - marker 위치와 offset 계산값 요약

## 적용 순서

### 1. Isaac Sim에서 marker 생성

`visual_test.usda`를 연 뒤, Script Editor에서 `cobot3_ws/isaacpjt/M0609/create_aruco_marker_grid_visual_test.py`를 실행합니다.

생성 위치:

```text
/World/aruco_vehicle_marker
```

기본 marker world center:

```text
[-0.40267, -0.77000, 1.20000]
```

`rqt_image_view`에서 `/rgb`를 보고 marker가 보이는지 확인합니다.
안 보이면 `MARKER_CENTER_WORLD`를 조정하세요.

### 2. 빌드

detector와 entry point는 이미 `fuel_port_perception` 패키지에 포함되어 있으므로 colcon build만 하면 됩니다.

```bash
cd ~/fuel_ws   # fuel_port_perception/src를 colcon workspace의 src로 심볼릭 링크/복사해둔 경로
colcon build --packages-select fuel_port_perception
source install/setup.bash
```

### 3. 실행

ArUco detector만 실행하면 됩니다(`multi_color_detector.py`는 더 이상 쓰지 않음).

```bash
ros2 run fuel_port_perception aruco_marker_detector \
  --ros-args --params-file install/fuel_port_perception/share/fuel_port_perception/config/aruco_detector_params_visual_test.yaml
```

확인:

```bash
ros2 topic echo /aruco_detector/target_locked
ros2 topic echo /aruco_detector/pose
rqt_image_view   # /aruco_detector/debug_image 선택
```

## mode별 의미

- `cap`: marker 기준 fuel_cap 위치 발행
- `hole`: marker 기준 fuel_port_hole mouth surface 위치 발행
- `door`: marker 기준 fuel_door 위치 발행

`hole`에서 mouth surface를 발행하는 이유는 메인 코드가 `apply_mouth_offset=True`로 주유구 중심 보정을 한 번 더 하기 때문입니다.

## 발표용 설명 문장

초기에는 색상 기반 contour로 마개와 주유구를 구분했지만 실제 차량 환경에서는 색상만으로 주유구를 안정적으로 구분하기 어렵다. 이를 보완하기 위해 차량 주유구 주변에 ArUco marker를 배치하고, marker pose를 기준 좌표계로 사용하여 마개와 주유구 입구의 상대 위치를 계산하도록 확장하였다. 기존 ROS2 인터페이스는 유지하여 로봇 제어부 수정은 최소화하였다.
