# Rokey F4 - 자동 주유 시스템

Isaac Sim과 ROS2를 기반으로 한 **자동 주유 로봇 팔 시뮬레이션 프로젝트**입니다.

---
1. M0609 로봇 암과 RealSense D455 카메라를 통한 자동 주유 시스템 구현을 목표로 한다.
2. 사용된 로봇 암의 레포지토리는 다음과 같다.
 - Robot Arm, EE Gripper : https://github.com/ahnisinc/rokey7.git
3. 주유 노즐, 주유소, NVIDIA Sim 모델, Universe의 차량 주유구 데이터셋이 사용된다.

본 프로젝트는 실제 주유 동작 전체를 물리적으로 완성하기보다, 자동 주유 시스템에서 가장 중요한 크리티컬 포인트인 **주유구 위치 인식**, **로봇팔 경로 생성**, **노즐 삽입 시퀀스**, **두 로봇 간 작업 순서 동기화**를 Isaac Sim 환경에서 검증하는 데 초점을 둡니다.

* Notion Doc Page : https://app.notion.com/p/515a6635480f83b89b6c01e30aced224

---

## 1. 시스템 설계 및 플로우 차트

### 1.1 시스템 구성도

<img width="1672" height="941" alt="ChatGPT Image Jun 24, 2026, 04_08_23 PM" src="https://github.com/user-attachments/assets/5e62505f-42bc-490c-9c4e-04dbc16eae46" />


### 1.2 주요 모듈 역할

| 모듈                                            | 역할                                                                                                     |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `multi_robot_oiling.py`                       | Isaac Sim 메인 시뮬레이션 실행, USD 로드, 로봇 A/B 등록, RMPFlow 제어, ROS2 토픽 송수신, 전체 상태 머신 실행                         |
| `aruco_marker_detector.py`                    | RGB 이미지와 CameraInfo를 받아 ArUco marker pose를 추정하고, fuel cap / fuel port / fuel door 목표 좌표를 ROS2 pose로 발행 |
| `m0609_rmpflow_controller.py`                 | Doosan M0609 로봇의 RMPFlow 기반 경로 제어                                                                      |
| `Collected_oiling_project/oiling_project.usd` | 주유소, 차량, 주유구, 주유구 덮개, 마개, 로봇 등이 포함된 Isaac Sim 씬                                                        |
| `rmpflow/*.yaml`                              | 로봇 RMPFlow 설정 파일                                                                                       |
| `doosan-robot2/urdf/m0609_isaac_sim.urdf`     | M0609 로봇 URDF 모델                                                                                       |

---

### 1.3 전체 동작 플로우

<img width="2720" height="3440" alt="fueling_flowchart_v4" src="https://github.com/user-attachments/assets/b00b98ae-1d8a-45e4-8955-00fdb80380c1" />


---

### 1.4 로봇 A/B 역할 분담

#### Robot A: 노즐 삽입 담당

Robot A는 Robot B가 마개 제거를 완료했다는 `/robot_b/done` 신호를 받은 뒤 동작합니다.

주요 역할은 다음과 같습니다.

1. `/color_detector/mode_switch`에 `green` 모드 요청
2. ArUco detector로부터 주유구 입구 위치 수신
3. 주유구 중심 좌표 기준 접근 경로 생성
4. 노즐 삽입
5. 노즐 후퇴
6. 초기 자세 복귀
7. `/robot_a/done` 발행

#### Robot B: 마개 및 덮개 처리 담당

Robot B는 시뮬레이션 시작 시 먼저 동작합니다.

주요 역할은 다음과 같습니다.

1. `/color_detector/mode_switch`에 `blue` 모드 요청
2. ArUco detector로부터 fuel cap 위치 수신
3. fuel cap 접근
4. 그리퍼 닫기
5. `joint_6` 회전을 통한 마개 풀기
6. 마개 추출 후 초기 위치 복귀
7. `/robot_b/done` 발행
8. Robot A의 `/robot_a/done` 수신 대기
9. 마개 재삽입
10. `joint_6` 반대 회전을 통한 마개 조이기
11. 그리퍼 열기
12. 주유구 덮개 닫기
13. 최종 초기 자세 복귀

---

## 2. 운영체제 환경

본 프로젝트는 다음 환경을 기준으로 구성합니다.

| 항목            | 버전 / 내용                                  |
| ------------- | ---------------------------------------- |
| OS            | Ubuntu 22.04 LTS                         |
| ROS2          | ROS2 Humble                              |
| Simulator     | NVIDIA Isaac Sim                         |
| Python        | Isaac Sim 내장 Python 또는 Python 3.10 계열 권장 |
| GPU           | NVIDIA GPU 권장                            |
| Middleware    | ROS2 DDS                                 |
| Robot Control | RMPFlow                                  |
| Vision        | OpenCV ArUco Marker Detection            |

> Isaac Sim은 일반 Python이 아니라 Isaac Sim에서 제공하는 `python.sh` 또는 Isaac Sim Python 환경으로 실행하는 것을 권장합니다.
> `isaacsim`, `omni`, `pxr` 관련 모듈은 일반 `pip install` 대상이 아니라 Isaac Sim 내부 모듈입니다.

---

## 3. 사용한 장비 목록

### 3.1 시뮬레이션 장비

| 장비                       | 용도                        |
| ------------------------ | ------------------------- |
| Doosan M0609 Robot Arm A | 주유 노즐 접근 및 삽입             |
| Doosan M0609 Robot Arm B | 주유구 마개 제거, 복원, 덮개 닫기      |
| Parallel Gripper         | 마개 파지 및 조작                |
| Fuel Nozzle USD          | 노즐 삽입 동작 검증               |
| Vehicle USD              | 차량 및 주유구 위치 검증 대상         |
| Gas Station USD          | 주유소 환경 구성                 |
| Fuel Door                | 주유구 덮개 열림/닫힘 대상           |
| Fuel Cap                 | 마개 제거 및 복원 대상             |
| Fuel Port Hole           | 노즐 삽입 목표 위치               |
| Wall Camera / RGB Camera | ArUco marker 인식용 카메라      |
| ArUco Marker             | 주유구 주변 목표 좌표 추정을 위한 기준 마커 |

### 3.2 소프트웨어 구성 요소

| 구성 요소                 | 설명                        |
| --------------------- | ------------------------- |
| Isaac Sim             | 로봇, 차량, 주유소 환경 시뮬레이션      |
| ROS2 Humble           | 노드 간 토픽 통신                |
| Isaac Sim ROS2 Bridge | Isaac Sim과 ROS2 간 데이터 송수신 |
| OpenCV ArUco          | Marker 기반 pose estimation |
| NumPy                 | 좌표, 벡터, 행렬 계산             |
| RMPFlow               | 로봇팔 경로 생성 및 제어            |

---

## 4. 의존성

### 4.1 Python requirements.txt 예시

아래 내용은 ROS2와 Isaac Sim을 제외한 Python 패키지 중심의 예시입니다.

```txt
numpy
opencv-contrib-python
```

`cv_bridge`, `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`는 일반적으로 ROS2 패키지로 설치합니다.

---

### 4.2 ROS2 패키지 의존성

Ubuntu 22.04 + ROS2 Humble 기준 예시는 다음과 같습니다.

```bash
sudo apt update

sudo apt install -y \
  ros-humble-rclpy \
  ros-humble-std-msgs \
  ros-humble-geometry-msgs \
  ros-humble-sensor-msgs \
  ros-humble-cv-bridge \
  ros-humble-image-transport
```

OpenCV ArUco 기능을 사용하기 위해 `opencv-contrib`가 필요합니다.

```bash
pip install opencv-contrib-python numpy
```

> Isaac Sim 내부 Python 환경을 사용할 경우, 위 pip 설치는 Isaac Sim의 `python.sh -m pip install ...` 형태로 설치해야 할 수 있습니다.

예시:

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh -m pip install opencv-contrib-python numpy
```

---

### 4.3 Isaac Sim 관련 의존성

다음 모듈은 Isaac Sim 내부에서 제공됩니다.

```python
isaacsim
omni.usd
pxr
isaacsim.core.api
isaacsim.robot.manipulators
isaacsim.core.utils
```

따라서 일반 Python 환경에서 `python multi_robot_oiling.py`로 실행하면 import 오류가 발생할 수 있습니다.
반드시 Isaac Sim Python 환경에서 실행해야 합니다.

---

## 5. 사용 설명

### 5.1 프로젝트 폴더 구조 예시

```bash
automatic-oiling-robot/
├── multi_robot_oiling.py
├── aruco_marker_detector.py
├── requirements.txt
├── rmpflow/
│   ├── m0609_rmpflow_controller.py
│   ├── m0609_description.yaml
│   └── m0609_rmpflow_common.yaml
├── doosan-robot2/
│   └── urdf/
│       └── m0609_isaac_sim.urdf
└── Collected_oiling_project/
    └── oiling_project.usd
```

---

### 5.2 실행 전 확인 사항

실행 전 다음 항목을 확인합니다.

1. ROS2 Humble이 설치되어 있어야 합니다.
2. Isaac Sim이 설치되어 있어야 합니다.
3. Isaac Sim에서 ROS2 Bridge 사용이 가능해야 합니다.
4. `Collected_oiling_project/oiling_project.usd` 경로가 올바르게 존재해야 합니다.
5. `rmpflow/` 폴더 안에 M0609 RMPFlow 설정 파일이 있어야 합니다.
6. `doosan-robot2/urdf/m0609_isaac_sim.urdf` 파일이 존재해야 합니다.
7. 카메라에서 `/rgb`, `/camera_info` 토픽이 발행되어야 합니다.
8. ArUco marker 크기와 코드의 `marker_size_m` 값이 일치해야 합니다.

---

### 5.3 ROS2 환경 설정

새 터미널을 열고 ROS2 환경을 source 합니다.

```bash
source /opt/ros/humble/setup.bash
```

워크스페이스를 사용하는 경우:

```bash
cd ~/cobot3_ws
source install/setup.bash
```

---

### 5.4 ArUco Marker Detector 실행

ArUco 인식 노드를 먼저 실행합니다.

```bash
python3 aruco_marker_detector.py
```

또는 ROS2 패키지로 구성한 경우:

```bash
ros2 run <package_name> aruco_marker_detector
```

실행 후 다음 토픽을 구독합니다.

| Subscribe Topic               | Message Type                 | 설명                    |
| ----------------------------- | ---------------------------- | --------------------- |
| `/rgb`                        | `sensor_msgs/msg/Image`      | Isaac Sim 카메라 RGB 이미지 |
| `/camera_info`                | `sensor_msgs/msg/CameraInfo` | 카메라 내부 파라미터           |
| `/color_detector/mode_switch` | `std_msgs/msg/String`        | 탐지 대상 전환 명령           |

다음 토픽을 발행합니다.

| Publish Topic                   | Message Type                    | 설명           |
| ------------------------------- | ------------------------------- | ------------ |
| `/color_detector/pose`          | `geometry_msgs/msg/PoseStamped` | 카메라 기준 목표 좌표 |
| `/color_detector/target_locked` | `std_msgs/msg/Bool`             | 목표 좌표 안정화 여부 |
| `/color_detector/current_mode`  | `std_msgs/msg/String`           | 현재 인식 모드     |
| `/color_detector/debug_image`   | `sensor_msgs/msg/Image`         | 디버그 이미지      |

---

### 5.5 Isaac Sim 메인 시뮬레이션 실행

Isaac Sim Python 환경에서 메인 스크립트를 실행합니다.

예시:

```bash
cd automatic-oiling-robot

~/.local/share/ov/pkg/isaac-sim-*/python.sh multi_robot_oiling.py
```

Isaac Sim 설치 경로가 다른 경우, 본인 환경에 맞게 `python.sh` 경로를 수정합니다.

예시:

```bash
/path/to/isaac-sim/python.sh multi_robot_oiling.py
```

---

### 5.6 실행 순서 요약

권장 실행 순서는 다음과 같습니다.

```bash
# Terminal 1: ROS2 환경 설정
source /opt/ros/humble/setup.bash
source ~/cobot3_ws/install/setup.bash

# Terminal 2: ArUco detector 실행
python3 aruco_marker_detector.py

# Terminal 3: Isaac Sim 메인 시뮬레이션 실행
~/.local/share/ov/pkg/isaac-sim-*/python.sh multi_robot_oiling.py
```

실행 후 Isaac Sim 창에서 Play를 누르면 자동 주유 시퀀스가 진행됩니다.

---

### 5.7 주요 토픽 확인 명령어

현재 발행 중인 ROS2 토픽을 확인합니다.

```bash
ros2 topic list
```

카메라 이미지 토픽 확인:

```bash
ros2 topic echo /camera_info
```

ArUco detector의 목표 좌표 확인:

```bash
ros2 topic echo /color_detector/pose
```

Target lock 상태 확인:

```bash
ros2 topic echo /color_detector/target_locked
```

모드 전환 확인:

```bash
ros2 topic echo /color_detector/current_mode
```

Robot A 완료 신호 확인:

```bash
ros2 topic echo /robot_a/done
```

Robot B 완료 신호 확인:

```bash
ros2 topic echo /robot_b/done
```

디버그 이미지를 확인하려면 `rqt_image_view`를 사용할 수 있습니다.

```bash
rqt_image_view
```

이후 `/color_detector/debug_image` 토픽을 선택합니다.

---

### 5.8 수동 모드 전환 테스트

ArUco detector는 다음 mode 값을 사용합니다.

| Mode     | 의미                   |
| -------- | -------------------- |
| `blue`   | fuel cap 위치 추정       |
| `green`  | fuel port hole 위치 추정 |
| `yellow` | fuel door 위치 추정      |

수동으로 모드를 바꾸려면 다음 명령어를 사용합니다.

```bash
ros2 topic pub --once /color_detector/mode_switch std_msgs/msg/String "{data: 'blue'}"
```

```bash
ros2 topic pub --once /color_detector/mode_switch std_msgs/msg/String "{data: 'green'}"
```

```bash
ros2 topic pub --once /color_detector/mode_switch std_msgs/msg/String "{data: 'yellow'}"
```

---

### 5.9 정상 동작 기준

시뮬레이션이 정상적으로 실행되면 다음 순서로 로그와 동작이 나타납니다.

1. Isaac Sim에서 USD 씬이 로드됩니다.
2. `m0609_A`, `m0609_B` 로봇이 등록됩니다.
3. ROS2 Bridge가 활성화됩니다.
4. Robot B가 `blue` 모드로 fuel cap 위치를 요청합니다.
5. ArUco detector가 fuel cap 위치를 발행합니다.
6. Robot B가 마개를 잡고 제거한 뒤 `/robot_b/done`을 발행합니다.
7. Robot A가 `green` 모드로 주유구 위치를 요청합니다.
8. Robot A가 노즐을 주유구 방향으로 접근 및 삽입합니다.
9. Robot A가 복귀한 뒤 `/robot_a/done`을 발행합니다.
10. Robot B가 마개를 다시 끼우고 주유구 덮개를 닫습니다.
11. 전체 시퀀스가 종료되고 시뮬레이션이 일시정지됩니다.

---

### 5.10 문제 해결

#### `/rgb` 또는 `/camera_info`가 보이지 않는 경우

```bash
ros2 topic list
```

명령어로 토픽이 발행되고 있는지 확인합니다.
토픽이 없다면 Isaac Sim 카메라 또는 ROS2 Bridge 설정을 확인해야 합니다.

#### ArUco marker가 인식되지 않는 경우

다음을 확인합니다.

1. marker가 카메라 시야 안에 있는지 확인합니다.
2. `marker_id` 값이 실제 marker ID와 같은지 확인합니다.
3. `marker_size_m` 값이 실제 marker 크기와 같은지 확인합니다.
4. `aruco_dictionary` 값이 marker 생성 시 사용한 dictionary와 같은지 확인합니다.
5. `/color_detector/debug_image`에서 marker box가 표시되는지 확인합니다.

#### 로봇이 잘못된 위치로 이동하는 경우

다음을 확인합니다.

1. ArUco marker 기준 offset 값이 실제 USD 배치와 맞는지 확인합니다.
2. `/color_detector/pose` 값이 정상 범위인지 확인합니다.
3. 메인 스크립트의 camera 좌표계 변환 함수가 현재 카메라 배치와 맞는지 확인합니다.
4. fuel cap, fuel port hole, fuel door prim의 실제 world position이 하드코딩 fallback 값과 크게 다르지 않은지 확인합니다.

#### Isaac Sim import 오류가 발생하는 경우

일반 Python이 아닌 Isaac Sim Python으로 실행했는지 확인합니다.

잘못된 예시:

```bash
python3 multi_robot_oiling.py
```

권장 예시:

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh multi_robot_oiling.py
```

---

## 6. 향후 개선 사항

* 웹 클라이언트 연동
* 사용자 주유량 입력 기능 추가
* 관리자 관제 페이지 추가
* 실제 주유량 상태 모니터링 UI 추가
* ArUco marker 없이 주유구를 직접 인식하는 vision 모델 적용
* LiDAR 또는 depth camera 기반 주유구 방향 추정
* 충돌 감지 및 비상 정지 로직 강화
* 주유구 위치 오차에 따른 자동 보정 경로 생성
* 실제 로봇 제어 환경과의 연동 검증

---

## 7. 프로젝트 목표 요약

이 프로젝트는 자동 주유 로봇 팔 시스템의 핵심 기능을 Isaac Sim에서 검증하는 것을 목표로 합니다.

핵심 검증 항목은 다음과 같습니다.

1. 차량 주유구 주변 목표 위치 인식
2. ROS2 토픽 기반 인식-제어 연동
3. 두 로봇 간 작업 순서 동기화
4. RMPFlow 기반 로봇팔 경로 제어
5. 마개 제거, 노즐 삽입, 마개 복원 시퀀스 구현
6. 자동 주유 시스템의 전체 동작 흐름 시뮬레이션
