# Rokey F4 - 자동 주유 시스템

Isaac Sim과 ROS2를 기반으로 한 **자동 주유 로봇 팔 시뮬레이션 프로젝트**입니다.

---

1. M0609 로봇 암과 RealSense D455 카메라를 통한 자동 주유 시스템 구현을 목표로 한다.
2. 사용된 로봇 암의 레포지토리는 다음과 같다.
 - Robot Arm, EE Gripper : https://github.com/ahnisinc/rokey7.git
3. 주유 노즐, 주유소, NVIDIA Sim 모델, Universe의 차량 주유구 데이터셋이 사용된다.

본 프로젝트는 실제 주유 동작 전체를 물리적으로 완성하기보다, 자동 주유 시스템에서 가장 중요한 크리티컬 포인트인 **주유구 위치 인식**, **로봇팔 경로 생성**, **노즐 삽입 시퀀스**, **두 로봇 간 작업 순서 동기화**를 Isaac Sim 환경에서 검증하는 데 초점을 둡니다.

웹 클라이언트를 연동하여 사용자가 목표 주유량을 입력하고 주유 시작/정지 명령을 보낼 수 있으며, 브라우저에서 시뮬레이션 로그와 주유 진행 상태를 확인할 수 있도록 구성했습니다.

* Notion Doc Page : https://app.notion.com/p/515a6635480f83b89b6c01e30aced224

---

## 🙋 내 역할 & 배운 점

### 담당 역할
- **Robot B 전체 시퀀스 설계 및 구현**  
  연료 도어 감지 → 캡 제거 → 캡 복원까지 11단계 시퀀스 직접 설계 및 코드 작성 후 통합

- **웹 서버 UI 설계 및 개발**  
  FastAPI + SSE 기반 소비자용 인터페이스 구현 (주유 설정 → 실시간 진행 → 영수증 화면)

- **웹 ↔ ROS2 연동 구조 설계**  
  Isaac Sim의 프로세스 격리 구조를 파악하고 subprocess + ROS2 토픽 방식으로 연동 설계,  
  `/start_fueling` 토픽 메시지로 시뮬레이션 트리거

### 배운 점

- **RMPFlow의 한계와 직접 제어**
  joint_6 단독 회전 시 RMPFlow가 로봇 전체를 회전시키는 문제를 직접 진단하고,
  `ArticulationAction`으로 전환하여 원하는 관절만 제어하는 방법 습득

- **FastAPI + SSE 기반 웹 서버 구축**
  FastAPI로 서버를 구축하고 SSE(Server-Sent Events)로 실시간 진행 상태를 클라이언트에 스트리밍,
  ROS2 토픽(`/start_fueling`)으로 시뮬레이션과 메시지 통신까지 연동
  
- **로봇 End-Effector orientation 고정 제어**
  위치(position)만 지정하면 손목이 자유롭게 회전하는 문제를 경험하고,
  삽입 방향의 법선 벡터를 기준으로 orientation을 고정하여 정밀 삽입 경로 구현

---

## 1. 시스템 설계 및 플로우 차트

### 1.1 시스템 구성도

<img width="1672" height="941" alt="ChatGPT Image Jun 27, 2026, 03_44_09 PM" src="https://github.com/user-attachments/assets/cbaf5ba9-5fca-49fe-8bff-9b50db98ac92" />


### 1.2 주요 모듈 역할

| 모듈 | 역할 |
| ---- | ---- |
| `cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py` | Isaac Sim 메인 시뮬레이션 실행, USD 로드, 로봇 A/B 등록, RMPFlow 제어, ROS2 토픽 송수신, 전체 상태 머신 실행 |
| `fuel_port_perception/.../aruco_marker_detector.py` | RGB 이미지와 CameraInfo를 받아 ArUco marker pose를 추정하고, fuel cap / fuel port / fuel door 목표 좌표를 ROS2 pose로 발행 |
| `cobot3_ws/isaacpjt/M0609/rmpflow/m0609_rmpflow_controller.py` | Doosan M0609 로봇의 RMPFlow 기반 경로 제어 |
| `web/server.py` | FastAPI 기반 웹 서버. `receipt.html` 서빙, `/start`·`/stop` API, `/events` SSE 로그 스트리밍 담당 |
| `web/templates/receipt.html` | 사용자 웹 클라이언트. 목표 금액/리터 입력, 주유 시작/정지, 진행 상태 표시, 완료 후 영수증 화면 제공 |
| `cobot3_ws/isaacpjt/M0609/Collected_oiling_project/oiling_project.usd` | 주유소, 차량, 주유구, 주유구 덮개, 마개, 로봇 등이 포함된 Isaac Sim 씬 |
| `cobot3_ws/isaacpjt/M0609/rmpflow/*.yaml` | 로봇 RMPFlow 설정 파일 |

---

### 1.3 전체 동작 플로우

<img width="1672" height="941" alt="ChatGPT Image Jun 26, 2026, 07_53_40 PM" src="https://github.com/user-attachments/assets/f0afed26-94fc-45ab-95da-c582af72bc85" />

---

### 1.4 로봇 A/B 역할 분담

#### Robot A: 노즐 삽입 담당

Robot A는 Robot B가 마개 제거를 완료했다는 `/robot_b/done` 신호를 받은 뒤 동작합니다.

1. `/aruco_detector/mode_switch`에 `hole` 모드 요청
2. ArUco detector로부터 주유구 입구 위치 수신
3. 주유구 중심 좌표 기준 접근 경로 생성
4. 노즐 삽입
5. 노즐 후퇴
6. 초기 자세 복귀
7. `/robot_a/done` 발행

#### Robot B: 마개 및 덮개 처리 담당

Robot B는 시뮬레이션 시작 시 먼저 동작합니다.

1. `/aruco_detector/mode_switch`에 `cap` 모드 요청
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

| 항목 | 버전 / 내용 |
| ---- | ----------- |
| OS | Ubuntu 22.04 LTS |
| ROS2 | ROS2 Humble |
| Simulator | NVIDIA Isaac Sim |
| Python | Isaac Sim 내장 Python (python.sh) |
| GPU | NVIDIA GPU 필수 |
| Robot Control | RMPFlow |
| Vision | OpenCV ArUco Marker Detection |
| Web Server | FastAPI, Uvicorn |

---

## 3. 사용한 장비 목록

### 3.1 시뮬레이션 장비

| 장비 | 용도 |
| ---- | ---- |
| Doosan M0609 Robot Arm A | 주유 노즐 접근 및 삽입 |
| Doosan M0609 Robot Arm B | 주유구 마개 제거, 복원, 덮개 닫기 |
| Parallel Gripper (OnRobot RG2) | 마개 파지 및 조작 |
| Fuel Nozzle USD | 노즐 삽입 동작 검증 |
| Vehicle USD | 차량 및 주유구 위치 검증 대상 |
| Gas Station USD | 주유소 환경 구성 |
| Fuel Door | 주유구 덮개 열림/닫힘 대상 |
| Fuel Cap | 마개 제거 및 복원 대상 |
| Fuel Port Hole | 노즐 삽입 목표 위치 |
| RGB Camera | ArUco marker 인식용 카메라 |
| ArUco Marker | 주유구 주변 목표 좌표 추정을 위한 기준 마커 |

---

## 4. 의존성 설치

### 4.1 ROS2 패키지

Ubuntu 22.04 + ROS2 Humble 기준:

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

### 4.2 Python 패키지

```bash
pip install opencv-contrib-python numpy fastapi uvicorn
```

> Isaac Sim 내부 Python 환경을 사용할 경우 아래처럼 설치합니다.

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh -m pip install opencv-contrib-python numpy fastapi uvicorn
```

### 4.3 isaac_python 명령어 등록

Isaac Sim Python을 간편하게 호출하기 위해 `~/.bashrc`에 alias를 등록합니다.

```bash
echo 'alias isaac_python="$HOME/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh"' >> ~/.bashrc
source ~/.bashrc
```

> Isaac Sim 버전에 따라 경로가 다를 수 있습니다. 실제 경로를 확인하세요.

```bash
ls ~/.local/share/ov/pkg/
```

---

## 5. 프로젝트 파일 구조

```
fuel_project/                                  ← 이 저장소 루트
├── README.md
├── web/                                       ← 웹 서버
│   ├── server.py                              ← FastAPI 서버 (uvicorn으로 실행)
│   └── templates/
│       └── receipt.html                       ← 웹 클라이언트 UI
│
├── fuel_port_perception/                      ← ROS2 워크스페이스 (ArUco 인식 패키지)
│   └── src/
│       └── fuel_port_perception/
│           ├── fuel_port_perception/
│           │   ├── aruco_marker_detector.py   ← ArUco 인식 ROS2 노드
│           │   └── multi_color_detector.py
│           ├── config/
│           │   └── aruco_detector_params_visual_test.yaml
│           ├── package.xml
│           └── setup.py
│
├── cobot3_ws/                                 ← Isaac Sim 워크스페이스
│   └── isaacpjt/
│       └── M0609/
│           ├── multi_robot_oiling.py          ← Isaac Sim 메인 실행 파일
│           ├── Collected_oiling_project/
│           │   └── oiling_project.usd         ← Isaac Sim 씬 파일
│           ├── onrobot_rg2/                   ← 그리퍼 URDF/USD 모델
│           └── rmpflow/                       ← RMPFlow 설정 및 컨트롤러
│               ├── m0609_rmpflow_controller.py
│               ├── m0609_description.yaml
│               └── m0609_rmpflow_common.yaml
│
└── procces_image/                             ← 개발 과정 이미지/영상
```

---

## 6. 실행 방법

터미널 3개를 열어 아래 순서대로 실행합니다.

---

### Terminal 1 — ROS2 ArUco Detector 빌드 및 실행

#### (최초 1회) ROS2 패키지 빌드

```bash
# ROS2 기본 환경 소스
source /opt/ros/humble/setup.bash

# fuel_port_perception 워크스페이스로 이동
cd ~/Desktop/project/fuel_project/fuel_port_perception

# 빌드
colcon build

# 빌드 결과 소스
source install/setup.bash
```

#### ArUco Detector 실행

```bash
# 환경변수 설정
export ROS_DOMAIN_ID=156
source /opt/ros/humble/setup.bash

# 워크스페이스로 이동 후 소스
cd ~/Desktop/project/fuel_project/fuel_port_perception
source install/setup.bash

# ArUco 인식 노드 실행
ros2 run fuel_port_perception aruco_marker_detector --ros-args \
  --params-file src/fuel_port_perception/config/aruco_detector_params_visual_test.yaml
```

정상 실행 시 아래와 같은 로그가 출력됩니다.

```
[INFO] [aruco_marker_detector_node]: ArUco detector node started. Initial mode: cap
[INFO] [aruco_marker_detector_node]: Subscribed to /rgb, /camera_info
```

---

### Terminal 2 — Isaac Sim + 웹 서버 동시 실행

Isaac Sim의 표준 출력(stdout)을 웹 서버로 파이프하여 브라우저에서 실시간 로그를 확인합니다.

```bash
# Isaac Sim + ROS2 브리지 환경변수 설정
export ROS_DOMAIN_ID=156
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib

# 프로젝트 루트로 이동
cd ~/Desktop/project/fuel_project

# Isaac Sim 실행 결과를 웹 서버로 파이프
isaac_python cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py 2>&1 | \
  (cd web && uvicorn server:app --host 0.0.0.0 --port 8000)
```

> `isaac_python`은 4단계에서 등록한 alias입니다.  
> 등록하지 않은 경우 full path를 사용하세요.  
> 예: `~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh`

---

### Terminal 3 — 웹 브라우저 접속

웹 서버가 실행된 후 브라우저에서 아래 주소로 접속합니다.

```
http://localhost:8000
```

다른 PC나 모바일에서 접속하는 경우 서버 PC의 IP를 사용합니다.

```
http://<서버-IP>:8000
```

서버 IP 확인 방법:

```bash
ip addr show | grep "inet " | grep -v "127.0.0.1"
```

---

### 실행 순서 요약

```
[Terminal 1]  ArUco detector 빌드 → 실행 (ROS2 노드)
[Terminal 2]  Isaac Sim + 웹 서버 파이프 실행
[Browser]     http://localhost:8000 → 주유량 입력 → 주유 시작
```

---

### 웹 서버만 단독 실행 (개발/테스트용)

Isaac Sim 없이 웹 화면만 확인할 경우:

```bash
cd ~/Desktop/project/fuel_project/web
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

이 경우 웹 화면은 정상 표시되지만 Isaac Sim 로그 스트리밍은 동작하지 않습니다.

---

## 7. 웹 클라이언트 사용법

브라우저에서 `http://localhost:8000` 접속 후:

1. 목표 주유량을 금액 또는 리터 단위로 입력
2. `주유 시작` 버튼 클릭 → 시뮬레이션 트리거 전송
3. 경과 시간, 주유량, 예상 금액, 진행률 게이지 실시간 표시
4. Isaac Sim 로그가 SSE로 화면에 스트리밍됨
5. `주유 정지` 버튼으로 중간 정지 가능
6. 주유 완료 후 영수증 화면으로 전환

---

## 8. ROS2 토픽 확인

### ArUco Detector 토픽

| 방향 | 토픽명 | 메시지 타입 | 설명 |
| ---- | ------ | ----------- | ---- |
| Subscribe | `/rgb` | `sensor_msgs/msg/Image` | Isaac Sim 카메라 RGB 이미지 |
| Subscribe | `/camera_info` | `sensor_msgs/msg/CameraInfo` | 카메라 내부 파라미터 |
| Subscribe | `/aruco_detector/mode_switch` | `std_msgs/msg/String` | 탐지 대상 전환 명령 |
| Publish | `/aruco_detector/pose` | `geometry_msgs/msg/PoseStamped` | 카메라 기준 목표 좌표 |
| Publish | `/aruco_detector/target_locked` | `std_msgs/msg/Bool` | 목표 좌표 안정화 여부 |
| Publish | `/aruco_detector/current_mode` | `std_msgs/msg/String` | 현재 인식 모드 |
| Publish | `/aruco_detector/debug_image` | `sensor_msgs/msg/Image` | 디버그 이미지 |

### 주유 제어 토픽

| 토픽명 | 타입 | 설명 |
| ------ | ---- | ---- |
| `/start_fueling` | `std_msgs/msg/Bool` | 주유 시작 트리거 |
| `/stop_fueling` | `std_msgs/msg/Bool` | 주유 정지 트리거 |
| `/fuel_target` | `std_msgs/msg/Float64` | 목표 주유량(리터) |
| `/robot_a/done` | `std_msgs/msg/Bool` | Robot A 시퀀스 완료 신호 |
| `/robot_b/done` | `std_msgs/msg/Bool` | Robot B 시퀀스 완료 신호 |

### 유용한 확인 명령어

```bash
# 현재 활성화된 토픽 목록
ros2 topic list

# ArUco 인식 목표 좌표 확인
ros2 topic echo /aruco_detector/pose

# ArUco lock 상태 확인
ros2 topic echo /aruco_detector/target_locked

# 디버그 이미지 시각화
rqt_image_view
# → /aruco_detector/debug_image 선택
```

### 수동 모드 전환

| 모드 값 | 인식 대상 |
| ------- | --------- |
| `cap` | fuel cap(마개) 위치 |
| `hole` | fuel port hole(주유 입구) 위치 |
| `door` | fuel door(덮개) 위치 |

```bash
# 마개 인식 모드
ros2 topic pub --once /aruco_detector/mode_switch std_msgs/msg/String "{data: 'cap'}"

# 주유 입구 인식 모드
ros2 topic pub --once /aruco_detector/mode_switch std_msgs/msg/String "{data: 'hole'}"

# 덮개 인식 모드
ros2 topic pub --once /aruco_detector/mode_switch std_msgs/msg/String "{data: 'door'}"
```

---

## 9. 정상 동작 기준

시뮬레이션이 정상적으로 실행되면 다음 순서로 동작합니다.

1. 브라우저에 `서버에 연결되었습니다` 메시지 표시
2. 목표 주유량 입력 후 `주유 시작` 클릭
3. Isaac Sim에서 USD 씬 로드
4. `m0609_A`, `m0609_B` 로봇 등록
5. ROS2 Bridge 활성화
6. Robot B가 `cap` 모드로 마개 위치 요청 → ArUco detector가 좌표 발행
7. Robot B가 마개를 잡고 제거 → `/robot_b/done` 발행
8. Robot A가 `hole` 모드로 주유구 위치 요청
9. Robot A가 노즐을 삽입 → 웹 화면에 진행 상태 표시
10. Robot A 복귀 → `/robot_a/done` 발행
11. Robot B가 마개를 재삽입하고 덮개를 닫음
12. 완료 후 브라우저에 영수증 화면 표시

---

## 10. 문제 해결

### Isaac Sim import 오류

```
ModuleNotFoundError: No module named 'isaacsim'
```

일반 Python으로 실행했을 때 발생합니다. 반드시 Isaac Sim Python 환경으로 실행해야 합니다.

```bash
# 잘못된 예
python3 cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py

# 올바른 예
isaac_python cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py
```

---

### 웹 페이지가 열리지 않는 경우

1. `uvicorn server:app` 실행이 `web/` 디렉토리에서 이루어졌는지 확인합니다.
2. 8000번 포트가 사용 중인지 확인합니다.

```bash
sudo lsof -i :8000
```

3. 포트를 변경하려면:

```bash
cd web && uvicorn server:app --host 0.0.0.0 --port 8080
# 브라우저 접속: http://localhost:8080
```

---

### 웹 로그가 갱신되지 않는 경우

`server.py`는 Isaac Sim stdout을 파이프로 받아 SSE로 브라우저에 전달합니다.  
반드시 파이프 형태로 실행해야 합니다.

```bash
isaac_python cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py 2>&1 | \
  (cd web && uvicorn server:app --host 0.0.0.0 --port 8000)
```

웹 서버만 단독 실행한 경우 로그 스트리밍은 동작하지 않습니다.

---

### ArUco marker가 인식되지 않는 경우

1. `/rgb` 토픽이 발행되는지 확인합니다.

```bash
ros2 topic list | grep rgb
```

2. `aruco_detector_params_visual_test.yaml`의 `marker_id`, `marker_size_m`이 실제 마커와 일치하는지 확인합니다.
3. `/aruco_detector/debug_image`에서 마커 박스가 표시되는지 `rqt_image_view`로 확인합니다.

---

### ROS2 토픽이 보이지 않는 경우

터미널마다 `ROS_DOMAIN_ID`가 동일한지 확인합니다. 이 프로젝트는 `156`을 사용합니다.

```bash
export ROS_DOMAIN_ID=156
```

---

### `isaac_python` 명령어를 찾을 수 없는 경우

alias가 등록되지 않은 경우 Isaac Sim 설치 경로를 직접 사용합니다.

```bash
# Isaac Sim 설치 경로 확인
ls ~/.local/share/ov/pkg/

# 직접 실행 (버전 번호는 실제 설치 버전에 맞게 수정)
~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh \
  cobot3_ws/isaacpjt/M0609/multi_robot_oiling.py
```

---

## 11. 향후 개선 사항

* 관리자 관제 페이지 추가
* 실제 주유량을 시뮬레이션 상태 또는 센서 데이터 기반으로 계산하도록 개선
* ArUco marker 없이 주유구를 직접 인식하는 vision 모델 적용
* LiDAR 또는 depth camera 기반 주유구 방향 추정
* 충돌 감지 및 비상 정지 로직 강화
* 실제 로봇 제어 환경과의 연동 검증

---

## 12. 프로젝트 목표 요약

이 프로젝트는 자동 주유 로봇 팔 시스템의 핵심 기능을 Isaac Sim에서 검증하는 것을 목표로 합니다.

1. 차량 주유구 주변 목표 위치 인식
2. ROS2 토픽 기반 인식-제어 연동
3. 두 로봇 간 작업 순서 동기화
4. RMPFlow 기반 로봇팔 경로 제어
5. 마개 제거, 노즐 삽입, 마개 복원 시퀀스 구현
6. 웹 클라이언트를 통한 사용자 입력 및 시뮬레이션 상태 모니터링
7. 자동 주유 시스템의 전체 동작 흐름 시뮬레이션
