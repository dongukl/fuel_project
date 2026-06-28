# 자동 주유 시스템

차에서 내리지 않고 결제까지 완료할 수 있는 자율 주유 로봇 시스템입니다.  
Isaac Sim 기반 시뮬레이션 환경에서 두 대의 로봇 팔이 주유구 마개 제거 → 노즐 삽입 → 마개 복원까지 자동으로 수행하며, 웹 클라이언트를 통해 주유량 입력 및 결제까지 처리합니다.

---

## 환경

| 항목 | 버전 / 내용 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS2 | ROS2 Humble |
| Simulator | NVIDIA Isaac Sim |
| Python | Isaac Sim 내장 Python 또는 Python 3.10 계열 권장 |
| GPU | NVIDIA GPU 권장 |
| Middleware | ROS2 DDS |
| Robot Control | RMPFlow |
| Vision | OpenCV ArUco Marker Detection |
| Web Server | FastAPI, Uvicorn |
| Web Client | HTML, CSS, JavaScript |
| Realtime Log | Server-Sent Events (SSE) |

---

## 폴더 구조

```
fuel_project/
├── cobot3_ws/isaacpjt/
│   ├── M0609/
│   │   ├── multi_robot_oiling.py       # Isaac Sim 메인 실행 스크립트
│   │   ├── rmpflow/                    # RMPFlow 설정 파일
│   │   └── doosan-robot2/urdf/         # 로봇 URDF 모델
│   └── hand/
│       ├── multi_robot_oiling_hand.py  # 핸드 로봇 시뮬레이션 스크립트
│       └── Collected_oiling_project/   # Isaac Sim USD 씬
├── fuel_port_perception/               # ROS2 ArUco 인식 패키지
│   └── src/fuel_port_perception/
│       └── fuel_port_perception/
│           └── aruco_marker_detector.py
└── web/
    ├── server.py                       # FastAPI 웹 서버
    └── templates/receipt.html          # 웹 클라이언트 UI
```

---

## 설치

### 1. 레포지토리 클론

```bash
git clone https://github.com/dongukl/fuel_project.git
cd fuel_project
```

### 2. ROS2 패키지 의존성 설치

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

### 3. Python 패키지 설치

```bash
pip install opencv-contrib-python numpy fastapi uvicorn
```

> Isaac Sim 내부 Python 환경을 사용하는 경우:
> ```bash
> ~/.local/share/ov/pkg/isaac-sim-*/python.sh -m pip install opencv-contrib-python numpy fastapi uvicorn
> ```

### 4. ROS2 패키지 빌드

```bash
cd fuel_port_perception
colcon build
source install/setup.bash
cd ..
```

---

## 실행

터미널을 3개 열어서 순서대로 실행합니다.

### Terminal 1 — ROS2 환경 설정 및 ArUco 인식 노드 실행

```bash
source /opt/ros/humble/setup.bash
source ~/fuel_project/fuel_port_perception/install/setup.bash

ros2 run fuel_port_perception aruco_marker_detector
```

### Terminal 2 — Isaac Sim 시뮬레이션 실행

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh ~/fuel_project/cobot3_ws/isaacpjt/hand/multi_robot_oiling_hand.py
```

### Terminal 3 — 웹 서버 실행

```bash
cd ~/fuel_project/web
uvicorn server:app --host 0.0.0.0 --port 8000
```

브라우저에서 접속:

```
http://localhost:8000
```

---

## 사용 방법

1. 브라우저에서 `http://localhost:8000` 접속
2. 목표 주유량(금액 또는 리터) 입력
3. `주유 시작` 버튼 클릭
4. 로봇이 자동으로 주유구 마개 제거 → 노즐 삽입 → 마개 복원 수행
5. 완료 후 영수증 화면에서 결제 확인
