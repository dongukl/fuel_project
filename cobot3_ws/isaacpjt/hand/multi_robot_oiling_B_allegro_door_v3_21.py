# multi_robot_oiling.py
#
# Isaac Sim에서 m0609 협동로봇 2대(A, B)를 동시에 시뮬레이션하는 메인 스크립트.
#   - m0609_A: 기존 "주유구에 노즐 꽂기" 동작만 수행 (기존 단일로봇 코드를 거의 그대로 사용)
#   - m0609_B: 주유구 커버를 열고, 마개를 풀어서 빼고, A가 주유를 마치면 마개를 다시 끼우고
#              커버를 닫는 새로운 동작을 수행
# 두 로봇은 ROS2 토픽으로 서로 "내 할 일 끝났어" 신호를 주고받으며 순서를 맞춘다(state machine).
# 카메라로 어떤 표적(door/cap/hole)을 찾을지는 ArUco marker 기반 detector 노드
# (aruco_marker_detector.py)에 /aruco_detector/mode_switch로 모드 전환 명령을 보내서 정한다.

# SimulationApp은 Isaac Sim을 파이썬에서 띄우는 진입점. 다른 isaacsim/omni 모듈을 import하기 전에
# 반드시 가장 먼저 생성해야 한다 (안 그러면 import 자체가 실패함).
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # headless=False -> 시뮬레이션 창을 화면에 띄움

# ROS2 브리지 확장 기능을 켜야 Isaac Sim 쪽에서 ROS2 토픽을 주고받을 수 있다.
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()  # 확장 기능이 실제로 로드되도록 한 프레임 갱신해줌

from dataclasses import dataclass  # FuelStage처럼 "데이터만 담는 클래스"를 짧게 정의하기 위한 도구
from pathlib import Path           # 운영체제에 상관없이 안전하게 파일 경로를 다루기 위한 도구
import sys
import time

import rclpy                       # ROS2 파이썬 클라이언트 라이브러리 (노드, 통신의 기본)
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped  # 위치+방향 메시지 타입 (카메라가 찾은 좌표를 받을 때 사용)
from std_msgs.msg import Bool, String      # 참/거짓, 문자열 메시지 타입 (lock 신호, 모드 전환 명령 등)

import numpy as np      # 벡터/행렬 연산 (3D 좌표 계산에 필수)
import omni.usd         # 현재 열려있는 USD Stage(3D 씬)에 접근하기 위한 Isaac Sim API
from pxr import Usd, UsdGeom, UsdPhysics, Gf  # USD(3D 씬 포맷)와 PhysX(물리엔진) 관련 저수준 라이브러리

from isaacsim.core.api import World                      # 시뮬레이션 전체를 관리하는 객체(물리 스텝, reset 등)
from isaacsim.core.api.objects import VisualCuboid        # 디버깅용으로 화면에 표시할 작은 박스 마커
from isaacsim.core.api.tasks import BaseTask               # "씬 구성 + 초기화"를 표준화해주는 베이스 클래스
from isaacsim.core.utils.rotations import euler_angles_to_quat  # 오일러각(롤/피치/요)을 쿼터니언으로 변환
from isaacsim.core.utils.types import ArticulationAction  # 조인트 일부만 골라 드라이브 목표를 보낼 때 사용
from isaacsim.robot.manipulators.grippers import ParallelGripper      # 그리퍼(손가락 2개) 제어 클래스
from isaacsim.robot.manipulators.manipulators import SingleManipulator  # 로봇팔 1대를 다루는 클래스


_THIS_DIR = Path(__file__).resolve().parent
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    # rmpflow 폴더 안의 m0609_rmpflow_controller.py를 import하려면
    # 그 폴더가 파이썬의 모듈 검색 경로(sys.path)에 들어있어야 한다.
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_rmpflow_controller import RMPFlowController  # RMPFlow 기반 역기구학/모션플래닝 컨트롤러

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 두 로봇이 공유하는 환경 파라미터                            ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_oiling_project/oiling_project.usd")
# m0609_A, m0609_B, fuel_door(Revolution Joint 포함), fuel_cap, fuel_port_hole이
# 이미 이 USD 안에 모델링되어 있다고 가정한다 (Isaac Sim 에디터에서 직접 구성).
EE_LINK_NAME    = "link_6"  # 엔드이펙터(손목 끝) 링크 이름 - 로봇팔의 "손" 역할을 하는 부분
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]  # 그리퍼를 움직이는 조인트 2개 이름

# PhysX 관절 드라이브(모터) 설정값. 값이 클수록 "더 단단하고 힘있게" 목표 위치를 따라간다.
DRIVE_STIFFNESS = 1e8  # 강성(stiffness): 목표 각도로 얼마나 강하게 끌어당길지
DRIVE_DAMPING   = 1e4  # 댐핑(damping): 움직임의 진동/흔들림을 얼마나 빨리 죽일지
DRIVE_MAX_FORCE = 1e8  # 낼 수 있는 최대 힘/토크

GRIPPER_OPEN    = [0.0, 0.0]   # 그리퍼 완전히 벌린 상태의 조인트 값
GRIPPER_CLOSE   = [0.5, 0.5]   # 그리퍼 완전히 닫은 상태의 조인트 값
GRIPPER_DELTA   = [-0.5, -0.5]  # ParallelGripper가 동작 명령을 만들 때 쓰는 변화량 기준값

M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")  # 로봇 골격 정의 파일
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")            # RMPFlow용 로봇 설명
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")          # RMPFlow 동작 파라미터

# 로봇이 처음 시작할 때 각 조인트가 가져야 할 각도(도 단위). "기본 자세(home pose)"를 정의한다.
# A/B는 베이스 위치/방향이 달라서 같은 기본 자세를 쓰면 손목이 서로 다른 곳을 향하게 되므로 따로 정의한다.
INITIAL_ARM_JOINT_DEG_A = {
    "joint_1": -170.0,
    "joint_2": -66.0,
    "joint_3": 150.0,
    "joint_4": 3.5,
    "joint_5": -75.0,
    "joint_6": 180.0,
}
INITIAL_ARM_JOINT_DEG_B = {
    "joint_1": 10.0,
    "joint_2": -66.0,
    "joint_3": 150.0,
    "joint_4": 3.5,
    "joint_5": -85.0,
    "joint_6": -5.0,
}
INITIAL_GRIPPER_JOINTS = {
    "finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  A-1. m0609_B wrist + Allegro hand 설정                       ║
# ╚══════════════════════════════════════════════════════════════╝
# 주의:
# - 이 섹션은 m0609_B만 위한 설정이다.
# - A 로봇 / cap 접근 waypoint / 속도 / 메인 루프 배속은 건드리지 않는다.
# - cap 접근 중에는 wrist pitch를 꺾지 않는다.
USE_B_ALLEGRO_WRIST = True

WRIST_PITCH_JOINT_CANDIDATES = [
    "WristPitchJoint",
    "wrist_pitch_joint",
    "wrist_pitch",
]

WRIST_PITCH_NEUTRAL_DEG = 0.0
WRIST_PITCH_CAP_GRASP_DEG = -45.0

# 문 닫기 길이 보정
CAP_GRASP_EXTRA_INWARD_M = 0.04
CAP_RESTORE_EXTRA_INWARD_M = 0.18

HAND_POSE_OPEN = "open"
HAND_POSE_CAP_HOLD = "cap_grasp"
HAND_POSE_DOOR_PALM_PUSH = "door_palm_press"
HAND_POSE_DOOR_FINAL_PRESS = "press"

WRIST_PITCH_DOOR_BENT_DEG = -20.0
WRIST_PITCH_DOOR_PRESS_DEG = -10.0

# Allegro joint 순서가 프로젝트 USD에서 다를 수 있으므로,
# 이 값은 "초기 시도용 pose"다. 손가락이 반대로 움직이면 부호/크기를 조정해야 한다.
ALLEGRO_POSES = {
    "open": {
        "index":  [0.0200, 0.2000, 0.2000, 0.2000],
        "middle":  [0.0000, 0.2000, 0.2000, 0.2000],
        "ring":  [-0.0200, 0.2000, 0.2000, 0.2000],
        "thumb":  [0.4200, 0.1800, 1.0000, 0.5000],
    },
    
    # cap grasp는 degree 값을 deg2rad로 자동 변환하지 않고,
    # Isaac/ArticulationAction에 바로 넣는 radian pose로 직접 관리한다.
    # "cap_grasp": {
    #     # small block grip style:
    #     # 야구공처럼 둥글게 감싸기보다, 작은 블럭을 잡듯이
    #     # index/middle/ring을 비교적 평행하게 접고 thumb이 옆면을 받치는 형태.
    #     "index":  [0.02, 0.52, 0.62, 0.22],
    #     "middle": [0.00, 0.54, 0.64, 0.22],
    #     "ring":   [-0.02, 0.52, 0.62, 0.22],
    #     "thumb":  [0.50, 0.34, 0.24, 0.08],
    # },
    "cap_grasp": {
        "index":  [0.0600, 0.0000, 1.4000, 0.6200],
        "middle":  [0.0000, -0.3000, 1.4000, 0.6200],
        "ring":  [-0.0600, 0.0000, 1.4000, 0.6200],
        "thumb":  [1.1700, 1.0000, 1.0000, -0.1500],
    },

    # door close 초반: 손을 과하게 말지 않고 살짝 구부린 palm/push 형태.
    # 끝에서 끝점 : 대강 0.29m
    "door_palm_press": {
        "index":  [0.02, 0.20, 0.20, 0.75],
        "middle": [0.00, 0.20, 0.20, 0.75],
        "ring":   [-0.02, 0.20, 0.20, 0.75],
        "thumb":  [0.2, 0.50, 1.1, 0.50],
    },

    # 마지막 누르기: palm보다 조금 더 굽히되, 주먹처럼 과하게 말리지 않게 제한.
    "press": {
        "index":  [0.0300, 0.3000, 0.3000, 0.5000],
        "middle":  [0.0000, 0.3000, 0.3000, 0.5000],
        "ring":  [-0.0300, 0.3000, 0.3000, 0.5000],
        "thumb":  [0.6900, 0.4900, 1.0000, 0.7900],
    },

}

# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 로봇 A/B 배치                                               ║
# ╚══════════════════════════════════════════════════════════════╝
# 로봇이 USD 씬에서 어디에 있는지(prim 경로), 어디에 세워질지(world 위치/회전)를 정의한다.
ROBOT_A_PRIM_PATH    = "/World/m0609_A"                      # USD 안에서 A 로봇을 가리키는 경로
ROBOT_A_BASE_WORLD   = np.array([0.8, 0.0, 1.0], dtype=float)  # A 로봇 받침대의 world 좌표(x,y,z)
ROBOT_A_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)  # A 로봇 받침대의 회전(오일러각, 도)
ROBOT_A_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_A_BASE_EULER_DEG))  # 위 회전을 쿼터니언으로

ROBOT_B_PRIM_PATH    = "/World/m0609_B"
ROBOT_B_BASE_WORLD   = np.array([0.0, 0.0, 1.0], dtype=float)
ROBOT_B_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)
ROBOT_B_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_B_BASE_EULER_DEG))

# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 씬 오브젝트 위치 / 이름                                     ║
# ╚══════════════════════════════════════════════════════════════╝
# 아래 세 좌표는 "최초 설계값"으로 둔 하드코딩 상수다. 실제로는 USD 씬에서 prim 이름으로
# 찾은 실제 world 위치를 우선 사용하고(MultiRobotOilingTask._resolve_world_position 참고),
# 혹시 prim을 못 찾았을 때만 이 값으로 대체(fallback)한다.
FUEL_DOOR_CENTER      = np.array([-0.40267, -0.81071, 1.02293], dtype=float)  # 주유구 커버 중심
FUEL_CAP_CENTER       = np.array([-0.40301, -0.89268, 1.06441], dtype=float)  # 마개 중심
FUEL_PORT_HOLE_CENTER = np.array([-0.39580, -1.11245, 1.00525], dtype=float)  # 주유구 입구(구멍) 중심

FUEL_DOOR_PRIM_NAME = "fuel_door"            # USD 씬에서 커버 prim을 찾을 때 쓰는 이름
FUEL_CAP_PRIM_NAME = "fuel_cap"              # USD 씬에서 마개 prim을 찾을 때 쓰는 이름
FUEL_PORT_HOLE_PRIM_NAME = "fuel_port_hole"  # USD 씬에서 주유구 구멍 prim을 찾을 때 쓰는 이름
SCENE_SEARCH_ROOT = "/World"                  # 위 prim들을 찾기 시작할 루트 경로 (이 아래를 전부 탐색)

# 차체(car_visual)와 ArUco 마커(aruco_vehicle_marker)를 한 덩어리로 묶어둔 부모 prim 이름.
# 이 prim 하나만 옮기면 차체+마커가 항상 같이 움직인다(CarArrivalController).
CAR_ROOT_PRIM_NAME = "car"
CAR_ARRIVAL_START_OFFSET = np.array([9.5, 0.0, 0.0], dtype=float)  # 출발 지점 = 도착 지점 + 이 오프셋(m)
CAR_ARRIVAL_SPEED_MPS = 2.0      # 차가 들어오는 속도(m/s). 10m 기준 약 5초
CAR_ARRIVAL_TOLERANCE_M = 0.01   # 도착 판정 허용 오차(m)
# 매 Play마다 도착 지점의 x,y에 주는 랜덤 오차 범위(+-, m). 항상 똑같은 자리에 서지 않고 매번
# 조금씩 다르게 주차되도록 해서, ArUco 인식이 고정된 좌표가 아니라 실제 측정값을 따라가는지
# 일반화 테스트를 할 수 있게 한다. SEARCH_GATE_HALF_EXTENT(아래, x/y=0.35)보다 충분히 작아야
# lock 게이트가 정상적으로 통과한다.
CAR_PARK_XY_NOISE_M = 0.00

# 벽면 기준 바깥 방향 / 삽입 방향. door/cap/hole이 같은 차체 벽면에 있다고 가정하고 공유한다.
# 실제 USD 배치가 다르면 이 두 값만 조정하면 된다.
_OUTWARD_ANGLE_DEG = 105.0
# 벽면이 정확히 수직(90도)이 아니라 약간 기울어 있다고 보고, 그 기울기를 반영해
# "벽에서 바깥쪽으로 멀어지는 방향" 단위벡터를 만든다.
PORT_OUTWARD_NORMAL = np.array(
    [0.0, np.sin(np.deg2rad(_OUTWARD_ANGLE_DEG)), -np.cos(np.deg2rad(_OUTWARD_ANGLE_DEG))], dtype=float
)
INSERTION_DIRECTION = -PORT_OUTWARD_NORMAL  # 삽입 방향은 바깥 방향의 정반대(안쪽으로 들어가는 방향)

FUEL_PORT_DEPTH = 0.10               # 주유구 구멍 깊이(m)
INSERT_DISTANCE = FUEL_PORT_DEPTH / 2  # 실제로 노즐/마개를 얼마나 깊이 밀어넣을지(절반만)

VIRTUAL_NOZZLE_LENGTH = 0.63     # 로봇 손목(link_6)에서 노즐 끝까지의 가상 길이(m)
VIRTUAL_NOZZLE_Z_OFFSET = -0.25  # 노즐 길이를 고려해 손목 목표점을 z방향으로 보정하는 오프셋

# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 제어 파라미터 (prompt 지정값)                                ║
# ╚══════════════════════════════════════════════════════════════╝
PHYSICS_DT = 1.0 / 60.0       # 물리 시뮬레이션 한 스텝의 시간(초). 60Hz -> 약 16.7ms마다 한 번
POSITION_TOLERANCE = 0.10     # "목표 위치에 도착했다"고 판단할 기본 허용 오차(m)
MAX_STEPS_PER_STAGE = 300     # 한 단계(stage)가 이 스텝 수를 넘으면 타임아웃으로 보고 강제로 넘어감 (~10초 @ 60fps, 디버깅 속도 우선)

# 손목(EE) 목표를 향해 이동할 때의 속도(m/s). 상황별로 다른 속도를 쓴다.
DEFAULT_TARGET_SPEED = 0.360   # 평범하게 이동할 때
NEAR_TARGET_SPEED    = 0.240   # 목표에 가까워졌을 때 (더 정밀하게, 천천히)
INSERT_TARGET_SPEED  = 0.120   # 구멍에 삽입할 때 (충돌 위험 있어 가장 느리게)
RETREAT_TARGET_SPEED = 0.300   # 빼거나 후퇴할 때

HOME_JOINT_SPEED_ALPHA = 0.024  # 기본 자세로 복귀할 때, 현재값과 목표값 차이의 몇 %씩 매 스텝 움직일지
HOME_JOINT_TOLERANCE   = 0.05   # 기본 자세에 "도착했다"고 볼 조인트 각도 오차(rad)
HOME_HOLD_STEPS        = 40     # 도착 판정 후에도 이만큼의 스텝 동안 안정적으로 유지되어야 진짜 완료로 인정

# 주유구/마개/구멍에 접근할 때 거리 단계 (멀리서 -> 중간 -> 가까이 순서로 단계적으로 다가감)
FAR_DISTANCE  = 0.28
MID_DISTANCE  = 0.18
NEAR_DISTANCE = 0.09

COVER_CLEARANCE_DISTANCE = 0.55  # 마개로 가기 전에 들르는 경유점의 벽 바깥쪽 거리(m) - 열린 커버를 피해서 돌아가기 위함
                                 # 0.35였을 때 이 경유점을 지나가는 중 손이 열린 덮개를 뚫고 지나가는 문제가
                                 # 있어 로봇 쪽(바깥쪽)으로 더 당겼다. 그래도 뚫리면 이 값을 더 키운다.

GRIPPER_LENGTH_B = 0.32  # B 로봇 그리퍼 길이(m) - 손목(link_6)에서 손가락 끝까지 거리.
                         # 마개에 접근할 때 손목이 아니라 "손가락 끝"이 표면에 닿아야 하므로,
                         # 손목 목표점은 이 길이만큼 표면보다 바깥쪽에 둬야 그리퍼가 차체를 파고들지 않는다.

# 커버(fuel_door)가 열리는 각도 단계: 30도(닫힘 기준) -> 120도(완전열림).
# 여는 동작은 USD 씬에 이미 설정된 velocity 드라이브(stiffness=0, damping>0, targetVelocity>0)와
# RevoluteJoint limit(lower=0, upper=120)이 "열림 방향으로 계속 밀다가 upper limit에 닿으면
# 멈추는" 방식으로 자동 처리한다 - 코드는 이 drive/limit 값 자체를 다시 만들거나 덮어쓰지 않는다.
# 닫는 동작(CLOSE_COVER)은 로봇이 실제로 밀면서, 그 진행률에 맞춰 덮개 각도를
# 120 -> 75 -> 30도(=COVER_START_DEG, 닫힘 기준)로 같이 갱신한다 (set_door_angle_deg).
# 단, 이 velocity 드라이브는 항상 "열림" 방향으로 힘을 내고 있어서 CLOSE_COVER 동안/이후에도
# 덮개를 도로 열려고 계속 밀기 때문에, CLOSE_COVER 시작 시점에 targetVelocity를 0으로 꺼서
# (DOOR_AUTO_OPEN_DRIVE_VELOCITY -> 0) 더 이상 못 열게 막고, 다음 Play/리셋(on_play_reset)에서
# 다시 DOOR_AUTO_OPEN_DRIVE_VELOCITY로 복원해 자동 열기가 그대로 동작하게 한다.
COVER_START_DEG = 0.0
COVER_OPEN_DEG  = 130.0
COVER_FULL_CLOSE_DEG = 0.0  # 30도까지 1차로 닫은 뒤, 중앙으로 옮겨 마지막으로 완전히 미는 목표 각도
DOOR_ANGLE_TOLERANCE_DEG = 4.0  # 목표 도어 각도에 도착했다고 볼 허용 오차(도)
DOOR_AUTO_OPEN_DRIVE_VELOCITY = 10.0  # USD에 이미 설정된 덮개 자동 열기 velocity 드라이브의 targetVelocity 값

# current_door_angle_deg()가 일부 씬(힌지에 실제 회전 limit이 없어 180도를 넘어가면
# axis-angle 표현이 wrap되어 음수로 튀는 등)에서 신뢰할 수 없는 값을 줘서, 각도 기반 정체
# 감지가 오작동하는 경우가 있었다. 그래서 각도를 보는 대신 그냥 고정 시간만큼 기다린 뒤
# "다 열렸다"고 가정하고 다음 단계(ArUco door lock)로 넘어간다.
DOOR_OPEN_FIXED_WAIT_SECONDS = 5.0
DOOR_OPEN_WAIT_STEPS = int(DOOR_OPEN_FIXED_WAIT_SECONDS / PHYSICS_DT)  # 5초 @ 60Hz = 300스텝

# build_close_cover_sequence의 각 단계 목표점에 world x/y/z로 직접 더하는 보정값.
# PORT_OUTWARD_NORMAL_UNIT(x성분이 항상 0)만으로는 x축 방향 깊이를 조절할 수 없어서 둔다.
DOOR_TOUCH_OFFSET = np.array([0.0, 0.15, 0.0], dtype=float)
DOOR_CLOSE_PUSH_OFFSET = 0.15  # 닫는 동작에서 door_world_position 기준 x축으로 더하는 오프셋(m)
                               # - 여는 동작(반대쪽 접근)과 다른 쪽에서 밀어야 해서 부호가 다르다.

# 30도까지 닫은 뒤 손을 한번 빼서(retreat) 도어 중앙으로 정렬(center_align)한 다음,
# 그 중앙 지점을 그대로 0도(완전 닫힘)까지 직선으로 쭉 밀어주는 마무리 단계용 거리(m).
DOOR_FULL_CLOSE_RETREAT_DISTANCE = 0.15

# build_close_cover_sequence의 p()/p_center() 목표점은 원래 DOOR_TOUCH_OFFSET(고정 0.15m)만
# 보정으로 들어가 있어서, 그리퍼(EE)가 차체 안으로 파고들어가는 문제가 있었다(B10_01~06 전 구간).
# PORT_OUTWARD_NORMAL_UNIT 방향으로 이만큼 추가로 바깥쪽에 목표점을 둔다.
DOOR_CLOSE_OUTWARD_CLEARANCE = 0.14

# fuel_door를 130도까지 더 열어둔 상태에 맞춘 닫기 접근 보정.
# cap 복원 후 바로 문을 밀지 않고, 한 번 뒤로 빠진 상태에서 오른쪽(+x)으로 이동한 뒤
# 앞쪽(door 접촉점)으로 들어가고, 그 다음 왼쪽 방향으로 쓸어 밀며 닫는다.
# 좌/우 방향이 화면 기준과 반대로 보이면 DOOR_CLOSE_RIGHT_SHIFT_M의 부호만 바꿔서 테스트한다.
DOOR_CLOSE_RIGHT_SHIFT_M = -0.15
DOOR_CLOSE_FORWARD_BACKOFF_M = 0.05

# 힌지 축 방향 부호가 USD/PhysX 쪽과 반대로 측정될 수 있다.
# 리셋 직후 로그의 "door angle"이 COVER_START_DEG와 다르게 튀면 -1.0으로 바꿔서 맞춘다.
DOOR_ANGLE_SIGN = 1.0

# 카메라로 lock한(또는 timeout 시 하드코딩 기준값인) 마개 위치가 실제 마개 중심과 살짝
# 어긋나서 grasp/insert가 부정확할 때, world x/y/z로 직접 더해 보정하는 값(m).
# locked_cap_center에 한 번만 더해지므로 마개를 열 때(B5)와 닫을 때(B9) 모두 동일하게 적용된다.
CAP_POSITION_OFFSET = np.array([0.0, 0.0, -0.04], dtype=float)

# 마개를 풀고(unscrew) 다시 조일 때(screw) joint_6을 얼마나 회전시킬지
CAP_JOINT6_UNSCREW_DEG = -360.0   # 마개 풀기: -360도(한 바퀴 반대로)
CAP_JOINT6_SCREW_DEG   = 360.0    # 마개 조이기: +360도(한 바퀴)
CAP_JOINT6_DEG_PER_STEP = 4.0     # 한 스텝마다 최대 몇 도씩 회전시킬지 (너무 빨리 돌면 부자연스러움)

# joint_6 직접 제어(rotate/screw sub_phase)용 라디안 버전. RMPFlow를 거치지 않고 매 스텝
# joint_6에만 이 스텝만큼을 더해 회전시키고, accumulated가 TOTAL의 절댓값에 도달하면 멈춘다.
UNSCREW_ANGLE_STEP_RAD = -np.deg2rad(CAP_JOINT6_DEG_PER_STEP)  # 한 스텝당 회전량(라디안). 음수=푸는 방향
UNSCREW_TOTAL_ANGLE_RAD = np.deg2rad(CAP_JOINT6_UNSCREW_DEG)   # 풀기 총 회전량(라디안) = -2*pi

EXTRACT_TOTAL_STEPS = 60  # GRIP_UNSCREW.extract: 현재 EE 위치 기준 0.20m 후퇴를 이 스텝 수에 걸쳐 선형 보간

GRIPPER_ACTION_HOLD_STEPS = 30  # 그리퍼를 열거나 닫는 동작이 "완료됐다"고 보기까지 유지할 스텝 수

USE_TARGET_ORIENTATION = True  # RMPFlow에 위치만 줄지, 방향(orientation)까지 같이 고정해서 줄지

# True면 MOVE_TO_CAP/RUN_SEQUENCE에서 WaypointSequence 대신, 뷰포트에서 직접 옮길 수
# 있는 마커 prim의 현재 world 위치를 매 스텝 RMPFlow 목표로 사용한다 (수동 디버깅/튜닝용).
# (덮개 열기는 코드가 구동하지 않고 USD 씬 설정으로 시작하므로 door_push 마커는 더 이상 제어에 안 쓰임)
USE_MARKER_CONTROL = False
MARKER_PRIM_PATHS = {
    "door_push":    "/World/marker_door_push",
    "cap_approach": "/World/marker_cap_approach",
    "fuel_port":    "/World/marker_fuel_port",
}

# 벽에 붙은 카메라 prim을 USD에서 찾을 때 시도해볼 경로 후보 목록 (위에서부터 순서대로 확인)
CAMERA_PRIM_CANDIDATES = [
    "/World/wall/rsd455/RSD455",
    "/World/wall/rsd455/Camera",
    "/World/wall/rsd455/camera",
]
REQUIRE_TARGET_LOCK = True                 # True면 "lock=True"가 아닌 pose는 아예 사용하지 않음
CONTROLLER_REQUIRED_LOCK_SAMPLES = 5       # 안정적이라고 판단하기 위해 모아야 하는 최소 샘플 개수
CONTROLLER_WORLD_STD_TOLERANCE = 0.025     # 모은 샘플들의 흔들림(표준편차) 허용치(m)
SEARCH_GATE_HALF_EXTENT = np.array([0.35, 0.35, 0.18], dtype=float)  # 기준 위치 대비 허용 오차 범위(게이트)
WAIT_LOCK_TIMEOUT_STEPS = 600                # 이 스텝 수가 지나도 lock이 안 되면 하드코딩 기준값으로 폴백 (~10초 @ 60fps)

# ╔══════════════════════════════════════════════════════════════╗
# ║  E. ROS2 토픽 이름                                              ║
# ╚══════════════════════════════════════════════════════════════╝
TOPIC_ARUCO_POSE = "/aruco_detector/pose"           # 카메라가 찾은 ArUco 표적의 3D 위치
TOPIC_ARUCO_LOCK = "/aruco_detector/target_locked"  # 그 위치가 "안정적으로 확정됐는지" 여부
TOPIC_MODE_SWITCH = "/aruco_detector/mode_switch"   # 카메라에게 "이제 이 표적을 찾아" 명령
TOPIC_ROBOT_A_DONE = "/robot_a/done"                # A 로봇이 자기 작업을 끝냈다는 신호
TOPIC_ROBOT_B_DONE = "/robot_b/done"                # B 로봇이 자기 작업을 끝냈다는 신호


# ============================================================
# 유틸
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """벡터를 길이 1인 단위벡터로 만든다. 길이가 거의 0이면(eps보다 작으면) 0벡터를 반환해서 0으로 나누는 에러를 피한다."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


# 모듈이 로드될 때 한 번만 계산해두는 단위벡터들 (매번 normalize를 다시 부를 필요 없게 캐싱).
PORT_OUTWARD_NORMAL_UNIT = normalize(PORT_OUTWARD_NORMAL)
INSERTION_DIRECTION_UNIT = normalize(INSERTION_DIRECTION)


def make_outward_point(center: np.ndarray, distance: float) -> np.ndarray:
    """어떤 중심점(center)에서 벽 바깥쪽 방향으로 distance(m)만큼 떨어진 점을 계산한다.
    (로봇이 마개/구멍에 부딫히지 않고 그 앞에서 멈춰야 할 때 쓰는 "접근 대기 지점"을 만드는 용도)"""
    return center + PORT_OUTWARD_NORMAL_UNIT * distance


def find_prim_path_by_name(root_path: str, name: str):
    """root_path 아래를 전부 훑어서(재귀 탐색) 이름이 name과 정확히 같은 prim의 경로를 찾는다.
    못 찾으면 None을 반환한다. (예: "/World" 아래에서 "fuel_door"라는 이름의 prim 찾기)"""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def get_prim_world_position(prim_path: str):
    """어떤 prim의 "world 좌표계 기준" 위치(x,y,z)를 구한다.
    prim은 보통 부모 prim 기준 local 위치만 가지고 있어서, XformCache로 부모->World까지의
    모든 변환을 누적해 곱해야 진짜 world 위치가 나온다."""
    if not prim_path:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def get_prim_world_rotation(prim_path: str):
    """get_prim_world_position과 같은 방식으로, 위치 대신 world 기준 "회전"을 구한다."""
    if not prim_path:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    return mat.ExtractRotation()


def get_prim_world_matrix(prim_path: str):
    """prim의 world transform matrix를 반환한다. fuel_door를 힌지 기준으로 직접 회전시킬 때 사용한다."""
    if not prim_path:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    return cache.GetLocalToWorldTransform(prim)


def _matrix_translate(vec: np.ndarray):
    """Gf.Matrix4d translation matrix 생성용 헬퍼."""
    m = Gf.Matrix4d(1.0)
    m.SetTranslate(Gf.Vec3d(float(vec[0]), float(vec[1]), float(vec[2])))
    return m


def _matrix_rotate_axis(axis: np.ndarray, angle_deg: float):
    """world axis 기준 회전 matrix 생성용 헬퍼."""
    axis = normalize(np.array(axis, dtype=float))
    m = Gf.Matrix4d(1.0)
    if np.linalg.norm(axis) < 1e-9:
        return m
    m.SetRotate(Gf.Rotation(Gf.Vec3d(float(axis[0]), float(axis[1]), float(axis[2])), float(angle_deg)))
    return m


def set_prim_world_matrix(prim_path: str, target_world_matrix) -> bool:
    """prim의 local xform을 target world matrix와 일치하도록 직접 설정한다.

    RevoluteJoint의 state 값만 바꾸면 화면상 fuel_door transform이 즉시 갱신되지 않는 경우가 있다.
    그래서 닫는 동안 fuel_door prim transform도 매 스텝 직접 갱신해서 로봇 모션과 같이 자연스럽게
    닫히는 것처럼 보이게 한다.
    """
    if not prim_path or target_world_matrix is None:
        return False
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False

    cache = UsdGeom.XformCache()
    parent = prim.GetParent()
    if parent and parent.IsValid():
        parent_world = cache.GetLocalToWorldTransform(parent)
        # USD/Gf는 row-vector convention을 사용하므로 local = world * inverse(parent_world)
        local_matrix = target_world_matrix * parent_world.GetInverse()
    else:
        local_matrix = target_world_matrix

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(local_matrix)
    return True


def set_door_prim_visual_angle_deg(
    door_prim_path: str,
    closed_world_matrix,
    pivot_world: np.ndarray,
    axis_world: np.ndarray,
    visual_angle_deg: float,
    label: str = "fuel_door",
    verbose: bool = True,
) -> bool:
    """fuel_door prim 자체를 닫힘 기준 transform에서 원하는 시각 각도로 직접 회전시킨다.

    COVER_START_DEG는 닫힌 상태를 30도로 표현한 값이므로 실제 회전량은
    visual_angle_deg - COVER_START_DEG이다."""
    if closed_world_matrix is None:
        if verbose:
            print(f"[DOOR][WARN] {label} 기준 world matrix가 없어 prim 직접 회전을 건너뜀", flush=True)
        return False

    joint_delta_deg = float(visual_angle_deg - COVER_START_DEG)
    pivot_world = np.array(pivot_world, dtype=float)
    axis_world = normalize(np.array(axis_world, dtype=float))

    # row-vector 기준: p' = ((p - pivot) * R) + pivot 이므로 M = T(-pivot) * R * T(pivot)
    around_hinge = (
        _matrix_translate(-pivot_world)
        * _matrix_rotate_axis(axis_world, joint_delta_deg)
        * _matrix_translate(pivot_world)
    )
    target_world_matrix = closed_world_matrix * around_hinge
    ok = set_prim_world_matrix(door_prim_path, target_world_matrix)
    if verbose:
        if ok:
            print(f"[DOOR] {label} prim transform 직접 적용: visual_angle={visual_angle_deg:.1f}deg "
                  f"(delta={joint_delta_deg:.1f}deg)")
        else:
            print(f"[DOOR][WARN] {label} prim transform 직접 적용 실패", flush=True)
    return ok


def find_dof_index(robot, dof_name: str):
    """로봇의 조인트(DOF) 이름으로 그 조인트가 몇 번째 인덱스인지 찾는다.
    get_joint_positions()가 반환하는 배열에서 어느 위치가 어느 조인트인지 알아야
    joint_6처럼 "딱 하나의 조인트만" 골라서 제어할 수 있다."""
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        return robot.dof_names.index(dof_name)
    return None


def find_first_dof_index(robot, candidate_names: list[str]):
    """후보 이름 목록 중 실제 robot.dof_names에 존재하는 첫 DOF를 찾는다."""
    if not hasattr(robot, "dof_names"):
        return None, None
    for name in candidate_names:
        if name in robot.dof_names:
            return robot.dof_names.index(name), name
    return None, None


def flatten_allegro_pose(pose_name: str) -> np.ndarray:
    """ALLEGRO_POSES의 index/middle/ring/thumb 값을 16개 배열로 펼친다."""
    pose = ALLEGRO_POSES.get(pose_name, ALLEGRO_POSES["open"])
    values = []
    for key in ("index", "middle", "ring", "thumb"):
        values.extend(pose.get(key, [0.0, 0.0, 0.0, 0.0]))
    return np.array(values, dtype=float)


def find_b_allegro_finger_joint_indices(robot) -> list[int]:
    """m0609_B Allegro finger DOF를 pose 배열 순서와 정확히 맞춰 찾는다.

    이전 방식은 robot.dof_names에 나온 순서를 그대로 사용해서,
    index/middle/ring에 같은 값을 넣어도 실제 적용 대상이 섞일 수 있었다.
    여기서는 이미지에 보이는 명칭 기준으로 반드시 아래 순서를 만든다.

    index_joint_0..3,
    middle_joint_0..3,
    ring_joint_0..3,
    thumb_joint_0..3
    """
    if not hasattr(robot, "dof_names"):
        return []

    ordered_names = []
    for finger in ("index", "middle", "ring", "thumb"):
        for j in range(4):
            ordered_names.append(f"{finger}_joint_{j}")

    indices = []
    missing = []
    for name in ordered_names:
        idx = find_dof_index(robot, name)
        if idx is None:
            missing.append(name)
        else:
            indices.append(idx)

    if missing:
        print(f"[B][ALLEGRO][WARN] finger DOF missing={missing}", flush=True)
        print(f"[B][ALLEGRO][DIAG] available dof_names={getattr(robot, 'dof_names', [])}", flush=True)

    return indices



def build_initial_joint_positions(robot, base_positions=None, arm_joint_deg: dict = INITIAL_ARM_JOINT_DEG_B) -> np.ndarray:
    """로봇의 "기본 자세(home pose)"에 해당하는 전체 조인트 각도 배열을 만든다.
    arm_joint_deg(로봇 A/B마다 다름)/INITIAL_GRIPPER_JOINTS에 정의된 이름으로 값을 채워 넣되,
    이름으로 못 찾으면 순서(인덱스)로 폴백한다."""
    if base_positions is None:
        q = np.zeros(robot.num_dof, dtype=float)
    else:
        q = np.array(base_positions, dtype=float).copy()
        if len(q) != robot.num_dof:
            q = np.zeros(robot.num_dof, dtype=float)

    missing_arm = []
    for joint_name, deg in arm_joint_deg.items():
        idx = find_dof_index(robot, joint_name)
        if idx is None:
            missing_arm.append(joint_name)
        else:
            q[idx] = np.deg2rad(deg)  # 각도는 도 단위로 적었지만 USD/물리엔진은 라디안을 쓰므로 변환

    if missing_arm and robot.num_dof >= 6:
        # 이름으로 매칭이 안 된 팔 조인트가 있으면, 마지막 수단으로 "앞에서부터 순서대로" 채운다.
        fallback_values = list(arm_joint_deg.values())
        for i, deg in enumerate(fallback_values):
            q[i] = np.deg2rad(deg)

    for joint_name, value in INITIAL_GRIPPER_JOINTS.items():
        idx = find_dof_index(robot, joint_name)
        if idx is not None:
            q[idx] = value

    return q


def apply_robot_start_state(robot, base_world: np.ndarray, base_orientation: np.ndarray, arm_joint_deg: dict):
    """로봇을 지정한 world 위치/방향에 세우고, 조인트를 기본 자세로 맞춘다."""
    robot.set_world_pose(position=base_world, orientation=base_orientation)
    current = robot.get_joint_positions()
    q0 = build_initial_joint_positions(robot, current, arm_joint_deg)
    robot.set_joint_positions(q0)
    return q0


def initialize_robot(robot, world, base_world: np.ndarray, base_orientation: np.ndarray, arm_joint_deg: dict):
    """로봇 articulation과 그리퍼를 초기화하고, 시작 위치/자세/그리퍼 상태까지 한 번에 셋업한다.
    매 reset마다(시뮬레이션 재시작마다) 이 함수를 다시 호출해 깨끗한 상태로 되돌린다."""
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    q0 = apply_robot_start_state(robot, base_world, base_orientation, arm_joint_deg)
    robot.gripper.set_joint_positions(np.array(GRIPPER_OPEN, dtype=float))
    return q0


def initialize_robot_b(robot, world, base_world: np.ndarray, base_orientation: np.ndarray, arm_joint_deg: dict):
    """m0609_B 전용 초기화.

    B가 Allegro hand/wrist 모델로 교체된 경우 ParallelGripper가 없을 수 있으므로,
    A용 initialize_robot()는 그대로 두고 B만 안전하게 초기화한다.
    """
    robot.initialize()
    if getattr(robot, "gripper", None) is not None:
        robot.gripper.initialize(
            physics_sim_view=world.physics_sim_view,
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
    q0 = apply_robot_start_state(robot, base_world, base_orientation, arm_joint_deg)
    if getattr(robot, "gripper", None) is not None:
        robot.gripper.set_joint_positions(np.array(GRIPPER_OPEN, dtype=float))
    return q0


def step_home_return(robot, target_joint_positions: np.ndarray,
                      alpha: float = HOME_JOINT_SPEED_ALPHA, tol: float = HOME_JOINT_TOLERANCE) -> bool:
    """로봇을 목표 조인트 각도(보통 기본 자세)로 한 스텝씩 천천히 이동시킨다.
    alpha는 "현재값과 목표값 차이의 몇 %씩 이번 스텝에 움직일지" 비율 -> 지수적으로 감속하며 부드럽게 도착.
    반환값: 팔의 앞 6개 조인트가 목표에 충분히(tol 이내) 가까워졌으면 True."""
    current = robot.get_joint_positions()
    next_joints = current + alpha * (target_joint_positions - current)
    robot.set_joint_positions(next_joints)
    joint_err = np.linalg.norm(next_joints[:6] - target_joint_positions[:6])
    return bool(joint_err < tol)


# ============================================================
# 카메라 좌표 변환 (단일 벽 부착 카메라를 A/B가 공유)
# ============================================================
def find_camera_prim_path():
    """벽에 붙은 카메라(rsd455)의 USD prim 경로를 찾는다.
    우선 CAMERA_PRIM_CANDIDATES에 적어둔 후보 경로들을 순서대로 확인하고,
    하나도 안 맞으면 /World 아래를 전부 훑어서 타입이 Camera이거나 이름이 "camera"/"rsd455"인
    prim을 찾는다. 끝까지 못 찾으면 None (이 경우 이후 모든 좌표 변환이 동작하지 않는다)."""
    stage = omni.usd.get_context().get_stage()
    for path in CAMERA_PRIM_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path
    root = stage.GetPrimAtPath(SCENE_SEARCH_ROOT)
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Camera" or prim.GetName().lower() in ["camera", "rsd455"]:
                return str(prim.GetPath())
    return None


def transform_camera_point_to_world(point_camera_ros: np.ndarray, camera_prim_path: str):
    """카메라가 보고하는 "카메라 기준 3D 좌표(x,y,z)"를 받아서 "world 기준 3D 좌표"로 바꾼다.

    ArUco detector 노드(aruco_marker_detector.py)가 보내는
    pose.position은 카메라 로컬 좌표계 기준값이므로,
    여기서 (1) 카메라 자신이 world의 어디에 있는지 구하고 (2) ROS 좌표축 -> world 좌표축으로
    바꾼 변위(delta)를 더해서 최종 world 좌표를 만든다."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    camera_origin_world = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)

    x_cam = float(point_camera_ros[0])
    y_cam = float(point_camera_ros[1])
    z_cam = float(point_camera_ros[2])

    # 카메라가 벽에 붙어 아래를 내려보는 배치이기 때문에 축이 단순 대응되지 않고
    # (x,y,z)_camera -> (-x, -z, -y)_world 형태로 매핑된다 (실제 씬 배치에 맞춰 캘리브레이션된 값).
    delta_world = np.array([-x_cam, -z_cam, -y_cam], dtype=float)
    return camera_origin_world + delta_world


def detected_world_point_to_mouth_center(detected_world_point: np.ndarray) -> np.ndarray:
    """카메라가 본 건 구멍의 "입구 표면"인데, 실제 목표로 삼고 싶은 건 그보다 약간 안쪽인
    "구멍의 중심"이다. 바깥 방향 단위벡터를 빼서 깊이의 절반만큼 안쪽으로 보정한다."""
    return detected_world_point - PORT_OUTWARD_NORMAL_UNIT * (FUEL_PORT_DEPTH / 2.0)


def validate_detected_center_world(center_world: np.ndarray, reference_center: np.ndarray,
                                    gate_half_extent: np.ndarray = SEARCH_GATE_HALF_EXTENT):
    """카메라가 찾은 좌표가 "말이 되는 위치"인지 검사한다(게이트 체크).
    NaN/inf처럼 숫자가 깨졌으면 즉시 reject. 그리고 기준 위치(reference_center)에서
    각 축(x,y,z)별로 gate_half_extent를 넘게 벗어나면 오탐(다른 물체를 잘못 본 것)으로 보고 reject한다.
    이 게이트가 너무 좁으면 진짜 표적도 계속 reject되고, 너무 넓으면 오탐도 통과되니
    기준 위치(reference_center)가 실제 씬과 맞는 게 가장 중요하다."""
    if center_world is None or not np.all(np.isfinite(center_world)):
        return False, "non-finite target"
    delta = center_world - reference_center
    if np.any(np.abs(delta) > gate_half_extent):
        return False, f"outside gate: delta={np.round(delta, 3)}"
    return True, f"inside gate: delta={np.round(delta, 3)}"


# ============================================================
# 마개 표시/숨김: 물리적으로 EE에 붙이는 대신, 잡고 있는 동안은 시각적으로만 숨겼다가
# 다시 끼울 때 보여준다 (FixedJoint로 물리 부착하던 방식은 마개가 따라오지 않는 문제가 있어 제거).
# ============================================================
def set_prim_visibility(prim_path: str, visible: bool):
    """USD prim의 표시 여부를 켠다/끈다. 그리퍼가 마개를 잡고 도는 동안은 숨겨서 "손에 들고
    사라진 것처럼" 보이게 하고, 다시 끼울 때 보이게 해서 "마개가 복원된 것처럼" 표현한다."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
        print(f"[VISIBLE] {prim_path} 표시", flush=True)
    else:
        imageable.MakeInvisible()
        print(f"[HIDDEN] {prim_path} 숨김", flush=True)


# ============================================================
# 도어 힌지 geometry: Revolution Joint의 축/피벗을 USD에서 읽어온다
# ============================================================
def find_revolute_joint_for_body(body_prim_path: str):
    """씬 전체를 훑어서, body0 또는 body1이 body_prim_path(=fuel_door)를 가리키는
    RevoluteJoint(회전 관절) prim을 찾는다. 이게 바로 커버가 매달려 돌아가는 "힌지"다."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(SCENE_SEARCH_ROOT)
    if not root.IsValid() or not body_prim_path:
        return None
    for prim in Usd.PrimRange(root):
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue
        targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
        if any(str(t) == body_prim_path for t in targets):
            return prim
    return None


def get_joint_world_axis_and_pivot(joint_prim):
    """RevoluteJoint의 axis/localPos0를 body0 world transform 기준으로 변환한다."""
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    axis_token = joint.GetAxisAttr().Get() or "X"
    local_axis = {
        "X": Gf.Vec3d(1, 0, 0),
        "Y": Gf.Vec3d(0, 1, 0),
        "Z": Gf.Vec3d(0, 0, 1),
    }.get(str(axis_token), Gf.Vec3d(1, 0, 0))

    body0_targets = joint.GetBody0Rel().GetTargets()
    stage = omni.usd.get_context().get_stage()
    if body0_targets:
        body0_prim = stage.GetPrimAtPath(body0_targets[0])
        cache = UsdGeom.XformCache()
        body0_world = cache.GetLocalToWorldTransform(body0_prim)
    else:
        body0_world = Gf.Matrix4d(1.0)

    local_pos0 = joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
    local_rot0 = joint.GetLocalRot0Attr().Get()

    if local_rot0 is not None:
        rot = Gf.Rotation(Gf.Quatd(local_rot0.GetReal(), Gf.Vec3d(local_rot0.GetImaginary())))
        axis_in_body0 = rot.TransformDir(local_axis)
    else:
        axis_in_body0 = local_axis

    axis_world_gf = body0_world.TransformDir(axis_in_body0)
    axis_world = normalize(np.array([axis_world_gf[0], axis_world_gf[1], axis_world_gf[2]], dtype=float))

    pivot_world_gf = body0_world.Transform(Gf.Vec3d(local_pos0))
    pivot_world = np.array([pivot_world_gf[0], pivot_world_gf[1], pivot_world_gf[2]], dtype=float)
    return axis_world, pivot_world


def rotate_point_around_axis(point: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """한 점(point)을 피벗(pivot)을 중심으로 axis 축 둘레로 angle_deg도만큼 회전시킨 새 위치를 구한다.
    커버가 30->75->120도로 열릴 때, "그 각도에서 로봇 손이 닿아야 할 위치"를 미리 계산하는 데 쓴다."""
    rot = Gf.Rotation(Gf.Vec3d(*[float(a) for a in axis]), float(angle_deg))
    rel = Gf.Vec3d(*[float(p) for p in (point - pivot)])  # 피벗을 원점으로 옮긴 상대좌표에서 회전
    rotated_rel = rot.TransformDir(rel)
    return pivot + np.array([rotated_rel[0], rotated_rel[1], rotated_rel[2]], dtype=float)


def signed_angle_about_axis_deg(rest_rotation, current_rotation, axis_world: np.ndarray) -> float:
    """리셋 시점의 회전(rest_rotation)과 지금 회전(current_rotation)의 차이를 구해서,
    "기준 자세에서 몇 도 회전했는지"를 +/- 부호가 있는 각도로 반환한다.
    부호는 회전축이 axis_world와 같은 방향이면 +, 반대 방향이면 -로 정한다.
    DOOR_ANGLE_SIGN은 USD/PhysX가 측정하는 부호와 우리가 원하는 부호가 반대일 때 뒤집는 보정값."""
    if rest_rotation is None or current_rotation is None:
        return 0.0
    # 두 회전의 "차이"는 (현재 회전) * (기준 회전의 역) 으로 구한다 (쿼터니언/회전 합성 공식).
    rel = current_rotation * rest_rotation.GetInverse()
    angle = float(rel.GetAngle())
    axis = rel.GetAxis()
    axis_np = np.array([axis[0], axis[1], axis[2]], dtype=float)
    if np.linalg.norm(axis_np) < 1e-6:
        # 회전량이 거의 0이면 회전축 자체가 정의되지 않으므로(0벡터) 그냥 각도 0으로 처리.
        return 0.0
    sign = 1.0 if np.dot(axis_np, axis_world) >= 0.0 else -1.0
    return DOOR_ANGLE_SIGN * sign * angle


# ============================================================
# Stage / WaypointSequence: 속도 제한 RMPFlow 목표점 state machine
# ============================================================
@dataclass
class FuelStage:
    """하나의 "이동 목표 단계"를 표현하는 데이터 묶음.
    예: "마개에서 0.28m 떨어진 곳까지 천천히 이동, 도착하면 그걸로 끝" 같은 한 단계가 FuelStage 하나."""
    name: str                                  # 이 단계의 고유 이름 (로그 출력/식별용)
    target_position: "np.ndarray | None"       # 도착해야 할 world 좌표. None이면 "위치 이동 없음"(예: home 복귀 전용 단계)
    hold_steps: int = 0                        # 목표에 도착한 뒤 이만큼의 스텝을 더 버텨야 진짜 완료로 인정 (떨림 방지)
    tolerance: float = POSITION_TOLERANCE      # "도착했다"고 볼 거리 오차(m)
    max_steps: int = MAX_STEPS_PER_STAGE       # 이 스텝 수를 넘기면 도착 못해도 강제로 다음 단계로 넘어감(타임아웃)
    speed: float = DEFAULT_TARGET_SPEED        # 이 단계에서 움직일 속도(m/s)
    use_orientation: bool = True               # RMPFlow에 목표 방향(orientation)도 같이 줄지 여부
    target_door_angle: "float | None" = None   # 이 단계가 끝나려면 도어가 이 각도까지 와야 하는지(없으면 검사 안 함)
    door_angle_tolerance: float = DOOR_ANGLE_TOLERANCE_DEG  # 위 도어 각도의 허용 오차(도)


class WaypointSequence:
    """link_6 목표를 속도 제한된 중간 목표로 천천히 이동시키는 범용 stage state machine.

    동작 원리: stages 리스트를 순서대로 하나씩 처리한다. 각 단계마다
      1) update()를 매 스텝 호출해서 "이번 단계 끝났는지" 판정하고 끝났으면 다음 단계로 넘어감
      2) get_command_target()을 호출해서 "이번 스텝에 RMPFlow에게 보낼 중간 목표점"을 구함
         (목표까지 한 번에 점프하지 않고 stage.speed로 제한된 만큼씩만 다가가게 함)
    모든 단계가 끝나면 self.done = True가 되고 이후 update()는 항상 True만 반환한다."""

    def __init__(self, stages: list, stage_log: dict | None = None):
        self.stages = stages
        self.stage_log = stage_log or {}  # 단계 이름(영문 코드) -> 사람이 읽기 좋은 한글 설명, 로그용
        self.index = 0          # 지금 몇 번째 단계를 진행 중인지
        self.stage_step = 0     # 현재 단계에서 몇 스텝이 지났는지 (타임아웃 판정용)
        self.hold_count = 0     # 목표에 도착한 상태로 몇 스텝이나 유지됐는지
        self.done = False       # 전체 시퀀스가 다 끝났는지
        self.command_target = None  # 이번 스텝에 실제로 명령으로 내보낼 중간 목표점 (속도 제한 적용된 값)

    @property
    def current(self) -> FuelStage:
        # 주의: done=True가 되면 self.index == len(self.stages)라서 이 프로퍼티를 부르면
        # IndexError가 난다. done 체크 없이 호출하는 곳이 없는지 항상 주의해야 한다.
        return self.stages[self.index]

    def reset(self):
        """처음 단계(0번)부터 다시 시작하도록 모든 진행 상태를 초기화한다."""
        self.index = 0
        self.stage_step = 0
        self.hold_count = 0
        self.done = False
        self.command_target = None

    def update(self, ee_position: np.ndarray, extra_condition_ok: bool = True) -> bool:
        """현재 위치(ee_position)를 보고 "이번 단계가 끝났는지" 판정한다.
        extra_condition_ok: 위치 도착 말고 추가로 만족해야 하는 조건(예: 도어 각도가 맞는지).
        반환값: 전체 시퀀스(모든 단계)가 끝났으면 True, 아니면 False."""
        if self.done:
            return True
        stage = self.current
        if stage.target_position is None:
            # 목표 위치가 없는 단계(예: home 복귀)는 이 함수가 처리하지 않고 호출하는 쪽에서 별도로 처리한다.
            return False

        self.stage_step += 1
        err = np.linalg.norm(stage.target_position - ee_position)  # 목표까지 남은 거리
        reached = (err < stage.tolerance) and extra_condition_ok
        timed_out = self.stage_step >= stage.max_steps

        if reached:
            self.hold_count += 1
        else:
            # 도착 조건이 한 번이라도 깨지면 "유지 카운트"를 처음부터 다시 센다 (계속 안정적으로 있어야 함).
            self.hold_count = 0

        if (reached and self.hold_count >= stage.hold_steps) or timed_out:
            label = self.stage_log.get(stage.name, stage.name)
            if timed_out and not reached:
                print(f"\n[⚠️ 타임아웃] {label} 남은거리={err:.3f}m", flush=True)
            else:
                print(f"\n[✅ {label}] 위치={np.round(ee_position, 3)}", flush=True)
            self.index += 1
            self.stage_step = 0
            self.hold_count = 0
            self.command_target = None
            if self.index >= len(self.stages):
                self.done = True
                return True
        return self.done

    def get_command_target(self, ee_position: np.ndarray):
        """이번 스텝에 RMPFlow 컨트롤러로 보낼 "중간 목표점"을 계산한다.
        목표(stage.target_position)까지 한 번에 이동시키지 않고, 매 스텝
        stage.speed * PHYSICS_DT 만큼씩만 다가가게 해서 부드럽고 일정한 속도로 움직이게 만든다."""
        if self.done:
            return None
        stage = self.current
        if stage.target_position is None:
            return None
        if self.command_target is None:
            # 새 단계가 시작될 때, 중간 목표의 출발점을 "현재 실제 EE 위치"로 잡는다.
            self.command_target = np.array(ee_position, dtype=float)

        delta = stage.target_position - self.command_target
        dist = np.linalg.norm(delta)
        max_step = max(stage.speed * PHYSICS_DT, 1e-5)  # 이번 스텝에 이동 가능한 최대 거리
        if dist <= max_step:
            # 한 스텝 안에 목표에 도착할 수 있으면 그냥 목표로 스냅
            self.command_target = np.array(stage.target_position, dtype=float)
        else:
            # 아니면 목표 방향으로 max_step 만큼만 전진
            self.command_target = self.command_target + delta / dist * max_step
        return self.command_target

    def debug_string(self, ee_position: np.ndarray) -> str:
        """현재 진행 상태를 한 줄 로그 문자열로 만든다 (디버깅용)."""
        if self.done:
            return "[DONE]"
        stage = self.current
        if stage.target_position is None:
            return f"[stage={self.index}:{stage.name}] hold={self.hold_count}/{stage.hold_steps}"
        err = np.linalg.norm(stage.target_position - ee_position)
        return (
            f"[stage={self.index}:{stage.name}] target={np.round(stage.target_position, 3)} "
            f"ee={np.round(ee_position, 3)} err={err:.4f} speed={stage.speed:.3f} "
            f"hold={self.hold_count}/{stage.hold_steps}"
        )


def single_stage_sequence(name: str, target: np.ndarray, **kwargs) -> WaypointSequence:
    """단계가 딱 하나뿐인 WaypointSequence를 만드는 짧은 헬퍼 함수."""
    return WaypointSequence([FuelStage(name, target, **kwargs)])


# ============================================================
# A 로봇: 기존 주유 시퀀스 (기존 FuelPortSequence 로직 그대로, 모드만 hole)
# ============================================================
def build_fuel_port_sequence(fuel_port_center: np.ndarray, fuel_target_liters: float = 0.0) -> WaypointSequence:
    """A 로봇이 "가상의 노즐"을 들고 주유구에 다가가 꽂고 다시 빼는 8단계 시퀀스를 만든다.
    fuel_port_center: 카메라로 lock한 주유구 구멍의 실제 world 좌표.
    fuel_target_liters: 웹UI에서 전달된 목표 주유량. 0이면 기본 hold_steps 사용."""
    outward = PORT_OUTWARD_NORMAL_UNIT
    insertion = INSERTION_DIRECTION_UNIT

    # 노즐의 "끝(tip)"이 있어야 할 위치들 (구멍에서 바깥쪽으로 점점 가까워짐 -> 안쪽으로 삽입)
    tip_far  = fuel_port_center + outward * FAR_DISTANCE
    tip_mid  = fuel_port_center + outward * MID_DISTANCE
    tip_near = fuel_port_center + outward * NEAR_DISTANCE
    tip_insert = fuel_port_center + insertion * INSERT_DISTANCE

    # 로봇이 실제로 움직이는 건 "노즐 끝"이 아니라 "손목(link_6)"이므로, 가상 노즐 길이만큼
    # 뒤로 뺀 오프셋을 더해서 손목이 가야 할 목표(approach_*)를 계산한다.
    control_offset = outward * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])
    approach_far  = tip_far + control_offset
    approach_mid  = tip_mid + control_offset
    approach_near = tip_near + control_offset
    insert_target = tip_insert + control_offset

    # 주유 시간: 웹UI 가격 공식(1 + krw/5000초)과 동일하게 계산 (pricePerLiter=2000)
    if fuel_target_liters > 0:
        _fill_sec = 1.0 + (fuel_target_liters * 2000.0) / 5000.0
        _hold_steps_insert = max(1, int(_fill_sec / PHYSICS_DT))
    else:
        _fill_sec = 1.0 + 80 * PHYSICS_DT  # 기본값 80스텝 ≈ 1.3s
        _hold_steps_insert = 80
    print(f"\n[A] 주유 hold_steps={_hold_steps_insert} ({_fill_sec:.1f}초) "
          f"← target={fuel_target_liters:.1f}L\n", flush=True)

    stages = [
        FuelStage("01_axis_far_start", approach_far, tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        FuelStage("02_axis_mid", approach_mid, tolerance=0.07, speed=DEFAULT_TARGET_SPEED),
        FuelStage("03_axis_near_stop", approach_near, hold_steps=40, tolerance=0.06, speed=NEAR_TARGET_SPEED),
        FuelStage("04_insert_into_cylinder", insert_target, hold_steps=10, tolerance=0.05, speed=INSERT_TARGET_SPEED, max_steps=MAX_STEPS_PER_STAGE),

        FuelStage("05_fueling_dwell", insert_target, hold_steps=_hold_steps_insert, tolerance=0.06, speed=NEAR_TARGET_SPEED, max_steps=_hold_steps_insert + 120),

        FuelStage("06_retreat_near", approach_near, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("07_retreat_mid", approach_mid, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("08_retreat_far", approach_far, hold_steps=15, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("09_return_home", None),
    ]
    stage_log = {
        "01_axis_far_start": "A 주유구 접근 시작",
        "02_axis_mid": "A 주유구 접근 중(중간 지점)",
        "03_axis_near_stop": "A 주유구 근접 완료, 삽입 준비",
        "04_insert_into_cylinder": "A 노즐 삽입 완료, 주유 중",
        "05_fueling_dwell": "A 목표 주유량 도달, 노즐 후퇴 시작",
        "06_retreat_near": "A 주유 완료, 노즐 후퇴 시작",
        "07_retreat_mid": "A 노즐 후퇴 중(중간 지점)",
        "08_retreat_far": "A 노즐 후퇴 완료",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# B 로봇: 커버 닫기(push) 시퀀스 / 마개 접근-복원 시퀀스
# 커버 열기는 더 이상 로봇이 밀지 않고 USD 씬 설정으로 자동으로 열려있는 상태에서 시작하지만,
# 닫기는 로봇이 실제로 밀면서 그 진행률에 맞춰 덮개를 같이 회전시켜야 자연스러워서
# (multi_robot_oiling_start_open.py에서 이식) 로봇 푸시 + 시각 각도 동기화 방식을 그대로 쓴다.
# ============================================================
def build_close_cover_sequence(
    door_reference_point: np.ndarray, pivot: np.ndarray, axis: np.ndarray,
    avoid_point: "np.ndarray | None" = None,
    center_reference_point: "np.ndarray | None" = None,
) -> WaypointSequence:
    """B 로봇이 커버를 닫는 시퀀스.

    130도까지 더 열어둔 door에 맞춰, cap 복원 후 바로 정면으로 밀지 않고:
      1) 뒤로 빠진 위치에서 오른쪽(+x)으로 이동
      2) door 접촉점 앞쪽으로 들어감
      3) 왼쪽 방향으로 쓸어 밀며 130 -> 중간 -> 닫힘으로 진행
      4) 한 번 빠져나와 중앙으로 맞춘 뒤 PRESS pose로 마지막 완전 닫힘을 누름
    """
    right_vec = np.array([DOOR_CLOSE_RIGHT_SHIFT_M, 0.0, 0.0], dtype=float)

    def p(angle_deg):
        rotated = rotate_point_around_axis(
            door_reference_point, pivot, axis, angle_deg - COVER_START_DEG
        )
        return rotated + DOOR_TOUCH_OFFSET + PORT_OUTWARD_NORMAL_UNIT * DOOR_CLOSE_OUTWARD_CLEARANCE

    def p_center(angle_deg):
        rotated = rotate_point_around_axis(
            center_reference_point, pivot, axis, angle_deg - COVER_START_DEG
        )
        return rotated + DOOR_TOUCH_OFFSET + PORT_OUTWARD_NORMAL_UNIT * DOOR_CLOSE_OUTWARD_CLEARANCE

    mid_deg = (COVER_OPEN_DEG + COVER_START_DEG) / 2.0
    stages = []
    stage_log = {}

    if avoid_point is not None:
        stages.append(FuelStage("B10_00_avoid_door", avoid_point, tolerance=0.08, speed=DEFAULT_TARGET_SPEED, use_orientation=False))
        stage_log["B10_00_avoid_door"] = "B 열린 덮개 회피 경로 통과"

    # 오른쪽으로 이동한 뒤, 바깥쪽에서 door 접촉점 앞까지 들어간다.
    open_contact = p(COVER_OPEN_DEG)
    right_ready = open_contact + right_vec + PORT_OUTWARD_NORMAL_UNIT * DOOR_CLOSE_FORWARD_BACKOFF_M
    forward_entry = open_contact + right_vec

    # 실제 밀기 구간은 오른쪽 offset을 점차 줄여서 왼쪽으로 쓸어 미는 느낌을 준다.
    mid_push = p(mid_deg) + right_vec * 0.35
    start_push = p(COVER_START_DEG)

    stages += [
        FuelStage("B10_01_move_right", right_ready, tolerance=0.15, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B10_02_forward_entry", forward_entry, hold_steps=8, tolerance=0.15, speed=NEAR_TARGET_SPEED,
                  target_door_angle=COVER_OPEN_DEG),
        FuelStage("B10_03_push_left_mid", mid_push, hold_steps=20, tolerance=0.14, speed=NEAR_TARGET_SPEED,
                  target_door_angle=mid_deg),
        FuelStage("B10_04_push_left_start", start_push, hold_steps=30, tolerance=0.15, speed=NEAR_TARGET_SPEED,
                  target_door_angle=COVER_START_DEG, use_orientation=True),
    ]
    stage_log.update({
        "B10_01_move_right": "B 덮개 닫기 전 오른쪽 위치로 이동",
        "B10_02_forward_entry": "B 덮개 접촉점으로 전진",
        "B10_03_push_left_mid": f"B 덮개 왼쪽으로 밀기 중({mid_deg:.0f}도)",
        "B10_04_push_left_start": f"B 덮개 {COVER_START_DEG:.0f}도까지 밀어 닫음",
    })

    if center_reference_point is not None:
        retreat_point = start_push + PORT_OUTWARD_NORMAL_UNIT * DOOR_FULL_CLOSE_RETREAT_DISTANCE
        center_align_point = p_center(COVER_START_DEG) + PORT_OUTWARD_NORMAL_UNIT * DOOR_FULL_CLOSE_RETREAT_DISTANCE
        full_close_point = p_center(COVER_FULL_CLOSE_DEG)
        stages += [
            FuelStage("B10_05_retreat_for_press", retreat_point, tolerance=0.06, speed=RETREAT_TARGET_SPEED),
            FuelStage("B10_06_center_align_press", center_align_point, hold_steps=10, tolerance=0.06, speed=NEAR_TARGET_SPEED),
            FuelStage("B10_07_final_press", full_close_point, hold_steps=30, tolerance=0.10,
                      speed=NEAR_TARGET_SPEED, target_door_angle=COVER_FULL_CLOSE_DEG),
        ]
        stage_log.update({
            "B10_05_retreat_for_press": f"B 덮개 {COVER_START_DEG:.0f}도까지 닫고 후퇴",
            "B10_06_center_align_press": "B 덮개 중앙 PRESS 위치 정렬",
            "B10_07_final_press": f"B PRESS 자세로 덮개 완전 닫힘({COVER_FULL_CLOSE_DEG:.0f}도)",
        })
    return WaypointSequence(stages, stage_log)

def build_cap_depth_stages(center: np.ndarray, far_name: str, near_name: str, insert_name: str) -> list:
    """마개를 잡으러 가는 쪽(B5)과 다시 끼우는 쪽(B9)이 완전히 동일한 깊이 오프셋을 쓰도록
    공유하는 far/near/insert(grasp) 3단계. 닫기 쪽이 열기보다 더 깊이 들어가도록 따로
    설정되어 있어서 팔 길이 한계를 넘는 위치를 목표로 하던 문제가 있었어서, 이제는
    GRIPPER_LENGTH_B 기준 오프셋을 이 한 곳에서만 정의한다."""
    return [
        FuelStage(far_name, make_outward_point(center, FAR_DISTANCE + GRIPPER_LENGTH_B),
                  tolerance=0.12, speed=DEFAULT_TARGET_SPEED),
        FuelStage(near_name, make_outward_point(center, NEAR_DISTANCE + GRIPPER_LENGTH_B),
                  hold_steps=30, tolerance=0.12, speed=NEAR_TARGET_SPEED),
        # 그리퍼 scale을 2배로 키운 뒤에도 마개 잡기/복원 둘 다 파고드는 게 그대로라,
        # GRIPPER_LENGTH_B(0.32)만으로는 실제 스케일된 그리퍼 길이를 못 따라가는 것으로 보여
        # 안전 마진을 0.05 -> 0.15로 늘렸다. 그래도 파고들면 GRIPPER_LENGTH_B 자체를 더 키워야 한다.
        FuelStage(insert_name, center + PORT_OUTWARD_NORMAL_UNIT * (GRIPPER_LENGTH_B + 0.15 - CAP_RESTORE_EXTRA_INWARD_M),
                  hold_steps=40, tolerance=0.12, speed=INSERT_TARGET_SPEED),
    ]


def build_cap_approach_sequence(cap_center: np.ndarray) -> WaypointSequence:
    """B 로봇이 마개(fuel_cap)를 잡으러 멀리서부터 단계적으로 접근하는 4단계 시퀀스.
    가장 먼저 COVER_CLEARANCE_DISTANCE만큼 벽에서 멀리 떨어진 경유점(clearance_point)을 거쳐서,
    열어둔 커버(fuel_door)에 부딫히지 않고 돌아서 마개로 접근한다."""
    clearance_point = cap_center + PORT_OUTWARD_NORMAL_UNIT * COVER_CLEARANCE_DISTANCE
    stages = [
        FuelStage("B5_00_avoid_door", clearance_point, tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
    ] + build_cap_depth_stages(cap_center, "B5_01_far", "B5_02_near", "B5_03_grasp")
    stage_log = {
        "B5_00_avoid_door": "B 열린 덮개 회피 경로 통과",
        "B5_01_far":        "B 마개 접근 중(먼 지점)",
        "B5_02_near":       "B 마개 접근 중(가까운 지점)",
        "B5_03_grasp":      "B 마개 잡을 위치 도착, 그리퍼 닫기 준비",
    }
    return WaypointSequence(stages, stage_log)


def build_cap_restore_sequence(cap_center: np.ndarray) -> WaypointSequence:
    """A 로봇이 주유를 마친 뒤, B 로봇이 마개를 들고 다시 다가가 끼우는 3단계 시퀀스.
    build_cap_approach_sequence와 build_cap_depth_stages를 공유해서 열기/닫기 깊이가 항상 같다.
    cap_center는 hole_world_position(주유구 구멍)이 아니라 locked_cap_center(마개를 집을 때
    카메라로 lock했던 위치)를 받아야 한다 - hole은 cap보다 outward-normal 방향으로 약 0.227m
    더 안쪽이라, hole 기준으로 같은 오프셋을 쓰면 닫을 때가 열 때보다 훨씬 깊이 들어간다."""
    stages = build_cap_depth_stages(cap_center, "B9_01_far", "B9_02_near", "B9_03_insert")
    stage_log = {
        "B9_01_far": "B 주유구 접근 중(먼 지점)",
        "B9_02_near": "B 주유구 접근 중(가까운 지점)",
        "B9_03_insert": "B 마개 삽입 위치 도착, 조이기 준비",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# ROS2 bridge: ArUco detector 노드(aruco_marker_detector.py)
# 및 A/B 동기화 토픽을 한 노드에서 처리
# ============================================================
class MultiRobotRosBridge(Node):
    """이 시뮬레이션 스크립트 전체를 위한 ROS2 노드 하나.
    A/B 두 로봇 runner가 직접 ROS2 토픽을 구독/발행하지 않고, 이 클래스를 통해서만 통신한다
    (한 프로세스 안에 노드를 여러 개 만들 필요 없이 하나로 충분하기 때문)."""

    def __init__(self):
        super().__init__("multi_robot_oiling_ros_bridge")
        # BEST_EFFORT: 카메라 위치/lock 신호처럼 "초당 여러 번 오는, 최신 값이 중요한" 토픽에 사용.
        # 못 받은 메시지를 재전송하지 않아 더 빠르고 가볍다.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # mode_switch/robot_a_done/robot_b_done은 한 번만 발행되는 이벤트 플래그라,
        # 구독자가 그 시점에 늦게 join해도 마지막 값을 받을 수 있도록 latch 시킨다.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.latest_pose: "PoseStamped | None" = None  # 가장 최근에 받은 카메라 표적 위치
        self.pose_count = 0       # pose가 몇 번 도착했는지 누적 카운트 (새 메시지 도착 여부 판단용)
        self.target_locked = False  # 그 위치가 "안정적으로 확정됐다"는 디텍터의 판단
        self.lock_count = 0
        self.robot_a_done = False        # A 로봇이 done 신호를 보냈는지
        self.robot_a_done_count = 0
        self.robot_b_done = False        # B 로봇이 done 신호를 보냈는지
        self.robot_b_done_count = 0
        self.start_requested = False     # 웹UI "주유 시작" 버튼 수신 플래그
        self.fuel_target_liters = 0.0    # 웹UI에서 전달된 목표 주유량(리터)

        self.pose_sub = self.create_subscription(PoseStamped, TOPIC_ARUCO_POSE, self._pose_cb, sensor_qos)
        self.lock_sub = self.create_subscription(Bool, TOPIC_ARUCO_LOCK, self._lock_cb, sensor_qos)
        self.robot_a_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_A_DONE, self._robot_a_done_cb, latched_qos)
        self.robot_b_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_B_DONE, self._robot_b_done_cb, latched_qos)
        self.start_sub = self.create_subscription(Bool, "/start_fueling", self._start_cb, 10)

        self.mode_switch_pub = self.create_publisher(String, TOPIC_MODE_SWITCH, latched_qos)
        self.robot_a_done_pub = self.create_publisher(Bool, TOPIC_ROBOT_A_DONE, latched_qos)
        self.robot_b_done_pub = self.create_publisher(Bool, TOPIC_ROBOT_B_DONE, latched_qos)

        self.get_logger().info("MultiRobotRosBridge started")

    def _pose_cb(self, msg: PoseStamped):
        """카메라 위치 토픽이 도착할 때마다 자동으로 호출되는 콜백. 최신값만 저장하고 카운트를 올린다."""
        self.latest_pose = msg
        self.pose_count += 1

    def _lock_cb(self, msg: Bool):
        self.target_locked = bool(msg.data)
        self.lock_count += 1

    def _robot_a_done_cb(self, msg: Bool):
        self.robot_a_done = bool(msg.data)
        self.robot_a_done_count += 1

    def _robot_b_done_cb(self, msg: Bool):
        self.robot_b_done = bool(msg.data)
        self.robot_b_done_count += 1

    def _start_cb(self, msg: Bool):
        if msg.data:
            self.start_requested = True
            print("\n[ROS] /start_fueling 수신 → start_requested=True\n", flush=True)

    def get_pose_if_ready(self):
        """아직 pose를 한 번도 못 받았거나, lock이 필요한데 아직 lock=False면 None을 반환해서
        "쓸 수 없는 상태"임을 알려준다. 호출하는 쪽은 None이면 이번 틱은 그냥 건너뛰면 된다."""
        if self.latest_pose is None:
            return None
        if REQUIRE_TARGET_LOCK and not self.target_locked:
            return None
        return self.latest_pose

    def publish_mode_switch(self, mode: str):
        """디텍터에게 "이제부터 이 색만 찾아"라고 명령을 보낸다."""
        self.mode_switch_pub.publish(String(data=mode))
        self.get_logger().info(f"mode_switch -> {mode}")

    def publish_robot_a_done(self, flag: bool):
        self.robot_a_done_pub.publish(Bool(data=bool(flag)))

    def publish_robot_b_done(self, flag: bool):
        self.robot_b_done_pub.publish(Bool(data=bool(flag)))


class StableTargetLockAcquirer:
    """door/cap/hole 모드 공통: N개 샘플이 표준편차 이내로 모이면 평균 world 좌표를 반환한다."""

    def __init__(self, ros_bridge: MultiRobotRosBridge, camera_prim_path: "str | None",
                 reference_center: np.ndarray, apply_mouth_offset: bool = False,
                 lock_z_to_reference: bool = True,
                 required_samples: int = CONTROLLER_REQUIRED_LOCK_SAMPLES,
                 std_tolerance: float = CONTROLLER_WORLD_STD_TOLERANCE,
                 gate_half_extent: np.ndarray = SEARCH_GATE_HALF_EXTENT):
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.reference_center = reference_center      # 게이트 판정 기준이 되는 "예상 위치"
        self.apply_mouth_offset = apply_mouth_offset    # 구멍 입구->중심 보정을 적용할지(주유구에서만 True)
        self.lock_z_to_reference = lock_z_to_reference  # z(높이)는 카메라 추정값 대신 기준값으로 고정할지
        self.required_samples = required_samples        # 안정적이라고 보기 위해 모아야 하는 샘플 수
        self.std_tolerance = std_tolerance              # 샘플들의 흔들림 허용치
        self.gate_half_extent = gate_half_extent        # 게이트(허용 범위) 크기
        self.samples: list = []                          # 지금까지 모은 유효한 후보 위치들
        self.last_sampled_pose_count = -1               # 마지막으로 처리한 pose_count (중복 처리 방지)

    def reset(self):
        """모아둔 샘플을 전부 버리고 처음부터 다시 모으기 시작한다 (모드 전환/타임아웃 시 호출)."""
        self.samples = []
        self.last_sampled_pose_count = -1

    def update(self):
        """매 tick 호출. 안정화된 평균 world 좌표가 나오면 반환, 아니면 None.

        target_locked가 잠깐 False로 흔들려도(디텍터는 publish_hz로만 lock을 발행하므로
        그 순간의 깜빡임이 그대로 들어온다) 누적된 샘플을 통째로 비우지 않는다.
        get_pose_if_ready()가 이미 unlocked 상태에서는 새 샘플 추가를 막아주므로,
        여기서 추가로 버퍼를 비우면 5개를 채우기 전에 계속 리셋되어 영원히 lock이 안 된다.
        """
        pose_msg = self.ros_bridge.get_pose_if_ready()
        if pose_msg is None or self.ros_bridge.pose_count == self.last_sampled_pose_count:
            return None
        self.last_sampled_pose_count = self.ros_bridge.pose_count

        if self.camera_prim_path is None:
            print("[LOCK] camera_prim_path가 None이라 pose를 world 좌표로 변환할 수 없음 "
                  "(find_camera_prim_path()가 카메라 prim을 못 찾음 - CAMERA_PRIM_CANDIDATES 확인)")
            return None

        p_cam = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ], dtype=float)
        detected_world_point = transform_camera_point_to_world(p_cam, self.camera_prim_path)
        if detected_world_point is None:
            print(f"[LOCK] transform_camera_point_to_world() 실패 (camera_prim_path={self.camera_prim_path} 가 invalid)", flush=True)
            return None

        candidate = (
            detected_world_point_to_mouth_center(detected_world_point)
            if self.apply_mouth_offset else detected_world_point
        )
        if self.lock_z_to_reference:
            candidate[2] = self.reference_center[2]

        valid, reason = validate_detected_center_world(candidate, self.reference_center, self.gate_half_extent)
        if valid:
            self.samples.append(candidate)
            self.samples = self.samples[-self.required_samples:]
        else:
            self.samples = []
            print(f"[LOCK REJECT] p_cam={np.round(p_cam, 3)} world_pt={np.round(detected_world_point, 3)} "
                  f"candidate={np.round(candidate, 3)} ref={np.round(self.reference_center, 3)} reason={reason}")

        if len(self.samples) >= self.required_samples:
            # 충분한 샘플이 모이면 평균과 흔들림(표준편차)을 계산해서, 흔들림이 허용치 이내일 때만
            # "확정된 위치"로 인정하고 반환한다. 아직 흔들리고 있으면 None을 반환해 계속 더 모은다.
            arr = np.array(self.samples, dtype=float)
            mean = arr.mean(axis=0)
            std_norm = float(np.linalg.norm(arr.std(axis=0)))
            if self.lock_z_to_reference:
                mean[2] = self.reference_center[2]
            if std_norm <= self.std_tolerance:
                return mean
        return None


# ============================================================
# 차량 슬라이드-인: CAR_ROOT_PRIM_NAME("car") prim을 x축 +10m 지점에서 실제 주차 위치까지
# 직선으로 이동시킨다. 물리 드라이브 없이 매 스텝 prim transform을 직접 설정하는 kinematic
# 방식이라 바퀴는 회전하지 않는다(요청대로 바퀴는 안 움직여도 됨). A/B 로봇 시퀀스는
# main()의 메인 루프에서 이 컨트롤러가 끝날 때까지 시작하지 않는다.
# ============================================================
class CarArrivalController:
    """차체(car_visual)+ArUco 마커(aruco_vehicle_marker)를 묶은 /World/car prim을 이동시킨다."""

    def __init__(self, car_prim_path: "str | None"):
        self.car_prim_path = car_prim_path
        self.base_position = None       # 씬에 원래(authored) 저장된 "기준" 주차 위치. 최초 1회만 캡처.
        self.target_position = None     # 이번 판의 실제 도착 지점(base + 매판 새로 뽑는 랜덤 오차)
        self.current_position = None    # 지금 이동 중인 위치
        self.done = self.car_prim_path is None
        if self.car_prim_path is None:
            print(f"  [CAR][WARN] '{CAR_ROOT_PRIM_NAME}' prim을 못 찾아서 슬라이드-인 시뮬레이션을 건너뜀", flush=True)

    def is_done(self) -> bool:
        return self.done

    def _apply_position(self, position: np.ndarray):
        set_prim_world_matrix(self.car_prim_path, _matrix_translate(position))

    def on_play_reset(self):
        """매 Play/리셋마다 호출. 도착 지점에 랜덤 오차를 새로 뽑고, 차를 출발 지점
        (도착 지점 + CAR_ARRIVAL_START_OFFSET)으로 즉시 이동시킨 뒤 슬라이드-인을 다시 시작한다."""
        if self.car_prim_path is None:
            return
        if self.base_position is None:
            # 최초 1회만 "기준" 위치를 캡처해둔다 - 매번 새로 읽으면 이전 판에서 옮겨놓은 위치를
            # "기준"으로 잘못 캡처하게 된다.
            self.base_position = get_prim_world_position(self.car_prim_path)
        xy_noise = np.random.uniform(-CAR_PARK_XY_NOISE_M, CAR_PARK_XY_NOISE_M, size=2)
        self.target_position = self.base_position.copy()
        self.target_position[0] += xy_noise[0]
        self.target_position[1] += xy_noise[1]
        self.current_position = self.target_position + CAR_ARRIVAL_START_OFFSET
        self.done = False
        self._apply_position(self.current_position)
        print(f"\n[차량] 주유소 입고 시작: start={np.round(self.current_position, 3)} -> "
              f"주차위치={np.round(self.target_position, 3)} "
              f"(기준={np.round(self.base_position, 3)}, 주차오차={np.round(xy_noise, 3)})\n")

    def tick(self, step_count: int):
        """매 시뮬레이션 스텝마다 호출. 도착할 때까지 target_position 방향으로 일정 속도로 이동시킨다."""
        if self.done:
            return
        remaining = self.target_position - self.current_position
        distance = float(np.linalg.norm(remaining))
        if distance <= CAR_ARRIVAL_TOLERANCE_M:
            self.current_position = self.target_position.copy()
            self._apply_position(self.current_position)
            self.done = True
            print(f"\n[차량] 주차 완료: position={np.round(self.current_position, 3)}\n", flush=True)
            return
        step_dist = min(CAR_ARRIVAL_SPEED_MPS * PHYSICS_DT, distance)
        self.current_position = self.current_position + remaining / distance * step_dist
        self._apply_position(self.current_position)


# ============================================================
# Task: USD 로드, 두 로봇 등록, fuel_door/cap/hole 및 힌지 joint 탐색
# ============================================================
class MultiRobotOilingTask(BaseTask):
    """Isaac Sim의 World.add_task()에 등록되는 "씬 구성 담당" 클래스.
    set_up_scene()이 World.reset() 시점에 자동으로 호출되어, USD 로드부터 로봇 등록,
    주유구 관련 오브젝트 탐색까지 한 번에 처리한다."""

    def __init__(self, name):
        super().__init__(name=name, offset=None)

    def set_up_scene(self, scene):
        """씬을 만드는 전체 순서. 각 단계는 이전 단계의 결과(예: ee_path)에 의존하므로 순서가 중요하다."""
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_robot_links()
        self._setup_physics()
        self._register_robots(scene)
        self._discover_fuel_objects(scene)
        self._setup_control_markers(scene)
        print("\n  [완료] multi-robot oiling 씬 구성 성공!\n", flush=True)

    def _load_usd(self):
        """프로젝트 USD 파일을 /World 아래에 참조(reference)로 추가해서 씬에 불러온다.
        AddReference는 즉시 로드되는 게 아니라 비동기로 처리되므로, 로드가 끝나길 기다리려고
        simulation_app.update()를 여러 번 호출해 강제로 프레임을 진행시킨다."""
        print("\n" + "=" * 60, flush=True)
        print("[1.LOAD] USD 로드", flush=True)
        print("=" * 60, flush=True)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}", flush=True)
        print(f"  [NOTE] m0609_A={ROBOT_A_PRIM_PATH}, m0609_B={ROBOT_B_PRIM_PATH}가 USD에 이미 있다고 가정", flush=True)

    def _discover_robot_links(self):
        """A/B 로봇 USD 안에서 손목 끝(link_6) prim 경로를 찾는다.
        못 찾으면 이후 SingleManipulator 등록 자체가 불가능하므로 바로 에러를 내서 빨리 알게 한다."""
        print("\n" + "=" * 60, flush=True)
        print("[2.DISCOVER] 로봇 A/B 링크 경로 탐색", flush=True)
        print("=" * 60, flush=True)
        self.ee_path_a = find_prim_path_by_name(ROBOT_A_PRIM_PATH, EE_LINK_NAME)
        self.ee_path_b = find_prim_path_by_name(ROBOT_B_PRIM_PATH, EE_LINK_NAME)
        if self.ee_path_a is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_A_PRIM_PATH}")
        if self.ee_path_b is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_B_PRIM_PATH}")
        print(f"  A EE = {self.ee_path_a}", flush=True)
        print(f"  B EE = {self.ee_path_b}", flush=True)

    def _setup_physics(self):
        """A/B 로봇의 모든 관절 드라이브(모터)에 강성/댐핑/최대힘 값을 강하게 설정한다.
        USD에 기본으로 들어있는 드라이브 값이 너무 약하면 로봇이 목표 위치로 잘 따라가지 못하거나
        무거운 물체(마개 등)를 들 때 축 늘어지는 문제가 생길 수 있어 이 값들을 직접 키워준다."""
        print("\n" + "=" * 60, flush=True)
        print("[3.PHYSICS] 로봇 A/B drive 설정", flush=True)
        print("=" * 60, flush=True)
        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for root_path in (ROBOT_A_PRIM_PATH, ROBOT_B_PRIM_PATH):
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                raise RuntimeError(f"Robot prim not found: {root_path}")
            for prim in Usd.PrimRange(root):
                for dt in ["angular", "linear"]:
                    drive = UsdPhysics.DriveAPI.Get(prim, dt)
                    if drive:
                        drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                        drive.GetDampingAttr().Set(DRIVE_DAMPING)
                        drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                        drive_count += 1
        print(f"  [OK] drive updated: {drive_count}", flush=True)

    def _register_robots(self, scene):
        """Isaac Sim의 SingleManipulator/ParallelGripper 객체를 만들어 scene에 등록한다.
        이렇게 등록해야 robot.get_joint_positions(), robot.apply_action() 같은
        고수준 API를 쓸 수 있게 된다 (등록 전에는 USD prim일 뿐, 로봇 객체가 아니다)."""
        print("\n" + "=" * 60, flush=True)
        print("[4.REGISTER] SingleManipulator A/B 등록", flush=True)
        print("=" * 60, flush=True)
        gripper_a = ParallelGripper(
            end_effector_prim_path=self.ee_path_a,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self.robot_a = scene.add(
            SingleManipulator(
                prim_path=ROBOT_A_PRIM_PATH,
                name="m0609_A",
                end_effector_prim_path=self.ee_path_a,
                gripper=gripper_a,
            )
        )
        if USE_B_ALLEGRO_WRIST:
            # m0609_B는 USD 안에서 wrist+Allegro 모델로 교체되어 있다고 가정한다.
            # 기존 ParallelGripper는 등록하지 않는다.
            gripper_b = None
        else:
            gripper_b = ParallelGripper(
                end_effector_prim_path=self.ee_path_b,
                joint_prim_names=GRIPPER_JOINTS,
                joint_opened_positions=np.array(GRIPPER_OPEN),
                joint_closed_positions=np.array(GRIPPER_CLOSE),
                action_deltas=np.array(GRIPPER_DELTA),
            )
        self.robot_b = scene.add(
            SingleManipulator(
                prim_path=ROBOT_B_PRIM_PATH,
                name="m0609_B",
                end_effector_prim_path=self.ee_path_b,
                gripper=gripper_b,
            )
        )
        print(f"  [OK] m0609_A = {ROBOT_A_PRIM_PATH}", flush=True)
        print(f"  [OK] m0609_B = {ROBOT_B_PRIM_PATH}", flush=True)

    def _resolve_world_position(self, prim_path, fallback_constant, label):
        """prim_path가 실제로 존재하면 USD에서 읽은 "진짜 world 위치"를 쓰고,
        prim을 못 찾았을 때만 프롬프트에 적혀있던 하드코딩 상수(fallback_constant)로 대체한다.
        하드코딩 상수와 실제 위치가 얼마나 차이나는지(diff)도 로그로 남겨서,
        혹시 USD 씬이 바뀌었는데 상수를 안 고친 경우를 바로 알아챌 수 있게 한다."""
        if prim_path is not None:
            pos = get_prim_world_position(prim_path)
            if pos is not None:
                diff = pos - fallback_constant
                print(f"  [OK] {label} 실제 world 위치 = {np.round(pos, 4)} "
                      f"(하드코딩 상수와 차이 = {np.round(diff, 4)}, |diff|={np.linalg.norm(diff):.4f}m)")
                return pos
        print(f"  [WARN] {label} prim 위치를 못 읽음 -> 하드코딩 상수 {np.round(fallback_constant, 4)} 로 폴백", flush=True)
        return fallback_constant.copy()

    def _discover_fuel_objects(self, scene):
        """fuel_door/fuel_cap/fuel_port_hole의 실제 world 위치와, 도어 힌지의 회전축/피벗을
        USD 씬에서 직접 읽어온다. 이 위치들이 게이트 판정 기준(reference_center)이자
        로봇이 실제로 움직여야 할 목표 좌표의 출발점이 되므로, 이 함수의 정확도가
        전체 시스템이 제대로 작동하는지의 핵심이다."""
        print("\n" + "=" * 60, flush=True)
        print("[5.SCENE] fuel_door / fuel_cap / fuel_port_hole 탐색", flush=True)
        print("=" * 60, flush=True)
        self.door_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_DOOR_PRIM_NAME)
        self.cap_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_CAP_PRIM_NAME)
        self.hole_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_PORT_HOLE_PRIM_NAME)
        print(f"  fuel_door = {self.door_prim_path}", flush=True)
        print(f"  fuel_cap  = {self.cap_prim_path}", flush=True)
        print(f"  fuel_port_hole = {self.hole_prim_path}", flush=True)

        # 게이트 판정/실제 모션 목표는 프롬프트의 하드코딩 좌표가 아니라 USD 씬에서 직접 읽은
        # 실제 world 위치를 우선 사용한다. 하드코딩 값은 해당 prim을 못 찾았을 때만 폴백으로 쓴다.
        self.door_world_position = self._resolve_world_position(self.door_prim_path, FUEL_DOOR_CENTER, "fuel_door")
        self.cap_world_position = self._resolve_world_position(self.cap_prim_path, FUEL_CAP_CENTER, "fuel_cap")
        self.hole_world_position = self._resolve_world_position(self.hole_prim_path, FUEL_PORT_HOLE_CENTER, "fuel_port_hole")

        # fuel_door의 "닫힌(30도 기준) 상태" world transform을 따로 저장해둔다 - 지금(Play 시작 전)
        # 시점에 캡처해야 진짜 닫힌 자세가 기준이 된다. CLOSE_COVER에서 set_door_angle_deg()가
        # 이 matrix를 기준으로 prim을 힌지축 회전시켜서, 로봇이 미는 진행률에 맞춰 자연스럽게
        # 닫히는 것처럼 보이게 한다.
        self.door_closed_world_matrix = get_prim_world_matrix(self.door_prim_path)

        self.door_joint_prim = None
        self.door_axis_world = np.array([0.0, 0.0, 1.0], dtype=float)
        self.door_pivot_world = self.door_world_position.copy()
        if self.door_prim_path is not None:
            self.door_joint_prim = find_revolute_joint_for_body(self.door_prim_path)
            if self.door_joint_prim is not None:
                self.door_axis_world, self.door_pivot_world = get_joint_world_axis_and_pivot(self.door_joint_prim)
                print(f"  [OK] door hinge axis(world)={np.round(self.door_axis_world, 3)} "
                      f"pivot(world)={np.round(self.door_pivot_world, 3)}")
            else:
                print("  [WARN] fuel_door에 연결된 RevoluteJoint를 찾지 못함. "
                      "기본 축([0,0,1])과 FUEL_DOOR_CENTER를 피벗으로 가정함 - 실제 동작 전 검증 필요.")

        # 리셋 시 door의 "기준 회전"을 캐시해서 이후 상대 각도를 측정한다 (COVER_START_DEG로 가정).
        self.door_rest_rotation = get_prim_world_rotation(self.door_prim_path)
        self._setup_door_drive()

        # 디버깅용으로 door/cap/hole 위치에 작은 색깔 박스를 띄워서, Isaac Sim 화면에서
        # "코드가 인식한 위치"가 실제 USD 모델과 맞는지 눈으로 바로 확인할 수 있게 한다.
        markers = [
            ("fuel_marker_door", self.door_world_position, np.array([1.0, 1.0, 0.0])),  # 노란 박스
            ("fuel_marker_cap", self.cap_world_position, np.array([0.0, 0.5, 1.0])),     # 파란 박스
            ("fuel_marker_hole", self.hole_world_position, np.array([0.0, 1.0, 0.3])),   # 초록 박스
        ]
        self.debug_marker_prim_paths = [f"/World/{marker_name}" for marker_name, _, _ in markers]
        for marker_name, pos, color in markers:
            scene.add(
                VisualCuboid(
                    prim_path=f"/World/{marker_name}",
                    name=marker_name,
                    position=pos,
                    scale=np.array([0.02, 0.02, 0.02]),
                    color=color,
                )
            )

    def hide_debug_markers(self):
        """fuel_marker_door/cap/hole 디버그 박스 및 제어용 마커 3개를 숨긴다.
        Play를 누르면(실제 시뮬레이션 시작) 시야를 가리지 않게 꺼둔다."""
        for prim_path in self.debug_marker_prim_paths:
            set_prim_visibility(prim_path, False)
        for key in ("door_push", "cap_approach", "fuel_port"):
            set_prim_visibility(MARKER_PRIM_PATHS[key], False)

    def _setup_door_drive(self):
        """덮개 힌지의 RevoluteJoint limit/드라이브는 이미 USD 씬에 직접 설정되어 있으므로
        코드에서 다시 만들거나 값을 덮어쓰지 않는다. set_up_scene 시점에 door_joint_prim을
        찾아서 캐시해두는 역할만 한다 (실제 캐시는 호출부인 _discover_fuel_objects에서 함).
        USD에 이미 설정된 targetVelocity도 여기서 읽어서 door_open_drive_velocity_from_usd에
        저장해둔다 - CLOSE_COVER 종료 후 복원할 때 코드의 하드코딩 상수(DOOR_AUTO_OPEN_DRIVE_VELOCITY)
        대신 이 값을 써야, USD에서 속도를 바꿔도(예: 10 -> 50) 코드가 매 Play마다 덮어써서
        도로 느려지는 일이 없다."""
        self.door_open_drive_velocity_from_usd = DOOR_AUTO_OPEN_DRIVE_VELOCITY
        if self.door_joint_prim is None:
            print("  [COVER][WARN] door_joint_prim이 없어 덮개 드라이브 설정을 건너뜀", flush=True)
            return
        joint = UsdPhysics.RevoluteJoint(self.door_joint_prim)
        if joint:
            print(f"  [COVER][DIAG] RevoluteJoint limit 확인(읽기 전용): "
                  f"lower={joint.GetLowerLimitAttr().Get()} upper={joint.GetUpperLimitAttr().Get()}")
        drive = UsdPhysics.DriveAPI.Get(self.door_joint_prim, "angular")
        if drive:
            usd_velocity = drive.GetTargetVelocityAttr().Get()
            if usd_velocity is not None:
                self.door_open_drive_velocity_from_usd = float(usd_velocity)
                print(f"  [COVER][DIAG] USD에 설정된 덮개 자동 열기 targetVelocity={usd_velocity} "
                      f"를 그대로 사용 (코드 상수 DOOR_AUTO_OPEN_DRIVE_VELOCITY={DOOR_AUTO_OPEN_DRIVE_VELOCITY}는 무시)")

    def set_door_auto_open_drive_velocity(self, target_velocity: float):
        """USD에 미리 설정된 덮개 자동 열기 velocity 드라이브(angular DriveAPI)의 targetVelocity만
        바꾼다. stiffness/damping/limit 등 나머지 속성은 USD에 설정해둔 값을 그대로 둔다.
        이 드라이브는 always-on이라 CLOSE_COVER 중에도 계속 "열림" 방향으로 힘을 내고 있어서,
        로봇이 덮개를 다 닫고 손을 떼면(set_door_angle_deg 호출이 멈추면) 덮개가 도로 열려버린다.
        그래서 CLOSE_COVER 시작 시점에 0으로 꺼주고, 다음 Play/리셋(on_play_reset)에서
        DOOR_AUTO_OPEN_DRIVE_VELOCITY로 복원해 자동 열기가 다시 동작하게 한다."""
        if self.door_joint_prim is None:
            print("  [COVER][WARN] door_joint_prim이 None이라 "
                  "set_door_auto_open_drive_velocity를 적용할 수 없음")
            return
        drive = UsdPhysics.DriveAPI.Get(self.door_joint_prim, "angular")
        if not drive:
            print("  [COVER][WARN] door_joint_prim에 angular DriveAPI가 없어 "
                  "set_door_auto_open_drive_velocity를 적용할 수 없음")
            return
        drive.GetTargetVelocityAttr().Set(float(target_velocity))
        joint = UsdPhysics.RevoluteJoint(self.door_joint_prim)
        print(f"  [COVER][DIAG] door drive targetVelocity={drive.GetTargetVelocityAttr().Get()} "
              f"(요청값={target_velocity}) type={drive.GetTypeAttr().Get()} "
              f"stiffness={drive.GetStiffnessAttr().Get()} damping={drive.GetDampingAttr().Get()} "
              f"lower={joint.GetLowerLimitAttr().Get() if joint else None} "
              f"upper={joint.GetUpperLimitAttr().Get() if joint else None}")

    def _setup_control_markers(self, scene):
        """USE_MARKER_CONTROL=True일 때 RMPFlow 목표로 직접 쓰이는 "제어용" 마커 3개를 만든다.
        _discover_fuel_objects()가 만드는 작은 디버그 표시용 박스와는 별개로, 사용자가 Isaac Sim
        뷰포트에서 이 박스를 직접 드래그하면 그 위치가 매 스텝 목표점으로 읽혀서 로봇이 따라온다."""
        scene.add(VisualCuboid(
            prim_path="/World/marker_door_push",
            name="marker_door_push",
            position=self.door_world_position,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),  # 빨간색
        ))
        scene.add(VisualCuboid(
            prim_path="/World/marker_cap_approach",
            name="marker_cap_approach",
            position=self.cap_world_position,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),  # 빨간색
        ))
        scene.add(VisualCuboid(
            prim_path="/World/marker_fuel_port",
            name="marker_fuel_port",
            position=self.hole_world_position,
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),  # 빨간색
        ))

    def current_door_angle_deg(self) -> float:
        """지금 이 순간 커버가 몇 도 열려있는지를 계산한다.
        기준 회전(리셋 시점, 30도라고 가정)에서 지금까지 회전한 양(delta)을 구해서 더한다."""
        current_rotation = get_prim_world_rotation(self.door_prim_path)
        delta = signed_angle_about_axis_deg(self.door_rest_rotation, current_rotation, self.door_axis_world)
        return COVER_START_DEG + delta

    def set_door_angle_deg(
        self,
        visual_angle_deg: float,
        label: str = "fuel_door",
        refresh_viewport: bool = True,
        verbose: bool = True,
    ) -> bool:
        """fuel_door를 원하는 시각적 각도로 맞춘다 (CLOSE_COVER에서 로봇 손 위치에 맞춰 매 스텝 호출).

        물리(RevoluteJoint state/DriveAPI)는 건드리지 않고 fuel_door prim transform만 힌지축
        기준으로 직접 회전시킨다. 닫는 동안 물리 조인트 state/limit을 매 프레임 강제로 다시
        쓰면 물리 솔버가 같은 프레임에 다른 목표로 끌어당기면서 튐/관통/역방향 같은 비정상
        움직임이 생기므로, 이 경로에서는 순수 시각적(kinematic) 회전만 적용한다."""
        prim_ok = set_door_prim_visual_angle_deg(
            self.door_prim_path,
            self.door_closed_world_matrix,
            self.door_pivot_world,
            self.door_axis_world,
            visual_angle_deg,
            label=label,
            verbose=verbose,
        )
        if refresh_viewport:
            try:
                simulation_app.update()
            except Exception:
                pass
        return prim_ok

    def post_reset(self):
        """시뮬레이션을 Play로 (재)시작할 때마다 호출. 그리퍼를 열어두고 도어의 "기준 회전"을
        다시 캐시한다(door_rest_rotation은 COVER_START_DEG=30도 상태를 기준으로 가정).
        덮개 자동 열기는 이제 코드가 아니라 USD 씬 쪽에 이미 구성되어 있으므로 여기서는
        건드리지 않는다."""
        self.robot_a.gripper.set_joint_positions(self.robot_a.gripper.joint_opened_positions)
        if getattr(self.robot_b, "gripper", None) is not None:
            self.robot_b.gripper.set_joint_positions(self.robot_b.gripper.joint_opened_positions)
        self.door_rest_rotation = get_prim_world_rotation(self.door_prim_path)


# ============================================================
# Robot A runner: WAIT_LOCK_HOLE -> RUN_SEQUENCE(기존 주유 로직)
# ============================================================
class RobotARunner:
    """A 로봇(주유 로봇)의 동작을 매 시뮬레이션 스텝마다 한 단계씩 진행시키는 state machine.
    상태는 두 가지뿐: B가 끝나길 기다리는 IDLE_WAIT_B/WAIT_LOCK_HOLE, 그리고 실제로
    주유 동작을 실행하는 RUN_SEQUENCE."""

    def __init__(self, robot, controller, ros_bridge: MultiRobotRosBridge, camera_prim_path,
                 task: "MultiRobotOilingTask"):
        self.robot = robot
        self.controller = controller
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.task = task
        self.run_state = "IDLE_WAIT_B"  # 처음엔 아무것도 안 하고 B가 끝나길 기다림
        self.lock_acquirer = None
        self.sequence = None
        self.locked_target_orientation = None  # 리셋 시점의 손목 방향을 고정해서 계속 그 방향을 유지
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = 0  # 리셋 시점의 robot_b_done_count (그 이전 신호는 무시하기 위함)

    def on_play_reset(self):
        """시뮬레이션을 Play로 시작할 때마다 호출되어 모든 상태를 깨끗하게 초기화한다."""
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = ee_ori.copy()
        self.run_state = "IDLE_WAIT_B"
        self.lock_acquirer = None
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = self.ros_bridge.robot_b_done_count
        print("[A] 대기 중: B가 덮개를 열고 마개를 분리할 때까지 기다립니다", flush=True)

    def _robot_b_done_received(self) -> bool:
        """리셋 이후에 "새로" 도착한 robot_b/done=True 신호가 있는지 확인한다.
        latched(TRANSIENT_LOCAL) QoS 때문에 리셋 전에 발행된 오래된 done 신호를
        구독 시작과 동시에 다시 받을 수 있어서, count 비교로 "이번 판"의 신호인지 구분해야 한다."""
        return (
            self.ros_bridge.robot_b_done_count > self._b_done_count_at_reset
            and self.ros_bridge.robot_b_done
        )

    def tick(self, step_count: int):
        """매 시뮬레이션 스텝마다 한 번 호출. 현재 run_state에 맞는 동작을 한 스텝만 진행한다."""
        if self.task_done:
            return
        ee_pos, _ = self.robot.end_effector.get_world_pose()

        if self.run_state == "IDLE_WAIT_B":
            if self._robot_b_done_received():
                self.ros_bridge.publish_mode_switch("hole")
                self.lock_acquirer = StableTargetLockAcquirer(
                    self.ros_bridge, self.camera_prim_path, self.task.hole_world_position, apply_mouth_offset=True,
                )
                self.wait_steps = 0
                self.run_state = "WAIT_LOCK_HOLE"
                print("\n[A] B 작업 완료 확인 -> 주유구 위치 인식 시작 (mode=hole)\n", flush=True)
            return

        if self.run_state == "WAIT_LOCK_HOLE":
            # A-8: B로부터 신호를 받아 카메라를 hole 모드로 바꾼 뒤, 주유구 위치가
            # 안정적으로 확정(lock)될 때까지 매 스텝 lock_acquirer.update()를 호출해 기다린다.
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.sequence = build_fuel_port_sequence(mean, self.ros_bridge.fuel_target_liters)
                self.run_state = "RUN_SEQUENCE"
                print(f"\n[A] 주유구 위치 확인 완료: center={np.round(mean, 4)} -> 주유 시작\n", flush=True)
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                # 너무 오래 lock이 안 되면(카메라가 못 찾거나 계속 게이트 밖이면) 재시도하지 않고
                # USD에서 읽은 하드코딩 기준값(task.hole_world_position)으로 즉시 다음 단계로 넘어간다.
                self.sequence = build_fuel_port_sequence(self.task.hole_world_position, self.ros_bridge.fuel_target_liters)
                self.run_state = "RUN_SEQUENCE"
                print(f"\n[A][WARN] 주유구 위치 인식 실패(타임아웃) -> 기준값으로 진행: "
                      f"center={np.round(self.task.hole_world_position, 4)}\n")
                return
            return

        if self.run_state == "RUN_SEQUENCE":
            if USE_MARKER_CONTROL:
                # 디버그/튜닝용: WaypointSequence 대신 뷰포트의 marker_fuel_port 위치를 매 스텝
                # 그대로 RMPFlow 목표로 사용한다. 도착하면 주유 완료로 간주하고 종료한다.
                marker_pos = get_prim_world_position(MARKER_PRIM_PATHS["fuel_port"])
                if marker_pos is not None:
                    actions = self.controller.forward(
                        target_end_effector_position=marker_pos,
                        target_end_effector_orientation=self.locked_target_orientation,
                    )
                    self.robot.apply_action(actions)
                    err = np.linalg.norm(marker_pos - ee_pos)
                    if err < POSITION_TOLERANCE:
                        self.task_done = True
                        self.ros_bridge.publish_robot_a_done(True)
                        print("\n[A] 주유구 마커 도착 -> 주유 완료 처리\n", flush=True)
                return

            # 실제 8단계 주유 시퀀스 진행. 마지막 단계(08_return_home)는 위치 목표가 없는
            # 특수 단계라 WaypointSequence가 처리하지 못하므로 여기서 별도 분기로 처리한다.
            stage = self.sequence.current
            if stage.name == "09_return_home":
                target_joints = build_initial_joint_positions(
                    self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_A,
                )
                reached = step_home_return(self.robot, target_joints)
                self.sequence.hold_count = self.sequence.hold_count + 1 if reached else 0
                if self.sequence.hold_count >= HOME_HOLD_STEPS:
                    self.task_done = True
                    self.ros_bridge.publish_robot_a_done(True)
                    print("\n[A] 주유 완료, 초기 위치 복귀 완료 -> robot_a/done 발행\n", flush=True)
                return

            done = self.sequence.update(ee_pos)
            if done:
                return
            stage = self.sequence.current
            # 이번 스텝에 가야 할 "속도제한 적용된 중간 목표점"을 구해서 RMPFlow에 넘긴다.
            cmd = self.sequence.get_command_target(ee_pos)
            if cmd is not None:
                if stage.use_orientation and USE_TARGET_ORIENTATION:
                    actions = self.controller.forward(
                        target_end_effector_position=cmd,
                        target_end_effector_orientation=self.locked_target_orientation,
                    )
                else:
                    actions = self.controller.forward(target_end_effector_position=cmd)
                self.robot.apply_action(actions)


# ============================================================
# Robot B runner: 9단계 상태머신
# WAIT_DOOR_OPEN -> WAIT_CAP -> MOVE_TO_CAP -> GRIP_UNSCREW -> RETURN_HOME_WITH_CAP ->
# WAIT_ROBOT_A -> RESTORE_CAP -> CLOSE_COVER -> FINAL_HOME
# (덮개 열기 동작 자체는 코드가 구동하지 않고 USD 씬의 자동 열기 velocity 드라이브로 진행된다.
#  current_door_angle_deg()가 이 씬에서 신뢰할 수 없는 값을 줄 수 있어, WAIT_DOOR_OPEN은 각도를
#  보는 대신 DOOR_OPEN_FIXED_WAIT_SECONDS만큼 고정 대기한 뒤 "다 열렸다"고 가정하고, ArUco
#  door lock으로 덮개의 실제 위치를 한 번 확정한다 - 이 값을
#  CLOSE_COVER에서 하드코딩 상수 대신 닫기 기준점으로 사용한다.)
# ============================================================
class RobotBRunner:
    """B 로봇(커버/마개 담당)의 동작을 매 스텝 진행시키는 state machine.
    run_state가 큰 단계를, sub_phase가 한 단계 안의 더 작은 하위 단계를 나타낸다
    (예: GRIP_UNSCREW 안에서 close_grip -> rotate -> extract 순서로 진행)."""

    def __init__(self, robot, controller, ros_bridge: MultiRobotRosBridge, camera_prim_path, task: MultiRobotOilingTask):
        self.robot = robot
        self.controller = controller
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.task = task
        self.run_state = "WAIT_DOOR_OPEN"  # 고정 시간(DOOR_OPEN_FIXED_WAIT_SECONDS) 대기부터 시작
        self.sub_phase = None
        self.lock_acquirer = None
        self.door_lock_acquirer = None
        self.sequence = None
        self.locked_target_orientation = None
        self.locked_cap_center = task.cap_world_position.copy()
        self.locked_door_center = task.door_world_position.copy()  # WAIT_DOOR_OPEN에서 door lock으로 갱신됨
        self._door_open_wait_steps = 0        # WAIT_DOOR_OPEN 고정 대기용 카운터
        self._door_open_announced = False     # "문 열림 확정" 로그를 한 번만 찍기 위한 플래그
        self.task_done = False
        self.wait_steps = 0
        self.joint6_index = None        # joint_6(마개를 돌리는 조인트)의 배열 인덱스
        self.joint6_accumulated = 0.0   # rotate sub_phase에서 지금까지 누적된 회전량(라디안)
        self.gripper_hold_count = 0
        self.frozen_joint_positions = None  # rotate 완료/screw 시작 시점의 관절값을 캡처해서 고정
        self.extract_start_ee = None        # extract 시작 시점의 EE world 위치
        self.extract_target_ee = None       # extract_start_ee에서 outward로 0.20m 이동한 목표 위치
        self.extract_step = 0               # extract sub_phase의 보간 진행 스텝 수
        self.screw_step_count = 0           # screw sub_phase에서 frozen_joint_positions 기준 누적 스텝 수
        self._a_done_count_at_reset = 0
        self.wrist_pitch_index = None
        self.wrist_pitch_name = None
        self.hand_joint_indices = []
        self.current_wrist_pitch_deg = None
        self.current_hand_pose_name = None
        self._hand_map_printed = False
        # # door close 초반에 손가락으로 밀기 위해 link_6를 돌린 orientation을 따로 저장
        # self.door_push_target_orientation = None
        # self.door_push_joint6_target = None

        # cap 잡은 뒤, 중지-엄지 사이 중심을 cap 중심에 맞추고
        # 그 cap 중심을 기준으로 팔 전체가 회전하기 위한 runtime cache
        self.cap_roll_center_world = None
        self.cap_roll_base_orientation = None
        self.cap_roll_local_center_offset = None

        self.cap_grasp_align_steps = 0
        self.cap_grasp_align_max_steps = 60
        self.cap_grasp_align_tolerance = 0.015
        # CLOSE_COVER에서 로봇 푸시 진행률에 맞춰 덮개 각도를 보간하는 데 쓰는 상태값들.
        self.cover_anim_stage_index = None
        self.cover_anim_start_pos = None
        self.cover_anim_start_angle = COVER_OPEN_DEG
        self.cover_anim_last_angle = COVER_OPEN_DEG

    def on_play_reset(self):
        """시뮬레이션을 Play로 시작할 때마다 호출되어 모든 상태를 깨끗하게 초기화하고,
        곧바로 카메라에 "door(덮개)부터 찾아"라고 명령(mode_switch=door)을 보낸다.
        덮개가 USD 씬 설정으로 자동 열리는 동안 로봇은 WAIT_DOOR_OPEN에서 가만히 기다리고,
        다 열리면 이 door lock으로 확정한 위치를 CLOSE_COVER의 닫기 기준점으로 쓴다.
        이전 판 CLOSE_COVER에서 0으로 꺼뒀을 수 있는 덮개 자동 열기 velocity 드라이브를
        다시 복원해, 이번 판에도 자동 열기가 동작하게 한다 - 코드의 하드코딩 상수가 아니라
        USD에 실제로 설정된 값(task.door_open_drive_velocity_from_usd)을 그대로 복원하므로,
        USD에서 속도를 바꿔도(예: 50) 코드가 매 Play마다 덮어쓰지 않는다."""
        self.task.set_door_auto_open_drive_velocity(self.task.door_open_drive_velocity_from_usd)
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = ee_ori.copy()
        self.run_state = "WAIT_DOOR_OPEN"
        self.sub_phase = None
        self.lock_acquirer = None  # WAIT_CAP 진입 시 생성
        # marker_to_door_xyz는 world.usda에 저장된 "닫힘(rest pose) 기준" fuel_door 좌표를
        # 기준으로 계산되어 있어서(yaml 주석 참고), 카메라로 매번 다시 재서도 결과는 항상
        # "문이 닫혀 있었다면 여기"라는 닫힘 등가 좌표가 나온다 - 문이 실제로 열려있어도 이
        # 회전을 전혀 반영하지 못한다. 그래서 게이트 기준도 열림 상태로 회전시키지 않고
        # task.door_world_position(닫힘 좌표) 그대로 써야 candidate와 같은 상태를 가리킨다.
        self.door_lock_acquirer = StableTargetLockAcquirer(
            self.ros_bridge, self.camera_prim_path, self.task.door_world_position, apply_mouth_offset=False,
        )
        self.locked_door_center = self.task.door_world_position.copy()
        self._door_open_wait_steps = 0
        self._door_open_announced = False
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self.gripper_hold_count = 0
        self.joint6_index = find_dof_index(self.robot, "joint_6")
        self.wrist_pitch_index, self.wrist_pitch_name = find_first_dof_index(self.robot, WRIST_PITCH_JOINT_CANDIDATES)
        self.hand_joint_indices = find_b_allegro_finger_joint_indices(self.robot)
        self.current_wrist_pitch_deg = None
        self.current_hand_pose_name = None
        self._hand_map_printed = False
        self.door_push_target_orientation = None
        self.door_push_joint6_target = None

        # cap 잡은 뒤, 중지-엄지 사이 중심을 cap 중심에 맞추고
        # 그 cap 중심을 기준으로 팔 전체가 회전하기 위한 runtime cache
        self.cap_roll_center_world = None
        self.cap_roll_base_orientation = None
        self.cap_roll_local_center_offset = None

        self.cap_grasp_align_steps = 0
        self.cap_grasp_align_max_steps = 60
        self.cap_grasp_align_tolerance = 0.015

        self.cap_roll_base_ee_pos = None

        if USE_B_ALLEGRO_WRIST:
            print(f"[B][ALLEGRO] wrist={self.wrist_pitch_name} index={self.wrist_pitch_index}, "
                  f"hand_dofs={len(self.hand_joint_indices)}", flush=True)
            # 접근 시작 자세는 neutral wrist + open hand.
            self._set_wrist_pitch(WRIST_PITCH_NEUTRAL_DEG, force=True)
            self._set_hand_pose(HAND_POSE_OPEN, force=True)
        if self.joint6_index is None:
            print("[B][WARN] dof_names 안에서 'joint_6'을 찾지 못함 (joint6_index=None) "
                  "- rotate/screw가 잘못된 인덱스로 동작할 수 있음")
        elif self.robot.dof_names[self.joint6_index] != "joint_6":
            print(f"[B][WARN] joint6_index={self.joint6_index} 가 'joint_6'이 아니라 "
                  f"'{self.robot.dof_names[self.joint6_index]}'를 가리킴! rotate/screw 대상이 잘못됐을 수 있음")
        self._a_done_count_at_reset = self.ros_bridge.robot_a_done_count
        set_prim_visibility(self.task.cap_prim_path, True)  # 이전 판에서 숨겨둔 마개가 있을 수 있어 리셋 시 항상 보이게 정리
        self.ros_bridge.publish_mode_switch("door")
        print("[B] 덮개 자동 열림 시작, 문이 열리는 동안 대기합니다 (mode=door)", flush=True)

    def _robot_a_done_received(self) -> bool:
        """리셋 이후 새로 도착한 robot_a/done=True 신호가 있는지 확인 (RobotARunner의 같은 패턴과 동일한 이유)."""
        return (
            self.ros_bridge.robot_a_done_count > self._a_done_count_at_reset
            and self.ros_bridge.robot_a_done
        )

    def _set_wrist_pitch(self, angle_deg: float, force: bool = False):
        """B wrist pitch를 지정 각도로 보낸다.

        cap 접근 중에는 호출하지 않는다. cap 위치에 도착한 뒤 grip 단계에서만 사용한다.
        """
        if not USE_B_ALLEGRO_WRIST or self.wrist_pitch_index is None:
            return
        angle_deg = float(angle_deg)
        if (not force) and self.current_wrist_pitch_deg is not None:
            if abs(self.current_wrist_pitch_deg - angle_deg) < 1e-4:
                return
        self.current_wrist_pitch_deg = angle_deg
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.array([np.deg2rad(angle_deg)], dtype=float),
                joint_indices=np.array([self.wrist_pitch_index], dtype=np.int32),
            )
        )

    def _set_hand_pose(self, pose_name: str, force: bool = False):
        """B Allegro hand pose를 지정한다. 같은 pose면 재명령하지 않는다."""
        if not USE_B_ALLEGRO_WRIST or not self.hand_joint_indices:
            return
        if pose_name not in ALLEGRO_POSES:
            return
        if (not force) and self.current_hand_pose_name == pose_name:
            return

        pose = flatten_allegro_pose(pose_name)
        n = min(len(pose), len(self.hand_joint_indices))

        if not self._hand_map_printed and hasattr(self.robot, "dof_names"):
            mapped = [(int(i), self.robot.dof_names[int(i)]) for i in self.hand_joint_indices[:n]]
            print(f"[B][ALLEGRO][MAP] hand pose order={mapped}", flush=True)
            self._hand_map_printed = True

        if n != len(pose):
            print(f"[B][ALLEGRO][WARN] pose '{pose_name}' len={len(pose)} but mapped_dofs={n}", flush=True)

        self.robot.apply_action(
            ArticulationAction(
                joint_positions=pose[:n],
                joint_indices=np.array(self.hand_joint_indices[:n], dtype=np.int32),
            )
        )
        self.current_hand_pose_name = pose_name

    def _rotate_robot_joint6_only(self, target_joint6_value_rad: float):
        """M0609 arm의 joint_6만 회전시킨다.

        cap을 돌릴 때 WristPitchJoint가 아니라 robot arm joint_6가 돌아가야 한다.
        전체 joint 배열을 다시 보내면 Allegro/wrist DOF까지 같이 재명령되어 손 모양이 꼬일 수 있으므로,
        여기서는 joint_6 하나만 joint_indices로 지정한다.
        """
        if self.joint6_index is None:
            print("[B][WARN] joint_6 index가 없어 cap 회전을 건너뜀", flush=True)
            return
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.array([float(target_joint6_value_rad)], dtype=float),
                joint_indices=np.array([self.joint6_index], dtype=np.int32),
            )
        )
    # def _prepare_door_finger_push_orientation(self, ee_ori):
    #     """door close 초반에 손가락으로 밀기 위해 joint_6를 90도 돌리고,
    #     그때의 EE orientation을 door close 전용 orientation으로 저장한다.

    #     주의:
    #     - RMPFlow는 target_end_effector_orientation을 계속 따라가므로,
    #     joint_6만 돌리고 orientation을 안 바꾸면 다시 원래 방향으로 돌아가려 할 수 있다.
    #     - 그래서 joint_6 목표값과 door_push_target_orientation을 같이 관리한다.
    #     """
    #     if self.joint6_index is None:
    #         print("[B][DOOR_PUSH][WARN] joint_6 index가 없어 link_6 90도 회전 생략", flush=True)
    #         self.door_push_target_orientation = ee_ori.copy()
    #         return

    #     current_joints = self.robot.get_joint_positions()

    #     # 기존 joint_6 값은 주석으로 보존
    #     # original_joint6 = current_joints[self.joint6_index]

    #     self.door_push_joint6_target = current_joints[self.joint6_index] + np.deg2rad(90.0)

    #     self._rotate_robot_joint6_only(self.door_push_joint6_target)

    #     # 바로 이 프레임에서 get_world_pose가 완전히 갱신되지 않을 수 있으므로,
    #     # 일단 현재 orientation을 기준으로 저장한다.
    #     # 실제로 방향이 덜 반영되면 아래 5번의 재적용 코드가 필요하다.
    #     self.door_push_target_orientation = ee_ori.copy()

    #     print("[B][DOOR_PUSH] door_palm_press + joint_6 90deg finger push 준비", flush=True)

    def _cap_np_quat_to_gf(self, q: np.ndarray):
        """Isaac orientation([w,x,y,z]) -> Gf.Quatd."""
        q = np.array(q, dtype=float)
        return Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))

    def _cap_gf_quat_to_np(self, q) -> np.ndarray:
        """Gf.Quatd -> Isaac orientation([w,x,y,z])."""
        imag = q.GetImaginary()
        return np.array(
            [float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])],
            dtype=float,
        )

    def _cap_transform_dir_by_orientation(self, orientation: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
        """orientation 기준 local vector를 world vector로 변환."""
        rot = Gf.Rotation(self._cap_np_quat_to_gf(orientation))
        v = Gf.Vec3d(float(local_vec[0]), float(local_vec[1]), float(local_vec[2]))
        out = rot.TransformDir(v)
        return np.array([float(out[0]), float(out[1]), float(out[2])], dtype=float)

    def _cap_inverse_transform_dir_by_orientation(self, orientation: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
        """world vector를 orientation 기준 local vector로 변환."""
        rot = Gf.Rotation(self._cap_np_quat_to_gf(orientation)).GetInverse()
        v = Gf.Vec3d(float(world_vec[0]), float(world_vec[1]), float(world_vec[2]))
        out = rot.TransformDir(v)
        return np.array([float(out[0]), float(out[1]), float(out[2])], dtype=float)

    def _cap_rotate_orientation_about_axis(self, base_orientation: np.ndarray, angle_rad: float) -> np.ndarray:
        """cap 중심축(PORT_OUTWARD_NORMAL_UNIT)을 기준으로 EE 목표 orientation 회전."""
        axis = normalize(PORT_OUTWARD_NORMAL_UNIT)
        if np.linalg.norm(axis) < 1e-9:
            return np.array(base_orientation, dtype=float).copy()

        base_q = self._cap_np_quat_to_gf(base_orientation)
        roll_q = Gf.Rotation(
            Gf.Vec3d(float(axis[0]), float(axis[1]), float(axis[2])),
            float(np.degrees(angle_rad)),
        ).GetQuat()
        return self._cap_gf_quat_to_np(roll_q * base_q)

    def _get_finger_tip_position(self, candidate_names: list[str]):
        """후보 prim 이름들 중 존재하는 첫 finger tip/link 위치를 반환."""
        for name in candidate_names:
            path = find_prim_path_by_name(ROBOT_B_PRIM_PATH, name)
            if path is None:
                continue
            pos = get_prim_world_position(path)
            if pos is not None:
                return pos
        return None

    def _get_middle_thumb_grasp_center_world(self):
        """중지 끝과 엄지 끝 사이의 중심점을 대략적인 grasp center로 계산."""
        middle_pos = self._get_finger_tip_position([
            "middle_tip",
            "middle_fingertip",
            "middle_link_3",
            "middle_link_2",
            "middle_distal",
        ])
        thumb_pos = self._get_finger_tip_position([
            "thumb_tip",
            "thumb_fingertip",
            "thumb_link_3",
            "thumb_link_2",
            "thumb_distal",
        ])

        if middle_pos is None or thumb_pos is None:
            print("[B][CAP_GRASP][WARN] middle/thumb tip prim을 찾지 못해 grasp center 정렬을 건너뜀", flush=True)
            return None

        return 0.5 * (middle_pos + thumb_pos)

    def _drive_cap_grasp_center_alignment(self, ee_pos: np.ndarray, ee_ori: np.ndarray) -> bool:
        """중지-엄지 사이 중심을 locked_cap_center에 맞추는 짧은 보정 이동."""
        grasp_center = self._get_middle_thumb_grasp_center_world()
        if grasp_center is None:
            return True

        delta = self.locked_cap_center - grasp_center
        err = float(np.linalg.norm(delta))

        if err < self.cap_grasp_align_tolerance:
            print(f"[B][CAP_GRASP] grasp center 정렬 완료 err={err:.4f}m", flush=True)
            return True

        self.cap_grasp_align_steps += 1

        # 기존 속도 파라미터를 그대로 사용해서 과격하게 점프하지 않도록 제한
        max_step = max(NEAR_TARGET_SPEED * PHYSICS_DT, 1e-5)
        if err > max_step:
            target_pos = ee_pos + delta / err * max_step
        else:
            target_pos = ee_pos + delta

        actions = self.controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=self.locked_target_orientation,
        )
        self.robot.apply_action(actions)

        # RMPFlow action 이후 손 모양 유지
        if USE_B_ALLEGRO_WRIST:
            self._set_wrist_pitch(WRIST_PITCH_CAP_GRASP_DEG)
            self._set_hand_pose(HAND_POSE_CAP_HOLD)

        if self.cap_grasp_align_steps >= self.cap_grasp_align_max_steps:
            print(f"[B][CAP_GRASP][WARN] grasp center 정렬 timeout err={err:.4f}m -> 다음 단계 진행", flush=True)
            return True

        return False

    def _begin_cap_center_roll(self, ee_pos: np.ndarray, ee_ori: np.ndarray):
        """cap 중심을 고정점으로 잡고, 팔 전체가 그 주변을 돌기 위한 기준값 캡처."""
        self.cap_roll_center_world = np.array(self.locked_cap_center, dtype=float).copy()
        self.cap_roll_base_orientation = np.array(ee_ori, dtype=float).copy()
        self.cap_roll_base_ee_pos = np.array(ee_pos, dtype=float).copy()

        center_offset_world = self.cap_roll_center_world - np.array(ee_pos, dtype=float)
        self.cap_roll_local_center_offset = self._cap_inverse_transform_dir_by_orientation(
            self.cap_roll_base_orientation,
            center_offset_world,
        )

        print(
            f"[B][CAP_ROLL] center={np.round(self.cap_roll_center_world, 4)}, "
            f"local_offset={np.round(self.cap_roll_local_center_offset, 4)}",
            flush=True,
        )

    def _get_cap_center_roll_target(self, angle_rad: float):
        """angle_rad만큼 cap 축 기준 회전했을 때의 EE 목표 position/orientation 계산."""
        if (
            self.cap_roll_center_world is None
            or self.cap_roll_base_orientation is None
            or self.cap_roll_local_center_offset is None
        ):
            return None, None

        target_ori = self._cap_rotate_orientation_about_axis(
            self.cap_roll_base_orientation,
            angle_rad,
        )
        center_offset_world = self._cap_transform_dir_by_orientation(
            target_ori,
            self.cap_roll_local_center_offset,
        )
        target_pos = self.cap_roll_center_world - center_offset_world
        return target_pos, target_ori

    def _apply_cap_center_roll_step(self, angle_rad: float) -> bool:
        """cap 중심 기준 회전 목표를 RMPFlow로 한 스텝 적용."""
        target_pos, target_ori = self._get_cap_center_roll_target(angle_rad)
        if target_pos is None:
            return False

        actions = self.controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_ori,
        )
        self.robot.apply_action(actions)

        if USE_B_ALLEGRO_WRIST:
            self._set_wrist_pitch(WRIST_PITCH_CAP_GRASP_DEG)
            self._set_hand_pose(HAND_POSE_CAP_HOLD)

        return True

    def _hold_gripper(self, closed: bool):
        """B gripper/Allegro 상태 유지.

        Allegro 모드에서는 접근 중 wrist를 꺾지 않고 손 open/close pose만 유지한다.
        cap 회전은 wrist pitch가 아니라 기존 joint_6 회전으로 수행한다.
        """
        if USE_B_ALLEGRO_WRIST:
            self._set_hand_pose(HAND_POSE_CAP_HOLD if closed else HAND_POSE_OPEN)
            return
        if getattr(self.robot, "gripper", None) is None:
            return
        target = GRIPPER_CLOSE if closed else GRIPPER_OPEN
        self.robot.gripper.set_joint_positions(np.array(target, dtype=float))

    def _drive_sequence(self, ee_pos, extra_condition_ok=True):
        """현재 self.sequence(WaypointSequence)를 한 스텝 진행. 완료 여부를 반환.
        RobotARunner의 RUN_SEQUENCE 본문과 같은 패턴이라 여러 run_state에서 재사용한다."""
        done = self.sequence.update(ee_pos, extra_condition_ok=extra_condition_ok)
        if done:
            return True
        stage = self.sequence.current
        cmd = self.sequence.get_command_target(ee_pos)
        if cmd is not None:
            if stage.use_orientation and USE_TARGET_ORIENTATION:
                actions = self.controller.forward(
                    target_end_effector_position=cmd,
                    target_end_effector_orientation=self.locked_target_orientation,
                )
            else:
                actions = self.controller.forward(target_end_effector_position=cmd)
            self.robot.apply_action(actions)
        return False

    def _reset_cover_animation(self, start_angle: float = COVER_OPEN_DEG):
        """CLOSE_COVER에서 로봇 EE 이동 진행률에 맞춰 fuel_door를 부드럽게 회전시키기 위한 상태 초기화."""
        self.cover_anim_stage_index = None
        self.cover_anim_start_pos = None
        self.cover_anim_start_angle = float(start_angle)
        self.cover_anim_last_angle = float(start_angle)

    def _sync_cover_door_angle_from_hand(self, stage: FuelStage, ee_pos: np.ndarray):
        """fuel_door 각도를 로봇 손(EE)의 "실제" world 위치에 맞춰 갱신한다 (물리 미사용, 순수 시각적).

        cmd(속도 제한된 중간 목표)가 아니라 실제 ee_pos를 기준으로 진행률(0~1)을 구해서,
        그 비율만큼 door_angle을 stage 시작 각도 -> target_door_angle로 보간한다. RMPFlow가
        목표를 못 따라가서 손이 늦게 도착해도 문은 항상 "지금 손이 있는 자리"만큼만 닫히므로,
        문이 손보다 먼저 닫히거나 뚫고 지나가는 일 없이 손을 그대로 따라간다."""
        if stage is None or stage.target_position is None or stage.target_door_angle is None:
            return

        if self.cover_anim_stage_index != self.sequence.index:
            self.cover_anim_stage_index = self.sequence.index
            self.cover_anim_start_pos = np.array(ee_pos, dtype=float).copy()
            self.cover_anim_start_angle = float(self.cover_anim_last_angle)

        start_pos = np.array(self.cover_anim_start_pos, dtype=float)
        end_pos = np.array(stage.target_position, dtype=float)
        ee_pos = np.array(ee_pos, dtype=float)
        move_vec = end_pos - start_pos
        denom = float(np.dot(move_vec, move_vec))

        if denom < 1e-9:
            progress = 1.0
        else:
            progress = float(np.dot(ee_pos - start_pos, move_vec) / denom)
            progress = float(np.clip(progress, 0.0, 1.0))

        target_angle = float(stage.target_door_angle)
        angle = self.cover_anim_start_angle + (target_angle - self.cover_anim_start_angle) * progress

        # 닫는 중에는 수치 오차 때문에 각도가 다시 열리는 방향으로 튀지 않도록 범위를 제한한다.
        lo = min(self.cover_anim_start_angle, target_angle)
        hi = max(self.cover_anim_start_angle, target_angle)
        angle = float(np.clip(angle, lo, hi))

        self.task.set_door_angle_deg(
            angle,
            label="fuel_door close follow",
            refresh_viewport=False,
            verbose=False,
        )
        self.cover_anim_last_angle = angle

    def _door_close_pose_for_stage(self, stage_name: str | None):
        """fuel_door 닫기 stage별 wrist/hand pose."""
        if not USE_B_ALLEGRO_WRIST:
            return None, None
        if stage_name in {
            "B10_00_avoid_door",
            "B10_01_move_right",
        }:
            return HAND_POSE_OPEN, WRIST_PITCH_NEUTRAL_DEG
        if stage_name in {
            "B10_02_forward_entry",
        }:
            return HAND_POSE_DOOR_PALM_PUSH, WRIST_PITCH_NEUTRAL_DEG
        if stage_name in {
            "B10_03_push_left_mid",
            "B10_04_push_left_start",
            "B10_05_retreat_for_press",
        }:
            return HAND_POSE_DOOR_PALM_PUSH, WRIST_PITCH_DOOR_BENT_DEG
        if stage_name in {
            "B10_06_center_align_press",
            "B10_07_final_press",
        }:
            return HAND_POSE_DOOR_FINAL_PRESS, WRIST_PITCH_DOOR_PRESS_DEG
        return HAND_POSE_OPEN, WRIST_PITCH_NEUTRAL_DEG

    def _drive_close_cover_sequence(self, ee_pos):
        """CLOSE_COVER 전용 sequence 구동. 일반 _drive_sequence와 달리, 로봇 손의 실제 위치에
        맞춰 fuel_door 시각 각도를 매 스텝 동시에 갱신한다(물리 조인트는 건드리지 않음)."""
        done = self.sequence.update(ee_pos)
        if done:
            return True
        stage = self.sequence.current
        if USE_B_ALLEGRO_WRIST:
            pose_name, wrist_deg = self._door_close_pose_for_stage(stage.name)
            if wrist_deg is not None:
                self._set_wrist_pitch(wrist_deg)
            if pose_name is not None:
                self._set_hand_pose(pose_name)
        self._sync_cover_door_angle_from_hand(stage, ee_pos)
        cmd = self.sequence.get_command_target(ee_pos)
        if cmd is not None:
            if stage.use_orientation and USE_TARGET_ORIENTATION:
                actions = self.controller.forward(
                    target_end_effector_position=cmd,
                    target_end_effector_orientation=self.locked_target_orientation,
                )
            else:
                actions = self.controller.forward(target_end_effector_position=cmd)

            self.robot.apply_action(actions)
            # door close 초반 stage에서는 손가락으로 밀기 위해 joint_6 회전 목표를 유지
            if (
                USE_B_ALLEGRO_WRIST
                and self.door_push_joint6_target is not None
                and stage.name in {
                    "B10_01_move_right",
                    "B10_02_forward_entry",
                    "B10_03_push_left_mid",
                    "B10_04_push_left_start",
                }
            ):
                self._rotate_robot_joint6_only(self.door_push_joint6_target)
        return False

    def tick(self, step_count: int):
        """매 시뮬레이션 스텝마다 호출. run_state(와 필요하면 sub_phase)에 따라 분기해서 한 스텝 진행."""
        if self.task_done:
            return
        # ee_pos, _ = self.robot.end_effector.get_world_pose()
        ee_pos, ee_ori = self.robot.end_effector.get_world_pose()

        # ---------------- B-3: WAIT_DOOR_OPEN ----------------
        # 덮개가 USD 자동 열기 velocity 드라이브로 다 열리기 전에 마개 쪽으로 움직이면 아직
        # 열리는 중인 덮개와 부딫힌다. current_door_angle_deg()가 이 씬에서는 비정상적인 값
        # (예: -141.8도)을 줘서 각도 기반 정체 감지가 오작동했으므로, 그 측정을 신뢰하지 않고
        # 그냥 DOOR_OPEN_FIXED_WAIT_SECONDS(5초)만큼 고정으로 기다린 뒤 "다 열렸다"고 가정한다.
        if self.run_state == "WAIT_DOOR_OPEN":
            self._door_open_wait_steps += 1
            if self._door_open_wait_steps < DOOR_OPEN_WAIT_STEPS:
                return

            if not self._door_open_announced:
                self._door_open_announced = True
                print(f"\n[B] 덮개 열림 대기 {DOOR_OPEN_FIXED_WAIT_SECONDS:.0f}초 완료 -> "
                      "덮개 위치 확인 시작\n")

            # 고정 대기가 끝났으니, 이제부터 ArUco door lock으로 덮개의 실제 위치를 확정한다
            # (CLOSE_COVER에서 하드코딩 상수 대신 이 값을 닫기 기준점으로 쓴다).
            self.wait_steps += 1
            mean = self.door_lock_acquirer.update()
            if mean is not None:
                self.locked_door_center = mean
                print(f"\n[B] 덮개 위치 확인 완료: center={np.round(mean, 4)}\n", flush=True)
            elif self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                self.locked_door_center = self.task.door_world_position.copy()
                print(f"\n[B][WARN] 덮개 위치 확인 실패(타임아웃) -> 기준값으로 진행: "
                      f"center={np.round(self.locked_door_center, 4)}\n")
            else:
                return

            self.wait_steps = 0
            self.run_state = "WAIT_CAP"
            self.lock_acquirer = StableTargetLockAcquirer(
                self.ros_bridge, self.camera_prim_path, self.task.cap_world_position, apply_mouth_offset=False,
            )
            self.ros_bridge.publish_mode_switch("cap")
            print("[B] 마개 위치 인식 시작 (mode=cap)", flush=True)
            return

        # ---------------- B-4: WAIT_CAP ----------------
        # 문이 다 열린 뒤에만 이 상태에 들어오므로, cap(마개) lock에만 집중한다.
        if self.run_state == "WAIT_CAP":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.locked_cap_center = mean + CAP_POSITION_OFFSET
                self.sequence = build_cap_approach_sequence(self.locked_cap_center)
                self.run_state = "MOVE_TO_CAP"
                print(f"\n[B] 마개 위치 확인 완료: center={np.round(self.locked_cap_center, 4)} -> 마개로 이동\n", flush=True)
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                # cap(마개) lock이 끝까지 안정되지 않으면 재시도하지 않고, USD에서 읽은
                # 하드코딩 기준값(task.cap_world_position)으로 즉시 다음 단계로 넘어간다.
                self.locked_cap_center = self.task.cap_world_position.copy() + CAP_POSITION_OFFSET
                self.sequence = build_cap_approach_sequence(self.locked_cap_center)
                self.run_state = "MOVE_TO_CAP"
                print(f"\n[B][WARN] 마개 위치 확인 실패(타임아웃) -> 기준값으로 진행: "
                      f"center={np.round(self.locked_cap_center, 4)}\n")
                return
            return

        # ---------------- B-5: MOVE_TO_CAP ----------------
        # 마개 위치가 lock 되었으니 그리퍼를 연 채로 마개 쪽으로 접근한다(far -> near -> grasp 3단계).
        if self.run_state == "MOVE_TO_CAP":
            self._hold_gripper(closed=False)
            if USE_MARKER_CONTROL:
                # 디버그/튜닝용: WaypointSequence 대신 뷰포트의 marker_cap_approach 위치를 목표로 쓴다.
                marker_pos = get_prim_world_position(MARKER_PRIM_PATHS["cap_approach"])
                if marker_pos is not None:
                    actions = self.controller.forward(
                        target_end_effector_position=marker_pos,
                        target_end_effector_orientation=self.locked_target_orientation,
                    )
                    self.robot.apply_action(actions)
                    err = np.linalg.norm(marker_pos - ee_pos)
                    if err < POSITION_TOLERANCE:
                        self.sub_phase = "close_grip"
                        self.gripper_hold_count = 0
                        self.run_state = "GRIP_UNSCREW"
                        print("\n[B] 마개 위치 도착 -> 그리퍼 닫는 중\n", flush=True)
                return

            done = self._drive_sequence(ee_pos)
            if done:
                self.sub_phase = "close_grip"
                self.gripper_hold_count = 0
                self.run_state = "GRIP_UNSCREW"
                print("\n[B] 마개 분리 시작: 그리퍼를 닫는 중\n", flush=True)
            return

        # ---------------- B-6: GRIP_UNSCREW (close_grip -> rotate -> extract) ----------------
        # 그리퍼로 마개를 잡고(close_grip) -> joint_6을 -360도 돌려서 마개를 풀고(rotate)
        # -> 그리퍼를 닫은 채로 빼낸다(extract). rotate 구간은 RMPFlow를 호출하지 않고
        # joint_6만 apply_action으로 직접 증분시킨다 (나머지 조인트는 매 스텝 현재값을 그대로 재전송해서
        # 자세가 흐트러지지 않게 유지).
        if self.run_state == "GRIP_UNSCREW":
            if self.sub_phase == "close_grip":
                if USE_B_ALLEGRO_WRIST:
                    # 접근은 기존 waypoint 그대로 끝낸 뒤, cap 위치에 도착한 상태에서만
                    # wrist를 꺾고 손가락으로 cap을 잡는다.
                    self._set_wrist_pitch(WRIST_PITCH_CAP_GRASP_DEG)
                    self._set_hand_pose(HAND_POSE_CAP_HOLD)
                else:
                    self._hold_gripper(closed=True)
                self.gripper_hold_count += 1
                # if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                #     # 그리퍼가 마개를 잡았으니, 회전/추출/복귀 내내는 "손에 들고 빠진 것처럼"
                #     # 보이도록 마개를 숨긴다 (실제 물리 부착은 하지 않음).
                #     set_prim_visibility(self.task.cap_prim_path, False)
                #     self.joint6_accumulated = 0.0
                #     self.sub_phase = "rotate"
                #     print(f"\n[B] 그리퍼로 마개 고정 완료 -> 마개 풀기 시작 "
                #           f"(목표 {abs(np.degrees(UNSCREW_TOTAL_ANGLE_RAD)):.0f}도)\n")
                # return
                if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                    # 기존 코드 보존:
                    # set_prim_visibility(self.task.cap_prim_path, False)
                    #
                    # cap을 바로 숨기기 전에, 중지-엄지 사이 중심을 cap 중심에 맞추는 보정 단계로 이동.
                    self.cap_grasp_align_steps = 0
                    self.sub_phase = "align_grasp_center"
                    print("\n[B] cap grasp 완료 -> 중지/엄지 사이 중심을 cap 중심에 정렬 시작\n", flush=True)
            
            if self.sub_phase == "align_grasp_center":
                self._hold_gripper(closed=True)

                aligned = self._drive_cap_grasp_center_alignment(ee_pos, ee_ori)
                if aligned:
                    # 정렬이 끝난 현재 위치/방향을 기준으로 cap 중심 roll 기준을 캡처한다.
                    ee_pos_now, ee_ori_now = self.robot.end_effector.get_world_pose()
                    self._begin_cap_center_roll(ee_pos_now, ee_ori_now)

                    self.joint6_accumulated = 0.0
                    self.sub_phase = "rotate"
                    print(f"\n[B] cap 중심 정렬 완료 -> cap 중심 기준 팔 회전으로 마개 풀기 시작 "
                        f"(목표 {abs(np.degrees(UNSCREW_TOTAL_ANGLE_RAD)):.0f}도)\n", flush=True)
                return

            # if self.sub_phase == "rotate":
            #     self._hold_gripper(closed=True)
            #     current_joints = self.robot.get_joint_positions()
            #     target_joint6 = current_joints[self.joint6_index] + UNSCREW_ANGLE_STEP_RAD
            #     self._rotate_robot_joint6_only(target_joint6)
            #     self.joint6_accumulated += UNSCREW_ANGLE_STEP_RAD
            #     if abs(self.joint6_accumulated) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
            if self.sub_phase == "rotate":
                self._hold_gripper(closed=True)

                next_angle = self.joint6_accumulated + UNSCREW_ANGLE_STEP_RAD
                applied = self._apply_cap_center_roll_step(next_angle)

                if not applied:
                    # cap 중심 roll 기준 캡처가 실패하면 기존 joint_6 단독 회전으로 fallback
                    current_joints = self.robot.get_joint_positions()
                    target_joint6 = current_joints[self.joint6_index] + UNSCREW_ANGLE_STEP_RAD
                    self._rotate_robot_joint6_only(target_joint6)

                self.joint6_accumulated = next_angle

                if abs(self.joint6_accumulated) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
                    # 회전이 끝난 뒤부터 cap을 숨겨서 extract/복귀는 기존 흐름 유지
                    set_prim_visibility(self.task.cap_prim_path, False)
                    # frozen_joint_positions로 자세를 고정하는 방식이 extract에서 제대로 안 먹혀서
                    # RMPFlow가 끼어들 때 팔이 아래로 처지는 문제가 있었다. 이제 extract는
                    # WaypointSequence/_drive_sequence를 전혀 쓰지 않고, "지금 EE 위치에서
                    # 바깥쪽으로 0.20m"라는 상대 목표를 고정 스텝 수(EXTRACT_TOTAL_STEPS)에 걸쳐
                    # 직접 선형보간해서 매 스텝 RMPFlow에 넘긴다.
                    self.frozen_joint_positions = self.robot.get_joint_positions().copy()
                    self.extract_start_ee = ee_pos.copy()
                    self.extract_target_ee = ee_pos + PORT_OUTWARD_NORMAL_UNIT * 0.20
                    self.extract_step = 0
                    self.sub_phase = "extract"
                    print(f"\n[B] 마개 풀기 완료({abs(np.degrees(self.joint6_accumulated)):.0f}도) -> "
                          f"마개를 빼내는 중\n")
                return

            if self.sub_phase == "extract":
                self._hold_gripper(closed=True)
                self.extract_step += 1
                t = min(self.extract_step / EXTRACT_TOTAL_STEPS, 1.0)
                interp_pos = self.extract_start_ee + t * (self.extract_target_ee - self.extract_start_ee)
                actions = self.controller.forward(
                    target_end_effector_position=interp_pos,
                    target_end_effector_orientation=self.locked_target_orientation,
                )
                self.robot.apply_action(actions)
                if t >= 1.0:
                    # GRIP_UNSCREW의 extract는 마개를 쥔 채 끝까지 안 놓아야 하므로(놓는 건
                    # RESTORE_CAP의 open_grip에서) 다음은 RETURN_HOME_WITH_CAP으로 간다.
                    target_joints = build_initial_joint_positions(
                        self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_B,
                    )
                    self._home_target_joints = target_joints
                    self._home_hold_count = 0
                    self.run_state = "RETURN_HOME_WITH_CAP"
                    print(f"\n[B] 마개 추출 완료: ee_pos={np.round(ee_pos, 3)} -> 초기 위치로 복귀\n", flush=True)
                return

        # ---------------- B-7: RETURN_HOME_WITH_CAP ----------------
        # 마개를 그리퍼로 꼭 쥔 채(닫힌 상태 유지) 처음 기본 자세로 복귀한다.
        if self.run_state == "RETURN_HOME_WITH_CAP":
            self._hold_gripper(closed=True)
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.ros_bridge.publish_robot_b_done(True)
                self.run_state = "WAIT_ROBOT_A"
                print("\n[B] 마개를 든 채 초기 위치 복귀 완료 -> A의 주유 작업 대기\n", flush=True)
            return

        # ---------------- WAIT_ROBOT_A (A-8 동안 대기) ----------------
        # A가 주유를 끝낼 때까지 B는 마개를 그리퍼로 꼭 쥔 채 아무것도 안 하고 기다린다.
        if self.run_state == "WAIT_ROBOT_A":
            self._hold_gripper(closed=True)
            if self._robot_a_done_received():
                # hole_world_position(주유구 구멍)은 cap_center(마개 표면)보다 outward-normal
                # 방향으로 약 0.227m 더 안쪽이라, 똑같은 GRIPPER_LENGTH_B 오프셋을 적용해도
                # 닫을 때가 열 때보다 훨씬 깊이 들어가는 원인이었다. locked_cap_center(마개를
                # 집을 때 카메라로 lock했던 바로 그 위치) 기준으로 바꿔서 여는 쪽과 깊이를 맞춘다.
                self.sequence = build_cap_restore_sequence(self.locked_cap_center)
                self.sub_phase = "insert"
                self.run_state = "RESTORE_CAP"
                print(f"\n[B] A의 주유 완료 확인 -> 마개 재장착 시작: "
                      f"target={np.round(self.locked_cap_center, 4)}\n")
            return

        # ---------------- B-9: RESTORE_CAP (insert -> screw -> open_grip) ----------------
        # 주유구 구멍으로 마개를 다시 가져가 끼우고(insert) -> joint_6을 +360도 돌려서 조이고(screw)
        # -> 그리퍼를 열어 마개를 놓는다(open_grip). GRIP_UNSCREW의 정반대 순서.
        # (예전엔 "sub_phase != 'screw'"로 검사해서 open_grip 단계에서도 insert 코드가 잘못 실행되는
        #  버그가 있었다 - 지금은 각 sub_phase를 명시적으로 분기해서 그 문제를 막았다.)
        if self.run_state == "RESTORE_CAP":
            if self.sub_phase == "insert":
                self._hold_gripper(closed=True)
                done = self._drive_sequence(ee_pos)
                if done:
                    # screw 시작 시점의 관절값을 한 번 캡처해서 고정해두고, 이후 매 스텝은 이
                    # 고정된 기준값에서 joint_6만 누적 스텝 수만큼 더해 계산한다 (rotate처럼 매번
                    # get_joint_positions()를 다시 읽지 않으므로, 측정값 드리프트와 무관하게
                    # 나머지 관절(1~5)이 흔들리지 않는다).
                    # self.frozen_joint_positions = self.robot.get_joint_positions().copy()
                    # self.joint6_accumulated = 0.0
                    # self.screw_step_count = 0
                    # self.sub_phase = "screw"
                    # print(f"\n[B] 마개 삽입 위치 도착 -> 마개 조이기 시작 "
                    #       f"(목표 {abs(np.degrees(UNSCREW_TOTAL_ANGLE_RAD)):.0f}도)\n")
                    self.frozen_joint_positions = self.robot.get_joint_positions().copy()
                    self.joint6_accumulated = 0.0
                    self.screw_step_count = 0

                    # 조이기도 현재 위치를 기준으로 cap 중심 roll 기준을 다시 캡처
                    ee_pos_now, ee_ori_now = self.robot.end_effector.get_world_pose()
                    self._begin_cap_center_roll(ee_pos_now, ee_ori_now)

                    self.sub_phase = "screw"
                    print(f"\n[B] 마개 삽입 위치 도착 -> cap 중심 기준 팔 회전으로 마개 조이기 시작 "
                        f"(목표 {abs(np.degrees(UNSCREW_TOTAL_ANGLE_RAD)):.0f}도)\n")
                return

            # if self.sub_phase == "screw":
            #     self._hold_gripper(closed=True)
            #     # screw는 rotate의 정반대 방향으로 robot arm joint_6만 돌린다.
            #     # 전체 joint 배열을 보내지 않아 wrist/Allegro finger pose가 같이 꼬이는 것을 막는다.
            #     self.screw_step_count += 1
            #     base_joint6 = self.frozen_joint_positions[self.joint6_index]
            #     target_joint6 = base_joint6 - UNSCREW_ANGLE_STEP_RAD * self.screw_step_count
            #     self._rotate_robot_joint6_only(target_joint6)
            #     if abs(self.screw_step_count * UNSCREW_ANGLE_STEP_RAD) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
            if self.sub_phase == "screw":
                self._hold_gripper(closed=True)

                # rotate의 반대 방향으로 cap 중심 기준 팔 회전
                self.screw_step_count += 1
                next_angle = -UNSCREW_ANGLE_STEP_RAD * self.screw_step_count

                applied = self._apply_cap_center_roll_step(next_angle)

                if not applied:
                    # cap 중심 roll 기준 캡처 실패 시 기존 joint_6 단독 회전으로 fallback
                    base_joint6 = self.frozen_joint_positions[self.joint6_index]
                    target_joint6 = base_joint6 - UNSCREW_ANGLE_STEP_RAD * self.screw_step_count
                    self._rotate_robot_joint6_only(target_joint6)

                if abs(next_angle) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
                    # 마개가 다시 끼워졌으니 숨겨뒀던 마개를 다시 보이게 해서 "복원됐다"는 걸 표현한다.
                    set_prim_visibility(self.task.cap_prim_path, True)
                    self.gripper_hold_count = 0
                    self.sub_phase = "open_grip"
                    print(f"\n[B] 마개 조이기 완료({abs(np.degrees(self.screw_step_count * UNSCREW_ANGLE_STEP_RAD)):.0f}도) "
                          f"-> 그리퍼 여는 중\n")
                return

            if self.sub_phase == "open_grip":
                self._hold_gripper(closed=False)
                self.gripper_hold_count += 1
                if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                    # 그리퍼를 열어 마개를 놓은 직후 바로 닫기 시퀀스로 넘어가면 손이 마개/주유구
                    # 바로 옆에 있는 채로 다음 동작이 시작되므로, 바깥쪽으로 0.20m 직선 후퇴를
                    # 한 번 거친다. hole_world_position 같은 고정값 기준으로 잡으면 insert/screw가
                    # 타임아웃으로 목표에 못 미친 채 끝났을 때 EE와 거리가 멀어 또 타임아웃이 나므로,
                    # "지금 실제 EE 위치" 기준으로 동적으로 계산한다.
                    ee_pos_now, ee_ori_now = self.robot.end_effector.get_world_pose()
                    self.locked_target_orientation = ee_ori_now.copy()

                    retreat_target = ee_pos + PORT_OUTWARD_NORMAL_UNIT * 0.08
                    self.sequence = single_stage_sequence(
                        "B9_extract_retreat", retreat_target,
                        hold_steps=5, tolerance=0.15, speed=RETREAT_TARGET_SPEED, max_steps=450,
                        use_orientation=False,
                    )
                    self.sub_phase = "retreat"
                    print(f"\n[B] 마개 재장착 완료, 그리퍼 열림 -> 후퇴 중: target={np.round(retreat_target, 4)}\n", flush=True)
                return

            if self.sub_phase == "retreat":
                self._hold_gripper(closed=False)
                done = self._drive_sequence(ee_pos)
                if done:
                    # DriveAPI로 한번에 0도까지 구동하면 덮개가 로봇 모션과 무관하게 움직여서
                    # 부자연스러웠다. 그래서 로봇이 다시 덮개를 밀면서, 그 진행률에 맞춰
                    # set_door_angle_deg()로 덮개 각도를 같이 120 -> 75 -> 30도로 보간한다
                    # (multi_robot_oiling_start_open.py에서 이식).
                    # 덮개 자동 열기 velocity 드라이브는 always-on이라 그대로 두면 CLOSE_COVER
                    # 중에도 계속 "열림" 방향으로 힘을 내고 있다가, 로봇이 다 닫고 손을 떼는 순간
                    # (set_door_angle_deg 호출이 멈추는 순간) 덮개를 도로 열어버린다. 그래서 닫기
                    # 시작 시점에 0으로 꺼서 더 이상 못 열게 막는다(다음 판 on_play_reset에서 복원).
                    self.task.set_door_auto_open_drive_velocity(0.0)
                    # door_world_position(닫힘 기준 하드코딩/USD 스냅샷) 대신, WAIT_DOOR_OPEN에서
                    # ArUco door lock으로 확정한 위치(locked_door_center)를 쓴다. marker_to_door_xyz가
                    # world.usda의 "닫힘(rest pose)" 좌표 기준으로 계산되어 있어서, 이 lock 결과는
                    # door_world_position과 같은 "닫힘 등가" 좌표이므로 역회전 없이 그대로
                    # door_reference_point/center_reference_point로 쓸 수 있다(원본과 동일한 구조).

                    # ee_pos_now, ee_ori_now = self.robot.end_effector.get_world_pose()
                    # self.locked_target_orientation = ee_ori_now.copy()

                    door_close_push_point = self.locked_door_center.copy()
                    door_close_push_point[0] += DOOR_CLOSE_PUSH_OFFSET
                    avoid_point = self.locked_cap_center + PORT_OUTWARD_NORMAL_UNIT * COVER_CLEARANCE_DISTANCE
                    self.sequence = build_close_cover_sequence(
                        door_close_push_point, self.task.door_pivot_world, self.task.door_axis_world,
                        avoid_point=avoid_point,
                        center_reference_point=self.locked_door_center.copy(),
                    )
                    self.sub_phase = None
                    self._reset_cover_animation(COVER_OPEN_DEG)
                    self.task.set_door_angle_deg(
                        COVER_OPEN_DEG, label="fuel_door close start", refresh_viewport=False, verbose=False,
                    )
                    if USE_B_ALLEGRO_WRIST:
                        self._set_wrist_pitch(WRIST_PITCH_DOOR_BENT_DEG)
                        self._set_hand_pose(HAND_POSE_DOOR_PALM_PUSH)

                    #     # # door close 초반은 손가락으로 밀기 위해 link_6를 90도 돌린다.
                    #     # ee_pos_now, ee_ori_now = self.robot.end_effector.get_world_pose()
                    #     # self._prepare_door_finger_push_orientation(ee_ori_now)

                    self.run_state = "CLOSE_COVER"
                    print(f"\n[B] 후퇴 완료 -> 덮개 닫기 시작 "
                          f"({COVER_OPEN_DEG:.0f}도 -> {COVER_START_DEG:.0f}도 -> "
                          f"{COVER_FULL_CLOSE_DEG:.0f}도)\n")
            return

        # ---------------- B-10: CLOSE_COVER ----------------
        # 로봇이 덮개를 다시 밀면서 닫는다(120->75->30도 역순). 덮개 시각 각도는 마지막에 강제
        # 스냅하지 않고, _drive_close_cover_sequence() 안에서 로봇 손의 실제 위치 진행률에
        # 맞춰 계속 갱신된다(물리 조인트는 사용하지 않음 - 순수 시각적 회전).
        if self.run_state == "CLOSE_COVER":
            done = self._drive_close_cover_sequence(ee_pos)
            if done:
                # 이미 B10_03 진행 중에 덮개가 로봇 모션과 함께 닫혔으므로, 여기서 다시
                # set_door_angle_deg를 보정 호출하면 마지막에 튀어 보일 수 있어 호출하지 않는다.
                target_joints = build_initial_joint_positions(
                    self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_B,
                )
                self._home_target_joints = target_joints
                self._home_hold_count = 0
                self.run_state = "FINAL_HOME"
                print(f"\n[B] 덮개 닫힘 완료(목표 {COVER_FULL_CLOSE_DEG:.0f}도) -> 초기 위치로 복귀\n", flush=True)
            return

        # ---------------- B-11: FINAL_HOME ----------------
        # 모든 작업이 끝났으니 기본 자세로 돌아가 멈춘다. 여기가 끝나면 B의 전체 시퀀스가 종료된다.
        if self.run_state == "FINAL_HOME":
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.task_done = True
                print("\n[B] 초기 위치 복귀 완료, 작업 종료\n", flush=True)
            return


# ╔══════════════════════════════════════════════════════════════╗
# ║  F. 메인                                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    """프로그램 진입점. 씬을 만들고, 로봇/컨트롤러/ROS 노드를 준비한 다음
    Isaac Sim의 메인 루프를 돌면서 매 스텝마다 두 로봇의 state machine을 진행시킨다."""
    my_world = World(stage_units_in_meters=1.0)
    task = MultiRobotOilingTask(name="multi_robot_oiling_task")
    my_world.add_task(task)
    my_world.reset()  # 이 시점에 MultiRobotOilingTask.set_up_scene()이 내부적으로 호출된다

    robot_a = my_world.scene.get_object("m0609_A")
    robot_b = my_world.scene.get_object("m0609_B")

    initialize_robot(robot_a, my_world, ROBOT_A_BASE_WORLD, ROBOT_A_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_A)
    if USE_B_ALLEGRO_WRIST:
        initialize_robot_b(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)
    else:
        initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)

    # 초기화 직후 물리엔진이 안정화되도록 몇 프레임 그냥 흘려보낸다.
    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60, flush=True)
    print("[C-1] 초기 상태", flush=True)
    print("=" * 60, flush=True)
    print(f"  m0609_A base = {ROBOT_A_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG_A}", flush=True)
    print(f"  m0609_B base = {ROBOT_B_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG_B}", flush=True)

    # RMPFlow: 목표 위치/방향을 주면 그쪽으로 손목이 가도록 알아서 각 조인트 각도를 계산해주는
    # 역기구학(IK) 기반 모션 플래너. A/B 로봇마다 따로 하나씩 필요하다.
    controller_a = RMPFlowController(
        name="m0609_A_rmpflow_controller",
        robot_articulation=robot_a,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    controller_b = RMPFlowController(
        name="m0609_B_rmpflow_controller",
        robot_articulation=robot_b,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] RMPFlowController A/B 생성 완료", flush=True)

    if not rclpy.ok():
        rclpy.init(args=None)
    ros_bridge = MultiRobotRosBridge()
    camera_prim_path = find_camera_prim_path()
    print(f"  [ROS] camera_prim_path = {camera_prim_path}", flush=True)

    runner_a = RobotARunner(robot_a, controller_a, ros_bridge, camera_prim_path, task)
    runner_b = RobotBRunner(robot_b, controller_b, ros_bridge, camera_prim_path, task)

    car_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, CAR_ROOT_PRIM_NAME)
    print(f"  [CAR] car_prim_path = {car_prim_path}", flush=True)
    car_controller = CarArrivalController(car_prim_path)

    import os as _os
    import signal as _signal
    import pathlib as _pathlib
    _START_FLAG = _pathlib.Path("/tmp/autofuel_start")
    # 이전 실행에서 남은 트리거 파일이 있으면 삭제
    _START_FLAG.unlink(missing_ok=True)

    # KeyboardInterrupt / 창 닫기 등 모든 종료 경로에서 os._exit(0)이 반드시 호출되도록
    # SIGINT/SIGTERM 핸들러를 등록하고, finally 블록으로 보장한다.
    # carbOnPluginShutdown 이 Python atexit 을 통해 SIGSEGV 를 내는 Isaac Sim 내부 버그를
    # Py_FinalizeEx 자체를 건너뜀으로써 완전히 차단한다.
    def _force_exit(*_):
        if rclpy.ok():
            try:
                ros_bridge.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        _os._exit(0)

    _signal.signal(_signal.SIGINT,  _force_exit)
    _signal.signal(_signal.SIGTERM, _force_exit)

    was_playing = False  # 직전 프레임에 재생 중이었는지 (Play 버튼을 "막 눌렀는지" 판단용)
    step_count = 0
    _web_start_received = False  # 웹UI "주유 시작" 버튼 수신 여부

    print(f"\n[LOOP] 메인 루프 진입, is_running={simulation_app.is_running()}\n", flush=True)
    try:
        # Isaac Sim 메인 루프: 창이 떠 있는 동안 계속 반복된다.
        while simulation_app.is_running():
            my_world.step(render=True)  # 물리 시뮬레이션 한 스텝 진행 + 화면 렌더링
            time.sleep(0.01)
            is_playing = my_world.is_playing()
            if rclpy.ok():
                # timeout_sec=0.0: 새 메시지가 있으면 즉시 처리하고, 없으면 기다리지 않고 바로 통과.
                # (여기서 블로킹하면 시뮬레이션 루프 전체가 멈추기 때문에 반드시 논블로킹으로 호출해야 함)
                rclpy.spin_once(ros_bridge, timeout_sec=0.0)

            if is_playing and not was_playing:
                # Play 버튼을 막 누른 시점(정지->재생으로 전환) - 모든 상태를 처음부터 다시 시작한다.
                my_world.reset()
                initialize_robot(robot_a, my_world, ROBOT_A_BASE_WORLD, ROBOT_A_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_A)
                if USE_B_ALLEGRO_WRIST:
                    initialize_robot_b(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)
                else:
                    initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)
                controller_a.reset()
                controller_b.reset()
                task.post_reset()
                task.hide_debug_markers()
                car_controller.on_play_reset()
                runner_a.on_play_reset()
                runner_b.on_play_reset()
                step_count = 0
                _web_start_received = False  # 재시작 시 게이트 초기화
                print("\n[RESET] multi-robot oiling sequence 준비 완료\n", flush=True)
                # 차량 prim이 없는 환경에서도 웹UI step이 진행되도록 완료 메시지 출력
                if car_controller.is_done():
                    print("\n[차량] 주차 완료: 차량 prim 미감지\n", flush=True)

            # 웹UI "주유 시작" 버튼 수신 → 로봇 시퀀스 허가
            # 반드시 init 블록(is_playing and not was_playing) 뒤에 위치해야 한다.
            # 첫 이터레이션에 init 블록이 _web_start_received=False 로 리셋한 직후,
            # 같은 이터레이션에 웹 클릭이 도착했다면 init 이후에 처리해야 플래그가 살아있다.
            #
            # uvicorn 프로세스는 ROS2 환경이 소싱되지 않을 수 있으므로
            # /tmp/autofuel_start 파일을 주요 트리거로 사용하고
            # ROS2 start_requested 는 보조 수단으로 병행한다.
            if _START_FLAG.exists() or ros_bridge.start_requested:
                _fuel_liters = 0.0
                if _START_FLAG.exists():
                    try:
                        _fuel_liters = float(_START_FLAG.read_text().strip())
                    except (ValueError, OSError):
                        _fuel_liters = 0.0
                    try:
                        _START_FLAG.unlink()
                    except OSError:
                        pass
                ros_bridge.start_requested = False
                ros_bridge.fuel_target_liters = _fuel_liters
                _web_start_received = True
                # 덮개는 시뮬레이션 시작부터 자동으로 열리고 있으므로 고정 대기를 완료 처리해서
                # ArUco 위치 확인 단계로 바로 진입하도록 한다
                runner_b._door_open_wait_steps = DOOR_OPEN_WAIT_STEPS
                print(f"\n[웹UI] /start_fueling 수신 → 주유 시퀀스 시작 "
                      f"(차량도착={car_controller.is_done()}, step={step_count}, "
                      f"target={_fuel_liters:.1f}L)\n", flush=True)

            if is_playing:
                step_count += 1
                if not car_controller.is_done():
                    # 차가 슬라이드-인을 끝낼 때까지는 A/B 로봇 시퀀스를 아예 시작하지 않는다
                    car_controller.tick(step_count)
                else:
                    # 차량 도착 + 덮개 자동 열림 완료 대기 → 웹UI 버튼이 눌려야 시퀀스 진행
                    if _web_start_received:
                        runner_a.tick(step_count)
                        runner_b.tick(step_count)

                        if runner_a.task_done and runner_b.task_done:
                            # 두 로봇 모두 전체 시퀀스를 끝마쳤으면 시뮬레이션을 자동으로 일시정지한다.
                            print("\n[완료] A/B 전체 시퀀스 종료 - 시뮬레이션 일시정지\n", flush=True)
                            my_world.pause()

            was_playing = is_playing

    except BaseException as _exc:
        import traceback as _tb
        print(f"\n[LOOP ERROR] {type(_exc).__name__}: {_exc}", flush=True)
        _tb.print_exc()
    finally:
        print(f"\n[LOOP] 루프 종료 → _force_exit 호출\n", flush=True)
        _force_exit()


if __name__ == "__main__":
    main()
