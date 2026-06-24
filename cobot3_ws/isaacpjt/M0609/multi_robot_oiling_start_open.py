# multi_robot_oiling.py
#
# Isaac Sim에서 m0609 협동로봇 2대(A, B)를 동시에 시뮬레이션하는 메인 스크립트.
#   - m0609_A: 기존 "주유구에 노즐 꽂기" 동작만 수행 (기존 단일로봇 코드를 거의 그대로 사용)
#   - m0609_B: 시작 시 이미 130도 열린 주유구 커버 상태에서 마개를 풀어서 빼고,
#              A가 주유를 마치면 마개를 다시 끼운 뒤 커버를 닫는 동작을 수행
# 두 로봇은 ROS2 토픽으로 서로 "내 할 일 끝났어" 신호를 주고받으며 순서를 맞춘다(state machine).
# 카메라로 어떤 색(노란/파란/초록)을 찾을지는 multi_color_detector.py 노드에 모드 전환 명령을 보내서 정한다.

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
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf  # USD(3D 씬 포맷)와 PhysX(물리엔진) 관련 저수준 라이브러리

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
USD_PATH        = str(_THIS_DIR / "Collected_nozzletip_project/nozzletip_project.usd")
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
    "joint_5": -75.0,
    "joint_6": -5.0,
}
INITIAL_GRIPPER_JOINTS = {
    "finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
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
FUEL_DOOR_CENTER      = np.array([0.07594, -0.94727, 1.02293], dtype=float)  # 주유구 커버 중심
FUEL_CAP_CENTER       = np.array([0.07560, -1.02924, 1.06441], dtype=float)  # 마개 중심
FUEL_PORT_HOLE_CENTER = np.array([0.08281, -1.24901, 1.00525], dtype=float)  # 주유구 입구(구멍) 중심

FUEL_DOOR_PRIM_NAME = "fuel_door"            # USD 씬에서 커버 prim을 찾을 때 쓰는 이름
FUEL_CAP_PRIM_NAME = "fuel_cap"              # USD 씬에서 마개 prim을 찾을 때 쓰는 이름
FUEL_PORT_HOLE_PRIM_NAME = "fuel_port_hole"  # USD 씬에서 주유구 구멍 prim을 찾을 때 쓰는 이름
SCENE_SEARCH_ROOT = "/World"                  # 위 prim들을 찾기 시작할 루트 경로 (이 아래를 전부 탐색)

# 벽면 기준 바깥 방향 / 삽입 방향. door/cap/hole이 같은 차체 벽면에 있다고 가정하고 공유한다.
#
# [수정] 기존 값은 [0, sin(105), -cos(105)]라서 y축과 z축이 동시에 변했다.
# 그래서 A/B 로봇이 마개/주유구로 접근할 때 "대각선으로 내려가며" 접근하는 것처럼 보였다.
# 주유구 축을 따라 마커 경로를 추적하려면 접근축은 실제 주유구 normal과 일치해야 한다.
# 현재 USD 배치에서는 차체 바깥 방향을 world +Y로 두고, 삽입 방향은 world -Y로 둔다.
# 만약 실제 씬에서 주유구가 다른 축을 향하면 이 벡터만 예: [1,0,0], [0,-1,0] 등으로 바꾸면 된다.
PORT_OUTWARD_NORMAL = np.array([0.0, 1.0, 0.0], dtype=float)
INSERTION_DIRECTION = -PORT_OUTWARD_NORMAL  # 삽입 방향은 바깥 방향의 정반대(안쪽으로 들어가는 방향)

# 기존 A 단독 코드처럼 "툴/노즐 축"도 주유구 삽입 벡터에 맞춘다.
# TOOL_FORWARD_LOCAL_AXIS는 link_6 기준에서 노즐/그리퍼가 앞으로 향하는 로컬 축이다.
# 실행했을 때 손목이 반대로 뒤집히면 [0,0,-1] 또는 [1,0,0]처럼 실제 툴 축에 맞게 조정한다.
ALIGN_TOOL_AXIS_TO_PORT = True
TOOL_FORWARD_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)

# 최종 접근 전에 주유구/마개 축 위로 먼저 들어가는 entry 지점.
# 멀리서 바로 near로 대각선 접근하지 않고, entry -> far -> mid/near -> insert 순으로 축을 따라 움직인다.
AXIS_ENTRY_EXTRA_DISTANCE = 0.18

FUEL_PORT_DIAMETER = 0.10            # 주유구 구멍 지름(m) - 참고용 치수
FUEL_PORT_DEPTH = 0.10               # 주유구 구멍 깊이(m)
INSERT_DISTANCE = FUEL_PORT_DEPTH / 2  # 실제로 노즐/마개를 얼마나 깊이 밀어넣을지(절반만)

VIRTUAL_NOZZLE_LENGTH = 0.63     # 로봇 손목(link_6)에서 노즐 끝까지의 가상 길이(m)
VIRTUAL_NOZZLE_Z_OFFSET = -0.25  # 노즐 길이를 고려해 손목 목표점을 z방향으로 보정하는 오프셋

# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 제어 파라미터 (prompt 지정값)                                ║
# ╚══════════════════════════════════════════════════════════════╝
PHYSICS_DT = 1.0 / 60.0       # 물리 시뮬레이션 한 스텝의 시간(초). 60Hz -> 약 16.7ms마다 한 번
POSITION_TOLERANCE = 0.10     # "목표 위치에 도착했다"고 판단할 기본 허용 오차(m)
MAX_STEPS_PER_STAGE = 600     # 한 단계(stage)가 이 스텝 수를 넘으면 타임아웃으로 보고 강제로 넘어감 (~10초 @ 60fps, 디버깅 속도 우선)
PRINT_EVERY_N_STEPS = 20      # 디버그 로그를 매 스텝마다 찍지 않고 N스텝마다 한 번만 찍음 (로그 폭주 방지)

# 손목(EE) 목표를 향해 이동할 때의 속도(m/s). 상황별로 다른 속도를 쓴다.
DEFAULT_TARGET_SPEED = 0.060   # 평범하게 이동할 때
NEAR_TARGET_SPEED    = 0.040   # 목표에 가까워졌을 때 (더 정밀하게, 천천히)
INSERT_TARGET_SPEED  = 0.020   # 구멍에 삽입할 때 (충돌 위험 있어 가장 느리게)
RETREAT_TARGET_SPEED = 0.050   # 빼거나 후퇴할 때

HOME_JOINT_SPEED_ALPHA = 0.012  # 기본 자세로 복귀할 때, 현재값과 목표값 차이의 몇 %씩 매 스텝 움직일지
HOME_JOINT_TOLERANCE   = 0.05   # 기본 자세에 "도착했다"고 볼 조인트 각도 오차(rad)
HOME_HOLD_STEPS        = 40     # 도착 판정 후에도 이만큼의 스텝 동안 안정적으로 유지되어야 진짜 완료로 인정

# 주유구/마개/구멍에 접근할 때 거리 단계 (멀리서 -> 중간 -> 가까이 순서로 단계적으로 다가감)
FAR_DISTANCE  = 0.28
MID_DISTANCE  = 0.18
NEAR_DISTANCE = 0.09

COVER_CLEARANCE_DISTANCE = 0.35  # 마개로 가기 전에 들르는 경유점의 벽 바깥쪽 거리(m) - 열린 커버를 피해서 돌아가기 위함

GRIPPER_LENGTH_B = 0.16  # B 로봇 그리퍼 길이(m, 실측값으로 조정) - 손목(link_6)에서 손가락 끝까지 거리.
                         # 마개에 접근할 때 손목이 아니라 "손가락 끝"이 표면에 닿아야 하므로,
                         # 손목 목표점은 이 길이만큼 표면보다 바깥쪽에 둬야 그리퍼가 차체를 파고들지 않는다.

# 커버(fuel_door)가 열리는 각도 단계.
# 이번 버전은 시작하자마자 덮개를 130도 열린 상태로 만든 뒤,
# B 로봇은 덮개 열기 과정을 건너뛰고 마개 작업부터 시작한다.
COVER_START_DEG = 30.0    # 닫힘 기준 각도
COVER_MID_DEG   = 80.0    # 30도와 130도 사이의 닫기 중간 각도
COVER_OPEN_DEG  = 130.0   # 시작 시 강제로 열어둘 각도
DOOR_ANGLE_TOLERANCE_DEG = 8.0  # 목표 도어 각도에 도착했다고 볼 허용 오차(도)

RETURN_MID_OUTWARD_OFFSET = 0.15  # RETURN_MID에서 추가로 차체 바깥쪽으로 더 빠지는 거리(m) - 카메라에 마개(blue)가 보이게
DOOR_PUSH_OFFSET = -0.35    # 커버 lock 중심에서 x축으로 빼는 오프셋(m) - 힌지에서 먼 쪽(손잡이 쪽)을 밀도록 보정 (여는 동작용)
DOOR_CLOSE_PUSH_OFFSET = 0.15  # 닫는 동작(CLOSE_COVER)에서 lock 중심에 x축으로 더하는 오프셋(m)

# build_cover_sequence의 각 단계 목표점에 world x/y/z로 직접 더하는 보정값.
# PORT_OUTWARD_NORMAL_UNIT 방향 보정만으로는 씬의 다른 깊이 축을
# 깊이를 절대 바꿀 수 없으므로, 세 축을 독립적으로 조절해서 실제 "깊이" 축을 찾아 고친다.
DOOR_TOUCH_OFFSET = np.array([0.0, 0.15, 0.0], dtype=float)
# 힌지 축 방향 부호가 USD/PhysX 쪽과 반대로 측정될 수 있다.
# 리셋 직후 로그의 "door angle"이 COVER_START_DEG와 다르게 튀면 -1.0으로 바꿔서 맞춘다.
DOOR_ANGLE_SIGN = 1.0

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

# True면 OPEN_COVER/MOVE_TO_CAP/RUN_SEQUENCE에서 WaypointSequence 대신, 뷰포트에서 직접 옮길 수
# 있는 마커 prim의 현재 world 위치를 매 스텝 RMPFlow 목표로 사용한다 (수동 디버깅/튜닝용).
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
CAMERA_POINT_CONVENTION = "ros_optical"  # 카메라가 보내는 좌표계 규칙 (ROS 광학 좌표계 기준)

REQUIRE_TARGET_LOCK = True                 # True면 "lock=True"가 아닌 pose는 아예 사용하지 않음
CONTROLLER_REQUIRED_LOCK_SAMPLES = 5       # 안정적이라고 판단하기 위해 모아야 하는 최소 샘플 개수
CONTROLLER_WORLD_STD_TOLERANCE = 0.025     # 모은 샘플들의 흔들림(표준편차) 허용치(m)
SEARCH_GATE_HALF_EXTENT = np.array([0.35, 0.35, 0.18], dtype=float)  # 기준 위치 대비 허용 오차 범위(게이트)
WAIT_LOCK_TIMEOUT_STEPS = 600                # 이 스텝 수가 지나도 lock이 안 되면 하드코딩 기준값으로 폴백 (~10초 @ 60fps)

# ╔══════════════════════════════════════════════════════════════╗
# ║  E. ROS2 토픽 이름                                              ║
# ╚══════════════════════════════════════════════════════════════╝
TOPIC_COLOR_POSE = "/color_detector/pose"          # 카메라가 찾은 색 표적의 3D 위치
TOPIC_COLOR_LOCK = "/color_detector/target_locked"  # 그 위치가 "안정적으로 확정됐는지" 여부
TOPIC_MODE_SWITCH = "/color_detector/mode_switch"   # 카메라에게 "이제 이 색을 찾아" 명령
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
    그래서 시작 시 열린 상태를 확실히 보이게 하려고 prim transform도 함께 설정한다.
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
    visual_angle_deg - COVER_START_DEG이다. 예: 130도 open -> +100도 회전.
    """
    if closed_world_matrix is None:
        if verbose:
            print(f"[DOOR][WARN] {label} 기준 world matrix가 없어 prim 직접 회전을 건너뜀")
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
            print(f"[DOOR][WARN] {label} prim transform 직접 적용 실패")
    return ok


def angle_deg_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9):
    """두 벡터 사이의 각도(도 단위)를 구한다. 내적 공식(cos세타 = (v1·v2)/(|v1||v2|))을 이용.
    둘 중 하나라도 길이가 거의 0이면 각도를 정의할 수 없으므로 None을 반환한다."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    # 부동소수점 오차로 cos값이 [-1,1] 범위를 살짝 벗어나면 arccos가 NaN을 내므로 clip으로 방지.
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))



def quat_normalize(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Isaac Sim의 wxyz 쿼터니언을 정규화한다."""
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_multiply(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언 합성. 반환 회전은 q1을 적용한 뒤 q2를 적용한 회전이다."""
    w2, x2, y2, z2 = quat_normalize(q2)
    w1, x1, y1, z1 = quat_normalize(q1)
    return quat_normalize(np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
    ], dtype=float))


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언 q로 로컬 벡터 v를 world 방향 벡터로 회전한다."""
    q = quat_normalize(q)
    w, x, y, z = q
    # 회전행렬 R(q) @ v
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R @ np.array(v, dtype=float)


def shortest_arc_quat(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """from_vec 방향을 to_vec 방향으로 돌리는 최소 회전 쿼터니언(wxyz)을 만든다."""
    f = normalize(np.array(from_vec, dtype=float))
    t = normalize(np.array(to_vec, dtype=float))
    if np.linalg.norm(f) < 1e-9 or np.linalg.norm(t) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    dot = float(np.clip(np.dot(f, t), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot < -0.9999:
        # 180도 반대 방향이면 f와 수직인 임의 축을 골라 회전한다.
        axis = np.cross(f, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(f, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)
    axis = np.cross(f, t)
    q = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float)
    return quat_normalize(q)


def align_orientation_to_port_axis(current_orientation: np.ndarray) -> np.ndarray:
    """현재 EE 자세를 기준으로, TOOL_FORWARD_LOCAL_AXIS가 INSERTION_DIRECTION_UNIT을 향하게 보정한다.

    기존 코드는 reset 시점의 EE 방향을 그대로 고정했기 때문에 위치 경로만 바뀌고
    노즐/그리퍼 축은 주유구 축과 맞지 않을 수 있었다. 이 함수는 현재 자세에서 툴의 forward 축이
    world에서 어디를 향하는지 계산한 뒤, 그 방향을 삽입 방향으로 최소 회전시킨다.
    """
    if not ALIGN_TOOL_AXIS_TO_PORT:
        return np.array(current_orientation, dtype=float).copy()
    current_orientation = quat_normalize(current_orientation)
    current_tool_axis_world = quat_rotate_vector(current_orientation, TOOL_FORWARD_LOCAL_AXIS)
    delta_q = shortest_arc_quat(current_tool_axis_world, INSERTION_DIRECTION_UNIT)
    return quat_multiply(delta_q, current_orientation)


def find_dof_index(robot, dof_name: str):
    """로봇의 조인트(DOF) 이름으로 그 조인트가 몇 번째 인덱스인지 찾는다.
    get_joint_positions()가 반환하는 배열에서 어느 위치가 어느 조인트인지 알아야
    joint_6처럼 "딱 하나의 조인트만" 골라서 제어할 수 있다."""
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        return robot.dof_names.index(dof_name)
    return None


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


def camera_ros_point_to_usd_camera_local(point_camera_ros: np.ndarray) -> np.ndarray:
    """ROS 광학(optical) 좌표계(x=오른쪽, y=아래, z=앞)를 USD 카메라 로컬 좌표계로 변환한다.
    (현재 transform_camera_point_to_world에서 직접 변환을 하고 있어서 이 함수는 보조/참고용으로 남아있다)"""
    x, y, z = [float(v) for v in point_camera_ros]
    if CAMERA_POINT_CONVENTION == "ros_optical":
        return np.array([x, -y, -z], dtype=float)
    return np.array([x, y, z], dtype=float)


def transform_camera_point_to_world(point_camera_ros: np.ndarray, camera_prim_path: str):
    """카메라가 보고하는 "카메라 기준 3D 좌표(x,y,z)"를 받아서 "world 기준 3D 좌표"로 바꾼다.

    multi_color_detector.py가 보내는 pose.position은 카메라 로컬 좌표계 기준값이므로,
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
        print(f"[VISIBLE] {prim_path} 표시")
    else:
        imageable.MakeInvisible()
        print(f"[HIDDEN] {prim_path} 숨김")


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
    커버가 30->80->130도로 움직일 때, "그 각도에서 로봇 손이 닿아야 할 위치"를 미리 계산하는 데 쓴다."""
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


def set_revolute_joint_visual_angle_deg(
    joint_prim, visual_angle_deg: float, label: str = "door", verbose: bool = True
) -> bool:
    """RevoluteJoint를 원하는 시각적 각도(예: 닫힘 30도, 열림 130도)로 초기화한다.

    이 코드에서 COVER_START_DEG=30은 "닫힌 상태"를 사람이 읽기 쉽게 표현한 값이다.
    USD RevoluteJoint의 실제 state position은 닫힌 자세를 0도로 두는 경우가 많으므로,
    실제 joint state에는 (visual_angle_deg - COVER_START_DEG) 값을 기록한다.
    """
    if joint_prim is None or not joint_prim.IsValid():
        if verbose:
            print(f"[DOOR][WARN] {label} RevoluteJoint가 없어 각도를 설정할 수 없음")
        return False

    joint_delta_deg = float(visual_angle_deg - COVER_START_DEG)
    applied = False

    # 130도 열린 상태를 허용하도록 joint limit이 너무 좁으면 자동으로 넓힌다.
    try:
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        lower_attr = joint.GetLowerLimitAttr()
        upper_attr = joint.GetUpperLimitAttr()
        lower = lower_attr.Get()
        upper = upper_attr.Get()
        if lower is not None and joint_delta_deg < float(lower):
            lower_attr.Set(float(joint_delta_deg))
        if upper is not None and joint_delta_deg > float(upper):
            upper_attr.Set(float(joint_delta_deg))
    except Exception as exc:
        if verbose:
            print(f"[DOOR][WARN] joint limit 보정 실패: {exc}")

    # 가장 직접적인 초기 상태 지정: state:angular:physics:position/velocity.
    try:
        state_api = UsdPhysics.JointStateAPI.Apply(joint_prim, "angular")
        state_api.GetPositionAttr().Set(joint_delta_deg)
        state_api.GetVelocityAttr().Set(0.0)
        applied = True
    except Exception as exc:
        if verbose:
            print(f"[DOOR][WARN] JointStateAPI 설정 실패: {exc}")
        try:
            joint_prim.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(joint_delta_deg)
            joint_prim.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
            applied = True
        except Exception as exc2:
            if verbose:
                print(f"[DOOR][WARN] joint state attribute 직접 설정 실패: {exc2}")

    # 이미 angular drive가 있는 joint라면 target도 같은 값으로 맞춘다.
    # 새 drive를 만들지는 않아서, 원래 drive가 없던 도어의 물리 특성을 강제로 바꾸지 않는다.
    try:
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(joint_delta_deg)
            drive.GetTargetVelocityAttr().Set(0.0)
    except Exception as exc:
        if verbose:
            print(f"[DOOR][WARN] angular drive target 설정 실패: {exc}")

    if applied and verbose:
        print(f"[DOOR] {label} visual_angle={visual_angle_deg:.1f}deg "
              f"(joint_delta={joint_delta_deg:.1f}deg) 설정 완료")
    return applied


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
                print(f"\n[⚠️ 타임아웃] {label} 남은거리={err:.3f}m")
            else:
                print(f"\n[✅ {label}] 위치={np.round(ee_position, 3)}")
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
# A 로봇: 기존 주유 시퀀스 (기존 FuelPortSequence 로직 그대로, 색만 green)
# ============================================================
def build_fuel_port_sequence(fuel_port_center: np.ndarray) -> WaypointSequence:
    """A 로봇이 "가상의 노즐"을 들고 주유구에 다가가 꽂고 다시 빼는 8단계 시퀀스를 만든다.
    fuel_port_center: 카메라로 lock한 주유구 구멍의 실제 world 좌표."""
    outward = PORT_OUTWARD_NORMAL_UNIT
    insertion = INSERTION_DIRECTION_UNIT

    # 노즐의 "끝(tip)"이 있어야 할 위치들.
    # entry -> far -> mid -> near -> insert는 모두 같은 outward/insertion 축 위에 놓인다.
    # 이렇게 해야 기존 A 단독 코드처럼 주유구 축을 따라 마커 경로를 추적하는 형태가 된다.
    tip_entry = fuel_port_center + outward * (FAR_DISTANCE + AXIS_ENTRY_EXTRA_DISTANCE)
    tip_far  = fuel_port_center + outward * FAR_DISTANCE
    tip_mid  = fuel_port_center + outward * MID_DISTANCE
    tip_near = fuel_port_center + outward * NEAR_DISTANCE
    tip_insert = fuel_port_center + insertion * INSERT_DISTANCE

    # 로봇이 실제로 움직이는 건 "노즐 끝"이 아니라 "손목(link_6)"이므로, 가상 노즐 길이만큼
    # 뒤로 뺀 오프셋을 더해서 손목이 가야 할 목표(approach_*)를 계산한다.
    control_offset = outward * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])
    approach_entry = tip_entry + control_offset
    approach_far  = tip_far + control_offset
    approach_mid  = tip_mid + control_offset
    approach_near = tip_near + control_offset
    insert_target = tip_insert + control_offset

    stages = [
        FuelStage("00_axis_entry", approach_entry, tolerance=0.10, speed=DEFAULT_TARGET_SPEED),
        FuelStage("01_axis_far_start", approach_far, tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        FuelStage("02_axis_mid", approach_mid, tolerance=0.07, speed=DEFAULT_TARGET_SPEED),
        FuelStage("03_axis_near_stop", approach_near, hold_steps=40, tolerance=0.06, speed=NEAR_TARGET_SPEED),
        FuelStage("04_insert_into_cylinder", insert_target, hold_steps=80, tolerance=0.05, speed=INSERT_TARGET_SPEED),
        FuelStage("05_retreat_near", approach_near, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("06_retreat_mid", approach_mid, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("07_retreat_far", approach_far, hold_steps=15, tolerance=0.08, speed=RETREAT_TARGET_SPEED),
        FuelStage("08_return_home", None),
    ]
    stage_log = {
        "00_axis_entry": "A 주유구 축 진입",
        "01_axis_far_start": "A 접근 시작",
        "02_axis_mid": "A 중간 지점",
        "03_axis_near_stop": "A 근처 대기",
        "04_insert_into_cylinder": "A 삽입 완료",
        "05_retreat_near": "A 후퇴 시작",
        "06_retreat_mid": "A 중간 후퇴",
        "07_retreat_far": "A 후퇴 완료",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# B 로봇: 커버 push 시퀀스 / 마개 접근-복원 시퀀스
# ============================================================
def build_cover_sequence(direction: str, door_reference_point: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> WaypointSequence:
    """B 로봇이 커버를 밀어서 여닫는 3단계 시퀀스를 만든다.
    direction: 'open' (30->80->130도로 열기) 또는 'close' (130->80->30도로 닫기).
    door_reference_point: 카메라로 lock한 "30도 상태일 때"의 커버 표면 위 한 점.
    pivot/axis: 힌지(RevoluteJoint)의 회전 중심점과 회전축 - 이 축을 기준으로 점을 돌려서
    "도어가 각 각도일 때 그 표면 위 점이 어디로 이동하는지"를 미리 계산한다."""
    def p(angle_deg):
        # door_reference_point는 30도 기준이므로, 목표각도와 30도의 차이만큼만 추가로 회전시킨다.
        rotated = rotate_point_around_axis(
            door_reference_point, pivot, axis, angle_deg - COVER_START_DEG
        )
        # PORT_OUTWARD_NORMAL_UNIT 방향(x성분이 0)으로만 보정하면 x축 방향 깊이는 전혀
        # 조절되지 않는다 - 그래서 DOOR_TOUCH_OFFSET을 x/y/z 각각 독립적으로 조절 가능한
        # world 좌표 보정값으로 둔다. 어느 축이 실제 "깊이"인지 값을 바꿔보며 찾는다.
        return rotated + DOOR_TOUCH_OFFSET

    if direction == "open":
        stages = [
            FuelStage("B2_01_approach_30", p(COVER_START_DEG), tolerance=0.12, speed=DEFAULT_TARGET_SPEED,
                       target_door_angle=COVER_START_DEG),
            FuelStage("B2_02_push_mid", p(COVER_MID_DEG), hold_steps=20, tolerance=0.12, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_MID_DEG),
            FuelStage("B2_03_push_open", p(COVER_OPEN_DEG), hold_steps=30, tolerance=0.10, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_OPEN_DEG),
        ]
        stage_log = {
            "B2_01_approach_30": "B 커버 접촉",
            "B2_02_push_mid": f"B 커버 {COVER_MID_DEG:.0f}도",
            "B2_03_push_open": f"B 커버 {COVER_OPEN_DEG:.0f}도 완전열림",
        }
    else:
        stages = [
            FuelStage("B10_01_reengage_open", p(COVER_OPEN_DEG), tolerance=0.08, speed=DEFAULT_TARGET_SPEED,
                       target_door_angle=COVER_OPEN_DEG),
            FuelStage("B10_02_push_mid", p(COVER_MID_DEG), hold_steps=20, tolerance=0.07, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_MID_DEG),
            FuelStage("B10_03_push_30", p(COVER_START_DEG), hold_steps=30, tolerance=0.07, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_START_DEG),
        ]
        stage_log = {
            "B10_01_reengage_open": f"B 커버 재접촉({COVER_OPEN_DEG:.0f})",
            "B10_02_push_mid": f"B 커버 {COVER_MID_DEG:.0f}도(닫는중)",
            "B10_03_push_30": "B 커버 30도 닫힘완료",
        }
    return WaypointSequence(stages, stage_log)


def build_cap_approach_sequence(cap_center: np.ndarray) -> WaypointSequence:
    """B 로봇이 마개(fuel_cap)를 잡으러 멀리서부터 단계적으로 접근하는 시퀀스.
    먼저 열린 덮개를 피하는 경유점을 거친 뒤, 마개 중심의 접근축 위 entry 지점으로 들어간다.
    이후 far -> near -> grasp는 모두 같은 축 위에서 움직이므로 대각선 접근을 줄일 수 있다."""
    clearance_point = cap_center + PORT_OUTWARD_NORMAL_UNIT * COVER_CLEARANCE_DISTANCE
    axis_entry = make_outward_point(cap_center, FAR_DISTANCE + AXIS_ENTRY_EXTRA_DISTANCE)
    stages = [
        FuelStage("B5_00_avoid_door", clearance_point,
                  tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B5_00_axis_entry", axis_entry,
                  tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B5_01_far", make_outward_point(cap_center, FAR_DISTANCE),
                  tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        # near/grasp 모두 그리퍼 길이(GRIPPER_LENGTH_B)만큼 바깥쪽으로 더 띄워서, 그리퍼(손목이 아니라
        # 손가락 끝)가 자동차 표면을 파고들지 않게 한다.
        FuelStage("B5_02_near", make_outward_point(cap_center, NEAR_DISTANCE + GRIPPER_LENGTH_B),
                  hold_steps=30, tolerance=0.06, speed=NEAR_TARGET_SPEED),
        FuelStage("B5_03_grasp", cap_center + PORT_OUTWARD_NORMAL_UNIT * (GRIPPER_LENGTH_B + 0.05),
                  hold_steps=40, tolerance=0.05, speed=INSERT_TARGET_SPEED),
    ]
    stage_log = {
        "B5_00_avoid_door": "B 덮개 회피",
        "B5_00_axis_entry": "B 마개 축 진입",
        "B5_01_far":        "B 마개 접근(far)",
        "B5_02_near":       "B 마개 접근(near)",
        "B5_03_grasp":      "B 마개 grasp 위치 도착",
    }
    return WaypointSequence(stages, stage_log)


def build_cap_restore_sequence(cap_center: np.ndarray) -> WaypointSequence:
    """A 로봇이 주유를 마친 뒤, B 로봇이 마개를 원래 잡았던 마개 중심으로 다시 가져가 조이는 시퀀스.

    주의: 여기서는 fuel_port_hole 중심이 아니라 cap_center를 사용한다.
    기존에는 RESTORE_CAP에서 self.task.hole_world_position을 넘겨서, 마개를 열 때 잡았던 위치와
    다시 닫을 때 목표 위치가 서로 달라져 타임아웃이 발생할 수 있었다.
    마개 prim은 실제로 그리퍼에 물리 부착되는 것이 아니라 숨김/표시로 처리하므로, 닫을 때도
    마개를 처음 잡았던 중심점(locked_cap_center 또는 cap_world_position)을 기준으로 복원해야 한다.
    entry -> far -> near -> insert는 모두 같은 접근축 위에 배치한다."""
    stages = [
        FuelStage("B9_00_axis_entry", make_outward_point(cap_center, FAR_DISTANCE + AXIS_ENTRY_EXTRA_DISTANCE),
                  tolerance=0.10, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B9_01_far", make_outward_point(cap_center, FAR_DISTANCE), tolerance=0.08, speed=DEFAULT_TARGET_SPEED),
        # 여는 쪽(B5_02_near/B5_03_grasp)과 같은 cap_center 기준으로 맞춘다.
        # 이렇게 해야 B5에서 마개를 잡았던 축과 B9에서 마개를 다시 끼우는 축이 일치한다.
        FuelStage("B9_02_near", make_outward_point(cap_center, NEAR_DISTANCE + GRIPPER_LENGTH_B),
                  hold_steps=30, tolerance=0.10, speed=NEAR_TARGET_SPEED),
        FuelStage("B9_03_insert", cap_center + PORT_OUTWARD_NORMAL_UNIT * (GRIPPER_LENGTH_B + 0.05),
                  hold_steps=40, tolerance=0.10, speed=INSERT_TARGET_SPEED),
    ]
    stage_log = {
        "B9_00_axis_entry": "B 주유구 축 진입",
        "B9_01_far": "B 주유구 접근(far)",
        "B9_02_near": "B 주유구 접근(near)",
        "B9_03_insert": "B 마개 삽입 위치 도착",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# ROS2 bridge: multi_color_detector.py 및 A/B 동기화 토픽을 한 노드에서 처리
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

        self.pose_sub = self.create_subscription(PoseStamped, TOPIC_COLOR_POSE, self._pose_cb, sensor_qos)
        self.lock_sub = self.create_subscription(Bool, TOPIC_COLOR_LOCK, self._lock_cb, sensor_qos)
        self.robot_a_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_A_DONE, self._robot_a_done_cb, latched_qos)
        self.robot_b_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_B_DONE, self._robot_b_done_cb, latched_qos)

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
    """노란/파란/초록 색 모드 공통: N개 샘플이 표준편차 이내로 모이면 평균 world 좌표를 반환한다."""

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
            print(f"[LOCK] transform_camera_point_to_world() 실패 (camera_prim_path={self.camera_prim_path} 가 invalid)")
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
        print("\n  [완료] multi-robot oiling 씬 구성 성공!\n")

    def _load_usd(self):
        """프로젝트 USD 파일을 /World 아래에 참조(reference)로 추가해서 씬에 불러온다.
        AddReference는 즉시 로드되는 게 아니라 비동기로 처리되므로, 로드가 끝나길 기다리려고
        simulation_app.update()를 여러 번 호출해 강제로 프레임을 진행시킨다."""
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}")
        print(f"  [NOTE] m0609_A={ROBOT_A_PRIM_PATH}, m0609_B={ROBOT_B_PRIM_PATH}가 USD에 이미 있다고 가정")

    def _discover_robot_links(self):
        """A/B 로봇 USD 안에서 손목 끝(link_6) prim 경로를 찾는다.
        못 찾으면 이후 SingleManipulator 등록 자체가 불가능하므로 바로 에러를 내서 빨리 알게 한다."""
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 로봇 A/B 링크 경로 탐색")
        print("=" * 60)
        self.ee_path_a = find_prim_path_by_name(ROBOT_A_PRIM_PATH, EE_LINK_NAME)
        self.ee_path_b = find_prim_path_by_name(ROBOT_B_PRIM_PATH, EE_LINK_NAME)
        if self.ee_path_a is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_A_PRIM_PATH}")
        if self.ee_path_b is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_B_PRIM_PATH}")
        print(f"  A EE = {self.ee_path_a}")
        print(f"  B EE = {self.ee_path_b}")

    def _setup_physics(self):
        """A/B 로봇의 모든 관절 드라이브(모터)에 강성/댐핑/최대힘 값을 강하게 설정한다.
        USD에 기본으로 들어있는 드라이브 값이 너무 약하면 로봇이 목표 위치로 잘 따라가지 못하거나
        무거운 물체(마개 등)를 들 때 축 늘어지는 문제가 생길 수 있어 이 값들을 직접 키워준다."""
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 로봇 A/B drive 설정")
        print("=" * 60)
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
        print(f"  [OK] drive updated: {drive_count}")

    def _register_robots(self, scene):
        """Isaac Sim의 SingleManipulator/ParallelGripper 객체를 만들어 scene에 등록한다.
        이렇게 등록해야 robot.get_joint_positions(), robot.apply_action() 같은
        고수준 API를 쓸 수 있게 된다 (등록 전에는 USD prim일 뿐, 로봇 객체가 아니다)."""
        print("\n" + "=" * 60)
        print("[4.REGISTER] SingleManipulator A/B 등록")
        print("=" * 60)
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
        print(f"  [OK] m0609_A = {ROBOT_A_PRIM_PATH}")
        print(f"  [OK] m0609_B = {ROBOT_B_PRIM_PATH}")

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
        print(f"  [WARN] {label} prim 위치를 못 읽음 -> 하드코딩 상수 {np.round(fallback_constant, 4)} 로 폴백")
        return fallback_constant.copy()

    def _discover_fuel_objects(self, scene):
        """fuel_door/fuel_cap/fuel_port_hole의 실제 world 위치와, 도어 힌지의 회전축/피벗을
        USD 씬에서 직접 읽어온다. 이 위치들이 게이트 판정 기준(reference_center)이자
        로봇이 실제로 움직여야 할 목표 좌표의 출발점이 되므로, 이 함수의 정확도가
        전체 시스템이 제대로 작동하는지의 핵심이다."""
        print("\n" + "=" * 60)
        print("[5.SCENE] fuel_door / fuel_cap / fuel_port_hole 탐색")
        print("=" * 60)
        self.door_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_DOOR_PRIM_NAME)
        self.cap_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_CAP_PRIM_NAME)
        self.hole_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_PORT_HOLE_PRIM_NAME)
        print(f"  fuel_door = {self.door_prim_path}")
        print(f"  fuel_cap  = {self.cap_prim_path}")
        print(f"  fuel_port_hole = {self.hole_prim_path}")

        # 게이트 판정/실제 모션 목표는 프롬프트의 하드코딩 좌표가 아니라 USD 씬에서 직접 읽은
        # 실제 world 위치를 우선 사용한다. 하드코딩 값은 해당 prim을 못 찾았을 때만 폴백으로 쓴다.
        self.door_world_position = self._resolve_world_position(self.door_prim_path, FUEL_DOOR_CENTER, "fuel_door")
        self.cap_world_position = self._resolve_world_position(self.cap_prim_path, FUEL_CAP_CENTER, "fuel_cap")
        self.hole_world_position = self._resolve_world_position(self.hole_prim_path, FUEL_PORT_HOLE_CENTER, "fuel_port_hole")

        # fuel_door의 원래 닫힌 transform을 별도로 저장한다.
        # RevoluteJoint state만으로는 시작 화면에서 바로 열려 보이지 않을 수 있으므로,
        # 이 닫힘 기준 matrix를 바탕으로 prim 자체를 힌지축 기준 회전시킨다.
        self.door_closed_world_matrix = get_prim_world_matrix(self.door_prim_path)
        self.door_closed_world_rotation = get_prim_world_rotation(self.door_prim_path)

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

        # 리셋 시 door의 "기준 회전"을 캐시해서 이후 상대 각도를 측정한다 (COVER_START_DEG=닫힘 기준).
        self.door_rest_rotation = self.door_closed_world_rotation or get_prim_world_rotation(self.door_prim_path)

        # 디버깅용으로 door/cap/hole 위치에 작은 색깔 박스를 띄워서, Isaac Sim 화면에서
        # "코드가 인식한 위치"가 실제 USD 모델과 맞는지 눈으로 바로 확인할 수 있게 한다.
        markers = [
            ("fuel_marker_door", self.door_world_position, np.array([1.0, 1.0, 0.0])),  # 노란 박스
            ("fuel_marker_cap", self.cap_world_position, np.array([0.0, 0.5, 1.0])),     # 파란 박스
            ("fuel_marker_hole", self.hole_world_position, np.array([0.0, 1.0, 0.3])),   # 초록 박스
        ]
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

    def set_door_angle_deg(
        self,
        visual_angle_deg: float,
        label: str = "fuel_door",
        refresh_viewport: bool = True,
        verbose: bool = True,
    ) -> bool:
        """fuel_door를 원하는 시각적 각도로 맞춘다.

        기존처럼 RevoluteJoint state/drive target도 설정하지만, 그것만으로는 시작 화면에서
        transform이 바로 갱신되지 않는 경우가 있어 fuel_door prim transform도 힌지축 기준으로
        직접 회전시킨다.
        """
        joint_ok = set_revolute_joint_visual_angle_deg(
            self.door_joint_prim, visual_angle_deg, label=label, verbose=verbose
        )
        prim_ok = set_door_prim_visual_angle_deg(
            self.door_prim_path,
            self.door_closed_world_matrix,
            self.door_pivot_world,
            self.door_axis_world,
            visual_angle_deg,
            label=label,
            verbose=verbose,
        )
        # 시작/reset처럼 즉시 화면 갱신이 필요한 경우만 한 프레임 갱신한다.
        # CLOSE_COVER 중에는 매 스텝 호출되므로 refresh_viewport=False로 사용한다.
        if refresh_viewport:
            try:
                simulation_app.update()
            except Exception:
                pass
        return bool(joint_ok or prim_ok)

    def current_door_angle_deg(self) -> float:
        """지금 이 순간 커버가 몇 도 열려있는지를 계산한다.
        기준 회전(리셋 직후 닫힌 자세, 30도라고 가정)에서 지금까지 회전한 양(delta)을 구해서 더한다."""
        current_rotation = get_prim_world_rotation(self.door_prim_path)
        delta = signed_angle_about_axis_deg(self.door_rest_rotation, current_rotation, self.door_axis_world)
        return COVER_START_DEG + delta

    def post_reset(self):
        """시뮬레이션을 Play로 (재)시작할 때마다 호출.
        그리퍼를 열어두고, 닫힌 자세를 기준 회전으로 캐시한 뒤 fuel_door를 130도 열린 상태로 만든다."""
        self.robot_a.gripper.set_joint_positions(self.robot_a.gripper.joint_opened_positions)
        self.robot_b.gripper.set_joint_positions(self.robot_b.gripper.joint_opened_positions)

        # reset 직후 현재 화면 transform이 아니라, 처음 로드된 닫힘 기준 회전을 사용한다.
        # 그래야 시작 시 130도로 직접 열어도 current_door_angle_deg()가 30도 기준에서 계산된다.
        self.door_rest_rotation = self.door_closed_world_rotation or get_prim_world_rotation(self.door_prim_path)
        self.set_door_angle_deg(COVER_OPEN_DEG, label="초기 fuel_door open")


# ============================================================
# Robot A runner: WAIT_LOCK(green) -> RUN_SEQUENCE(기존 주유 로직)
# ============================================================
class RobotARunner:
    """A 로봇(주유 로봇)의 동작을 매 시뮬레이션 스텝마다 한 단계씩 진행시키는 state machine.
    상태는 두 가지뿐: B가 끝나길 기다리는 IDLE_WAIT_B/WAIT_LOCK_GREEN, 그리고 실제로
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
        self.locked_target_orientation = None  # 주유구 삽입축에 맞춘 손목 방향을 고정해서 계속 유지
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = 0  # 리셋 시점의 robot_b_done_count (그 이전 신호는 무시하기 위함)

    def on_play_reset(self):
        """시뮬레이션을 Play로 시작할 때마다 호출되어 모든 상태를 깨끗하게 초기화한다."""
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = align_orientation_to_port_axis(ee_ori)
        self.run_state = "IDLE_WAIT_B"
        self.lock_acquirer = None
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = self.ros_bridge.robot_b_done_count
        print("[A] RESET -> IDLE_WAIT_B (robot_b/done 대기)")

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
                self.ros_bridge.publish_mode_switch("green")
                self.lock_acquirer = StableTargetLockAcquirer(
                    self.ros_bridge, self.camera_prim_path, self.task.hole_world_position, apply_mouth_offset=True,
                )
                self.wait_steps = 0
                self.run_state = "WAIT_LOCK_GREEN"
                print("\n[A-8] robot_b/done 수신 -> mode_switch=green, 주유구 lock 대기 시작\n")
            return

        if self.run_state == "WAIT_LOCK_GREEN":
            # A-8: B로부터 신호를 받아 카메라를 초록 모드로 바꾼 뒤, 주유구 위치가
            # 안정적으로 확정(lock)될 때까지 매 스텝 lock_acquirer.update()를 호출해 기다린다.
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.sequence = build_fuel_port_sequence(mean)
                self.run_state = "RUN_SEQUENCE"
                print(f"\n[A] green lock 완료. fuel_port_hole center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                # 너무 오래 lock이 안 되면(카메라가 못 찾거나 계속 게이트 밖이면) 재시도하지 않고
                # USD에서 읽은 하드코딩 기준값(task.hole_world_position)으로 즉시 다음 단계로 넘어간다.
                self.sequence = build_fuel_port_sequence(self.task.hole_world_position)
                self.run_state = "RUN_SEQUENCE"
                print("\n[A] WAIT_LOCK_GREEN timeout -> 하드코딩 기준값으로 다음 단계 진행\n")
                return
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[A][WAIT_LOCK_GREEN] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
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
                    if step_count % PRINT_EVERY_N_STEPS == 0:
                        print(f"[A][RUN_SEQUENCE][MARKER] target={np.round(marker_pos, 3)} err={err:.3f}")
                    if err < POSITION_TOLERANCE:
                        self.task_done = True
                        self.ros_bridge.publish_robot_a_done(True)
                        print("\n[A][MARKER] fuel_port 마커 도착 -> 주유 완료 처리\n")
                return

            # 실제 8단계 주유 시퀀스 진행. 마지막 단계(08_return_home)는 위치 목표가 없는
            # 특수 단계라 WaypointSequence가 처리하지 못하므로 여기서 별도 분기로 처리한다.
            stage = self.sequence.current
            if stage.name == "08_return_home":
                target_joints = build_initial_joint_positions(
                    self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_A,
                )
                reached = step_home_return(self.robot, target_joints)
                self.sequence.hold_count = self.sequence.hold_count + 1 if reached else 0
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[A][08_return_home] hold={self.sequence.hold_count}/{HOME_HOLD_STEPS}")
                if self.sequence.hold_count >= HOME_HOLD_STEPS:
                    self.task_done = True
                    self.ros_bridge.publish_robot_a_done(True)
                    print("\n[A] 주유 완료 -> robot_a/done = True 발행\n")
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
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print("[A] " + self.sequence.debug_string(ee_pos))


# ============================================================
# Robot B runner: 8단계 상태머신
# 시작 시 fuel_door는 이미 130도 열린 상태로 세팅되어 있으므로
# WAIT_YELLOW / OPEN_COVER / RETURN_MID는 건너뛰고 WAIT_BLUE부터 시작한다.
# WAIT_BLUE -> MOVE_TO_CAP -> GRIP_UNSCREW -> RETURN_HOME_WITH_CAP ->
# WAIT_ROBOT_A -> RESTORE_CAP -> CLOSE_COVER -> FINAL_HOME
# ============================================================
class RobotBRunner:
    """B 로봇(마개/커버 닫기 담당)의 동작을 매 스텝 진행시키는 state machine.
    fuel_door는 시작 시 130도 열린 상태로 강제 세팅하고, B는 마개를 열고 닫은 뒤 마지막에 커버를 닫는다.
    run_state가 큰 단계를, sub_phase가 한 단계 안의 더 작은 하위 단계를 나타낸다."""

    def __init__(self, robot, controller, ros_bridge: MultiRobotRosBridge, camera_prim_path, task: MultiRobotOilingTask):
        self.robot = robot
        self.controller = controller
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.task = task
        self.run_state = "WAIT_BLUE"  # 시작 시 덮개가 이미 130도 열려 있으므로 마개 lock부터 시작
        self.sub_phase = None
        self.lock_acquirer = None
        self.sequence = None
        self.locked_target_orientation = None
        self.locked_door_center = task.door_world_position.copy()  # lock 되기 전까지 쓸 임시 기본값
        self.locked_cap_center = task.cap_world_position.copy()
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
        self.cover_anim_stage_index = None
        self.cover_anim_start_pos = None
        self.cover_anim_start_angle = COVER_OPEN_DEG
        self.cover_anim_last_angle = COVER_OPEN_DEG

    def on_play_reset(self):
        """시뮬레이션을 Play로 시작할 때마다 호출되어 모든 상태를 깨끗하게 초기화한다.
        fuel_door는 task.post_reset()에서 이미 130도 열린 상태가 되므로,
        B 로봇은 노란색 커버 lock/열기 단계를 건너뛰고 곧바로 파란색 마개 lock부터 시작한다."""
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = align_orientation_to_port_axis(ee_ori)
        self.run_state = "WAIT_BLUE"
        self.sub_phase = None
        self.locked_door_center = self.task.door_world_position.copy()
        self.locked_cap_center = self.task.cap_world_position.copy()
        self.lock_acquirer = StableTargetLockAcquirer(
            self.ros_bridge, self.camera_prim_path, self.task.cap_world_position, apply_mouth_offset=False,
        )
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self.gripper_hold_count = 0
        self.joint6_index = find_dof_index(self.robot, "joint_6")
        print(f"[B-6] joint6_index={self.joint6_index}, dof_names={self.robot.dof_names}")
        if self.joint6_index is None:
            print("[B-6][WARN] dof_names 안에서 'joint_6'을 찾지 못함 (joint6_index=None) "
                  "- rotate/screw가 잘못된 인덱스로 동작할 수 있음")
        elif self.robot.dof_names[self.joint6_index] != "joint_6":
            print(f"[B-6][WARN] joint6_index={self.joint6_index} 가 'joint_6'이 아니라 "
                  f"'{self.robot.dof_names[self.joint6_index]}'를 가리킴! rotate/screw 대상이 잘못됐을 수 있음")
        self._a_done_count_at_reset = self.ros_bridge.robot_a_done_count
        self._reset_cover_animation(COVER_OPEN_DEG)
        set_prim_visibility(self.task.cap_prim_path, True)  # 이전 판에서 숨겨둔 마개가 있을 수 있어 리셋 시 항상 보이게 정리
        self.ros_bridge.publish_mode_switch("blue")
        print(f"[B] RESET -> fuel_door {COVER_OPEN_DEG:.0f}도 열린 상태, WAIT_BLUE (mode_switch=blue 발행)")

    def _robot_a_done_received(self) -> bool:
        """리셋 이후 새로 도착한 robot_a/done=True 신호가 있는지 확인 (RobotARunner의 같은 패턴과 동일한 이유)."""
        return (
            self.ros_bridge.robot_a_done_count > self._a_done_count_at_reset
            and self.ros_bridge.robot_a_done
        )

    def _hold_gripper(self, closed: bool):
        """그리퍼를 열거나 닫은 상태로 "계속 유지"시킨다. RMPFlow와 별개로 매 스텝 직접 명령해야
        한 번 닫은 그리퍼가 의도치 않게 풀리지 않는다."""
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

    def _sync_cover_door_angle_from_command(self, stage: FuelStage, tracked_pos: np.ndarray):
        """닫기 시퀀스의 현재 waypoint 진행률에 맞춰 fuel_door 각도를 갱신한다.

        기존에는 CLOSE_COVER가 끝난 뒤 set_door_angle_deg(COVER_START_DEG)를 한 번 호출해서
        덮개가 마지막 순간에 갑자기 닫히는 것처럼 보였다. 이제는 B10_01/B10_02/B10_03
        각 단계의 command target 진행률을 사용해서 130 -> 80 -> 30도로 계속 회전시킨다.
        """
        if stage is None or stage.target_position is None or stage.target_door_angle is None:
            return

        if self.cover_anim_stage_index != self.sequence.index:
            self.cover_anim_stage_index = self.sequence.index
            self.cover_anim_start_pos = np.array(tracked_pos, dtype=float).copy()
            self.cover_anim_start_angle = float(self.cover_anim_last_angle)

        start_pos = np.array(self.cover_anim_start_pos, dtype=float)
        end_pos = np.array(stage.target_position, dtype=float)
        tracked_pos = np.array(tracked_pos, dtype=float)
        move_vec = end_pos - start_pos
        denom = float(np.dot(move_vec, move_vec))

        if denom < 1e-9:
            progress = 1.0
        else:
            progress = float(np.dot(tracked_pos - start_pos, move_vec) / denom)
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

    def _drive_close_cover_sequence(self, ee_pos):
        """CLOSE_COVER 전용 sequence 구동.

        일반 _drive_sequence와 달리 RMPFlow에 넘기는 command target 진행률에 맞춰 fuel_door
        시각 각도를 동시에 갱신한다. 따라서 로봇팔 모션이 끝난 뒤 덮개가 강제로 튀어 닫히지 않고,
        로봇이 미는 경로를 따라 덮개도 같이 닫힌다.
        """
        done = self.sequence.update(ee_pos)
        if done:
            return True
        stage = self.sequence.current
        cmd = self.sequence.get_command_target(ee_pos)
        if cmd is not None:
            self._sync_cover_door_angle_from_command(stage, cmd)
            if stage.use_orientation and USE_TARGET_ORIENTATION:
                actions = self.controller.forward(
                    target_end_effector_position=cmd,
                    target_end_effector_orientation=self.locked_target_orientation,
                )
            else:
                actions = self.controller.forward(target_end_effector_position=cmd)
            self.robot.apply_action(actions)
        return False

    def _door_angle_ok(self) -> bool:
        """OPEN_COVER/CLOSE_COVER 단계에서, 위치만 맞으면 끝이 아니라 "실제 도어 각도"도
        목표 각도에 도달해야 다음 단계로 넘어가게 한다 (밀고 있는데 힌지가 안 움직이는 상황 방지)."""
        stage = self.sequence.current
        if stage.target_door_angle is None:
            return True
        current_angle = self.task.current_door_angle_deg()
        return abs(current_angle - stage.target_door_angle) <= stage.door_angle_tolerance

    def tick(self, step_count: int):
        """매 시뮬레이션 스텝마다 호출. run_state(와 필요하면 sub_phase)에 따라 분기해서 한 스텝 진행."""
        if self.task_done:
            return
        ee_pos, _ = self.robot.end_effector.get_world_pose()

        # ---------------- B-1: WAIT_YELLOW ----------------
        if self.run_state == "WAIT_YELLOW":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.locked_door_center = mean
                # 힌지에서 먼 쪽(손잡이 쪽)을 밀도록 lock 중심에서 x축으로 DOOR_PUSH_OFFSET만큼 뺀다.
                door_push_point = self.locked_door_center.copy()
                door_push_point[0] -= DOOR_PUSH_OFFSET
                self.sequence = build_cover_sequence(
                    "open", door_push_point, self.task.door_pivot_world, self.task.door_axis_world,
                )
                self.run_state = "OPEN_COVER"
                print(f"\n[B-1] yellow lock 완료. fuel_door center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                # 카메라 lock이 끝까지 안정되지 않으면 재시도하지 않고, USD에서 읽은 하드코딩
                # 기준값(task.door_world_position)으로 즉시 다음 단계로 넘어간다.
                print("[B] WAIT_YELLOW timeout -> 하드코딩 기준값으로 다음 단계 진행")
                door_push_point = self.task.door_world_position.copy()
                door_push_point[0] -= DOOR_PUSH_OFFSET
                self.locked_door_center = self.task.door_world_position.copy()
                self.sequence = build_cover_sequence(
                    "open", door_push_point,
                    self.task.door_pivot_world, self.task.door_axis_world,
                )
                self.run_state = "OPEN_COVER"
                return
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][WAIT_YELLOW] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
            return

        # ---------------- B-2: OPEN_COVER ----------------
        # 커버 표면 위 한 점을 30->80->130도 위치로 차례차례 밀고 가서 실제로 힌지를 회전시킨다.
        # (이전엔 위치 도착 + 도어 각도(_door_angle_ok)가 둘 다 맞아야 완료로 인정했는데, 실제로
        # 힌지가 안 따라와도(door_angle=30.0 고정) EE는 위치에 도달하므로, 그 경우 다음 단계로
        # 못 넘어가고 멈춰버렸다. 이제는 EE 위치 도달만으로 완료로 인정한다.)
        if self.run_state == "OPEN_COVER":
            if USE_MARKER_CONTROL:
                # 디버그/튜닝용: WaypointSequence 대신 뷰포트의 marker_door_push 위치를 목표로 쓴다.
                marker_pos = get_prim_world_position(MARKER_PRIM_PATHS["door_push"])
                if marker_pos is not None:
                    actions = self.controller.forward(
                        target_end_effector_position=marker_pos,
                        target_end_effector_orientation=self.locked_target_orientation,
                    )
                    self.robot.apply_action(actions)
                    err = np.linalg.norm(marker_pos - ee_pos)
                    if step_count % PRINT_EVERY_N_STEPS == 0:
                        print(f"[B][OPEN_COVER][MARKER] door_angle={self.task.current_door_angle_deg():.1f} err={err:.3f}")
                    if err < POSITION_TOLERANCE:
                        mid_point = rotate_point_around_axis(
                            self.locked_door_center, self.task.door_pivot_world, self.task.door_axis_world,
                            COVER_MID_DEG - COVER_START_DEG,
                        )
                        mid_point = mid_point + PORT_OUTWARD_NORMAL_UNIT * RETURN_MID_OUTWARD_OFFSET
                        self.sequence = single_stage_sequence(
                            "B3_return_mid", mid_point, hold_steps=20, tolerance=0.035, speed=RETREAT_TARGET_SPEED,
                        )
                        self.run_state = "RETURN_MID"
                        print("\n[B-2][MARKER] door_push 마커 도착 -> RETURN_MID\n")
                return

            done = self._drive_sequence(ee_pos)
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][OPEN_COVER] door_angle={self.task.current_door_angle_deg():.1f} "
                      + self.sequence.debug_string(ee_pos))
            if done:
                mid_point = rotate_point_around_axis(
                    self.locked_door_center, self.task.door_pivot_world, self.task.door_axis_world,
                    COVER_MID_DEG - COVER_START_DEG,
                )
                # 마개(blue)가 카메라 시야에 들어오도록 차체 바깥쪽으로 더 빠져나간 위치까지 후퇴한다.
                mid_point = mid_point + PORT_OUTWARD_NORMAL_UNIT * RETURN_MID_OUTWARD_OFFSET
                self.sequence = single_stage_sequence(
                    "B3_return_mid", mid_point, hold_steps=20, tolerance=0.035, speed=RETREAT_TARGET_SPEED,
                )
                self.run_state = "RETURN_MID"
                print(f"\n[B-2] 커버 완전열림({COVER_OPEN_DEG:.0f}도) 완료 -> RETURN_MID\n")
            return

        # ---------------- B-3: RETURN_MID ----------------
        # 커버를 완전히 열어둔 채로 EE만 살짝 뒤로 빼서 중간 지점까지 이동해 마개 쪽 공간을 만든다.
        if self.run_state == "RETURN_MID":
            done = self._drive_sequence(ee_pos)
            if done:
                self.lock_acquirer = StableTargetLockAcquirer(
                    self.ros_bridge, self.camera_prim_path, self.task.cap_world_position, apply_mouth_offset=False,
                )
                self.wait_steps = 0
                self.ros_bridge.publish_mode_switch("blue")
                self.run_state = "WAIT_BLUE"
                print("\n[B-3] 중간 지점 복귀 완료 -> mode_switch=blue, 마개 lock 대기\n")
            return

        # ---------------- B-4: WAIT_BLUE ----------------
        if self.run_state == "WAIT_BLUE":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.locked_cap_center = mean
                self.sequence = build_cap_approach_sequence(self.locked_cap_center)
                self.run_state = "MOVE_TO_CAP"
                print(f"\n[B-4] blue lock 완료. fuel_cap center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                # 파란색(마개) lock이 끝까지 안정되지 않으면 재시도하지 않고, USD에서 읽은
                # 하드코딩 기준값(task.cap_world_position)으로 즉시 다음 단계로 넘어간다.
                self.locked_cap_center = self.task.cap_world_position.copy()
                self.sequence = build_cap_approach_sequence(self.locked_cap_center)
                self.run_state = "MOVE_TO_CAP"
                print("\n[B] WAIT_BLUE timeout -> 하드코딩 기준값으로 다음 단계 진행\n")
                return
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][WAIT_BLUE] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
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
                    if step_count % PRINT_EVERY_N_STEPS == 0:
                        print(f"[B][MOVE_TO_CAP][MARKER] err={err:.3f}")
                    if err < POSITION_TOLERANCE:
                        self.sub_phase = "close_grip"
                        self.gripper_hold_count = 0
                        self.run_state = "GRIP_UNSCREW"
                        print("\n[B-5][MARKER] cap_approach 마커 도착 -> 그리퍼 닫기 시작\n")
                return

            done = self._drive_sequence(ee_pos)
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print("[B][MOVE_TO_CAP] " + self.sequence.debug_string(ee_pos))
            if done:
                self.sub_phase = "close_grip"
                self.gripper_hold_count = 0
                self.run_state = "GRIP_UNSCREW"
                print("\n[B-5] 마개 grasp 위치 도착 -> 그리퍼 닫기 시작\n")
            return

        # ---------------- B-6: GRIP_UNSCREW (close_grip -> rotate -> extract) ----------------
        # 그리퍼로 마개를 잡고(close_grip) -> joint_6을 -360도 돌려서 마개를 풀고(rotate)
        # -> 그리퍼를 닫은 채로 빼낸다(extract). rotate 구간은 RMPFlow를 호출하지 않고
        # joint_6만 apply_action으로 직접 증분시킨다 (나머지 조인트는 매 스텝 현재값을 그대로 재전송해서
        # 자세가 흐트러지지 않게 유지).
        if self.run_state == "GRIP_UNSCREW":
            if self.sub_phase == "close_grip":
                self._hold_gripper(closed=True)
                self.gripper_hold_count += 1
                if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                    # 그리퍼가 마개를 잡았으니, 회전/추출/복귀 내내는 "손에 들고 빠진 것처럼"
                    # 보이도록 마개를 숨긴다 (실제 물리 부착은 하지 않음).
                    set_prim_visibility(self.task.cap_prim_path, False)
                    self.joint6_accumulated = 0.0
                    self.sub_phase = "rotate"
                    print(f"[B-6] joint6_index={self.joint6_index} -> "
                          f"dof_names[{self.joint6_index}]='{self.robot.dof_names[self.joint6_index]}'")
                    print("\n[B-6] 그리퍼 닫힘 -> joint_6 unscrew 회전 시작\n")
                return

            if self.sub_phase == "rotate":
                self._hold_gripper(closed=True)
                current_joints = self.robot.get_joint_positions()
                target_joints = current_joints.copy()
                target_joints[self.joint6_index] += UNSCREW_ANGLE_STEP_RAD
                self.robot.apply_action(ArticulationAction(joint_positions=target_joints))
                self.joint6_accumulated += UNSCREW_ANGLE_STEP_RAD
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[B][GRIP_UNSCREW.rotate] accumulated={np.degrees(self.joint6_accumulated):.1f}/{CAP_JOINT6_UNSCREW_DEG}")
                if abs(self.joint6_accumulated) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
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
                    print("\n[B-6] unscrew 360도 완료 -> 마개 빼는 중\n")
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
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[B][GRIP_UNSCREW.extract] t={t:.2f} ee_pos={np.round(ee_pos, 3)} "
                          f"target={np.round(self.extract_target_ee, 3)}")
                if t >= 1.0:
                    # GRIP_UNSCREW의 extract는 마개를 쥔 채 끝까지 안 놓아야 하므로(놓는 건
                    # RESTORE_CAP의 open_grip에서) 다음은 RETURN_HOME_WITH_CAP으로 간다.
                    target_joints = build_initial_joint_positions(
                        self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_B,
                    )
                    self._home_target_joints = target_joints
                    self._home_hold_count = 0
                    self.run_state = "RETURN_HOME_WITH_CAP"
                    print("\n[B-6] 마개 추출 완료 -> RETURN_HOME_WITH_CAP\n")
                return

        # ---------------- B-7: RETURN_HOME_WITH_CAP ----------------
        # 마개를 그리퍼로 꼭 쥔 채(닫힌 상태 유지) 처음 기본 자세로 복귀한다.
        if self.run_state == "RETURN_HOME_WITH_CAP":
            self._hold_gripper(closed=True)
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][RETURN_HOME_WITH_CAP] hold={self._home_hold_count}/{HOME_HOLD_STEPS}")
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.ros_bridge.publish_robot_b_done(True)
                self.run_state = "WAIT_ROBOT_A"
                print("\n[B-7] 초기 위치 복귀(마개 보유) 완료 -> robot_b/done = True 발행\n")
            return

        # ---------------- WAIT_ROBOT_A (A-8 동안 대기) ----------------
        # A가 주유를 끝낼 때까지 B는 마개를 그리퍼로 꼭 쥔 채 아무것도 안 하고 기다린다.
        if self.run_state == "WAIT_ROBOT_A":
            self._hold_gripper(closed=True)
            if self._robot_a_done_received():
                # 마개를 닫을 때는 주유구 구멍 중심(hole_world_position)이 아니라,
                # 마개를 열 때 실제로 lock/grasp했던 중심을 다시 사용한다.
                # fuel_cap은 물리적으로 그리퍼에 붙인 것이 아니라 숨김/표시로 표현하므로,
                # 복원 목표도 cap prim의 원래 위치와 일치해야 한다.
                restore_center = (
                    self.locked_cap_center.copy()
                    if self.locked_cap_center is not None
                    else self.task.cap_world_position.copy()
                )
                self.sequence = build_cap_restore_sequence(restore_center)
                self.sub_phase = "insert"
                self.run_state = "RESTORE_CAP"
                print("\n[B-9] robot_a/done 수신 -> 마개 복원 시작")
                print(f"[B-9] restore_center(cap)={np.round(restore_center, 4)}, "
                      f"hole_center={np.round(self.task.hole_world_position, 4)}\n")
            elif step_count % PRINT_EVERY_N_STEPS == 0:
                print("[B][WAIT_ROBOT_A] robot_a/done 대기 중")
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
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print("[B][RESTORE_CAP.insert] " + self.sequence.debug_string(ee_pos))
                if done:
                    # screw 시작 시점의 관절값을 한 번 캡처해서 고정해두고, 이후 매 스텝은 이
                    # 고정된 기준값에서 joint_6만 누적 스텝 수만큼 더해 계산한다 (rotate처럼 매번
                    # get_joint_positions()를 다시 읽지 않으므로, 측정값 드리프트와 무관하게
                    # 나머지 관절(1~5)이 흔들리지 않는다).
                    self.frozen_joint_positions = self.robot.get_joint_positions().copy()
                    self.joint6_accumulated = 0.0
                    self.screw_step_count = 0
                    self.sub_phase = "screw"
                    print(f"[B-9] joint6_index={self.joint6_index} -> "
                          f"dof_names[{self.joint6_index}]='{self.robot.dof_names[self.joint6_index]}'")
                    print("\n[B-9] 마개 삽입 위치 도착 -> joint_6 screw 회전 시작\n")
                return

            if self.sub_phase == "screw":
                self._hold_gripper(closed=True)
                # screw는 rotate의 정반대 방향이므로 같은 스텝 크기를 빼서(-=) 반대로 돌린다.
                # 매 스텝 frozen_joint_positions(고정된 기준값)에서 joint_6만 누적 스텝 수만큼
                # 계산해서 다시 써넣으므로, 다른 관절은 절대 흔들리지 않는다.
                target_joints = self.frozen_joint_positions.copy()
                self.screw_step_count += 1
                target_joints[self.joint6_index] -= UNSCREW_ANGLE_STEP_RAD * self.screw_step_count
                self.robot.apply_action(ArticulationAction(joint_positions=target_joints))
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    progress_deg = np.degrees(UNSCREW_ANGLE_STEP_RAD * self.screw_step_count)
                    print(f"[B][RESTORE_CAP.screw] progress={abs(progress_deg):.1f}/{CAP_JOINT6_SCREW_DEG}")
                if abs(self.screw_step_count * UNSCREW_ANGLE_STEP_RAD) >= abs(UNSCREW_TOTAL_ANGLE_RAD):
                    # 마개가 다시 끼워졌으니 숨겨뒀던 마개를 다시 보이게 해서 "복원됐다"는 걸 표현한다.
                    set_prim_visibility(self.task.cap_prim_path, True)
                    self.gripper_hold_count = 0
                    self.sub_phase = "open_grip"
                    print("\n[B-9] screw 360도 완료 -> 그리퍼 열기\n")
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
                    retreat_target = ee_pos + PORT_OUTWARD_NORMAL_UNIT * 0.20
                    self.sequence = single_stage_sequence(
                        "B9_extract_retreat", retreat_target,
                        hold_steps=15, tolerance=0.10, speed=RETREAT_TARGET_SPEED,
                    )
                    self.sub_phase = "retreat"
                    print("\n[B-9] 그리퍼 열기 완료 -> 바깥쪽으로 후퇴\n")
                return

            if self.sub_phase == "retreat":
                self._hold_gripper(closed=False)
                done = self._drive_sequence(ee_pos)
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print("[B][RESTORE_CAP.retreat] " + self.sequence.debug_string(ee_pos))
                if done:
                    # 닫는 동작은 여는 동작과 반대쪽에서 접근하므로, lock 중심에서
                    # x축으로 DOOR_CLOSE_PUSH_OFFSET만큼 더한 지점을 미는 지점으로 쓴다.
                    door_close_push_point = self.locked_door_center.copy()
                    door_close_push_point[0] += DOOR_CLOSE_PUSH_OFFSET
                    self.sequence = build_cover_sequence(
                        "close", door_close_push_point, self.task.door_pivot_world, self.task.door_axis_world,
                    )
                    self.sub_phase = None
                    self._reset_cover_animation(COVER_OPEN_DEG)
                    self.task.set_door_angle_deg(
                        COVER_OPEN_DEG,
                        label="fuel_door close start",
                        refresh_viewport=False,
                        verbose=False,
                    )
                    self.run_state = "CLOSE_COVER"
                    print("\n[B-9] 마개 복원 완료 -> CLOSE_COVER (덮개 각도는 로봇 모션에 맞춰 130→30도로 동기화)\n")
            return

        # ---------------- B-10: CLOSE_COVER ----------------
        # OPEN_COVER와 같은 방식이지만 역순(130->80->30도)으로 밀어서 커버를 닫는다.
        # 덮개 각도는 마지막에 강제 스냅하지 않고, _drive_close_cover_sequence() 안에서
        # command target 진행률에 맞춰 130 -> 80 -> 30도로 계속 갱신한다.
        if self.run_state == "CLOSE_COVER":
            done = self._drive_close_cover_sequence(ee_pos)
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][CLOSE_COVER] door_angle={self.task.current_door_angle_deg():.1f} "
                      + self.sequence.debug_string(ee_pos))
            if done:
                # 여기서 set_door_angle_deg(COVER_START_DEG)를 다시 호출하지 않는다.
                # 이미 B10_03 진행 중에 덮개가 로봇 모션과 함께 닫혔기 때문에,
                # 별도 보정 호출을 하면 화면상 덮개가 마지막 순간에 튀어 보일 수 있다.
                target_joints = build_initial_joint_positions(
                    self.robot, self.robot.get_joint_positions(), INITIAL_ARM_JOINT_DEG_B,
                )
                self._home_target_joints = target_joints
                self._home_hold_count = 0
                self.run_state = "FINAL_HOME"
                print("\n[B-10] 커버 닫힘 모션 완료 -> FINAL_HOME\n")
            return

        # ---------------- B-11: FINAL_HOME ----------------
        # 모든 작업이 끝났으니 기본 자세로 돌아가 멈춘다. 여기가 끝나면 B의 전체 시퀀스가 종료된다.
        if self.run_state == "FINAL_HOME":
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][FINAL_HOME] hold={self._home_hold_count}/{HOME_HOLD_STEPS}")
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.task_done = True
                print("\n[B-11] 초기 위치 복귀 완료. 전체 시퀀스 종료.\n")
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
    initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)
    task.post_reset()  # 시작 화면/첫 Play 모두 fuel_door가 130도 열린 상태가 되도록 맞춘다.

    # 초기화 직후 물리엔진이 안정화되도록 몇 프레임 그냥 흘려보낸다.
    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-1] 초기 상태")
    print("=" * 60)
    print(f"  m0609_A base = {ROBOT_A_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG_A}")
    print(f"  m0609_B base = {ROBOT_B_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG_B}")

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
    print("  [OK] RMPFlowController A/B 생성 완료")

    if not rclpy.ok():
        rclpy.init(args=None)
    ros_bridge = MultiRobotRosBridge()
    camera_prim_path = find_camera_prim_path()
    print(f"  [ROS] camera_prim_path = {camera_prim_path}")

    runner_a = RobotARunner(robot_a, controller_a, ros_bridge, camera_prim_path, task)
    runner_b = RobotBRunner(robot_b, controller_b, ros_bridge, camera_prim_path, task)

    was_playing = False  # 직전 프레임에 재생 중이었는지 (Play 버튼을 "막 눌렀는지" 판단용)
    step_count = 0

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
            initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION, INITIAL_ARM_JOINT_DEG_B)
            controller_a.reset()
            controller_b.reset()
            task.post_reset()
            runner_a.on_play_reset()
            runner_b.on_play_reset()
            step_count = 0
            print("\n[RESET] multi-robot oiling sequence 준비 완료\n")

        if is_playing:
            step_count += 1
            runner_a.tick(step_count)
            runner_b.tick(step_count)

            if runner_a.task_done and runner_b.task_done:
                # 두 로봇 모두 전체 시퀀스를 끝마쳤으면 시뮬레이션을 자동으로 일시정지한다.
                print("\n[완료] A/B 전체 시퀀스 종료 - 시뮬레이션 일시정지\n")
                my_world.pause()

        was_playing = is_playing

    # 창을 닫으면(simulation_app.is_running()이 False가 되면) ROS2와 Isaac Sim을 정리하고 종료.
    if rclpy.ok():
        ros_bridge.destroy_node()
        rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
