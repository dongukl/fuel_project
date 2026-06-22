from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
# 창생성

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from dataclasses import dataclass
# 데이터 묶음으로 간편하게 하게 도구
from pathlib import Path
# 위치 찾는 도구
import sys
# 입력값 받아오기
import time
# 시간 제어
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# Qos설정시 필요
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import numpy as np
import omni.usd
# 아이작심에 열려있는 씬에 접근할때 사용
from pxr import Usd, UsdGeom, UsdPhysics, Gf
# Usd : 부품 순회
# UsdGeom : 공간및 도형 생성
# UsdPhysics : 관절 드라이브
# Gf : 백터 생성
# Prim >> USD에서 모든 객체의 기본 단위

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
# 물체 생성
from isaacsim.core.api.tasks import BaseTask
# 테스크 틀
from isaacsim.core.utils.rotations import euler_angles_to_quat
# 오일러 각을 쿼터니언으로 변경
from isaacsim.robot.manipulators.grippers import ParallelGripper
# 그리퍼 제어
from isaacsim.robot.manipulators.manipulators import SingleManipulator


_THIS_DIR = Path(__file__).resolve().parent
# 상위 파일 경우 가져오기
# rmpflow 인프라 폴더 경로 등록


RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
# RMPFLOW 파일에서 값 가져오기
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)
# 파일 없다면 가장 앞에 넣주기 

from m0609_rmpflow_controller import RMPFlowController
# 직접 동작을 제어할때 쓰는 컨트롤러

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 기존 Pick & Place 코드 기반 환경 파라미터                  ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_nozzletip_project/nozzletip_project.usd")
#USD 파일 경로
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"       # nozzle_tip 추가 전까지는 link_6 기준 제어
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8 # 강도
DRIVE_DAMPING   = 1e4 # 완충제 
DRIVE_MAX_FORCE = 1e8 # 하중

GRIPPER_OPEN    = [0.0, 0.0] # 그리퍼 열린 상태
GRIPPER_CLOSE   = [0.5, 0.5] # 그리퍼 닫힌 상태
GRIPPER_DELTA   = [-0.5, -0.5] # 그리퍼 상태 변환 

# RMPFlow 설정 파일 경로
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 벽 부착형 자동 주유 테스트 파라미터                         ║
# ╚══════════════════════════════════════════════════════════════╝

# 목표: 로봇 베이스를 벽면에 부착된 형태로 두고, y=-0.95 위치의 실린더 주유구에 삽입한다.
ROBOT_BASE_WORLD = np.array([0.0, 0.0, 1.0], dtype=float)
# 로봇 베이스 위치 z축으로 1미터
ROBOT_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)\
# x축으로 90도 (DEG ~도의미)
ROBOT_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_BASE_EULER_DEG))
# 쿼터니언으로 변환과정
# 도 >> 라디안 >> 쿼터니언 

# 사용자가 지정한 초기 관절각. 
# Isaac/URDF 관절 명령은 radian이므로 내부에서 deg->rad 변환한다.
INITIAL_ARM_JOINT_DEG = {
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

# ------------------------------------------------------------------
# 탐색 기준점과 실제 표적 위치를 분리한다.
#
# SEARCH_REFERENCE_CENTER:
#   로봇이 "주유구가 있을 법한 영역"으로 먼저 이동할 때 쓰는 고정 기준점.
#   실제 표적 위치를 알기 위해 쓰는 값이 아니라, 탐색 자세를 만들기 위한 rough prior이다.
#
# FUEL_PORT_CENTER:
#   시뮬레이션에 생성되는 실제 표적 실린더 위치.
#   실제 시스템에서는 모르는 값이며, detector가 찾아야 하는 값이다.
#   랜덤 테스트를 켜면 SEARCH_REFERENCE_CENTER 주변 범위 안에서 매 실행마다 달라진다.
# ------------------------------------------------------------------
# SEARCH_REFERENCE_CENTER = np.array([0.32, -1.2, 1.0], dtype=float)

# # 주유구 랜덤 생성 
# USE_RANDOM_FUEL_PORT_CENTER = False # 랜덤 생성 
# FUEL_PORT_RANDOM_SEED = None  # 재현하고 싶으면 예: 7
# FUEL_PORT_RANDOM_X_RANGE = (-0.18, 0.18)    # x값 범위
# FUEL_PORT_RANDOM_Y_RANGE = (-0.10, 0.10)    # y값 범위 
# FUEL_PORT_RANDOM_Z_RANGE = (0.00, 0.00)     # z값 범위

# MANUAL_FUEL_PORT_CENTER = np.array([0.32, -1.2, 1.0], dtype=float)


# def sample_fuel_port_center() -> np.ndarray:
#     if not USE_RANDOM_FUEL_PORT_CENTER:
#         return np.array(MANUAL_FUEL_PORT_CENTER, dtype=float)

#     rng = random.Random(FUEL_PORT_RANDOM_SEED) if FUEL_PORT_RANDOM_SEED is not None else random.Random()
#     offset = np.array([
#         rng.uniform(*FUEL_PORT_RANDOM_X_RANGE),
#         rng.uniform(*FUEL_PORT_RANDOM_Y_RANGE),
#         rng.uniform(*FUEL_PORT_RANDOM_Z_RANGE),
#     ], dtype=float)
#     return SEARCH_REFERENCE_CENTER + offset


FUEL_PORT_CENTER = np.array([0.32, -1.2, 1.0], dtype=float)
# 실린더 위치값 
FUEL_PORT_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)  
# 주유구 각도값
FUEL_PORT_DIAMETER = 0.10 # 지름
FUEL_PORT_RADIUS = FUEL_PORT_DIAMETER / 2.0 # 반지름
FUEL_PORT_DEPTH = 0.10 # 깊이 


# 105도 기준으로 각도값을 지정한다.
PORT_OUTWARD_NORMAL = np.array([0.0, np.sin(np.deg2rad(105.0)), -np.cos(np.deg2rad(105.0))], dtype=float)
INSERTION_DIRECTION = np.array([0.0, -np.sin(np.deg2rad(105.0)), np.cos(np.deg2rad(105.0))], dtype=float)

# 접근 거리. 모두 실린더 중심 기준 거리이다.
# 중간 지점을 두어서 원하는 모션을 만들기 위함.
FAR_DISTANCE     = 0.28 
MID_DISTANCE     = 0.18
NEAR_DISTANCE    = 0.09 
# 실린더 중심에서 -Y 방향으로 4.5cm 들어간 위치. 
INSERT_DISTANCE  = FUEL_PORT_DEPTH / 2.0 

# 현재 EE frame은 link_6이므로, 실제 실린더 안으로 들어가는 점은 가상의 nozzle_tip으로 둔다.

VIRTUAL_NOZZLE_LENGTH = 0.65
# 가상의 노즐 길이
VIRTUAL_NOZZLE_Z_OFFSET = -0.25
# 가상의 노즐 높이

# waypoint 판정 기준
POSITION_TOLERANCE = 0.060
# 목표에 도달했다고 판단하는 오차 허용값
MAX_STEPS_PER_STAGE = 1200
# 한 단계에서 최대 몇스탭까지 기다릴지 1200/60 >> 20초
PRINT_EVERY_N_STEPS = 20
# 몇스탭마다 터미널에 디버그를 출력할지 

# 속도 제한: RMPFlow 목표점을 바로 주지 않고, 가상의 command target을 조금씩 이동시킨다.
PHYSICS_DT = 1.0 / 60.0
# 1초에 60번 업데이트 
DEFAULT_TARGET_SPEED = 0.060
# 일반 이동 속도 : 0.06 * 60 = 3.6m/s
NEAR_TARGET_SPEED    = 0.040
# 근처 이동 : 2.4m/s
INSERT_TARGET_SPEED  = 0.020
# 삽입 : 1.2m/s
RETREAT_TARGET_SPEED = 0.050
# 후퇴 : 3m/s

# 복귀는 사용자가 지정한 초기 자세로 돌아간다.
HOME_JOINT_SPEED_ALPHA = 0.012
HOME_JOINT_TOLERANCE = 0.035
HOME_HOLD_STEPS = 80

# link_6 위치만 제어하면 손목/그리퍼가 기울어진 채로 접근하여 충돌할 수 있다.
# 하지만 euler 고정값을 중간 stage부터 갑자기 강제하면 손목이 과하게 정렬될 수 있다.
# 따라서 Play/reset 직후의 EE orientation을 "삽입 축 정렬 자세"로 잠그고,
# 모든 접근/삽입/후퇴 구간에서 같은 orientation을 유지한다.
# link_5를 직접 EE로 바꾸지는 않는다. 대신 link_5->link_6 축이 삽입축과 얼마나 어긋나는지 로그로 감시한다.
USE_TARGET_ORIENTATION = True
TARGET_ORIENTATION = None  # main loop에서 초기 EE orientation으로 설정

# ╔══════════════════════════════════════════════════════════════╗
# ║  C. ROS 인식 기반 target 연동 파라미터                          ║
# ╚══════════════════════════════════════════════════════════════╝
# fuel_port_detector_node_v2.py가 발행하는 안정화 pose를 받아서
# Camera frame 좌표를 Isaac Sim world 좌표로 변환한 뒤,
# 받아서 주유를 실행한다.


USE_ROS_DETECTED_TARGET = True
# Ros2 토픽으로 받은 주유구 위치를 사용 
ROS_POSE_TOPIC = "/fuel_port/pose_camera_filtered"
# 토픽 이름
ROS_LOCK_TOPIC = "/fuel_port/target_locked"
# 위치가 확정되었다는 토픽 신호
REQUIRE_TARGET_LOCK = True
# lock 신호가 와야만 주유 동작을 시작
USE_FIXED_TARGET_FALLBACK = False



DETECTED_POINT_IS_MOUTH_CENTER = True
# detector가 잡은 점이 실린더 입구면 중심이라고 가정

# Isaac USD Camera local axis와 ROS optical camera frame의 축 차이를 보정한다.
# ROS optical: +X right, +Y down, +Z forward
# USD camera : +X right, +Y up,   -Z forward
CAMERA_POINT_CONVENTION = "ros_optical"

# 카메라 주소 
CAMERA_PRIM_CANDIDATES = [
    "/World/wall/rsd455/RSD455",
    "/World/wall/rsd455/Camera",
    "/World/wall/rsd455/camera",
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 탐색 자세 → 인식 lock → 주유 sequence 파라미터              ║
# ╚══════════════════════════════════════════════════════════════╝
# 주유구가 처음부터 카메라에 보인다고 가정하지 않는다.
# 먼저 주유구가 있을 법한 위치 앞까지 이동한 뒤, 그 자세에서 detector가 안정화된 좌표를 낼 때까지 기다린다.
# 탐색 중 들어오는 검출값은 "참고"만 하고, 즉시 주유 sequence를 시작하지 않는다.
# SEARCH_STANDOFF_DISTANCE = 0.36       # fuel center 기준, 가상 nozzle_tip이 대기할 거리(+Y)
# SEARCH_SCAN_X = 0.12                  # 탐색 실패 시 좌우로 훑는 폭
# SEARCH_MOVE_SPEED = 0.045
# SEARCH_SCAN_SPEED = 0.030
# SEARCH_STAGE_HOLD_STEPS = 40


CONTROLLER_REQUIRED_LOCK_SAMPLES = 5
# detector가 보내는 위치값을 5번 모아서 평균을 냄. 
# 1번만 받으면 오탐지일 수 있으니까.
CONTROLLER_WORLD_STD_TOLERANCE = 0.025   
# 모은 5개 샘플의 표준편차가 2.5cm 이하일 때만 "안정적"이라고 판단.
CONTROLLER_TARGET_GATE_RADIUS = 0.45     
# 옛날 방식의 거리 gate. 지금은 안 쓰고 아래 박스 gate를 씀.

# 검출된 target이 "탐색 영역" 안에 있는지 확인하는 gate.
# 중요한 점: 이 gate는 실제 FUEL_PORT_CENTER가 아니라 SEARCH_REFERENCE_CENTER 기준이다.
# 따라서 표적 위치를 랜덤으로 바꿔도 탐색 로직은 실제 위치를 미리 알지 않는다.
SEARCH_GATE_HALF_EXTENT = np.array([0.35, 0.35, 0.18], dtype=float)  
# x/y/z 허용 범위 [m]
LOCK_Z_TO_SEARCH_REFERENCE = True  # Camera->World z축 검증 전에는 높이 급락 방지용으로 True 권장
# 카메라→월드 z축 변환이 불안정할 수 있어서, 높이값은 기준점 높이로 고정.

# WAIT 상태에서 너무 오래 lock이 안 되면 다시 탐색 scan을 반복한다.
WAIT_LOCK_TIMEOUT_STEPS = 900            # 대략 15초 @ 60 Hz


# ============================================================
# 유틸
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n
# 방향 벡터만 알고 싶어서 길이를 1로 만드어서 계싼 


# 부품을 찾고 주소를 가져옴 
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    # 도면 가져옴 
    root_prim = stage.GetPrimAtPath(root_path)
    # root_path = "/World/m0609" 이면 로봇 전체를 가져옴 
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    # 지정한 이름과 동일한 면 주소 반환 
    return None


# 특정 월드좌표를 가져옴 
def get_prim_world_position(prim_path: str) -> np.ndarray | None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    # 변환행렬을 저장하는 도구 
    mat = cache.GetLocalToWorldTransform(prim)
    # 월드 좌표의 행렬 가져옴
    t = mat.ExtractTranslation()
    # 위치만 필요하니까 ExtractTranslation()으로 꺼냄
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)



# 5번 조인트와 6번 조인트의 방향이 얼마나 틀어져 있는지 알려주는 함수
# 직선으로 움직이기 위해서 두 조인트를 동일하게 움직이게 하기 위해서 
# 두 백터사이의 각도를 -1, 1사이로 강제 제한 
def angle_deg_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

# 관절이름으로 인덱스를 찾는 함수
def find_dof_index(robot, dof_name: str):
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        # robot 객체에 dof_names 속성이 있는지 확인
        # 없으면 에러 방지
        return robot.dof_names.index(dof_name)
    return None

# 초기 관절값 배열을 만드는 함수 
def build_initial_joint_positions(robot, base_positions=None) -> np.ndarray:
    """현재 robot.num_dof에 맞춰 초기 관절 벡터를 만든다."""
    if base_positions is None:
        q = np.zeros(robot.num_dof, dtype=float)
    # 빈 배열 생성 
    else:
        q = np.array(base_positions, dtype=float).copy()
        if len(q) != robot.num_dof:
            q = np.zeros(robot.num_dof, dtype=float)

    # 1차: dof_names로 정확히 매핑
    missing_arm = []
    for joint_name, deg in INITIAL_ARM_JOINT_DEG.items():
        # 조인트 이름 값 꺼내기 >> 이렇게 쓰려고 딕셔너리를 썻군..
        idx = find_dof_index(robot, joint_name)
        # 조인트 인덱스값 찾기 
        if idx is None:
            missing_arm.append(joint_name)
        else:
            q[idx] = np.deg2rad(deg)

    # 방어: dof_names가 다르게 들어오는 경우, 앞 6개를 arm joint로 가정해 fallback
    if missing_arm and robot.num_dof >= 6:
        fallback_values = [
            INITIAL_ARM_JOINT_DEG["joint_1"],
            INITIAL_ARM_JOINT_DEG["joint_2"],
            INITIAL_ARM_JOINT_DEG["joint_3"],
            INITIAL_ARM_JOINT_DEG["joint_4"],
            INITIAL_ARM_JOINT_DEG["joint_5"],
            INITIAL_ARM_JOINT_DEG["joint_6"],
        ]
        # 해당이름의 조인트 값만 다시 리스트로 
        for i, deg in enumerate(fallback_values):
            q[i] = np.deg2rad(deg)
            # 라이안 값으로 바꿔서 다시 값 지정 

    # gripper는 0으로 열린 상태 유지
    for joint_name, value in INITIAL_GRIPPER_JOINTS.items():
        idx = find_dof_index(robot, joint_name)
        if idx is not None:
            q[idx] = value

    return q


# 로봇 시작 상태 
# 벽에 부착이라서 지정하는 함수 
def apply_robot_start_state(robot):
    """벽 부착형 root pose와 사용자가 지정한 초기 관절각을 적용한다."""
    robot.set_world_pose(
        position=ROBOT_BASE_WORLD,
        orientation=ROBOT_BASE_ORIENTATION,
    )
    current = robot.get_joint_positions()
    q0 = build_initial_joint_positions(robot, current)
    robot.set_joint_positions(q0)
    return q0


# 로봇 초기화 
def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    q0 = apply_robot_start_state(robot)
    robot.gripper.set_joint_positions(np.array(GRIPPER_OPEN, dtype=float))
    return q0


# 실린더 생성 
def create_usd_visual_cylinder(prim_path: str, position: np.ndarray, radius: float, height: float,
                               euler_deg: np.ndarray, color: np.ndarray):
    """VisualCylinder import 호환성 이슈를 피하기 위해 pxr UsdGeom.Cylinder로 직접 생성한다."""
    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    # z축 방향으로 세운다
    cyl.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    # 컬러값 
    cyl.CreateDisplayOpacityAttr([0.85])
    # 투명도 

    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])))
    return cyl




class RosFuelPortTargetReceiver(Node):
    """detector로 부터 주유구 위치를 받는 클래스 """

    def __init__(self):
        super().__init__("m0609_fuel_port_ros_target_receiver")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.latest_pose: PoseStamped | None = None
        # detector에서 가장 최근에 받은 주유구 위치값. 처음엔 아무것도 안 받았으니 None.
        self.target_locked: bool = False
        # 정확한 위치를 받았는지. 처음엔 안받았으니까 False
        self.pose_count = 0
        self.lock_count = 0
        self.pose_sub = self.create_subscription(
            PoseStamped,
            ROS_POSE_TOPIC,
            self.pose_callback,
            qos,
        )
        self.lock_sub = self.create_subscription(
            Bool,
            ROS_LOCK_TOPIC,
            self.lock_callback,
            qos,
        )
        self.get_logger().info("RosFuelPortTargetReceiver started")
        self.get_logger().info(f"  위치 토픽이름 = {ROS_POSE_TOPIC}")
        self.get_logger().info(f"  최종 토픽이름 = {ROS_LOCK_TOPIC}")

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg
        self.pose_count += 1

    def lock_callback(self, msg: Bool):
        self.target_locked = bool(msg.data)
        self.lock_count += 1

    # 위치값도 있고, lock도 됐을 때만 값 반환 >> 위치값 
    def get_pose_if_ready(self) -> PoseStamped | None:
        if self.latest_pose is None:
            return None
        if REQUIRE_TARGET_LOCK and not self.target_locked:
            return None
        return self.latest_pose


def find_camera_prim_path() -> str | None:
    """카메라 주소 찾기 """
    stage = omni.usd.get_context().get_stage()
    for path in CAMERA_PRIM_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path
    # 주소들을 확인하고 유효하다면 반환 

    # 찾지 못했다면 로봇을 순회하면서 "camera", "rsd455" 이름이 있다면 반환 
    # 로봇에도 없고 주소도 잘못되어서 카메라가 연결이 안되는 이슈가 있던것
    # 내일 주소 수정

    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            # Isaac/Usd camera prim usually has type name Camera.
            if prim.GetTypeName() == "Camera" or prim.GetName().lower() in ["camera", "rsd455"]:
                return str(prim.GetPath())
    return None

# 카메라 좌표계가 ROS랑 USD가 서로 달라서 변환해주는 함수예요.
def camera_ros_point_to_usd_camera_local(point_camera_ros: np.ndarray) -> np.ndarray:
    """카메라 좌표 변환 >>ROS와 USD서로 달라서 변환 과정이 필요 """
    x, y, z = [float(v) for v in point_camera_ros]
    if CAMERA_POINT_CONVENTION == "ros_optical":
        # ROS optical: +X right, +Y down, +Z forward
        # USD camera : +X right, +Y up,   -Z forward
        return np.array([x, -y, -z], dtype=float)
    # If the topic is already in USD camera-local convention, set CAMERA_POINT_CONVENTION differently.
    return np.array([x, y, z], dtype=float)


def transform_camera_point_to_world(point_camera_ros: np.ndarray, camera_prim_path: str) -> np.ndarray | None:
    """Transform Camera-frame 3D point into Isaac Sim world coordinates.

    detector의 point_camera_ros는 ROS optical frame 기준이다.
      +X: image right
      +Y: image down
      +Z: camera forward

    현재 벽 부착형 카메라 배치에서는 camera forward가 world -Y 방향으로 가는 것이
    자연스럽다. 기존 USD Camera local transform을 그대로 쓰면 camera forward가 world -Z로
    들어가 z가 급락하는 문제가 생길 수 있어, 이 버전은 수동 축 매핑을 기본으로 쓴다.

    예상 매핑:
      camera +Z forward -> world -Y
      camera +Y down    -> world -Z
      camera +X right   -> world -X

    x 방향이 반대로 보이면 delta_world[0]의 부호만 바꾸면 된다.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        return None
    # 카메라 prim을 찾는다 
    cache = UsdGeom.XformCache()
    # 월드좌표값 
    mat = cache.GetLocalToWorldTransform(prim)
    # 변환 
    t = mat.ExtractTranslation()
    # 위치값 
    camera_origin_world = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)
    # 카메라를 월드좌표계 기준 어디에 있는지를 찾는다 
    x_cam = float(point_camera_ros[0])
    y_cam = float(point_camera_ros[1])
    z_cam = float(point_camera_ros[2])
    # detector가 보내준 카메라 좌표계 기준 위치값을 x, y, z로 분리

    delta_world = np.array([
        -x_cam,   # image right -> world -X. 반대면 +x_cam으로 변경
        -z_cam,   # camera forward -> world -Y
        -y_cam,   # image down -> world -Z
    ], dtype=float)
    # 카메라 좌표 → 월드 좌표로 축 변환. 카메라 앞쪽이 월드 -Y 방향이라서 이렇게 매핑해요.
    point_world = camera_origin_world + delta_world
    # 카메라 위치 + 변환된 오프셋 = 주유구의 월드 좌표
    return point_world
    # 카메라 주소만 있으면 위치값을 알아서 찾음

def detected_world_point_to_fuel_center(detected_world_point: np.ndarray) -> np.ndarray:
    """Convert detected visible/mouth point to cylinder center expected by FuelPortSequence."""
    if DETECTED_POINT_IS_MOUTH_CENTER:
        return detected_world_point - normalize(PORT_OUTWARD_NORMAL) * (FUEL_PORT_DEPTH / 2.0)
    return detected_world_point

@dataclass
class FuelStage:
    name: str
    target_position: np.ndarray | None
    hold_steps: int = 0
    tolerance: float = POSITION_TOLERANCE
    max_steps: int = MAX_STEPS_PER_STAGE
    speed: float = DEFAULT_TARGET_SPEED
    use_orientation: bool = False


# class SearchSequence:
#     """주유구 탐색용 사전 이동 sequence.

#     목적:
#     - 시작하자마자 detector pose로 직진하지 않는다.
#     - 먼저 주유구가 있을 법한 영역 앞까지 이동한다.
#     - 탐색 자세에서 잠깐 멈춘 뒤, 필요하면 좌우로 천천히 훑는다.
#     - 이 sequence가 끝난 뒤에만 ROS target lock을 주유 sequence에 반영한다.
#     """

#     def __init__(self, reference_center: np.ndarray | None = None):
#         self.reference_center = np.array(FUEL_PORT_CENTER if reference_center is None else reference_center, dtype=float)
#         self.port_outward_normal = normalize(PORT_OUTWARD_NORMAL)
#         self.command_target = None
#         self.index = 0
#         self.stage_step = 0
#         self.hold_count = 0
#         self.done = False

#         # 가상 nozzle_tip이 주유구 앞에서 대기할 위치.
#         tip_observe_center = self.reference_center + self.port_outward_normal * SEARCH_STANDOFF_DISTANCE

#         # RMPFlow 제어 프레임은 link_6이므로, link_6 target은 가상 tip보다 바깥쪽(+Y)으로 물린다.
#         control_offset = self.port_outward_normal * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])
#         observe_center = tip_observe_center + control_offset

#         # 차량 좌측 주유구 탐색을 가정해, X 방향으로 약간 훑는 waypoint를 둔다.
#         # detector lock은 이 sequence가 끝난 뒤 WAIT_LOCK 상태에서만 반영된다.
#         self.stages = [
#             FuelStage(
#                 "S01_move_to_search_center",
#                 observe_center,
#                 hold_steps=SEARCH_STAGE_HOLD_STEPS,
#                 tolerance=0.035,
#                 speed=SEARCH_MOVE_SPEED,
#                 use_orientation=True,
#             ),
#             FuelStage(
#                 "S02_scan_plus_x",
#                 observe_center + np.array([SEARCH_SCAN_X, 0.0, 0.0]),
#                 hold_steps=SEARCH_STAGE_HOLD_STEPS,
#                 tolerance=0.035,
#                 speed=SEARCH_SCAN_SPEED,
#                 use_orientation=True,
#             ),
#             FuelStage(
#                 "S03_scan_minus_x",
#                 observe_center + np.array([-SEARCH_SCAN_X, 0.0, 0.0]),
#                 hold_steps=SEARCH_STAGE_HOLD_STEPS,
#                 tolerance=0.035,
#                 speed=SEARCH_SCAN_SPEED,
#                 use_orientation=True,
#             ),
#             FuelStage(
#                 "S04_back_to_search_center",
#                 observe_center,
#                 hold_steps=SEARCH_STAGE_HOLD_STEPS,
#                 tolerance=0.035,
#                 speed=SEARCH_SCAN_SPEED,
#                 use_orientation=True,
#             ),
#         ]

    # @property
    # def current(self) -> FuelStage:
    #     return self.stages[self.index]

    # def reset(self):
    #     self.index = 0
    #     self.stage_step = 0
    #     self.hold_count = 0
    #     self.done = False
    #     self.command_target = None

    # def update(self, ee_position: np.ndarray) -> bool:
    #     if self.done:
    #         return True

    #     stage = self.current
    #     self.stage_step += 1
    #     err = np.linalg.norm(stage.target_position - ee_position)
    #     reached = err < stage.tolerance
    #     timed_out = self.stage_step >= stage.max_steps

    #     if reached:
    #         self.hold_count += 1
    #     else:
    #         self.hold_count = 0

    #     if (reached and self.hold_count >= stage.hold_steps) or timed_out:
    #         status = "reached" if reached else "timeout"
    #         print(f"\n[SEARCH STAGE END] {stage.name} -> {status}")
    #         self.index += 1
    #         self.stage_step = 0
    #         self.hold_count = 0
    #         self.command_target = None
    #         if self.index >= len(self.stages):
    #             self.done = True
    #             return True
    #     return self.done

    # def get_command_target(self, ee_position: np.ndarray) -> np.ndarray | None:
    #     if self.done:
    #         return None
    #     stage = self.current
    #     if self.command_target is None:
    #         self.command_target = np.array(ee_position, dtype=float)

    #     delta = stage.target_position - self.command_target
    #     dist = np.linalg.norm(delta)
    #     max_step = max(stage.speed * PHYSICS_DT, 1e-5)
    #     if dist <= max_step:
    #         self.command_target = np.array(stage.target_position, dtype=float)
    #     else:
    #         self.command_target = self.command_target + delta / dist * max_step
    #     return self.command_target

    # def debug_string(self, ee_position: np.ndarray) -> str:
    #     if self.done:
    #         return "[SEARCH DONE]"
    #     stage = self.current
    #     err = np.linalg.norm(stage.target_position - ee_position)
    #     cmd = self.command_target if self.command_target is not None else np.array([np.nan, np.nan, np.nan])
    #     return (
    #         f"[SEARCH stage={self.index}:{stage.name}] "
    #         f"target={np.round(stage.target_position, 3)} "
    #         f"cmd={np.round(cmd, 3)} "
    #         f"ee={np.round(ee_position, 3)} "
    #         f"err={err:.4f} "
    #         f"hold={self.hold_count}/{stage.hold_steps}"
    #     )


def validate_detected_fuel_center_world(center_world: np.ndarray) -> tuple[bool, str]:
    """오탐지 방지를 위한 world gating.

    이 버전에서는 실제 표적 위치 FUEL_PORT_CENTER를 기준으로 검사하지 않는다.
    실제 표적은 detector가 찾아야 하는 값이기 때문이다.

    대신 SEARCH_REFERENCE_CENTER 주변의 탐색 박스 안에 들어오는지만 본다.
    즉, "주유구가 있을 법한 영역"은 고정하되, 그 안에서 실제 표적 위치는 랜덤/가변이다.
    """
    if center_world is None or not np.all(np.isfinite(center_world)):
        # None이나 무한대 값 같은 이상한 값이라면 바로 False
        return False, "non-finite target"

    delta = center_world - FUEL_PORT_CENTER
    abs_delta = np.abs(delta)
    if np.any(abs_delta > SEARCH_GATE_HALF_EXTENT):
        return (
            False,
            "outside search gate: "
            f"center={np.round(center_world, 3)}, "
            f"delta={np.round(delta, 3)}, "
            f"half_extent={np.round(SEARCH_GATE_HALF_EXTENT, 3)}"
        )
    # 지정해 둔 범위 밖이라면 바로 false
    return True, f"inside search gate: delta={np.round(delta, 3)}"
    # 다 통과시 반환 

class FuelPortSequence:
    """벽면 실린더 주유구에 대한 자동 주유 waypoint state machine."""

    def __init__(self, fuel_port_center: np.ndarray | None = None):
        self.fuel_port_center = np.array(FUEL_PORT_CENTER if fuel_port_center is None else fuel_port_center, dtype=float)
        self.port_outward_normal = normalize(PORT_OUTWARD_NORMAL)
        self.insertion_direction = normalize(INSERTION_DIRECTION)

        # tip waypoint: 실제로 실린더에 들어간다고 가정하는 가상 nozzle_tip 목표.
        tip_approach_far  = self.fuel_port_center + self.port_outward_normal * FAR_DISTANCE
        tip_approach_mid  = self.fuel_port_center + self.port_outward_normal * MID_DISTANCE
        tip_approach_near = self.fuel_port_center + self.port_outward_normal * NEAR_DISTANCE
        tip_insert_target = self.fuel_port_center + self.insertion_direction * INSERT_DISTANCE

        # control waypoint: 현재 RMPFlow EE frame은 link_6이므로, link_6 목표는 tip 목표보다 바깥쪽으로 물러나야 한다.
        # nozzle_tip = link_6 - port_outward_normal * VIRTUAL_NOZZLE_LENGTH 라고 가정한다.
        control_offset = self.port_outward_normal * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])
        approach_far  = tip_approach_far + control_offset
        approach_mid  = tip_approach_mid + control_offset
        approach_near = tip_approach_near + control_offset
        insert_target = tip_insert_target + control_offset

        self.tip_targets = {
            "01_axis_far_start": tip_approach_far,
            "02_axis_mid": tip_approach_mid,
            "03_axis_near_stop": tip_approach_near,
            "04_insert_into_cylinder": tip_insert_target,
            "05_retreat_near": tip_approach_near,
            "06_retreat_mid": tip_approach_mid,
            "07_retreat_far": tip_approach_far,
        }

        self.stages = [
            # 벽 부착형 시나리오: 위쪽 점을 찍지 않고 처음부터 주유구 축과 같은 선상으로 직선 접근한다.
            # orientation은 main loop에서 reset 직후 EE 자세로 잠근 뒤 전 구간 유지한다.
            FuelStage("01_axis_far_start", approach_far, tolerance=0.030, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            #단계 이름, 목표위치, 접근 위치, 속도, 위치 고정
            FuelStage("02_axis_mid", approach_mid, tolerance=0.028, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            FuelStage("03_axis_near_stop", approach_near, hold_steps=80, tolerance=0.022, speed=NEAR_TARGET_SPEED, use_orientation=True),
            FuelStage("04_insert_into_cylinder", insert_target, hold_steps=180, tolerance=0.018, speed=INSERT_TARGET_SPEED, use_orientation=True),

            # 후퇴도 같은 orientation을 유지한 채 역순으로 빠져나온다.
            FuelStage("05_retreat_near", approach_near, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("06_retreat_mid", approach_mid, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("07_retreat_far", approach_far, hold_steps=30, tolerance=0.035, speed=RETREAT_TARGET_SPEED, use_orientation=True),

            # RMPFlow 위치 제어가 아니라 joint 직접 보간으로 초기 자세 복귀
            FuelStage("08_return_home", None, hold_steps=HOME_HOLD_STEPS, use_orientation=False),
        ]
        self.index = 0
        self.stage_step = 0
        self.hold_count = 0
        self.done = False
        self.command_target = None

    @property
    def current(self) -> FuelStage:
        return self.stages[self.index]

    def reset(self):
        self.index = 0
        self.stage_step = 0
        self.hold_count = 0
        self.done = False
        self.command_target = None

    def update(self, ee_position: np.ndarray, ee_orientation: np.ndarray = None) -> bool:
        """현재 stage 완료 여부를 판단하고, 필요하면 다음 stage로 전환한다."""
        if self.done:
            return True

        stage = self.current
        self.stage_step += 1

        reached = False
        # 도착 여부 
        if stage.target_position is None:
            # return_home은 main loop에서 별도 처리하므로 여기서 자동 완료시키지 않는다.
            reached = False
        else:
            err = np.linalg.norm(stage.target_position - ee_position)
            reached = err < stage.tolerance
            # 남은 거리가 허용 오차값 보다 작다면 도착으로 지정 

        timed_out = self.stage_step >= stage.max_steps

        if reached:
            self.hold_count += 1
        else:
            self.hold_count = 0

        STAGE_LOG = {
                "01_axis_far_start":       "접근 시작",
                "02_axis_mid":             "중간 지점 도착",
                "03_axis_near_stop":       "근처 도착 대기",
                "04_insert_into_cylinder": "삽입 완료",
                "05_retreat_near":         "후퇴 시작",
                "06_retreat_mid":          "중간 후퇴",
                "07_retreat_far":          "먼 후퇴 완료",
                "08_return_home":          "홈 복귀 완료",
            }
        
        if (reached and self.hold_count >= stage.hold_steps) or timed_out:
            if timed_out and not reached:
                print(f"\n[⚠️ 타임아웃] {STAGE_LOG.get(stage.name, stage.name)}")
                print(f"  위치 = {np.round(ee_position, 3)}")
                print(f"  목표까지 = {np.linalg.norm(stage.target_position - ee_position):.3f}m 남음\n")
            else:
                print(f"\n[✅ {STAGE_LOG.get(stage.name, stage.name)}]")
                print(f"  위치  = {np.round(ee_position, 3)}")
                if ee_orientation is not None:
                    print(f"  각도  = {np.round(ee_orientation, 3)}\n")

            self.index += 1
            self.stage_step = 0
            self.hold_count = 0
            self.command_target = None
            if self.index >= len(self.stages):
                self.done = True
                return True
        return self.done

    def get_command_target(self, ee_position: np.ndarray) -> np.ndarray | None:
        """실제 stage 목표까지 한 번에 보내지 않고, 속도 제한된 중간 목표를 반환한다."""
        if self.done:
            return None
        stage = self.current
        if stage.target_position is None:
            return None
        if self.command_target is None:
            self.command_target = np.array(ee_position, dtype=float)

        delta = stage.target_position - self.command_target
        # 목표까지의 백터
        dist = np.linalg.norm(delta)
        # 목표 거리
        max_step = max(stage.speed * PHYSICS_DT, 1e-5)
        # 이번 스탬에 최대 이동할 수 있는거리 
        if dist <= max_step:
            self.command_target = np.array(stage.target_position, dtype=float)
        else:
            self.command_target = self.command_target + delta / dist * max_step
        return self.command_target

    def debug_string(self, ee_position: np.ndarray) -> str:
        if self.done:
            return "[DONE] fuel sequence complete"
        stage = self.current
        if stage.target_position is None:
            return f"[stage={self.index}:{stage.name}] hold={self.hold_count}/{stage.hold_steps}"
        err = np.linalg.norm(stage.target_position - ee_position)
        cmd = self.command_target if self.command_target is not None else np.array([np.nan, np.nan, np.nan])
        tip = self.tip_targets.get(stage.name, None)
        tip_str = "None" if tip is None else str(np.round(tip, 3))
        return (
            f"[단계 ={self.index}:{stage.name}] "
            f"link6_목표={np.round(stage.target_position, 3)} "
            f"tip_목표={tip_str} "
            f"현재명령={np.round(cmd, 3)} "
            f"현재위치={np.round(ee_position, 3)} "
            f"남은거리={err:.4f} "
            f"속도={stage.speed:.3f} "
            f"방향고정={'ON' if stage.use_orientation else 'OFF'} "
            f"유지 카운트={self.hold_count}/{stage.hold_steps}"
        )


# ============================================================
# Task — Pick & Place 구조 유지, 작업물만 벽면 실린더 주유구로 변경
# ============================================================
class M0609FuelPortWallCylinderTask(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        # self.search_sequence = SearchSequence(FUEL_PORT_CENTER)
        self.sequence = FuelPortSequence(FUEL_PORT_CENTER)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_fuel_port_scene(scene)
        print("\n  [완료] 벽 부착형 자동 주유 테스트 씬 구성 성공!\n")
   
    # usd 로드
    def _load_usd(self):
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
        print("  [NOTE] 이번 버전은 로봇 root pose를 벽 부착형으로 코드에서 명시 적용함")
    
    # 링크 발견 >> 링크 주소 저장
    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        self._link5_path = find_prim_path_by_name(ROBOT_PRIM_PATH, "link_5")
        self._link6_path = find_prim_path_by_name(ROBOT_PRIM_PATH, "link_6")
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")
        print(f"  link_5 monitor = {self._link5_path}")
        print(f"  link_6 monitor = {self._link6_path}")
        for jn in GRIPPER_JOINTS:
            print(f"  {jn:<35} = {find_prim_path_by_name(ROBOT_PRIM_PATH, jn)}")
   
    # 로봇에 강도, 완충, 하중 설정 
    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 로봇 drive 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        drive_count = 0
        root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not root.IsValid():
            raise RuntimeError(f"Robot prim not found: {ROBOT_PRIM_PATH}")

        for prim in Usd.PrimRange(root):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    # 로봇 생성
    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] SingleManipulator 등록")
        print("=" * 60)
        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")
    # 씬 구상 
    def _create_fuel_port_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 벽면 실린더 주유구 + waypoint 생성")
        print("=" * 60)

        # 실린더 주유구 생성. radius 0.05, height 0.10 -> 0.1 x 0.1 x 0.1 크기.
        create_usd_visual_cylinder(
            prim_path="/World/fuel_port_cylinder",
            position=FUEL_PORT_CENTER,
            radius=FUEL_PORT_RADIUS,
            height=FUEL_PORT_DEPTH,
            euler_deg=FUEL_PORT_EULER_DEG,
            color=np.array([0.0, 1.0, 0.0]),
        )
        print(f"  [OK] 주유구 실린더 중심 @ {FUEL_PORT_CENTER}")
        # print(f"  [OK] search_reference_center @ {SEARCH_REFERENCE_CENTER}")
        print(f"  [OK] 탐색 범위 = {SEARCH_GATE_HALF_EXTENT}")
        print(f"  [OK] 실린더 크기 = diameter {FUEL_PORT_DIAMETER}, depth {FUEL_PORT_DEPTH}")
        print(f"  [INFO] 주유구 각도 = {FUEL_PORT_EULER_DEG}")
        print(f"  [INFO] 주유구 바깥 방향 = {np.round(self.sequence.port_outward_normal, 4)}")
        print(f"  [INFO] 주유구 삽입 방향 = {np.round(self.sequence.insertion_direction, 4)}")
        print(f"  [INFO] 가상 노즐 길이 = {VIRTUAL_NOZZLE_LENGTH} m")
        print("  [NOTE] marker는 link_6 제어 목표이고, 실제 삽입점은 tip_target 로그를 기준으로 판단")
        print("  [NOTE] 벽 부착형이므로 위쪽 waypoint 없이 축 방향으로 바로 접근")
        print("  [NOTE] 모든 접근/삽입/후퇴 구간에서 reset 직후 EE orientation을 유지")

        # 입구면 중심 표시: 실린더 전면, 즉 로봇 쪽 face.
        mouth_center = FUEL_PORT_CENTER + self.sequence.port_outward_normal * (FUEL_PORT_DEPTH / 2.0)
        scene.add(
            VisualCuboid(
                prim_path="/World/fuel_port_mouth_center",
                name="fuel_port_mouth_center",
                position=mouth_center,
                scale=np.array([0.035, 0.006, 0.035]),
                color=np.array([1.0, 1.0, 0.0]),
            )
        )
        print(f"  [OK] 주유구 마커 생성 @ {np.round(mouth_center, 4)}")
        # 탐색 기준점 표시. 이 점은 실제 표적이 아니라 "있을 법한 영역"의 기준이다.
        # scene.add(
        #     VisualCuboid(
        #         prim_path="/World/search_reference_center",
        #         name="search_reference_center",
        #         position=SEARCH_REFERENCE_CENTER,
        #         scale=np.array([0.04, 0.04, 0.04]),
        #         color=np.array([0.0, 1.0, 1.0]),
        #     )
        # )
        # print(f"  [OK] search_reference_center marker @ {np.round(SEARCH_REFERENCE_CENTER, 4)}")


        # waypoint marker 생성
        for i, stage in enumerate(self.sequence.stages):
            if stage.target_position is None:
                continue
            if "insert" in stage.name:
                color = np.array([1.0, 0.0, 1.0])
                scale = np.array([0.035, 0.035, 0.035])
            elif "retreat" in stage.name:
                color = np.array([0.0, 1.0, 0.3])
                scale = np.array([0.025, 0.025, 0.025])
            elif "near" in stage.name:
                color = np.array([1.0, 0.5, 0.0])
                scale = np.array([0.030, 0.030, 0.030])
            else:
                color = np.array([0.0, 0.3, 1.0])
                scale = np.array([0.025, 0.025, 0.025])

            # link_6 control target marker
            scene.add(
                VisualCuboid(
                    prim_path=f"/World/fuel_wp_{i:02d}_{stage.name}",
                    name=f"fuel_wp_{i:02d}_{stage.name}",
                    position=stage.target_position,
                    scale=scale,
                    color=color,
                )
            )

            # 가상 nozzle_tip marker: 실제 실린더에 들어가는 점을 따로 표시한다.
            tip_target = self.sequence.tip_targets.get(stage.name)
            if tip_target is not None:
                scene.add(
                    VisualCuboid(
                        prim_path=f"/World/fuel_tip_wp_{i:02d}_{stage.name}",
                        name=f"fuel_tip_wp_{i:02d}_{stage.name}",
                        position=tip_target,
                        scale=np.array([0.015, 0.015, 0.015]),
                        color=np.array([1.0, 1.0, 1.0]),
                    )
                )

            print(
                f"  [OK] {i:02d} {stage.name:<28} "
                f"link6_target={np.round(stage.target_position, 4)} "
                f"tip_target={np.round(tip_target, 4) if tip_target is not None else None}"
            )

    # 로봇 현재 상태 추적 
    def get_observations(self):
        ee_pos, ee_ori = self._robot.end_effector.get_world_pose()
        link5_pos = get_prim_world_position(getattr(self, "_link5_path", ""))
        link6_pos = get_prim_world_position(getattr(self, "_link6_path", ""))
        link5_to_link6_angle = None
        if link5_pos is not None and link6_pos is not None:
            link5_to_link6_axis = link6_pos - link5_pos
            link5_to_link6_angle = angle_deg_between(link5_to_link6_axis, INSERTION_DIRECTION)
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
                "ee_position": ee_pos,
                "ee_orientation": ee_ori,
                "link5_to_link6_angle_deg": link5_to_link6_angle,
            },
            "fuel_port": {
                "center": FUEL_PORT_CENTER,
                "outward_normal": self.sequence.port_outward_normal,
                "insertion_direction": self.sequence.insertion_direction,
            },
        }

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        self.sequence.reset()


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 메인 — RMPFlow waypoint 제어                              ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    my_world = World(stage_units_in_meters=1.0)
    # 월드 단위 1미터로 설정
    task = M0609FuelPortWallCylinderTask(name="m0609_fuel_port_wall_cylinder_task")
    # tset설정
    my_world.add_task(task)
    # 테스트 넣어주기
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
     # 로봇 가져오기
    q0 = initialize_robot(robot, my_world)
    # 설정한 관절값들 가져오기

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-1] 초기 상태")
    print("=" * 60)
    print(f"  로봇 베이스 위치     = {ROBOT_BASE_WORLD}")
    print(f"  로봇 베이스 각도 = {ROBOT_BASE_EULER_DEG}")
    print(f"  관절이름            = {robot.dof_names}")
    print(f"  각도 값(라디안)       = {np.round(q0, 4)}")
    print(f"  각도 값(도)  = {INITIAL_ARM_JOINT_DEG}")

    print("\n" + "=" * 60)
    print("[C-2] RMPFlowController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  EE frame    = {EE_LINK_NAME}")
    print(f"  orientation = {'ON' if USE_TARGET_ORIENTATION else 'OFF'}")
    print("  target_ori   = reset 직후 EE orientation을 사용해 전진 중 자세를 유지")

    # 주의: RMPFlowController는 생성 시점의 robot world pose를 base pose로 cache한다.
    # 그래서 반드시 initialize_robot()로 벽 부착 pose를 적용한 뒤 생성해야 한다.
    controller = RMPFlowController(
        name="m0609_fuel_port_wall_cylinder_rmpflow_controller",
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] RMPFlowController 생성 완료")

    # ROS target receiver. detector v2가 /fuel_port/pose_camera_filtered와 /fuel_port/target_locked를 발행해야 한다.
    if not rclpy.ok():
        rclpy.init(args=None)
    ros_receiver = RosFuelPortTargetReceiver()
    camera_prim_path = find_camera_prim_path()
    print(f"  [ROS] camera_prim_path = {camera_prim_path}")

    ee_pos, ee_ori = robot.end_effector.get_world_pose()
    locked_target_orientation = ee_ori.copy()
    print(f"\n  EE 초기 위치          = {np.round(ee_pos, 4)}")
    print(f"  locked EE orientation = {np.round(locked_target_orientation, 4)}")
    # print(f"  Search reference center = {SEARCH_REFERENCE_CENTER}")
    # print(f"  Actual fuel cylinder center = {FUEL_PORT_CENTER}")
    # print(f"  Random fuel target = {USE_RANDOM_FUEL_PORT_CENTER}")
    # print("\n[벽 부착형 축방향 직선 접근 테스트 시작 - v6 close/camera-flip - Play 버튼을 누르면 동작]\n")

    was_playing = False
    task_done = False
    step_count = 0

    # run_state:
    #   SEARCH_MOVE  : 주유구가 있을 법한 위치까지 이동/스캔한다. detector pose는 아직 주유 목표로 쓰지 않는다.
    #   WAIT_LOCK    : 탐색 자세에서 detector lock pose를 모아 world 좌표 안정성을 확인한다.
    #   RUN_SEQUENCE : lock된 world 좌표로 FuelPortSequence를 생성하고 주유 동작을 수행한다.
    run_state = "WAIT_LOCK"
    detected_fuel_center_world = None
    target_world_samples = []
    last_sampled_pose_count = -1
    wait_lock_steps = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()
        if USE_ROS_DETECTED_TARGET and rclpy.ok():
            rclpy.spin_once(ros_receiver, timeout_sec=0.0)

        # Play 시작 감지 → 리셋
        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            # task.search_sequence = SearchSequence(FUEL_PORT_CENTER)
            task.sequence = FuelPortSequence(FUEL_PORT_CENTER)
            # reset 직후의 EE orientation을 잠근다.
            # 이전 버전처럼 중간 stage에서 euler [90,0,0]을 갑자기 강제하지 않는다.
            _, locked_target_orientation = robot.end_effector.get_world_pose()
            locked_target_orientation = locked_target_orientation.copy()
            task_done = False
            step_count = 0
            run_state = "WAIT_LOCK"
            detected_fuel_center_world = None
            target_world_samples = []
            last_sampled_pose_count = -1
            wait_lock_steps = 0
            print("\n[RESET] 탐색 이동 → 인식 lock → 주유 sequence 준비")
            print(f"[RESET] locked EE orientation = {np.round(locked_target_orientation, 4)}")
            # print(f"[RESET] search_reference_center = {np.round(SEARCH_REFERENCE_CENTER, 4)}")
            print(f"[RESET] actual_target_for_sim_only = {np.round(FUEL_PORT_CENTER, 4)}")
            print("[RESET] detector pose가 들어와도 SEARCH_MOVE 중에는 주유 동작을 시작하지 않음\n")

        if is_playing and not task_done:
            obs = task.get_observations()
            ee_position = obs["m0609_robot"]["ee_position"]

            # ------------------------------------------------------------
            # STATE 1: 탐색 이동/스캔
            # ------------------------------------------------------------
            # if run_state == "SEARCH_MOVE":
            #     # detector는 계속 spin되고 있지만, 이 단계에서는 오탐 방지를 위해 pose를 주유 목표로 사용하지 않는다.
            #     search_done = task.search_sequence.update(ee_position)
            #     command_target = task.search_sequence.get_command_target(ee_position)

            #     if command_target is not None:
            #         if USE_TARGET_ORIENTATION:
            #             actions = controller.forward(
            #                 target_end_effector_position=command_target,
            #                 target_end_effector_orientation=locked_target_orientation,
            #             )
            #         else:
            #             actions = controller.forward(target_end_effector_position=command_target)
            #         robot.apply_action(actions)

            #     if step_count % PRINT_EVERY_N_STEPS == 0:
            #         print(
            #             task.search_sequence.debug_string(ee_position)
            #             + f" detector_lock={ros_receiver.target_locked} pose_count={ros_receiver.pose_count}"
            #         )

            #     if search_done:
            #         run_state = "WAIT_LOCK"
            #         wait_lock_steps = 0
            #         target_world_samples = []
            #         last_sampled_pose_count = -1
            #         print("\n[SEARCH DONE] 탐색 자세 도달. 이제 detector lock pose를 안정화한 뒤 target으로 사용함.\n")

            #     step_count += 1
            #     was_playing = is_playing
            #     continue

            # ------------------------------------------------------------
            # STATE 2: 탐색 자세에서 detector target lock 대기
            # ------------------------------------------------------------
            if run_state == "WAIT_LOCK":
                wait_lock_steps += 1
                # 대기 시간 카운트
                pose_msg = ros_receiver.get_pose_if_ready() if USE_ROS_DETECTED_TARGET else None
                # ROS사용중이라면 detector에서 pose가져오기 아니면 None
                # detector가 lock을 잃으면 buffer를 비운다.
                if USE_ROS_DETECTED_TARGET and not ros_receiver.target_locked:
                    target_world_samples = []
                # detector가 lock을 잃으면 buffer를 비운다.

                
                if pose_msg is not None and ros_receiver.pose_count != last_sampled_pose_count:
                    last_sampled_pose_count = ros_receiver.pose_count
                # 새로운 pose가 들어왔을 때만 처리, 중복 카운트 방지
                    if camera_prim_path is None:
                        print("[ERROR] Camera prim path를 찾지 못해서 Camera->World 변환 불가")
                        task_done = True
                        my_world.pause()
                        continue
                    # 경로 찾지 못하면 continue

                    p_cam = np.array([
                        pose_msg.pose.position.x,
                        pose_msg.pose.position.y,
                        pose_msg.pose.position.z,
                    ], dtype=float)
                    # detector가 보낸 위치값 
                    detected_world_point = transform_camera_point_to_world(p_cam, camera_prim_path)
                    # 월드 좌표로 변호나 
                    if detected_world_point is None:
                        print("[ERROR] Camera point transform 실패")
                        task_done = True
                        my_world.pause()
                        continue
                    # 실패시 continue
                    # 입구면 → 실린더 중심으로 변환, 높이는 고정값으로 유지
                    candidate_center_world = detected_world_point_to_fuel_center(detected_world_point)
                    if LOCK_Z_TO_SEARCH_REFERENCE:
                        # Camera->World z축 변환이 완전히 검증되기 전까지 높이 급락을 방지한다.
                        # 실제 높이 랜덤 테스트를 하고 싶으면 LOCK_Z_TO_SEARCH_REFERENCE=False로 바꾼다.
                        candidate_center_world[2] = FUEL_PORT_CENTER[2]

                    valid, reason = validate_detected_fuel_center_world(candidate_center_world)

                    if valid:
                        target_world_samples.append(candidate_center_world)
                        if len(target_world_samples) > CONTROLLER_REQUIRED_LOCK_SAMPLES:
                            target_world_samples = target_world_samples[-CONTROLLER_REQUIRED_LOCK_SAMPLES:]
                    else:
                        target_world_samples = []
                    # 유효하면 샘플 추가(최대 5개 유지), 오탐지면 전부 버림

                    if step_count % PRINT_EVERY_N_STEPS == 0:
                        print(
                            f"[WAIT_LOCK sample] valid={valid} {reason} "
                            f"p_cam={np.round(p_cam, 3)} "
                            f"center_world={np.round(candidate_center_world, 3)} "
                            # f"search_ref={np.round(SEARCH_REFERENCE_CENTER, 3)} "
                            f"samples={len(target_world_samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}"
                        )

                # 최근 N개 sample의 world 좌표가 충분히 안정적일 때만 주유 sequence 시작
                if len(target_world_samples) >= CONTROLLER_REQUIRED_LOCK_SAMPLES:
                    samples = np.array(target_world_samples, dtype=float)
                    mean_center = np.mean(samples, axis=0)
                    std_norm = float(np.linalg.norm(np.std(samples, axis=0)))
                    # 샘플 5개 모이면 표준 편차 
                    if LOCK_Z_TO_SEARCH_REFERENCE:
                        mean_center[2] = FUEL_PORT_CENTER[2]
                    # 높이는 고정값 
                    if std_norm <= CONTROLLER_WORLD_STD_TOLERANCE:
                        detected_fuel_center_world = mean_center
                        task.sequence = FuelPortSequence(detected_fuel_center_world)
                        run_state = "RUN_SEQUENCE"
                        print("\n[ROS TARGET LOCKED - CONTROLLER STABLE]")
                        print(f"  fuel_center_mean = {np.round(detected_fuel_center_world, 4)}")
                        print(f"  sample_std_norm  = {std_norm:.4f} m")
                        print("  -> 검출 좌표 기반 주유 waypoint sequence 시작\n")
                    else:
                        if step_count % PRINT_EVERY_N_STEPS == 0:
                            print(f"[WAIT_LOCK] samples exist but unstable: std_norm={std_norm:.4f} m")

                if run_state == "WAIT_LOCK" and step_count % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"[WAIT_LOCK] 감지잠금={ros_receiver.target_locked} "
                        f"포스 수신 횟수={ros_receiver.pose_count} "
                        f"샘플={len(target_world_samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES} "
                        f"대기스탭={wait_lock_steps}/{WAIT_LOCK_TIMEOUT_STEPS}"
                    )

                # 오래 lock이 안 되면 대기 카운트만 초기화하고 WAIT_LOCK을 계속 유지한다.
                # (SearchSequence가 비활성화된 버전이라 SEARCH_MOVE로는 돌아가지 않는다.)
                if run_state == "WAIT_LOCK" and wait_lock_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                    print("\n[WAIT_LOCK TIMEOUT] target lock 실패. 계속 대기함.\n")
                    wait_lock_steps = 0
                    target_world_samples = []
                    last_sampled_pose_count = -1

                step_count += 1
                was_playing = is_playing
                continue

            # ------------------------------------------------------------
            # STATE 3: lock된 target으로 주유 sequence 실행
            # ------------------------------------------------------------
            obs = task.get_observations()
            ee_position = obs["m0609_robot"]["ee_position"]
            ee_orientation = obs["m0609_robot"]["ee_orientation"]
            # 매 스텝마다 로봇 현재 위치/방향 가져오기

            stage = task.sequence.current

            # 08_return_home: RMPFlow 위치 제어가 아니라 joint 직접 보간 복귀
            if stage.name == "08_return_home":
                current_joints = robot.get_joint_positions()
                target_joints = build_initial_joint_positions(robot, current_joints)
                next_joints = current_joints + HOME_JOINT_SPEED_ALPHA * (target_joints - current_joints)
                robot.set_joint_positions(next_joints)
            #홈 복귀 단계에서는 RMPFlow 대신 관절 직접 보간으로 이동
                joint_err = np.linalg.norm(next_joints[:6] - target_joints[:6])
                if joint_err < HOME_JOINT_TOLERANCE:
                    task.sequence.hold_count += 1
                else:
                    task.sequence.hold_count = 0

                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"[단계={task.sequence.index}:{stage.name}] "
                        f"괄절 오차={joint_err:.4f} "
                        f"유지={task.sequence.hold_count}/{stage.hold_steps}"
                    )

                if task.sequence.hold_count >= stage.hold_steps:
                    print("\n[STAGE END] 08_return_home -> home reached")
                    task.sequence.done = True
                    print("\n[완료] 벽 부착형 자동 주유 sequence 종료")
                    task_done = True
                    my_world.pause()

                step_count += 1
                was_playing = is_playing
                continue

            # 일반 waypoint stage: RMPFlow 위치 제어
            task_done = task.sequence.update(ee_position, ee_orientation )
            if task_done:
                print("\n[완료] 벽 부착형 자동 주유 waypoint sequence 종료")
                my_world.pause()
                was_playing = is_playing
                continue

            stage = task.sequence.current
            command_target = task.sequence.get_command_target(ee_position)

            if command_target is not None:
                # stage별 orientation 제어.
                # 위치만 제어하면 link_6는 목표점에 가지만 손목/그리퍼가 기울어진 채 전진할 수 있다.
                # 이번 버전은 중간에 강제 정렬하지 않고, reset 직후 EE 자세를 끝까지 유지한다.
                if USE_TARGET_ORIENTATION and stage.use_orientation:
                    actions = controller.forward(
                        target_end_effector_position=command_target,
                        target_end_effector_orientation=locked_target_orientation,
                    )
                else:
                    actions = controller.forward(
                        target_end_effector_position=command_target,
                    )
                robot.apply_action(actions)

            step_count += 1
            if step_count % PRINT_EVERY_N_STEPS == 0:
                link_angle = obs["m0609_robot"].get("link5_to_link6_angle_deg")
                link_angle_str = "None" if link_angle is None else f"{link_angle:.2f}deg"
                print(task.sequence.debug_string(ee_position) + f" link5_to_link6_vs_insert={link_angle_str}")

        was_playing = is_playing

    if rclpy.ok():
        ros_receiver.destroy_node()
        rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
