from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent

# rmpflow 인프라 폴더 경로 등록
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_rmpflow_controller import RMPFlowController


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Isaac / Robot 기본 파라미터                               ║
# ╚══════════════════════════════════════════════════════════════╝
# m0609_oiling.py의 nozzle-tip 포함 USD를 기준으로 사용한다.
USD_PATH = str(_THIS_DIR / "Collected_nozzletip_project/nozzletip_project.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME = "link_6"  # 현재 RMPFlow EE frame은 link_6 기준
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN = [0.0, 0.0]
GRIPPER_CLOSE = [0.5, 0.5]
GRIPPER_DELTA = [-0.5, -0.5]

M0609_URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. oiling 분기 기준 로봇 / 주유구 / 노즐 파라미터              ║
# ╚══════════════════════════════════════════════════════════════╝
ROBOT_BASE_WORLD = np.array([0.0, 0.0, 1.0], dtype=float)
ROBOT_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)
ROBOT_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_BASE_EULER_DEG))

INITIAL_ARM_JOINT_DEG = {
    "joint_1": 10.0,
    "joint_2": -66.0,
    "joint_3": 150.0,
    "joint_4": 3.5,
    "joint_5": -85.0,
    # 카메라/그리퍼가 뒤집혀 보이는 문제를 줄이기 위해 wrist roll을 180도 뒤집는다.
    # 기존 175도 + 180도 = 355도이며, 같은 의미로 -5도로 입력한다.
    "joint_6": 0.0,
}
INITIAL_GRIPPER_JOINTS = {
    "finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
}

# 시뮬레이션에서 생성하는 기준 주유구 위치이다.
# 실제 RUN_SEQUENCE는 ROS 인식 좌표가 안정화되면 그 좌표로 새 FuelPortSequence를 생성한다.
FUEL_PORT_CENTER = np.array([0.32, -1.2, 1.0], dtype=float)
FUEL_PORT_EULER_DEG = np.array([105.0, 0.0, 0.0], dtype=float)
FUEL_PORT_DIAMETER = 0.10
FUEL_PORT_RADIUS = FUEL_PORT_DIAMETER / 2.0
FUEL_PORT_DEPTH = 0.10

# oiling 분기 기준: 주유구가 105도 기울어진 상태.
PORT_OUTWARD_NORMAL = np.array(
    [0.0, np.sin(np.deg2rad(105.0)), -np.cos(np.deg2rad(105.0))], dtype=float
)
INSERTION_DIRECTION = np.array(
    [0.0, -np.sin(np.deg2rad(105.0)), np.cos(np.deg2rad(105.0))], dtype=float
)

FAR_DISTANCE = 0.28
MID_DISTANCE = 0.18
NEAR_DISTANCE = 0.09
INSERT_DISTANCE = FUEL_PORT_DEPTH / 2.0

# oiling 분기에서 실제 노즐 모델을 고려한 link_6 -> nozzle_tip 보정.
VIRTUAL_NOZZLE_LENGTH = 0.65
VIRTUAL_NOZZLE_Z_OFFSET = -0.23

POSITION_TOLERANCE = 0.060
MAX_STEPS_PER_STAGE = 1800
PRINT_EVERY_N_STEPS = 20

PHYSICS_DT = 1.0 / 60.0
DEFAULT_TARGET_SPEED = 0.060
NEAR_TARGET_SPEED = 0.040
INSERT_TARGET_SPEED = 0.020
RETREAT_TARGET_SPEED = 0.050

HOME_JOINT_SPEED_ALPHA = 0.012
HOME_JOINT_TOLERANCE = 0.035
HOME_HOLD_STEPS = 80

USE_TARGET_ORIENTATION = True
TARGET_ORIENTATION = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 벽 고정 D455 카메라 기반 ROS target 연동 파라미터           ║
# ╚══════════════════════════════════════════════════════════════╝
USE_ROS_DETECTED_TARGET = True
ROS_POSE_TOPIC = "/fuel_port/pose_camera_filtered"
ROS_LOCK_TOPIC = "/fuel_port/target_locked"
REQUIRE_TARGET_LOCK = True

# 인식이 없을 때 고정 좌표로 바로 주유 테스트를 하고 싶으면 True.
# 실제 인식 테스트에서는 False 권장.
USE_FIXED_TARGET_FALLBACK = False

# detector가 주유구의 입구면 중심을 검출한다고 가정한다.
# FuelPortSequence는 실린더 중심을 기대하므로, center = mouth - outward_normal * depth/2 로 보정한다.
DETECTED_POINT_IS_MOUTH_CENTER = True

# 벽 고정 카메라에서는 USD prim의 world transform을 그대로 쓰는 것이 기본이다.
# 만약 USD Camera 축 변환이 이상하면 "manual_wall"로 바꿔서 아래 수동 매핑을 사용할 수 있다.
CAMERA_TRANSFORM_MODE = "usd_xform"  # "usd_xform" or "manual_wall"
CAMERA_POINT_CONVENTION = "ros_optical"

WALL_CAMERA_PRIM_PATH = "/World/wall/rsd455/RSD455/Camera_OmniVision_OV9782_Color"
CAMERA_PRIM_CANDIDATES = [
    WALL_CAMERA_PRIM_PATH,
    "/World/wall/rsd455/RSD455",
    "/World/wall/rsd455",
]

# # 실제 USD prim 이름을 아직 100% 확정할 수 없어 후보를 넓게 둔다.
# # 실행 로그의 [ROS] camera_prim_path 값을 보고, 실제 경로를 맨 위로 올려두면 된다.
# CAMERA_PRIM_CANDIDATES = [
#     "/World/realsense_d455/RSD455",
#     "/World/realsense_d455/Camera",
#     "/World/RSD455",
#     "/World/D455/Camera",
#     "/World/WallCamera",
#     "/World/Camera",
#     "/World/camera",
#     # 기존 EE 부착형 카메라 후보. oiling 분기에서는 보통 사용하지 않지만 fallback으로 남긴다.
#     "/World/m0609/onrobot_rg2ft/angle_bracket/realsense_d455/RSD455",
#     "/World/m0609/onrobot_rg2ft/angle_bracket/realsense_d455/Camera",
#     "/World/m0609/onrobot_rg2ft/angle_bracket/realsense_d455/camera",
# ]

# 벽 카메라는 처음부터 표적을 보고 있으므로 SEARCH_MOVE 없이 WAIT_LOCK부터 시작한다.
CONTROLLER_REQUIRED_LOCK_SAMPLES = 5
CONTROLLER_WORLD_STD_TOLERANCE = 0.025
EXPECTED_TARGET_CENTER = FUEL_PORT_CENTER.copy()
TARGET_GATE_HALF_EXTENT = np.array([0.45, 0.45, 0.30], dtype=float)
LOCK_Z_TO_EXPECTED_CENTER = True
WAIT_LOCK_TIMEOUT_STEPS = 1800  # 대략 30초 @ 60 Hz. timeout 후에도 움직이지 않고 다시 대기한다.


# ============================================================
# 유틸
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def get_prim_world_position(prim_path: str) -> np.ndarray | None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def angle_deg_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def find_dof_index(robot, dof_name: str):
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        return robot.dof_names.index(dof_name)
    return None


def build_initial_joint_positions(robot, base_positions=None) -> np.ndarray:
    if base_positions is None:
        q = np.zeros(robot.num_dof, dtype=float)
    else:
        q = np.array(base_positions, dtype=float).copy()
        if len(q) != robot.num_dof:
            q = np.zeros(robot.num_dof, dtype=float)

    missing_arm = []
    for joint_name, deg in INITIAL_ARM_JOINT_DEG.items():
        idx = find_dof_index(robot, joint_name)
        if idx is None:
            missing_arm.append(joint_name)
        else:
            q[idx] = np.deg2rad(deg)

    if missing_arm and robot.num_dof >= 6:
        fallback_values = [
            INITIAL_ARM_JOINT_DEG["joint_1"],
            INITIAL_ARM_JOINT_DEG["joint_2"],
            INITIAL_ARM_JOINT_DEG["joint_3"],
            INITIAL_ARM_JOINT_DEG["joint_4"],
            INITIAL_ARM_JOINT_DEG["joint_5"],
            INITIAL_ARM_JOINT_DEG["joint_6"],
        ]
        for i, deg in enumerate(fallback_values):
            q[i] = np.deg2rad(deg)

    for joint_name, value in INITIAL_GRIPPER_JOINTS.items():
        idx = find_dof_index(robot, joint_name)
        if idx is not None:
            q[idx] = value

    return q


def apply_robot_start_state(robot):
    robot.set_world_pose(
        position=ROBOT_BASE_WORLD,
        orientation=ROBOT_BASE_ORIENTATION,
    )
    current = robot.get_joint_positions()
    q0 = build_initial_joint_positions(robot, current)
    robot.set_joint_positions(q0)
    return q0


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


def create_usd_visual_cylinder(
    prim_path: str,
    position: np.ndarray,
    radius: float,
    height: float,
    euler_deg: np.ndarray,
    color: np.ndarray,
):
    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    cyl.CreateDisplayOpacityAttr([0.85])

    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])))
    return cyl


# ============================================================
# ROS target receiver / Camera -> World 변환
# ============================================================
class RosFuelPortTargetReceiver(Node):
    """Receives camera-frame fuel-port pose and lock signal from external perception node."""

    def __init__(self):
        super().__init__("m0609_oiling_ros_target_receiver")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.latest_pose: PoseStamped | None = None
        self.target_locked: bool = False
        self.pose_count = 0
        self.lock_count = 0
        self.pose_sub = self.create_subscription(PoseStamped, ROS_POSE_TOPIC, self.pose_callback, qos)
        self.lock_sub = self.create_subscription(Bool, ROS_LOCK_TOPIC, self.lock_callback, qos)
        self.get_logger().info("RosFuelPortTargetReceiver started")
        self.get_logger().info(f"  pose_topic = {ROS_POSE_TOPIC}")
        self.get_logger().info(f"  lock_topic = {ROS_LOCK_TOPIC}")

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg
        self.pose_count += 1

    def lock_callback(self, msg: Bool):
        self.target_locked = bool(msg.data)
        self.lock_count += 1

    def get_pose_if_ready(self) -> PoseStamped | None:
        if self.latest_pose is None:
            return None
        if REQUIRE_TARGET_LOCK and not self.target_locked:
            return None
        return self.latest_pose


def _is_camera_like_prim(prim) -> bool:
    if prim.GetTypeName() == "Camera":
        return True
    name = prim.GetName().lower()
    return any(key in name for key in ["camera", "cam", "d455", "rsd455", "realsense"])


def find_camera_prim_path() -> str | None:
    """Find the wall-mounted D455 camera prim used by the ROS image topics."""
    stage = omni.usd.get_context().get_stage()
    for path in CAMERA_PRIM_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path

    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for prim in Usd.PrimRange(world):
            if _is_camera_like_prim(prim):
                return str(prim.GetPath())
    return None


def camera_ros_point_to_usd_camera_local(point_camera_ros: np.ndarray) -> np.ndarray:
    """ROS optical frame -> USD camera local frame.

    ROS optical: +X right, +Y down, +Z forward
    USD Camera : +X right, +Y up,   -Z forward
    """
    x, y, z = [float(v) for v in point_camera_ros]
    if CAMERA_POINT_CONVENTION == "ros_optical":
        return np.array([x, -y, -z], dtype=float)
    return np.array([x, y, z], dtype=float)


def transform_camera_point_to_world(point_camera_ros: np.ndarray, camera_prim_path: str) -> np.ndarray | None:
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

    # 벽 고정 카메라용 수동 매핑
    # +Z forward -> world -Y
    # +Y down    -> world -Z
    # +X right   -> world +X
    delta_world = np.array([
        x_cam,
        -z_cam,
        -y_cam,
    ], dtype=float)

    point_world = camera_origin_world + delta_world

    print(
        f"[CAMERA TF DEBUG] origin={np.round(camera_origin_world, 3)} "
        f"p_cam={np.round(point_camera_ros, 3)} "
        f"delta={np.round(delta_world, 3)} "
        f"world={np.round(point_world, 3)}"
    )

    return point_world


def detected_world_point_to_fuel_center(detected_world_point: np.ndarray) -> np.ndarray:
    if DETECTED_POINT_IS_MOUTH_CENTER:
        return detected_world_point - normalize(PORT_OUTWARD_NORMAL) * (FUEL_PORT_DEPTH / 2.0)
    return detected_world_point


def validate_detected_fuel_center_world(center_world: np.ndarray) -> tuple[bool, str]:
    """Wall camera용 검출 좌표 gate.

    search_region_random 버전처럼 로봇이 이동하며 탐색하지 않으므로,
    여기서는 EXPECTED_TARGET_CENTER 주변의 넓은 박스 안에 들어오는지만 확인한다.
    """
    if center_world is None or not np.all(np.isfinite(center_world)):
        return False, "non-finite target"

    delta = center_world - EXPECTED_TARGET_CENTER
    abs_delta = np.abs(delta)
    if np.any(abs_delta > TARGET_GATE_HALF_EXTENT):
        return (
            False,
            "outside target gate: "
            f"center={np.round(center_world, 3)}, "
            f"delta={np.round(delta, 3)}, "
            f"half_extent={np.round(TARGET_GATE_HALF_EXTENT, 3)}",
        )
    return True, f"inside target gate: delta={np.round(delta, 3)}"


# ============================================================
# Fuel sequence
# ============================================================
@dataclass
class FuelStage:
    name: str
    target_position: np.ndarray | None
    hold_steps: int = 0
    tolerance: float = POSITION_TOLERANCE
    max_steps: int = MAX_STEPS_PER_STAGE
    speed: float = DEFAULT_TARGET_SPEED
    use_orientation: bool = False


class FuelPortSequence:
    """oiling 노즐 오프셋을 반영한 벽면 실린더 주유 waypoint state machine."""

    def __init__(self, fuel_port_center: np.ndarray | None = None):
        self.fuel_port_center = np.array(FUEL_PORT_CENTER if fuel_port_center is None else fuel_port_center, dtype=float)
        self.port_outward_normal = normalize(PORT_OUTWARD_NORMAL)
        self.insertion_direction = normalize(INSERTION_DIRECTION)

        tip_approach_far = self.fuel_port_center + self.port_outward_normal * FAR_DISTANCE
        tip_approach_mid = self.fuel_port_center + self.port_outward_normal * MID_DISTANCE
        tip_approach_near = self.fuel_port_center + self.port_outward_normal * NEAR_DISTANCE
        tip_insert_target = self.fuel_port_center + self.insertion_direction * INSERT_DISTANCE

        control_offset = (
            self.port_outward_normal * VIRTUAL_NOZZLE_LENGTH
            + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET], dtype=float)
        )
        approach_far = tip_approach_far + control_offset
        approach_mid = tip_approach_mid + control_offset
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
            FuelStage("01_axis_far_start", approach_far, tolerance=0.030, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            FuelStage("02_axis_mid", approach_mid, tolerance=0.028, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            FuelStage("03_axis_near_stop", approach_near, hold_steps=80, tolerance=0.022, speed=NEAR_TARGET_SPEED, use_orientation=True),
            FuelStage("04_insert_into_cylinder", insert_target, hold_steps=180, tolerance=0.018, speed=INSERT_TARGET_SPEED, use_orientation=True),
            FuelStage("05_retreat_near", approach_near, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("06_retreat_mid", approach_mid, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("07_retreat_far", approach_far, hold_steps=30, tolerance=0.035, speed=RETREAT_TARGET_SPEED, use_orientation=True),
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

    def update(self, ee_position: np.ndarray, ee_orientation: np.ndarray | None = None) -> bool:
        if self.done:
            return True

        stage = self.current
        self.stage_step += 1

        if stage.target_position is None:
            reached = False
            err = None
        else:
            err = np.linalg.norm(stage.target_position - ee_position)
            reached = err < stage.tolerance

        timed_out = self.stage_step >= stage.max_steps
        if reached:
            self.hold_count += 1
        else:
            self.hold_count = 0

        stage_log = {
            "01_axis_far_start": "접근 시작",
            "02_axis_mid": "중간 지점 도착",
            "03_axis_near_stop": "근처 도착 대기",
            "04_insert_into_cylinder": "삽입 완료",
            "05_retreat_near": "후퇴 시작",
            "06_retreat_mid": "중간 후퇴",
            "07_retreat_far": "먼 후퇴 완료",
            "08_return_home": "홈 복귀 완료",
        }

        if (reached and self.hold_count >= stage.hold_steps) or timed_out:
            if timed_out and not reached:
                print(f"\n[⚠️ 타임아웃] {stage_log.get(stage.name, stage.name)}")
                print(f"  위치 = {np.round(ee_position, 3)}")
                if err is not None:
                    print(f"  목표까지 = {err:.3f}m 남음\n")
            else:
                print(f"\n[✅ {stage_log.get(stage.name, stage.name)}]")
                print(f"  위치 = {np.round(ee_position, 3)}")
                if ee_orientation is not None:
                    print(f"  각도 = {np.round(ee_orientation, 3)}\n")

            self.index += 1
            self.stage_step = 0
            self.hold_count = 0
            self.command_target = None
            if self.index >= len(self.stages):
                self.done = True
                return True
        return self.done

    def get_command_target(self, ee_position: np.ndarray) -> np.ndarray | None:
        if self.done:
            return None
        stage = self.current
        if stage.target_position is None:
            return None
        if self.command_target is None:
            self.command_target = np.array(ee_position, dtype=float)

        delta = stage.target_position - self.command_target
        dist = np.linalg.norm(delta)
        max_step = max(stage.speed * PHYSICS_DT, 1e-5)
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
            f"[stage={self.index}:{stage.name}] "
            f"fuel_center={np.round(self.fuel_port_center, 3)} "
            f"link6_target={np.round(stage.target_position, 3)} "
            f"tip_target={tip_str} "
            f"cmd={np.round(cmd, 3)} "
            f"ee={np.round(ee_position, 3)} "
            f"err={err:.4f} "
            f"speed={stage.speed:.3f} "
            f"ori={'ON' if stage.use_orientation else 'OFF'} "
            f"hold={self.hold_count}/{stage.hold_steps}"
        )


# ============================================================
# Task
# ============================================================
class M0609OilingWallCameraTask(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self.sequence = FuelPortSequence(FUEL_PORT_CENTER)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_fuel_port_scene(scene)
        print("\n  [완료] oiling + 벽 고정 카메라 기반 자동 주유 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] nozzle-tip 포함 USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}")

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

    def _create_fuel_port_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 초록색 벽면 실린더 주유구 + 기준 waypoint 생성")
        print("=" * 60)

        create_usd_visual_cylinder(
            prim_path="/World/fuel_port_cylinder",
            position=FUEL_PORT_CENTER,
            radius=FUEL_PORT_RADIUS,
            height=FUEL_PORT_DEPTH,
            euler_deg=FUEL_PORT_EULER_DEG,
            color=np.array([0.0, 1.0, 0.0]),
        )
        print(f"  [OK] sim/reference fuel_port_cylinder center @ {FUEL_PORT_CENTER}")
        print(f"  [OK] cylinder size = diameter {FUEL_PORT_DIAMETER}, depth {FUEL_PORT_DEPTH}")
        print(f"  [INFO] fuel_port_euler_deg = {FUEL_PORT_EULER_DEG}")
        print(f"  [INFO] port_outward_normal = {np.round(self.sequence.port_outward_normal, 4)}")
        print(f"  [INFO] insertion_direction = {np.round(self.sequence.insertion_direction, 4)}")
        print(f"  [INFO] virtual_nozzle_length = {VIRTUAL_NOZZLE_LENGTH} m")
        print(f"  [INFO] virtual_nozzle_z_offset = {VIRTUAL_NOZZLE_Z_OFFSET} m")
        print("  [NOTE] 실제 RUN_SEQUENCE는 ROS lock 좌표로 새 waypoint를 생성함")

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
        print(f"  [OK] mouth_center marker @ {np.round(mouth_center, 4)}")

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

            scene.add(
                VisualCuboid(
                    prim_path=f"/World/fuel_wp_{i:02d}_{stage.name}",
                    name=f"fuel_wp_{i:02d}_{stage.name}",
                    position=stage.target_position,
                    scale=scale,
                    color=color,
                )
            )
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
                "center": self.sequence.fuel_port_center,
                "outward_normal": self.sequence.port_outward_normal,
                "insertion_direction": self.sequence.insertion_direction,
            },
        }

    def post_reset(self):
        self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        self.sequence.reset()


# ============================================================
# 메인
# ============================================================
def main():
    my_world = World(stage_units_in_meters=1.0)
    task = M0609OilingWallCameraTask(name="m0609_oiling_wall_camera_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    q0 = initialize_robot(robot, my_world)

    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-1] 초기 상태")
    print("=" * 60)
    print(f"  robot_base_world     = {ROBOT_BASE_WORLD}")
    print(f"  robot_base_euler_deg = {ROBOT_BASE_EULER_DEG}")
    print(f"  dof_names            = {robot.dof_names}")
    print(f"  initial_q(rad)       = {np.round(q0, 4)}")
    print(f"  initial_q(deg arm)   = {INITIAL_ARM_JOINT_DEG}")

    print("\n" + "=" * 60)
    print("[C-2] RMPFlowController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  EE frame    = {EE_LINK_NAME}")
    print(f"  orientation = {'ON' if USE_TARGET_ORIENTATION else 'OFF'}")
    print("  target_ori   = reset 직후 EE orientation을 사용해 전진 중 자세를 유지")

    controller = RMPFlowController(
        name="m0609_oiling_wall_camera_rmpflow_controller",
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] RMPFlowController 생성 완료")

    if USE_ROS_DETECTED_TARGET and not rclpy.ok():
        rclpy.init(args=None)
    ros_receiver = RosFuelPortTargetReceiver() if USE_ROS_DETECTED_TARGET else None
    camera_prim_path = find_camera_prim_path()
    print(f"  [ROS] camera_prim_path = {camera_prim_path}")
    print(f"  [ROS] camera_transform_mode = {CAMERA_TRANSFORM_MODE}")

    ee_pos, ee_ori = robot.end_effector.get_world_pose()
    locked_target_orientation = ee_ori.copy()
    print(f"\n  EE 초기 위치          = {np.round(ee_pos, 4)}")
    print(f"  locked EE orientation = {np.round(locked_target_orientation, 4)}")
    print(f"  Reference fuel cylinder center = {FUEL_PORT_CENTER}")
    print("\n[oiling + wall-camera target lock 시작 - Play 버튼을 누르면 WAIT_LOCK부터 시작]\n")

    was_playing = False
    task_done = False
    step_count = 0

    # 벽 카메라는 처음부터 표적을 보고 있으므로 SEARCH_MOVE를 제거한다.
    # WAIT_LOCK: ROS 좌표 안정화 대기
    # RUN_SEQUENCE: 안정화된 target으로 주유 sequence 실행
    run_state = "WAIT_LOCK"
    detected_fuel_center_world = None
    target_world_samples = []
    last_sampled_pose_count = -1
    wait_lock_steps = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        if USE_ROS_DETECTED_TARGET and rclpy.ok() and ros_receiver is not None:
            rclpy.spin_once(ros_receiver, timeout_sec=0.0)

        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            _, locked_target_orientation = robot.end_effector.get_world_pose()
            locked_target_orientation = locked_target_orientation.copy()

            task.sequence = FuelPortSequence(FUEL_PORT_CENTER)
            task_done = False
            step_count = 0
            run_state = "WAIT_LOCK" if USE_ROS_DETECTED_TARGET else "RUN_SEQUENCE"
            detected_fuel_center_world = None
            target_world_samples = []
            last_sampled_pose_count = -1
            wait_lock_steps = 0
            print("\n[RESET] 벽 고정 카메라 target lock 대기 → 주유 sequence 준비")
            print(f"[RESET] locked EE orientation = {np.round(locked_target_orientation, 4)}")
            print(f"[RESET] reference_target_for_sim = {np.round(FUEL_PORT_CENTER, 4)}")
            print("[RESET] SEARCH_MOVE 없음: lock 좌표 안정화 후 바로 주유 sequence 시작\n")

        if is_playing and not task_done:
            obs = task.get_observations()
            ee_position = obs["m0609_robot"]["ee_position"]
            ee_orientation = obs["m0609_robot"]["ee_orientation"]

            # ------------------------------------------------------------
            # STATE 1: 벽 카메라 detector lock 대기
            # ------------------------------------------------------------
            if run_state == "WAIT_LOCK":
                wait_lock_steps += 1
                pose_msg = ros_receiver.get_pose_if_ready() if (USE_ROS_DETECTED_TARGET and ros_receiver is not None) else None

                if USE_ROS_DETECTED_TARGET and ros_receiver is not None and not ros_receiver.target_locked:
                    target_world_samples = []

                if pose_msg is not None and ros_receiver.pose_count != last_sampled_pose_count:
                    last_sampled_pose_count = ros_receiver.pose_count

                    if camera_prim_path is None:
                        print("[ERROR] Camera prim path를 찾지 못해서 Camera->World 변환 불가")
                        if USE_FIXED_TARGET_FALLBACK:
                            detected_fuel_center_world = FUEL_PORT_CENTER.copy()
                            task.sequence = FuelPortSequence(detected_fuel_center_world)
                            run_state = "RUN_SEQUENCE"
                            print("[FALLBACK] 고정 FUEL_PORT_CENTER로 주유 sequence를 시작함")
                        else:
                            task_done = True
                            my_world.pause()
                        continue

                    p_cam = np.array([
                        pose_msg.pose.position.x,
                        pose_msg.pose.position.y,
                        pose_msg.pose.position.z,
                    ], dtype=float)
                    detected_world_point = transform_camera_point_to_world(p_cam, camera_prim_path)
                    if detected_world_point is None:
                        print("[ERROR] Camera point transform 실패")
                        task_done = True
                        my_world.pause()
                        continue

                    candidate_center_world = detected_world_point_to_fuel_center(detected_world_point)
                    if LOCK_Z_TO_EXPECTED_CENTER:
                        candidate_center_world[2] = EXPECTED_TARGET_CENTER[2]

                    valid, reason = validate_detected_fuel_center_world(candidate_center_world)
                    if valid:
                        target_world_samples.append(candidate_center_world)
                        if len(target_world_samples) > CONTROLLER_REQUIRED_LOCK_SAMPLES:
                            target_world_samples = target_world_samples[-CONTROLLER_REQUIRED_LOCK_SAMPLES:]
                    else:
                        target_world_samples = []

                    if step_count % PRINT_EVERY_N_STEPS == 0:
                        print(
                            f"[WAIT_LOCK sample] valid={valid} {reason} "
                            f"p_cam={np.round(p_cam, 3)} "
                            f"world_point={np.round(detected_world_point, 3)} "
                            f"center_world={np.round(candidate_center_world, 3)} "
                            f"samples={len(target_world_samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}"
                        )

                if len(target_world_samples) >= CONTROLLER_REQUIRED_LOCK_SAMPLES:
                    samples = np.array(target_world_samples, dtype=float)
                    mean_center = np.mean(samples, axis=0)
                    std_norm = float(np.linalg.norm(np.std(samples, axis=0)))
                    if LOCK_Z_TO_EXPECTED_CENTER:
                        mean_center[2] = EXPECTED_TARGET_CENTER[2]
                    if std_norm <= CONTROLLER_WORLD_STD_TOLERANCE:
                        detected_fuel_center_world = mean_center
                        task.sequence = FuelPortSequence(detected_fuel_center_world)
                        run_state = "RUN_SEQUENCE"
                        print("\n[ROS TARGET LOCKED - CONTROLLER STABLE]")
                        print(f"  fuel_center_mean = {np.round(detected_fuel_center_world, 4)}")
                        print(f"  sample_std_norm  = {std_norm:.4f} m")
                        print("  -> oiling 노즐 오프셋 기반 주유 waypoint sequence 시작\n")
                    elif step_count % PRINT_EVERY_N_STEPS == 0:
                        print(f"[WAIT_LOCK] samples exist but unstable: std_norm={std_norm:.4f} m")

                if run_state == "WAIT_LOCK" and step_count % PRINT_EVERY_N_STEPS == 0:
                    lock_state = ros_receiver.target_locked if ros_receiver is not None else None
                    pose_count = ros_receiver.pose_count if ros_receiver is not None else 0
                    print(
                        f"[WAIT_LOCK] detector_lock={lock_state} "
                        f"pose_count={pose_count} "
                        f"samples={len(target_world_samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES} "
                        f"wait_steps={wait_lock_steps}/{WAIT_LOCK_TIMEOUT_STEPS}"
                    )

                if run_state == "WAIT_LOCK" and wait_lock_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                    if USE_FIXED_TARGET_FALLBACK:
                        detected_fuel_center_world = FUEL_PORT_CENTER.copy()
                        task.sequence = FuelPortSequence(detected_fuel_center_world)
                        run_state = "RUN_SEQUENCE"
                        print("\n[WAIT_LOCK TIMEOUT] lock 실패. FALLBACK 고정 좌표로 sequence 시작.\n")
                    else:
                        print("\n[WAIT_LOCK TIMEOUT] target lock 실패. 로봇은 이동하지 않고 lock 대기를 다시 시작함.\n")
                        wait_lock_steps = 0
                        target_world_samples = []
                        last_sampled_pose_count = -1

                step_count += 1
                was_playing = is_playing
                continue

            # ------------------------------------------------------------
            # STATE 2: lock된 target으로 주유 sequence 실행
            # ------------------------------------------------------------
            stage = task.sequence.current

            if stage.name == "08_return_home":
                current_joints = robot.get_joint_positions()
                target_joints = build_initial_joint_positions(robot, current_joints)
                next_joints = current_joints + HOME_JOINT_SPEED_ALPHA * (target_joints - current_joints)
                robot.set_joint_positions(next_joints)

                joint_err = np.linalg.norm(next_joints[:6] - target_joints[:6])
                if joint_err < HOME_JOINT_TOLERANCE:
                    task.sequence.hold_count += 1
                else:
                    task.sequence.hold_count = 0

                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"[stage={task.sequence.index}:{stage.name}] "
                        f"joint_err={joint_err:.4f} "
                        f"hold={task.sequence.hold_count}/{stage.hold_steps}"
                    )

                if task.sequence.hold_count >= stage.hold_steps:
                    print("\n[STAGE END] 08_return_home -> home reached")
                    task.sequence.done = True
                    print("\n[완료] oiling 자동 주유 sequence 종료")
                    task_done = True
                    my_world.pause()

                step_count += 1
                was_playing = is_playing
                continue

            task_done = task.sequence.update(ee_position, ee_orientation)
            if task_done:
                print("\n[완료] oiling 자동 주유 waypoint sequence 종료")
                my_world.pause()
                was_playing = is_playing
                continue

            stage = task.sequence.current
            command_target = task.sequence.get_command_target(ee_position)
            if command_target is not None:
                if USE_TARGET_ORIENTATION and stage.use_orientation:
                    actions = controller.forward(
                        target_end_effector_position=command_target,
                        target_end_effector_orientation=locked_target_orientation,
                    )
                else:
                    actions = controller.forward(target_end_effector_position=command_target)
                robot.apply_action(actions)

            step_count += 1
            if step_count % PRINT_EVERY_N_STEPS == 0:
                link_angle = obs["m0609_robot"].get("link5_to_link6_angle_deg")
                link_angle_str = "None" if link_angle is None else f"{link_angle:.2f}deg"
                print(task.sequence.debug_string(ee_position) + f" link5_to_link6_vs_insert={link_angle_str}")

        was_playing = is_playing

    if USE_ROS_DETECTED_TARGET and ros_receiver is not None and rclpy.ok():
        ros_receiver.destroy_node()
        rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
