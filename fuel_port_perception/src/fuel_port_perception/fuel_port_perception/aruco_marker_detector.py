#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_marker_detector.py

M0609 multi-robot oiling project - ArUco based perception node.

[이 노드가 하는 일]
카메라로 ArUco 마커를 찾고, 마커의 위치/자세를 기반으로
주유구(fuel_door), 주유캡(fuel_cap), 주유홀(fuel_port_hole) 의 위치를 계산해서 발행한다.

[ArUco 마커란?]
QR코드처럼 생긴 정사각형 패턴으로, 카메라로 인식하면 3D 위치와 방향을 알 수 있다.
마커의 실제 크기(marker_size_m)를 알면 카메라와의 거리도 계산할 수 있다.

[동작 모드]
- mode_switch가 "cap"  → 마커 기준 주유캡 위치를 발행
- mode_switch가 "hole" → 마커 기준 주유홀(삽입 입구) 위치를 발행
- mode_switch가 "door" → 마커 기준 주유도어 위치를 발행

Input(받는 토픽):
  /rgb                          sensor_msgs/msg/Image      (카메라 RGB 이미지)
  /camera_info                  sensor_msgs/msg/CameraInfo (카메라 내부 파라미터)
  /aruco_detector/mode_switch   std_msgs/msg/String        ("door" / "cap" / "hole")

Output(보내는 토픽):
  /aruco_detector/pose          geometry_msgs/msg/PoseStamped  (카메라 좌표계 기준 목표 위치)
  /aruco_detector/target_locked std_msgs/msg/Bool              (안정적으로 감지됐는지 여부)
  /aruco_detector/current_mode  std_msgs/msg/String            (현재 모드)
  /aruco_detector/debug_image   sensor_msgs/msg/Image          (시각화용 디버그 이미지)

주의
- depth 이미지 없이, ArUco 마커 크기와 camera_info만으로 거리를 추정한다.
- 위치는 카메라 광학 좌표계(x: 오른쪽, y: 아래, z: 앞쪽)로 발행한다.
"""
from __future__ import annotations

# deque: 고정 크기의 큐(버퍼). 오래된 값이 자동으로 밀려난다.
from collections import deque
# 타입 힌트를 위한 모듈 (코드 가독성 향상용)
from typing import Optional, Tuple, Dict

import cv2          # OpenCV: 이미지 처리 및 ArUco 감지
import numpy as np  # NumPy: 행렬/벡터 연산

import rclpy                          # ROS2 파이썬 클라이언트 라이브러리
from rclpy.node import Node           # ROS2 노드 기본 클래스
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# QoS: 토픽의 통신 품질 설정 (신뢰성, 버퍼 크기 등)

from sensor_msgs.msg import Image, CameraInfo          # 카메라 관련 메시지 타입
from geometry_msgs.msg import PoseStamped, Vector3Stamped  # 위치/방향 메시지 타입
from std_msgs.msg import Bool, String                  # 기본 데이터 메시지 타입
from cv_bridge import CvBridge  # ROS 이미지 메시지 ↔ OpenCV 이미지 변환 도구


# 허용되는 모드 목록. 이 외의 값이 들어오면 무시한다.
VALID_MODES = ("door", "cap", "hole")


def _make_aruco_dictionary(name: str):
    """
    문자열로 된 ArUco 딕셔너리 이름을 OpenCV 딕셔너리 객체로 변환한다.

    예) "4X4_50" → cv2.aruco.DICT_4X4_50
        "DICT_4X4_50" 형태로 입력해도 동작한다.

    ArUco 딕셔너리란?
    - 마커가 어떤 패턴 집합에서 왔는지 정의한다.
    - 4X4_50이면 4×4 픽셀 패턴, 총 50가지 마커 ID가 존재한다.
    """
    name = str(name).strip().upper()
    # "DICT_"로 시작하지 않으면 앞에 붙여준다.
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    # cv2.aruco에 해당 이름의 속성이 없으면 잘못된 딕셔너리 이름이다.
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_detector_parameters():
    """
    ArUco 마커 감지에 사용할 파라미터 객체를 생성한다.

    왜 두 가지 방식을 처리하는가?
    - OpenCV 4.7 이상: DetectorParameters() 생성자 사용
    - 구버전 OpenCV:   DetectorParameters_create() 함수 사용
    설치된 OpenCV 버전에 따라 존재하는 함수가 다르므로 둘 다 시도한다.
    """
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise AttributeError(
        "cv2.aruco has neither DetectorParameters nor DetectorParameters_create. "
        "Install opencv-contrib-python or check the OpenCV build."
    )


def _make_aruco_detector(dictionary, detector_params):
    """
    ArUco 감지기 객체를 생성한다.

    - OpenCV 4.7+: ArucoDetector 클래스를 사용 (더 유연하고 빠름)
    - 구버전:      None을 반환하면 detectMarkers() 함수(legacy)를 직접 사용한다.
    """
    if hasattr(cv2.aruco, "ArucoDetector"):
        try:
            return cv2.aruco.ArucoDetector(dictionary, detector_params)
        except Exception:
            return None  # 생성 실패 시 legacy 방식으로 대체
    return None


class ArucoMarkerDetectorNode(Node):
    """
    ArUco 마커를 감지하고 목표 위치를 발행하는 ROS2 노드.

    [동작 흐름 요약]
    1. 카메라에서 RGB 이미지를 받는다.
    2. 이미지에서 ArUco 마커를 찾는다.
    3. 마커의 3D 위치/방향(pose)을 추정한다.
    4. 마커 좌표계에서 목표물(cap/hole/door)까지의 오프셋을 더해 카메라 좌표계 위치를 구한다.
    5. 여러 프레임에 걸쳐 안정된 값을 확인(stable_count)한 뒤 발행한다.
    """
    def __init__(self) -> None:
        super().__init__("aruco_marker_detector_node")

        # =====================================================================
        # ROS 파라미터 선언
        # =====================================================================
        # 파라미터란? 노드 실행 시 외부에서 값을 바꿀 수 있는 설정값이다.
        # ex) ros2 run ... --ros-args -p marker_id:=2

        # --- 토픽 이름 파라미터 ---
        self.declare_parameter("rgb_topic", "/rgb")                          # RGB 이미지 토픽
        self.declare_parameter("camera_info_topic", "/camera_info")          # 카메라 정보 토픽
        self.declare_parameter("mode_switch_topic", "/aruco_detector/mode_switch")  # 모드 전환 토픽
        self.declare_parameter("pose_topic", "/aruco_detector/pose")         # 목표 위치 발행 토픽
        self.declare_parameter("target_locked_topic", "/aruco_detector/target_locked")  # 잠금 상태 발행
        self.declare_parameter("current_mode_topic", "/aruco_detector/current_mode")    # 현재 모드 발행
        self.declare_parameter("debug_image_topic", "/aruco_detector/debug_image")      # 디버그 이미지 발행
        self.declare_parameter("direction_topic", "/aruco_detector/direction")          # 삽입 방향 벡터 발행
        self.declare_parameter("initial_mode", "cap")                        # 노드 시작 시 초기 모드

        # --- ArUco 마커 파라미터 ---
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")  # 마커 종류 (딕셔너리)
        self.declare_parameter("marker_id", 0)                     # 감지할 마커의 ID 번호
        # marker_size_m: 마커 한 변의 실제 길이(미터). 거리 계산에 필수다.
        # create_aruco_marker_grid_in_isaac.py의 MARKER_PATTERN_SIZE_M와 반드시 같게 맞춘다.
        self.declare_parameter("marker_size_m", 0.15)

        # --- 마커 기준 각 목표물까지의 오프셋(단위: 미터) ---
        # OpenCV ArUco 좌표계: x=마커 오른쪽, y=마커 아래, z=마커 법선(카메라 방향)
        # 아래 기본값은 visual_test.usda 시뮬레이션 기준으로 계산한 실측값이다.
        self.declare_parameter("marker_to_door_xyz", [0.240267, -1.110197, -1.279059])
        self.declare_parameter("marker_to_cap_xyz",  [0.238197, -1.090520, -1.302334])
        self.declare_parameter("marker_to_hole_xyz", [0.244428, -1.054349, -1.510171])

        # --- 노즐 삽입 방향 벡터 (마커 좌표계 기준) ---
        # [0, 0, -1]: 마커 법선의 반대 방향 = 차량 내부 방향 (노즐을 꽂는 방향)
        self.declare_parameter("direction_axis_marker_xyz", [0.0, 0.0, -1.0])

        # --- 안정화 파라미터 ---
        # 너무 흔들리는 값을 걸러내기 위해, 여러 프레임에서 안정적인 경우에만 발행한다.
        self.declare_parameter("required_stable_frames", 3)      # 안정 판정에 필요한 최소 프레임 수
        self.declare_parameter("stable_buffer_size", 8)          # 통계 계산에 사용할 과거 프레임 수
        self.declare_parameter("stable_std_threshold_m", 0.035)  # 허용 표준편차 임계값 (미터)
        self.declare_parameter("publish_hz", 10.0)               # 최대 발행 주파수 (Hz)
        self.declare_parameter("publish_debug", True)            # 디버그 이미지 발행 여부
        self.declare_parameter("draw_axes", True)                # 마커 축 그리기 여부

        # =====================================================================
        # 파라미터 값 읽기 및 멤버 변수 초기화
        # =====================================================================
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.mode_switch_topic = self.get_parameter("mode_switch_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.target_locked_topic = self.get_parameter("target_locked_topic").value
        self.current_mode_topic = self.get_parameter("current_mode_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.direction_topic = self.get_parameter("direction_topic").value

        # 초기 모드 유효성 검사: VALID_MODES에 없으면 기본값 "cap"으로 대체
        initial_mode = str(self.get_parameter("initial_mode").value).strip().lower()
        self.current_mode = initial_mode if initial_mode in VALID_MODES else "cap"

        self.marker_id = int(self.get_parameter("marker_id").value)
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.dictionary_name = str(self.get_parameter("aruco_dictionary").value)
        self.dictionary = _make_aruco_dictionary(self.dictionary_name)

        # 각 모드별 오프셋을 딕셔너리로 보관 (키: 모드 이름, 값: [x, y, z] numpy 배열)
        self.target_offsets: Dict[str, np.ndarray] = {
            "door": np.array(self.get_parameter("marker_to_door_xyz").value, dtype=np.float64),
            "cap":  np.array(self.get_parameter("marker_to_cap_xyz").value, dtype=np.float64),
            "hole": np.array(self.get_parameter("marker_to_hole_xyz").value, dtype=np.float64),
        }

        # 노즐 삽입 방향 벡터 읽기 및 정규화(길이를 1로 만들기)
        self.direction_axis_marker = np.array(
            self.get_parameter("direction_axis_marker_xyz").value,
            dtype=np.float64,
        ).reshape(3)
        axis_norm = float(np.linalg.norm(self.direction_axis_marker))
        if axis_norm < 1e-9:
            # 벡터 길이가 0이면 방향을 알 수 없으므로 기본값 사용
            self.get_logger().warn(
                "direction_axis_marker_xyz가 0 벡터입니다. 기본 삽입 방향 [0, 0, -1]을 사용합니다."
            )
            self.direction_axis_marker = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            # 단위 벡터로 정규화 (길이 = 1)
            self.direction_axis_marker = self.direction_axis_marker / axis_norm

        self.required_stable_frames = int(self.get_parameter("required_stable_frames").value)
        self.stable_buffer_size = int(self.get_parameter("stable_buffer_size").value)
        self.stable_std_threshold_m = float(self.get_parameter("stable_std_threshold_m").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        # 발행 주기를 나노초 단위로 변환 (1초 = 1,000,000,000 ns)
        self.publish_period_ns = int(1e9 / max(self.publish_hz, 0.1))
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.draw_axes = bool(self.get_parameter("draw_axes").value)

        # ROS 이미지 ↔ OpenCV 이미지 변환기
        self.bridge = CvBridge()
        # 카메라 정보: 처음에는 None이고, camera_info_callback에서 설정된다.
        self.camera_info: Optional[CameraInfo] = None
        # point_buffer: 최근 N개 프레임의 목표 위치를 저장하는 고정 크기 버퍼
        self.point_buffer: deque[np.ndarray] = deque(maxlen=self.stable_buffer_size)
        # 연속으로 안정 조건을 만족한 프레임 수
        self.stable_count = 0
        # 마지막으로 발행한 시각 (나노초)
        self.last_publish_time_ns = 0

        # OpenCV 버전에 맞는 감지 파라미터와 감지기 생성
        self.detector_params = _make_detector_parameters()
        self.aruco_detector = _make_aruco_detector(self.dictionary, self.detector_params)

        # =====================================================================
        # QoS(통신 품질) 설정
        # =====================================================================
        # sensor_qos: 카메라처럼 빠른 데이터에 적합. 일부 메시지가 손실돼도 괜찮다.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 손실 허용 (속도 우선)
            history=HistoryPolicy.KEEP_LAST,
            depth=5,  # 최대 5개 메시지를 버퍼에 보관
        )
        # latched_qos: 모드처럼 중요한 상태 정보에 사용. 새 구독자도 최신값을 즉시 받는다.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,           # 손실 없음 보장
            durability=DurabilityPolicy.TRANSIENT_LOCAL,      # 나중에 연결된 구독자에게도 전달
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # =====================================================================
        # Subscriber(구독자) 생성: 외부에서 데이터를 받아온다.
        # =====================================================================
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.image_callback, sensor_qos
        )
        self.mode_switch_sub = self.create_subscription(
            String, self.mode_switch_topic, self.mode_switch_callback, latched_qos
        )

        # =====================================================================
        # Publisher(발행자) 생성: 계산 결과를 외부로 내보낸다.
        # =====================================================================
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.lock_pub = self.create_publisher(Bool, self.target_locked_topic, 10)
        self.mode_pub = self.create_publisher(String, self.current_mode_topic, latched_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.direction_pub = self.create_publisher(Vector3Stamped, self.direction_topic, 10)

        # 시작 시 현재 모드를 즉시 발행
        self.publish_current_mode()

        # 시작 로그 출력
        self.get_logger().info("ArucoMarkerDetectorNode started")
        self.get_logger().info(f"  rgb_topic           = {self.rgb_topic}")
        self.get_logger().info(f"  camera_info_topic   = {self.camera_info_topic}")
        self.get_logger().info(f"  mode_switch_topic   = {self.mode_switch_topic}")
        self.get_logger().info(f"  pose_topic          = {self.pose_topic}")
        self.get_logger().info(f"  direction_topic     = {self.direction_topic}")
        self.get_logger().info(f"  dictionary          = {self.dictionary_name}")
        self.get_logger().info(f"  marker_id           = {self.marker_id}")
        self.get_logger().info(f"  marker_size_m       = {self.marker_size_m:.3f}")
        self.get_logger().info(f"  initial_mode        = {self.current_mode}")
        self.get_logger().info(f"  direction_axis_marker_xyz = {np.round(self.direction_axis_marker, 4).tolist()}")
        for mode, offset in self.target_offsets.items():
            self.get_logger().info(f"  marker_to_{mode}_xyz = {np.round(offset, 4).tolist()}")

    # =========================================================================
    # ROS 콜백 함수들: 토픽 메시지가 도착하면 자동으로 호출된다.
    # =========================================================================

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """카메라 내부 파라미터(초점거리, 주점 등)를 저장한다."""
        self.camera_info = msg

    def mode_switch_callback(self, msg: String) -> None:
        """
        동작 모드를 전환한다. ("door", "cap", "hole" 중 하나)

        모드가 바뀌면 안정화 버퍼를 초기화해 새 모드에서 다시 안정화를 시작한다.
        """
        new_mode = str(msg.data).strip().lower()
        if new_mode not in VALID_MODES:
            self.get_logger().warn(f"알 수 없는 mode_switch 값 무시: '{msg.data}'")
            return
        if new_mode == self.current_mode:
            return  # 같은 모드면 무시
        self.get_logger().info(f"mode_switch: {self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.reset_stability()       # 안정화 카운터와 버퍼 초기화
        self.publish_current_mode()  # 새 모드를 즉시 발행

    def publish_current_mode(self) -> None:
        """현재 모드를 토픽으로 발행한다."""
        self.mode_pub.publish(String(data=self.current_mode))

    def reset_stability(self) -> None:
        """안정화 버퍼와 카운터를 초기화한다. 모드 변경 또는 마커 소실 시 호출."""
        self.point_buffer.clear()
        self.stable_count = 0

    def image_callback(self, rgb_msg: Image) -> None:
        """
        메인 처리 루프: 새 이미지가 들어올 때마다 호출된다.

        처리 순서:
        1. 카메라 정보가 준비됐는지 확인
        2. ROS 이미지를 OpenCV 배열로 변환
        3. 카메라 행렬 추출
        4. 마커 감지 및 pose 추정
        5. 목표 위치 계산 및 안정성 판단
        6. 주기적으로 위치/방향/잠금 상태 발행
        7. 디버그 이미지 발행
        """
        # 카메라 정보가 아직 도착하지 않았으면 대기
        if self.camera_info is None:
            self.get_logger().warn("Waiting for /camera_info ...", throttle_duration_sec=2.0)
            return

        # ROS Image 메시지 → OpenCV numpy 배열 변환
        try:
            rgb = self.convert_rgb_image(rgb_msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        # 카메라 행렬(K)과 왜곡 계수(D) 추출
        camera_matrix, dist_coeffs = self.camera_calibration(self.camera_info)
        if camera_matrix is None:
            self.get_logger().warn("Invalid camera_info K matrix", throttle_duration_sec=2.0)
            return

        # ArUco 마커 감지 및 pose 추정
        detection = self.detect_marker_pose(rgb, camera_matrix, dist_coeffs)

        # 결과 변수 초기화
        locked = False           # 안정 잠금 여부
        target_camera = None     # 카메라 좌표계 기준 목표 위치
        direction_camera = None  # 카메라 좌표계 기준 삽입 방향
        status = "NO ARUCO MARKER"
        debug_payload = None     # 디버그 이미지에 그릴 데이터

        if detection is not None:
            corners, rvec, tvec = detection
            # --- 목표 위치 계산 ---
            # rvec: 마커의 회전 벡터 (Rodrigues 표현)
            # tvec: 마커 중심의 카메라 좌표계 위치 (단위: 미터)
            offset_marker = self.target_offsets[self.current_mode].reshape(3, 1)

            # Rodrigues 변환: 회전 벡터(rvec) → 3×3 회전 행렬(R)
            # R은 마커 좌표계 → 카메라 좌표계 변환 행렬이다.
            R_marker_to_camera, _ = cv2.Rodrigues(rvec.reshape(3, 1))

            # 목표 위치(카메라 좌표계) = R × offset(마커 좌표계) + 마커 위치
            target_camera = (R_marker_to_camera @ offset_marker + tvec.reshape(3, 1)).reshape(3)

            # 삽입 방향 벡터를 카메라 좌표계로 변환
            direction_camera = self.compute_direction_camera(R_marker_to_camera)

            # 안정성 판단: 버퍼에 쌓고 표준편차로 흔들림을 측정
            self.point_buffer.append(target_camera.astype(np.float64))
            mean, std_norm = self.filtered_point_stats()

            # 충분한 프레임이 쌓이고 흔들림이 작으면 stable_count 증가
            if len(self.point_buffer) >= self.required_stable_frames and std_norm <= self.stable_std_threshold_m:
                self.stable_count += 1
            else:
                self.stable_count = 0  # 조건 미충족 시 초기화

            # stable_count가 1 이상이면 "잠금됨(locked)" 상태
            locked = self.stable_count >= 1

            status = (
                f"id={self.marker_id} mode={self.current_mode} "
                f"target=({target_camera[0]:.3f},{target_camera[1]:.3f},{target_camera[2]:.3f}) "
                f"dir=({direction_camera[0]:.3f},{direction_camera[1]:.3f},{direction_camera[2]:.3f}) "
                f"std={std_norm:.3f} stable={self.stable_count}/{self.required_stable_frames}"
            )
            debug_payload = (corners, rvec, tvec, mean, target_camera, direction_camera)

            # 발행 주기(publish_hz) 제한: 마지막 발행 이후 충분한 시간이 지났을 때만 발행
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
                self.last_publish_time_ns = now_ns
                # 평균 위치(mean)와 마커 방향(R)으로 PoseStamped 메시지 생성 후 발행
                self.pose_pub.publish(self.make_pose(rgb_msg, mean, R_marker_to_camera))
                # 삽입 방향 벡터 발행
                self.direction_pub.publish(self.make_direction(rgb_msg, direction_camera))
                # 잠금 상태 발행
                self.lock_pub.publish(Bool(data=bool(locked)))

            self.get_logger().info(status + f" locked={locked}", throttle_duration_sec=0.5)
        else:
            # 마커가 사라지면 안정화 초기화
            self.reset_stability()

        # 마커가 없을 때도 주기에 맞춰 locked=False를 발행한다.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
            self.last_publish_time_ns = now_ns
            self.lock_pub.publish(Bool(data=bool(locked)))

        # 디버그 이미지 발행 (시각화용)
        if self.publish_debug:
            self.publish_debug_image(rgb_msg, rgb, debug_payload, status, locked, camera_matrix, dist_coeffs)

    # =========================================================================
    # 마커 감지 및 위치 계산 관련 함수들
    # =========================================================================

    def detect_marker_pose(self, rgb: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        RGB 이미지에서 지정된 ID의 ArUco 마커를 찾고 pose를 추정한다.

        반환값:
          - 마커 감지 성공: (corners, rvec, tvec)
            - corners: 마커의 4개 꼭짓점 픽셀 좌표
            - rvec: 회전 벡터 (Rodrigues, shape=(3,))
            - tvec: 이동 벡터 = 마커 중심의 카메라 좌표 (shape=(3,), 단위: 미터)
          - 마커 없음: None
        """
        # ArUco 감지는 흑백 이미지에서 더 빠르고 정확하다.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # OpenCV 버전에 따라 다른 방식으로 마커를 찾는다.
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)

        # 이미지에서 마커를 하나도 못 찾은 경우
        if ids is None or len(ids) == 0:
            return None

        # 찾은 마커들 중 원하는 marker_id와 일치하는 것을 찾는다.
        ids_flat = ids.flatten().astype(int)
        matches = np.where(ids_flat == self.marker_id)[0]
        if len(matches) == 0:
            # 다른 ID의 마커는 보이지만 원하는 ID가 없는 경우
            self.get_logger().warn(
                f"ArUco marker는 보이지만 marker_id={self.marker_id}가 아님. visible ids={ids_flat.tolist()}",
                throttle_duration_sec=1.0,
            )
            return None

        # 첫 번째 매칭 마커의 인덱스
        i = int(matches[0])
        marker_corners = [corners[i]]

        # estimatePoseSingleMarkers: 마커의 실제 크기와 카메라 파라미터로 3D pose를 추정한다.
        # rvecs[0]: 회전 벡터, tvecs[0]: 이동 벡터
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            self.marker_size_m,    # 마커의 실제 크기 (미터)
            camera_matrix,         # 카메라 내부 행렬 (초점거리, 주점)
            dist_coeffs,           # 렌즈 왜곡 계수
        )
        return corners[i], rvecs[0].reshape(3), tvecs[0].reshape(3)

    def filtered_point_stats(self) -> Tuple[np.ndarray, float]:
        """
        point_buffer에 저장된 최근 N개 위치의 평균과 표준편차 크기를 반환한다.

        표준편차가 크면 위치가 흔들리고 있다는 뜻이다.
        표준편차가 stable_std_threshold_m 이하이면 "안정적"으로 판단한다.
        """
        if not self.point_buffer:
            return np.zeros(3, dtype=np.float64), float("inf")  # 버퍼가 비어있으면 inf 반환

        # (N, 3) 형태의 배열로 변환: N개 프레임, 각 [x, y, z]
        pts = np.stack(list(self.point_buffer), axis=0)
        mean = np.mean(pts, axis=0)                           # 각 축의 평균
        std_norm = float(np.linalg.norm(np.std(pts, axis=0)))  # 표준편차 벡터의 크기
        return mean, std_norm

    @staticmethod
    def camera_calibration(info: CameraInfo):
        """
        CameraInfo 메시지에서 카메라 행렬(K)과 왜곡 계수(D)를 추출한다.

        카메라 행렬 K (3×3):
          [[fx,  0, cx],
           [ 0, fy, cy],
           [ 0,  0,  1]]
          fx, fy: 초점 거리 (픽셀 단위)
          cx, cy: 주점 (이미지 중심 좌표)

        왜곡 계수 D: 렌즈 왜곡 보정 파라미터
        """
        k = np.array(info.k, dtype=np.float64).reshape(3, 3)
        # 초점거리(fx, fy)가 0이면 유효하지 않은 카메라 정보
        if abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9:
            return None, None
        # 왜곡 계수가 없으면 0으로 채운다.
        d = np.array(info.d, dtype=np.float64).reshape(-1, 1) if info.d else np.zeros((5, 1), dtype=np.float64)
        return k, d

    def make_pose(self, image_msg: Image, point_camera: np.ndarray,
                  R_marker_to_camera: Optional[np.ndarray] = None) -> PoseStamped:
        """
        카메라 좌표계의 3D 위치와 마커 방향으로 PoseStamped 메시지를 만든다.

        PoseStamped = 헤더(시각, 좌표계 이름) + 위치(x, y, z) + 자세(quaternion)
        자세(orientation)는 마커의 회전을 quaternion으로 표현한다.
        """
        pose = PoseStamped()
        # 헤더: 메시지 시각과 좌표계 이름 설정
        pose.header.stamp = image_msg.header.stamp
        pose.header.frame_id = self.camera_info.header.frame_id or image_msg.header.frame_id or "sim_camera"

        # 위치 설정 (카메라 좌표계 기준)
        pose.pose.position.x = float(point_camera[0])
        pose.pose.position.y = float(point_camera[1])
        pose.pose.position.z = float(point_camera[2])

        # 자세(orientation) 설정: 회전 행렬 → quaternion 변환
        if R_marker_to_camera is not None:
            qx, qy, qz, qw = self.rotation_matrix_to_quaternion(R_marker_to_camera)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            pose.pose.orientation.w = float(qw)
        else:
            # 회전 정보가 없으면 항등 quaternion (회전 없음)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
        return pose

    def make_direction(self, image_msg: Image, direction_camera: np.ndarray) -> Vector3Stamped:
        """노즐 삽입 방향 벡터를 Vector3Stamped 메시지로 만든다."""
        msg = Vector3Stamped()
        msg.header.stamp = image_msg.header.stamp
        msg.header.frame_id = self.camera_info.header.frame_id or image_msg.header.frame_id or "sim_camera"
        msg.vector.x = float(direction_camera[0])
        msg.vector.y = float(direction_camera[1])
        msg.vector.z = float(direction_camera[2])
        return msg

    def compute_direction_camera(self, R_marker_to_camera: np.ndarray) -> np.ndarray:
        """
        마커 좌표계의 삽입 방향 벡터를 카메라 좌표계로 변환한다.

        direction_axis_marker: 마커 좌표계에서의 삽입 방향 (기본: [0, 0, -1])
        R_marker_to_camera: 마커 → 카메라 회전 행렬

        결과: 카메라 좌표계에서 노즐이 향해야 할 단위 방향 벡터
        """
        direction = R_marker_to_camera @ self.direction_axis_marker.reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)  # 비정상 케이스 처리
        return direction / norm  # 단위 벡터로 정규화

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
        """
        3×3 회전 행렬을 ROS quaternion (x, y, z, w)으로 변환한다.

        [Quaternion이란?]
        3D 회전을 4개의 숫자로 표현하는 방법. ROS에서 주로 사용한다.
        (x, y, z, w) 형태이며, 회전 없음 = (0, 0, 0, 1)이다.

        [왜 4가지 경우로 나누는가?]
        회전 행렬의 대각합(trace)이 음수이거나 특정 성분이 큰 경우,
        수치 안정성을 위해 가장 큰 값을 기준으로 다른 성분을 계산한다.
        """
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(R))  # 대각합 = R[0,0] + R[1,1] + R[2,2]

        if trace > 0.0:
            # 가장 일반적인 경우: trace가 양수
            s = np.sqrt(trace + 1.0) * 2.0  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            # R[0,0]이 대각 성분 중 가장 큰 경우: qx를 기준으로 계산
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            # R[1,1]이 가장 큰 경우: qy를 기준으로 계산
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            # R[2,2]가 가장 큰 경우: qz를 기준으로 계산
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        # 계산된 quaternion을 단위 벡터로 정규화 (수치 오차 보정)
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0  # 비정상 케이스: 항등 quaternion 반환
        q /= q_norm
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # =========================================================================
    # 이미지 변환 및 디버그 시각화
    # =========================================================================

    def convert_rgb_image(self, msg: Image) -> np.ndarray:
        """
        ROS Image 메시지를 RGB 형식의 OpenCV numpy 배열로 변환한다.

        카메라마다 인코딩 방식이 다를 수 있으므로(rgb8, bgr8, rgba8, bgra8),
        모두 RGB로 통일한다.
        """
        enc = msg.encoding.lower()
        if enc == "rgb8":
            # 이미 RGB 형식이므로 그대로 변환
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if enc == "bgr8":
            # BGR → RGB (파란색↔빨간색 순서 교환)
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if enc == "rgba8":
            # RGBA → RGB (알파 채널 제거)
            rgba = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgba8")
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        if enc == "bgra8":
            # BGRA → RGB
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgra8")
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        # 그 외 형식은 rgb8으로 강제 변환 시도
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray, payload, text: str, locked: bool,
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """
        디버그용 이미지를 생성하고 발행한다.

        이미지에 다음을 그린다:
        - 감지된 마커의 외곽선
        - 마커 좌표축 (X/Y/Z 방향)
        - 목표 위치 (흰 원)
        - 노즐 삽입 방향 화살표
        - 상태 텍스트 (잠금 여부, 모드 등)
        """
        debug = rgb.copy()  # 원본 이미지를 복사해서 위에 그린다.
        # 잠금됐으면 초록색, 아니면 주황색으로 표시
        color = (0, 255, 0) if locked else (255, 180, 0)

        if payload is not None:
            corners, rvec, tvec, mean, target_camera, direction_camera = payload

            # 마커 외곽선 그리기
            cv2.aruco.drawDetectedMarkers(debug, [corners], np.array([[self.marker_id]], dtype=np.int32), color)

            # 마커의 3D 좌표축 그리기 (X=빨강, Y=초록, Z=파랑)
            if self.draw_axes:
                try:
                    cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs,
                                      rvec.reshape(3, 1), tvec.reshape(3, 1),
                                      self.marker_size_m * 0.5)
                except Exception:
                    pass

            # 목표 위치를 2D 이미지로 투영하여 원으로 표시
            try:
                # projectPoints: 3D 점 → 2D 이미지 픽셀 좌표 변환
                # 카메라 좌표계 점이므로 회전/이동 없이 투영만 한다.
                projected, _ = cv2.projectPoints(
                    mean.reshape(1, 1, 3).astype(np.float64),
                    np.zeros((3, 1), dtype=np.float64),  # 회전 없음
                    np.zeros((3, 1), dtype=np.float64),  # 이동 없음
                    camera_matrix,
                    dist_coeffs,
                )
                u, v = projected.reshape(2)  # 픽셀 좌표 (가로, 세로)
                # 흰 채움 원 + 색상 테두리 원으로 목표 위치 표시
                cv2.circle(debug, (int(round(u)), int(round(v))), 7, (255, 255, 255), -1)
                cv2.circle(debug, (int(round(u)), int(round(v))), 9, color, 2)

                # 삽입 방향 화살표: 목표 위치에서 방향으로 뻗는 화살표를 그린다.
                arrow_len_m = max(self.marker_size_m * 0.7, 0.05)  # 화살표 길이 (미터)
                p0 = mean.reshape(1, 1, 3).astype(np.float64)  # 화살표 시작점
                p1 = (mean + direction_camera * arrow_len_m).reshape(1, 1, 3).astype(np.float64)  # 화살표 끝점

                # 두 3D 점을 2D 픽셀 좌표로 투영
                proj0, _ = cv2.projectPoints(p0,
                    np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64),
                    camera_matrix, dist_coeffs)
                proj1, _ = cv2.projectPoints(p1,
                    np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64),
                    camera_matrix, dist_coeffs)
                u0, v0 = proj0.reshape(2)
                u1, v1 = proj1.reshape(2)

                # 화살표 선 그리기
                cv2.arrowedLine(
                    debug,
                    (int(round(u0)), int(round(v0))),  # 시작점
                    (int(round(u1)), int(round(v1))),  # 끝점
                    color, 3, cv2.LINE_AA, tipLength=0.25,
                )
            except Exception:
                pass

            # 화면 상단에 상태 텍스트 출력
            cv2.putText(debug, f"ARUCO LOCK={locked} mode={self.current_mode}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug, text[:110], (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # 마커가 없을 때는 "NO ARUCO MARKER" 텍스트만 표시
            cv2.putText(debug, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # OpenCV 배열 → ROS Image 메시지로 변환 후 발행
        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)


def main(args=None) -> None:
    """
    노드 진입점(entry point).

    rclpy.init()으로 ROS2를 초기화하고, 노드를 생성한 뒤
    rclpy.spin()으로 콜백 루프를 시작한다.
    Ctrl+C(KeyboardInterrupt)를 누르면 안전하게 종료된다.
    """
    rclpy.init(args=args)
    node = ArucoMarkerDetectorNode()
    try:
        rclpy.spin(node)  # 종료 신호가 올 때까지 콜백을 계속 처리한다.
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # 노드 자원 해제
        rclpy.shutdown()     # ROS2 종료


if __name__ == "__main__":
    main()
