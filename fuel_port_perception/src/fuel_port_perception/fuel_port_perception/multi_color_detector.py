#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_color_detector.py

M0609 multi-robot oiling project - perception node.

기존 fuel_port_detector_node.py(초록 단일 감지)를 기반으로,
외부 모드 전환 명령(yellow/blue/green)에 따라 한 번에 한 색만 감지하도록 확장한 버전.

Input:
  /rgb                          sensor_msgs/msg/Image      (rgb8)
  /depth                        sensor_msgs/msg/Image      (32FC1, meter)
  /camera_info                  sensor_msgs/msg/CameraInfo
  /color_detector/mode_switch   std_msgs/msg/String        ("yellow" / "blue" / "green")

Output:
  /color_detector/pose          geometry_msgs/msg/PoseStamped  (현재 모드 색상의 3D 위치)
  /color_detector/target_locked std_msgs/msg/Bool
  /color_detector/current_mode  std_msgs/msg/String
  /color_detector/debug_image   sensor_msgs/msg/Image

감지 순서와 역할 (모드 전환은 외부 상태머신이 명령한다):
  yellow -> fuel_door (커버) 위치
  blue   -> fuel_cap (마개) 위치
  green  -> fuel_port_hole (주유구 입구) 위치

모드가 바뀌면 이전 색의 표적 샘플을 즉시 버린다(reset_stability).
"""

# 이 한 줄을 넣으면 파이썬이 타입힌트(예: list[int])를 실제로 실행하지 않고
# 그냥 "글자"로만 취급한다. 그래서 옛 파이썬 버전에서도 에러 없이 동작한다.
from __future__ import annotations

# deque: 앞/뒤에서 넣고 빼는게 빠른 리스트. "최근 N개만 유지"하는 버퍼로 쓴다.
from collections import deque
# Optional[X] = "X이거나 None일 수 있다"는 표시. Tuple = 튜플 타입 표시.
from typing import Optional, Tuple

import cv2          # 영상처리 라이브러리 (색 검출, 윤곽선 찾기 등)
import numpy as np  # 배열/행렬 연산 라이브러리

import rclpy
from rclpy.node import Node
# QoS = 메시지를 주고받을 때의 "약속" 설정. 신뢰성/이력 보존 방식 등을 정한다.
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# 카메라 RGB/Depth 이미지, 카메라 내부 파라미터(초점거리 등) 메시지 타입
from sensor_msgs.msg import Image, CameraInfo
# 위치+방향(3D pose) 메시지 타입
from geometry_msgs.msg import PoseStamped
# 단순 참/거짓, 문자열 메시지 타입
from std_msgs.msg import Bool, String

# ROS 이미지 메시지 <-> OpenCV(numpy) 이미지를 서로 변환해주는 도구
from cv_bridge import CvBridge
# 서로 다른 두 토픽(rgb, depth)의 타임스탬프를 맞춰서 "같은 순간"끼리 콜백을 묶어주는 도구
import message_filters


# 디텍터가 받아들일 수 있는 모드(색) 목록. 이 셋 말고 다른 값이 오면 무시한다.
VALID_MODES = ("yellow", "blue", "green")


class MultiColorDetectorNode(Node):
    def __init__(self) -> None:
        # 부모 클래스(Node)를 초기화하면서 이 노드의 ROS 이름을 정해준다.
        super().__init__("multi_color_detector_node")

        # -----------------------------
        # Topic parameters
        # 토픽 이름들을 "하드코딩"하지 않고 파라미터로 빼두면, 나중에 launch 파일이나
        # 커맨드라인에서 토픽 이름을 바꿔 쓸 수 있다 (코드 수정 없이).
        # -----------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("mode_switch_topic", "/color_detector/mode_switch")
        self.declare_parameter("pose_topic", "/color_detector/pose")
        self.declare_parameter("target_locked_topic", "/color_detector/target_locked")
        self.declare_parameter("current_mode_topic", "/color_detector/current_mode")
        self.declare_parameter("debug_image_topic", "/color_detector/debug_image")
        self.declare_parameter("initial_mode", "yellow")  # 처음 시작할 때 어떤 색을 볼지

        # -----------------------------
        # Detection / filtering parameters (fuel_port_detector.py와 동일)
        # -----------------------------
        self.declare_parameter("min_area", 300.0)     # 이 픽셀 면적보다 작은 색 덩어리는 무시 (노이즈 제거).
        # 마개(blue)가 화면에서 작게 잡혀 800 기준으로는 탈락하는 경우가 있어 300으로 낮춤.
        # 모든 색(yellow/blue/green)에 공통으로 적용되는 전역 임계값이라, 필요하면 ros2 param으로
        # 색상별 런타임 조정도 가능하다.
        self.declare_parameter("max_area", 20000.0)   # 이 면적보다 큰 덩어리도 무시 (너무 크면 오탐 가능성)
        self.declare_parameter("depth_window", 9)     # 깊이값을 읽을 때 한 점이 아니라 9x9 영역의 중앙값을 씀 (노이즈에 강하게)
        self.declare_parameter("min_depth_m", 0.25)   # 이보다 가까운 깊이는 무효 처리
        # 차를 여러 번 이동시키면서 카메라에서 더 멀어졌다 - 실측해보니 진짜 마개까지의 깊이가
        # 약 1.28m인데 기존 1.20m 컷오프에 걸려서, 먼 진짜 신호는 버려지고 더 가까운(잘못된) 파란
        # 덩어리가 대신 lock되는 문제가 있었다. 여유를 두고 2.0m로 늘린다.
        self.declare_parameter("max_depth_m", 2.0)    # 이보다 먼 깊이도 무효 처리
        self.declare_parameter("edge_margin_px", 30)  # 화면 가장자리에서 이 픽셀 이내면 "잘릴 위험 있음"으로 표시
        self.declare_parameter("reject_edge_for_lock", False)  # 가장자리에 있으면 lock을 막을지 여부
        self.declare_parameter("stable_buffer_size", 12)       # 최근 몇 프레임까지 위치를 기억해둘지
        self.declare_parameter("required_stable_frames", 8)    # 그중 최소 몇 프레임이 모여야 "안정적"이라 볼지
        self.declare_parameter("stable_std_threshold_m", 0.025)  # 모은 위치들의 흔들림(표준편차)이 이 값보다 작아야 안정적
        self.declare_parameter("publish_hz", 5.0)     # 1초에 몇 번 결과를 발행할지 (카메라 프레임 속도보다 낮게 제한)
        self.declare_parameter("publish_debug", True)  # 디버그용 화면(사각형/텍스트 그린 이미지)을 보낼지 여부

        # HSV 파라미터 기본값 (prompt 지정값)
        # HSV는 색을 표현하는 또 다른 방식 (Hue=색조, Saturation=채도, Value=밝기).
        # RGB보다 "이 범위의 색만 골라내기"가 훨씬 쉬워서 색 검출에 자주 쓴다.
        self.declare_parameter("yellow_low", [20, 100, 100])
        self.declare_parameter("yellow_high", [35, 255, 255])
        # 기존 [100,150,50]~[130,255,255]는 Saturation 하한(150)이 너무 높아 밝은 파란색을
        # 탈락시킬 수 있어 낮추고, Hue 범위도 90~140으로 넓혀서 재질/조명에 따른 색조 편차를 더 허용함.
        # 실제 마개 색을 rqt/debug_image로 확인한 뒤 필요하면 이 값을 더 좁혀 정밀도를 올릴 것.
        self.declare_parameter("blue_low", [90, 80, 50])
        self.declare_parameter("blue_high", [140, 255, 255])
        self.declare_parameter("green_low", [35, 80, 60])
        self.declare_parameter("green_high", [90, 255, 255])

        # 위에서 선언한 파라미터들의 "실제 값"을 꺼내서 self.xxx 변수에 저장한다.
        # (declare_parameter는 "이런 설정값이 있다"고 등록만 하는 것, get_parameter가 실제 값을 읽는 것)
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.mode_switch_topic = self.get_parameter("mode_switch_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.target_locked_topic = self.get_parameter("target_locked_topic").value
        self.current_mode_topic = self.get_parameter("current_mode_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.min_area = float(self.get_parameter("min_area").value)
        self.max_area = float(self.get_parameter("max_area").value)
        self.depth_window = int(self.get_parameter("depth_window").value)
        if self.depth_window % 2 == 0:
            # 깊이를 읽을 영역은 "중심 픽셀 기준 좌우 대칭"이어야 하므로 홀수로 강제 보정한다.
            self.depth_window += 1
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.edge_margin_px = int(self.get_parameter("edge_margin_px").value)
        self.reject_edge_for_lock = bool(self.get_parameter("reject_edge_for_lock").value)
        self.stable_buffer_size = int(self.get_parameter("stable_buffer_size").value)
        self.required_stable_frames = int(self.get_parameter("required_stable_frames").value)
        self.stable_std_threshold_m = float(self.get_parameter("stable_std_threshold_m").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        # publish_hz(초당 횟수)를 "한 번 발행하고 다음 발행까지 최소 몇 나노초 기다려야 하는지"로 환산.
        self.publish_period_ns = int(1e9 / max(self.publish_hz, 0.1))
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        # 색 이름("yellow" 등)을 키로 해서 (하한값, 상한값) HSV 배열을 묶어두는 딕셔너리.
        # detect_color_target()에서 현재 모드에 맞는 범위를 바로 꺼내 쓴다.
        self.hsv_ranges = {
            "yellow": (
                np.array(self.get_parameter("yellow_low").value, dtype=np.uint8),
                np.array(self.get_parameter("yellow_high").value, dtype=np.uint8),
            ),
            "blue": (
                np.array(self.get_parameter("blue_low").value, dtype=np.uint8),
                np.array(self.get_parameter("blue_high").value, dtype=np.uint8),
            ),
            "green": (
                np.array(self.get_parameter("green_low").value, dtype=np.uint8),
                np.array(self.get_parameter("green_high").value, dtype=np.uint8),
            ),
        }

        # 시작 모드 파라미터가 이상한 값이면(VALID_MODES에 없으면) 안전하게 "yellow"로 시작한다.
        initial_mode = str(self.get_parameter("initial_mode").value).strip().lower()
        self.current_mode = initial_mode if initial_mode in VALID_MODES else "yellow"

        self.bridge = CvBridge()
        # 카메라 내부 파라미터(초점거리 fx,fy, 중심점 cx,cy 등)는 /camera_info에서 받기 전까지는 None.
        self.camera_info: Optional[CameraInfo] = None
        # 최근 검출된 3D 위치들을 최대 stable_buffer_size개까지 저장하는 버퍼.
        # deque는 maxlen을 넘으면 가장 오래된 것을 자동으로 버려준다.
        self.point_buffer: deque[np.ndarray] = deque(maxlen=self.stable_buffer_size)
        self.stable_count = 0       # 최근에 "안정적이다"라고 판단된 연속 프레임 수
        self.last_publish_time_ns = 0  # 마지막으로 결과를 발행한 시각 (publish_hz 제한용)

        # BEST_EFFORT = "최대한 빨리 보내되 못 받아도 재전송은 안 함" (영상처럼 빠른 스트림에 적합)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # mode_switch/current_mode은 늦게 join하는 쪽(로봇 runner, 디텍터 자신)도
        # 마지막 값을 즉시 받을 수 있어야 하므로 TRANSIENT_LOCAL로 latch 시킨다.
        # (TRANSIENT_LOCAL = "발행자가 마지막으로 보낸 값을 저장해두고, 늦게 구독한 쪽에도 그 값을 줌"
        #  RELIABLE = "못 받으면 다시 보내서 반드시 전달되게 함" → 한 번만 발행되는 중요한 신호에 적합)
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # /camera_info는 카메라 내부 파라미터를 주는 토픽. 이게 와야 픽셀->3D 변환이 가능해진다.
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        # rgb, depth는 서로 "같은 순간"에 찍힌 프레임끼리 짝을 맞춰야 하므로
        # 일반 subscription이 아니라 message_filters.Subscriber로 만든다.
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        # ApproximateTimeSynchronizer: 두 토픽의 timestamp가 slop(0.10초) 이내로 가까우면
        # 그 둘을 묶어서 한 번에 image_callback을 호출해준다.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.10
        )
        self.sync.registerCallback(self.image_callback)

        # 외부(로봇 제어 코드)에서 "이제 파란색 찾아"같은 명령을 받는 토픽
        self.mode_switch_sub = self.create_subscription(
            String, self.mode_switch_topic, self.mode_switch_callback, latched_qos
        )

        # 결과를 내보내는 발행자(publisher)들. create_publisher(타입, 토픽이름, QoS또는depth)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.lock_pub = self.create_publisher(Bool, self.target_locked_topic, 10)
        self.mode_pub = self.create_publisher(String, self.current_mode_topic, latched_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # 시작하자마자 현재 모드를 한 번 발행해둔다 (다른 노드가 바로 알 수 있게).
        self.publish_current_mode()

        # 시작했다는 걸 터미널에 로그로 남긴다. 디버깅할 때 설정값이 제대로 들어갔는지 확인하기 좋다.
        self.get_logger().info("MultiColorDetectorNode started")
        self.get_logger().info(f"  rgb_topic           = {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic         = {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic   = {self.camera_info_topic}")
        self.get_logger().info(f"  mode_switch_topic   = {self.mode_switch_topic}")
        self.get_logger().info(f"  pose_topic          = {self.pose_topic}")
        self.get_logger().info(f"  initial_mode        = {self.current_mode}")
        self.get_logger().info(f"  publish_hz          = {self.publish_hz}")

    # ------------------------------------------------------------------
    # 모드 전환
    # ------------------------------------------------------------------
    def mode_switch_callback(self, msg: String) -> None:
        """누군가 /color_detector/mode_switch로 "yellow"/"blue"/"green"을 보내면 호출된다."""
        new_mode = str(msg.data).strip().lower()
        if new_mode not in VALID_MODES:
            # 오타나 잘못된 명령이 오면 무시하고 경고만 남긴다 (멈추거나 죽지 않게).
            self.get_logger().warn(f"알 수 없는 mode_switch 값 무시: '{msg.data}'")
            return
        if new_mode == self.current_mode:
            # 이미 그 모드면 할 일이 없다.
            return
        self.get_logger().info(f"mode_switch: {self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        # 색이 바뀌었으니 이전 색으로 모았던 샘플들은 의미가 없다 -> 버린다.
        self.reset_stability()
        self.publish_current_mode()

    def publish_current_mode(self) -> None:
        """현재 어떤 색을 보고 있는지를 토픽으로 알려준다."""
        self.mode_pub.publish(String(data=self.current_mode))

    def reset_stability(self) -> None:
        """모아둔 위치 샘플과 안정 카운트를 전부 초기화한다."""
        self.point_buffer.clear()
        self.stable_count = 0

    # ------------------------------------------------------------------
    # 메인 콜백
    # ------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo) -> None:
        """카메라 내부 파라미터(초점거리, 중심점 등)를 저장해둔다. 한 번만 와도 충분하다."""
        self.camera_info = msg

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """rgb, depth가 같은 순간에 짝지어졌을 때마다 호출되는 메인 처리 함수.

        흐름: 색 검출 -> 깊이 읽기 -> 3D 좌표 계산 -> 안정화(여러 프레임 평균) -> 발행
        """
        if self.camera_info is None:
            # 카메라 파라미터가 아직 없으면 픽셀->3D 변환을 할 수 없으니 그냥 대기.
            # throttle_duration_sec=2.0 : 같은 경고를 2초에 한 번만 찍어서 로그가 도배되지 않게 함.
            self.get_logger().warn("Waiting for /camera_info ...", throttle_duration_sec=2.0)
            return

        try:
            # ROS 이미지 메시지를 OpenCV(numpy)가 다룰 수 있는 배열로 변환.
            rgb = self.convert_rgb_image(rgb_msg)
            depth_m = self.convert_depth_image_to_meters(depth_msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        mode = self.current_mode
        # 현재 모드 색깔의 덩어리를 화면에서 찾는다. 못 찾으면 None.
        detection = self.detect_color_target(rgb, mode)
        status = f"NO {mode.upper()} TARGET"
        locked = False
        debug_detection = None

        if detection is not None:
            u, v, area, bbox, mask = detection  # u,v = 화면 픽셀 좌표, area = 면적, bbox = 사각 테두리, mask = 색 영역
            debug_detection = (u, v, bbox)
            edge_hit = self.is_near_edge(u, v, rgb.shape[1], rgb.shape[0])
            # 무게중심 한 점이 아니라 색 영역 전체에서 깊이(카메라로부터의 거리)를 읽는다.
            z = self.get_depth_from_mask(depth_m, mask)

            if z is None:
                # 깊이를 못 읽으면(반사/노이즈 등) 신뢰할 수 없으니 안정화 버퍼를 비운다.
                status = "NO VALID DEPTH"
                self.reset_stability()
            else:
                # 픽셀(u,v) + 깊이(z) -> 카메라 기준 3D 좌표(x,y,z)로 변환.
                point = self.pixel_depth_to_camera_point(u, v, z, self.camera_info)
                if point is None:
                    status = "BAD CAMERA INFO"
                    self.reset_stability()
                else:
                    point_np = np.array(point, dtype=np.float64)

                    if self.reject_edge_for_lock and edge_hit:
                        # 화면 가장자리는 잘려서 부정확할 수 있으니, 설정에 따라 lock을 막는다.
                        self.reset_stability()
                        status = f"EDGE WARN raw z={z:.2f}m"
                    else:
                        # 이번 프레임 위치를 버퍼에 추가하고, 최근 버퍼들의 평균/흔들림 정도를 구한다.
                        self.point_buffer.append(point_np)
                        mean, std_norm = self.filtered_point_stats()
                        if len(self.point_buffer) >= self.required_stable_frames and std_norm < self.stable_std_threshold_m:
                            # 충분히 많이 모였고 흔들림도 작으면 "안정적인 프레임"으로 카운트.
                            self.stable_count += 1
                        else:
                            self.stable_count = 0

                        # 한 번이라도 안정 카운트가 쌓이면 lock=True로 본다.
                        locked = self.stable_count >= 1
                        edge_txt = " EDGE" if edge_hit else ""
                        status = (
                            f"{edge_txt} raw=({point_np[0]:.3f},{point_np[1]:.3f},{point_np[2]:.3f}) "
                            f"std={std_norm:.3f} stable={self.stable_count}/{self.required_stable_frames}"
                        )

                        # publish_hz로 정한 주기보다 더 자주 발행하지 않도록 시간을 체크한다.
                        now_ns = self.get_clock().now().nanoseconds
                        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
                            self.last_publish_time_ns = now_ns
                            self.pose_pub.publish(self.make_pose(rgb_msg, mean))
                            self.lock_pub.publish(Bool(data=bool(locked)))

                        # 같은 로그도 throttle_duration_sec로 너무 자주 찍히지 않게 제한.
                        self.get_logger().info(
                            f"{mode}: pixel=({u},{v}) depth={z:.3f}m "
                            f"area={area:.1f} locked={locked} {status}",
                            throttle_duration_sec=0.5,
                        )
        else:
            # 이번 프레임에 색 자체를 못 찾았으면 안정화 진행상황을 초기화.
            self.reset_stability()

        # 검출에 실패한 프레임이라도, lock 상태(보통 False)는 주기적으로 한 번씩 알려줘야
        # 받는 쪽(robot)이 "지금 안 보인다"는 걸 알 수 있다.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
            self.last_publish_time_ns = now_ns
            self.lock_pub.publish(Bool(data=bool(locked)))

        if self.publish_debug:
            # 사람이 rqt 등으로 눈으로 확인할 수 있는 디버그 이미지를 만들어 보낸다.
            self.publish_debug_image(rgb_msg, rgb, debug_detection, f"[{mode}] {status}", locked)

    # ------------------------------------------------------------------
    # 보조 함수들 (fuel_port_detector_node.py와 동일)
    # ------------------------------------------------------------------
    def filtered_point_stats(self) -> Tuple[np.ndarray, float]:
        """버퍼에 모인 점들의 평균 위치와, 흔들림 정도(표준편차의 길이)를 계산한다."""
        if not self.point_buffer:
            return np.zeros(3, dtype=np.float64), float("inf")
        pts = np.stack(list(self.point_buffer), axis=0)  # 여러 개의 (x,y,z)를 한 배열로 쌓기
        mean = np.mean(pts, axis=0)
        std_norm = float(np.linalg.norm(np.std(pts, axis=0)))
        return mean, std_norm

    def is_near_edge(self, u: int, v: int, width: int, height: int) -> bool:
        """검출된 픽셀이 화면 가장자리(edge_margin_px 이내)에 있는지 확인."""
        return (
            u < self.edge_margin_px
            or v < self.edge_margin_px
            or u > width - self.edge_margin_px
            or v > height - self.edge_margin_px
        )

    def convert_rgb_image(self, msg: Image) -> np.ndarray:
        """ROS Image 메시지를 항상 RGB 순서의 numpy 배열로 바꿔준다 (인코딩이 달라도 통일)."""
        enc = msg.encoding.lower()
        if enc == "rgb8":
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if enc == "bgr8":
            # OpenCV는 기본이 BGR이라, RGB로 다시 바꿔준다.
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if enc == "rgba8":
            rgba = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgba8")
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        if enc == "bgra8":
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgra8")
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        # 알 수 없는 인코딩이면 일단 rgb8로 시도.
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def convert_depth_image_to_meters(self, msg: Image) -> np.ndarray:
        """깊이 이미지를 항상 "미터 단위 float32" 배열로 통일해서 반환한다."""
        enc = msg.encoding.lower()
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if enc in ("32fc1", "32fc"):
            # 이미 미터 단위 float이면 그대로 사용.
            return depth.astype(np.float32)
        if enc in ("16uc1", "mono16"):
            # 밀리미터 단위 정수(16uc1)인 경우가 많아서 0.001을 곱해 미터로 변환.
            return depth.astype(np.float32) * 0.001
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def detect_color_target(
        self, rgb: np.ndarray, mode: str
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int], np.ndarray]]:
        """현재 모드(mode) 색깔의 덩어리 중 가장 큰 것을 찾아서 (중심픽셀, 면적, 테두리, 마스크)를 반환한다.
        마스크는 "가장 큰 덩어리만 켜진" 단일 색 영역 마스크로, 깊이를 그 영역 전체에서 읽을 때 쓴다."""
        low, high = self.hsv_ranges[mode]
        # RGB -> HSV로 변환해야 "이 색 범위만 골라내기"가 쉬워진다.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # low~high 범위에 있는 픽셀만 흰색(255), 나머지는 검은색(0)인 마스크를 만든다.
        mask = cv2.inRange(hsv, low, high)

        # 작은 노이즈 점들을 지우고(OPEN), 덩어리 안의 작은 구멍을 메운다(CLOSE).
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 마스크에서 색 덩어리들의 윤곽선(테두리)을 모두 찾는다.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 너무 작거나(min_area) 너무 큰(max_area) 덩어리는 노이즈/오탐으로 보고 제외한다.
        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)  # 윤곽선을 감싸는 사각형
            candidates.append((area, c, (x, y, w, h)))
        if not candidates:
            return None

        # 후보들 중 면적이 가장 큰 것을 "진짜 표적"으로 선택한다.
        area, contour, bbox = max(candidates, key=lambda item: item[0])
        # moments로 덩어리의 "무게중심"(centroid) 픽셀을 계산한다.
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            # 혹시 모멘트 계산이 불안정하면(거의 0) 사각형의 중심으로 대체.
            x, y, w, h = bbox
            u = int(x + w / 2)
            v = int(y + h / 2)
        else:
            u = int(m["m10"] / m["m00"])
            v = int(m["m01"] / m["m00"])

        # 깊이를 무게중심 점 하나가 아니라 "선택된 덩어리 영역 전체"에서 읽을 수 있도록,
        # 그 덩어리(contour)만 채워진 단일 마스크를 따로 만들어 반환한다.
        target_mask = np.zeros_like(mask)
        cv2.drawContours(target_mask, [contour], -1, 255, thickness=cv2.FILLED)
        return u, v, area, bbox, target_mask

    def get_depth_from_mask(self, depth_m: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """무게중심 픽셀 한 점이 아니라, 색 마스크 전체 영역의 깊이 중앙값을 구한다.
        마개처럼 영역이 작거나 표면이 고르지 않을 때 한 픽셀 깊이보다 훨씬 안정적이다."""
        valid_depths = depth_m[mask > 0].astype(np.float32)
        valid_depths = valid_depths[
            (valid_depths > self.min_depth_m) &
            (valid_depths < self.max_depth_m) &
            np.isfinite(valid_depths)
        ]
        if valid_depths.size == 0:
            return None
        return float(np.median(valid_depths))

    @staticmethod
    def pixel_depth_to_camera_point(u: int, v: int, depth_m: float, info: CameraInfo) -> Optional[Tuple[float, float, float]]:
        """카메라 핀홀 모델 공식으로 픽셀(u,v)+깊이(z) -> 카메라 기준 3D 좌표(x,y,z)를 계산한다.

        info.k 안에는 [fx, 0, cx, 0, fy, cy, 0, 0, 1] 형태로 초점거리(fx,fy)와
        이미지 중심점(cx,cy)이 들어있다 (camera_info 표준 형식).
        """
        k = info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            # 초점거리가 0이면 (camera_info가 아직 비정상) 계산할 수 없다.
            return None
        z = float(depth_m)
        # 핀홀 카메라 모델의 역변환 공식: 실제 거리 = (픽셀위치 - 중심) * 깊이 / 초점거리
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return x, y, z

    def make_pose(self, image_msg: Image, point_camera: np.ndarray) -> PoseStamped:
        """3D 좌표를 ROS PoseStamped 메시지로 포장한다 (방향은 항상 회전 없음으로 둔다)."""
        pose = PoseStamped()
        pose.header.stamp = image_msg.header.stamp  # 이 좌표가 "언제" 찍힌 프레임인지 표시
        pose.header.frame_id = self.camera_info.header.frame_id or "Camera"  # "어떤 기준 좌표계"인지 표시
        pose.pose.position.x = float(point_camera[0])
        pose.pose.position.y = float(point_camera[1])
        pose.pose.position.z = float(point_camera[2])
        # 방향(orientation)은 따로 알아낼 정보가 없어서 "회전 없음"을 뜻하는 단위 쿼터니언으로 고정.
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray, detection, text: str, locked: bool) -> None:
        """검출된 위치에 사각형/원/텍스트를 그려서 사람이 눈으로 확인할 수 있는 이미지를 만들어 보낸다."""
        debug = rgb.copy()
        # lock 됐으면 초록색 테두리, 아니면 주황색 테두리로 표시.
        color = (0, 255, 0) if locked else (255, 180, 0)
        if detection is not None:
            u, v, bbox = detection
            x, y, w, h = bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)   # 검출 영역 사각형
            cv2.circle(debug, (u, v), 5, (255, 255, 255), -1)        # 중심점 표시
            cv2.putText(debug, f"LOCK={locked} ({u},{v})", (max(0, x), max(20, y - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, text[:90], (max(0, x), max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # 아무것도 못 찾았으면 화면 왼쪽 위에 상태 텍스트만 표시.
            cv2.putText(debug, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)


def main(args=None) -> None:
    """ros2 run으로 이 노드를 실행할 때 가장 먼저 호출되는 진입점 함수."""
    rclpy.init(args=args)              # ROS2 통신 시스템 초기화
    node = MultiColorDetectorNode()    # 노드(이 클래스) 인스턴스 생성 -> __init__ 실행됨
    try:
        # spin(): 토픽 콜백들이 계속 호출되도록 무한 대기하며 이벤트를 처리한다.
        # (Ctrl+C 같은 종료 신호가 오기 전까지 여기서 계속 머무른다)
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C로 종료할 때 에러 메시지 없이 깔끔하게 빠져나가게 함.
        pass
    finally:
        # 정상 종료든 예외든 항상 노드를 정리하고 ROS2를 종료시킨다.
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # "python3 multi_color_detector.py"로 직접 실행했을 때만 main()을 호출한다.
    main()
