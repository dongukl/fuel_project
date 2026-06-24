#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_marker_detector.py

M0609 multi-robot oiling project - ArUco based perception node.

목적
- 기존 multi_color_detector.py와 같은 ROS2 인터페이스를 유지한다.
- 색상 contour 대신 ArUco marker pose를 추정한다.
- mode_switch가 blue이면 marker 기준 fuel_cap offset을 pose로 발행한다.
- mode_switch가 green이면 marker 기준 fuel_port_hole/mouth offset을 pose로 발행한다.
- mode_switch가 yellow이면 marker 기준 fuel_door offset을 pose로 발행한다.

Input:
  /rgb                          sensor_msgs/msg/Image      (rgb8/bgr8/rgba8/bgra8)
  /camera_info                  sensor_msgs/msg/CameraInfo
  /color_detector/mode_switch   std_msgs/msg/String        ("yellow" / "blue" / "green")

Output:
  /color_detector/pose          geometry_msgs/msg/PoseStamped  (camera frame target position)
  /color_detector/target_locked std_msgs/msg/Bool
  /color_detector/current_mode  std_msgs/msg/String
  /color_detector/debug_image   sensor_msgs/msg/Image

주의
- 이 노드는 depth 이미지를 사용하지 않는다. ArUco marker의 known size와 camera_info로 pose를 추정한다.
- 기존 Isaac Sim 메인 코드는 /color_detector/pose를 camera 좌표계로 받아 world로 변환하므로,
  이 노드도 target point를 camera 좌표계(OpenCV/ROS optical: x right, y down, z forward)로 발행한다.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge


VALID_MODES = ("yellow", "blue", "green")


def _make_aruco_dictionary(name: str):
    """문자열 파라미터를 OpenCV ArUco dictionary로 변환한다."""
    name = str(name).strip().upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_detector_parameters():
    """OpenCV ArUco DetectorParameters API compatibility helper.

    OpenCV/contrib 빌드에 따라 다음 둘 중 하나만 존재할 수 있다.
      - OpenCV 4.7+ style: cv2.aruco.DetectorParameters()
      - legacy style:       cv2.aruco.DetectorParameters_create()
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
    """OpenCV 4.7+ ArucoDetector가 있으면 사용하고, 없으면 legacy detectMarkers로 fallback."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        try:
            return cv2.aruco.ArucoDetector(dictionary, detector_params)
        except Exception:
            return None
    return None


class ArucoMarkerDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_marker_detector_node")

        # -----------------------------
        # Topic parameters: 기존 color detector와 이름을 맞춘다.
        # -----------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("mode_switch_topic", "/color_detector/mode_switch")
        self.declare_parameter("pose_topic", "/color_detector/pose")
        self.declare_parameter("target_locked_topic", "/color_detector/target_locked")
        self.declare_parameter("current_mode_topic", "/color_detector/current_mode")
        self.declare_parameter("debug_image_topic", "/color_detector/debug_image")
        self.declare_parameter("initial_mode", "blue")

        # -----------------------------
        # ArUco parameters
        # -----------------------------
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_id", 0)
        # create_aruco_marker_grid_in_isaac.py의 MARKER_PATTERN_SIZE_M와 반드시 같게 맞춘다.
        self.declare_parameter("marker_size_m", 0.15)

        # marker coordinate convention used by OpenCV estimatePoseSingleMarkers:
        #   x: marker right, y: marker down, z: marker normal toward camera.
        # 아래 기본값은 create_aruco_marker_grid_in_isaac.py의 기본 위치
        # visual_test.usda 기준 MARKER_CENTER_WORLD=[-0.40267,-0.77000,1.20000] 및 fuel_cap/fuel_port_hole 위치를 기준으로 계산했다.
        # green은 기존 Isaac 메인 코드가 apply_mouth_offset=True로 다시 안쪽 보정을 하므로,
        # 여기서는 "hole center"가 아니라 "mouth surface"에 해당하는 offset을 기본값으로 둔다.
        self.declare_parameter("marker_to_door_xyz", [0.240267, -1.110197, -1.279059])
        self.declare_parameter("marker_to_cap_xyz",  [0.238197, -1.090520, -1.302334])
        self.declare_parameter("marker_to_hole_xyz", [0.244428, -1.054349, -1.510171])

        # 안정화/발행 파라미터
        self.declare_parameter("required_stable_frames", 3)
        self.declare_parameter("stable_buffer_size", 8)
        self.declare_parameter("stable_std_threshold_m", 0.035)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("draw_axes", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.mode_switch_topic = self.get_parameter("mode_switch_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.target_locked_topic = self.get_parameter("target_locked_topic").value
        self.current_mode_topic = self.get_parameter("current_mode_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        initial_mode = str(self.get_parameter("initial_mode").value).strip().lower()
        self.current_mode = initial_mode if initial_mode in VALID_MODES else "blue"

        self.marker_id = int(self.get_parameter("marker_id").value)
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.dictionary_name = str(self.get_parameter("aruco_dictionary").value)
        self.dictionary = _make_aruco_dictionary(self.dictionary_name)

        self.target_offsets: Dict[str, np.ndarray] = {
            "yellow": np.array(self.get_parameter("marker_to_door_xyz").value, dtype=np.float64),
            "blue":   np.array(self.get_parameter("marker_to_cap_xyz").value, dtype=np.float64),
            "green":  np.array(self.get_parameter("marker_to_hole_xyz").value, dtype=np.float64),
        }

        self.required_stable_frames = int(self.get_parameter("required_stable_frames").value)
        self.stable_buffer_size = int(self.get_parameter("stable_buffer_size").value)
        self.stable_std_threshold_m = float(self.get_parameter("stable_std_threshold_m").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.publish_period_ns = int(1e9 / max(self.publish_hz, 0.1))
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.draw_axes = bool(self.get_parameter("draw_axes").value)

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.point_buffer: deque[np.ndarray] = deque(maxlen=self.stable_buffer_size)
        self.stable_count = 0
        self.last_publish_time_ns = 0

        # OpenCV ArUco API 호환 처리.
        # 일부 환경은 DetectorParameters()가 없고 DetectorParameters_create()만 있다.
        self.detector_params = _make_detector_parameters()
        self.aruco_detector = _make_aruco_detector(self.dictionary, self.detector_params)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, sensor_qos)
        self.mode_switch_sub = self.create_subscription(
            String, self.mode_switch_topic, self.mode_switch_callback, latched_qos
        )

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.lock_pub = self.create_publisher(Bool, self.target_locked_topic, 10)
        self.mode_pub = self.create_publisher(String, self.current_mode_topic, latched_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.publish_current_mode()
        self.get_logger().info("ArucoMarkerDetectorNode started")
        self.get_logger().info(f"  rgb_topic           = {self.rgb_topic}")
        self.get_logger().info(f"  camera_info_topic   = {self.camera_info_topic}")
        self.get_logger().info(f"  mode_switch_topic   = {self.mode_switch_topic}")
        self.get_logger().info(f"  pose_topic          = {self.pose_topic}")
        self.get_logger().info(f"  dictionary          = {self.dictionary_name}")
        self.get_logger().info(f"  marker_id           = {self.marker_id}")
        self.get_logger().info(f"  marker_size_m       = {self.marker_size_m:.3f}")
        self.get_logger().info(f"  initial_mode        = {self.current_mode}")
        for mode, offset in self.target_offsets.items():
            self.get_logger().info(f"  marker_to_{mode}_xyz = {np.round(offset, 4).tolist()}")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def mode_switch_callback(self, msg: String) -> None:
        new_mode = str(msg.data).strip().lower()
        if new_mode not in VALID_MODES:
            self.get_logger().warn(f"알 수 없는 mode_switch 값 무시: '{msg.data}'")
            return
        if new_mode == self.current_mode:
            return
        self.get_logger().info(f"mode_switch: {self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.reset_stability()
        self.publish_current_mode()

    def publish_current_mode(self) -> None:
        self.mode_pub.publish(String(data=self.current_mode))

    def reset_stability(self) -> None:
        self.point_buffer.clear()
        self.stable_count = 0

    def image_callback(self, rgb_msg: Image) -> None:
        if self.camera_info is None:
            self.get_logger().warn("Waiting for /camera_info ...", throttle_duration_sec=2.0)
            return

        try:
            rgb = self.convert_rgb_image(rgb_msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        camera_matrix, dist_coeffs = self.camera_calibration(self.camera_info)
        if camera_matrix is None:
            self.get_logger().warn("Invalid camera_info K matrix", throttle_duration_sec=2.0)
            return

        detection = self.detect_marker_pose(rgb, camera_matrix, dist_coeffs)
        locked = False
        target_camera = None
        status = "NO ARUCO MARKER"
        debug_payload = None

        if detection is not None:
            corners, rvec, tvec = detection
            offset_marker = self.target_offsets[self.current_mode].reshape(3, 1)
            R_marker_to_camera, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            target_camera = (R_marker_to_camera @ offset_marker + tvec.reshape(3, 1)).reshape(3)

            self.point_buffer.append(target_camera.astype(np.float64))
            mean, std_norm = self.filtered_point_stats()
            if len(self.point_buffer) >= self.required_stable_frames and std_norm <= self.stable_std_threshold_m:
                self.stable_count += 1
            else:
                self.stable_count = 0
            locked = self.stable_count >= 1
            status = (
                f"id={self.marker_id} mode={self.current_mode} "
                f"target=({target_camera[0]:.3f},{target_camera[1]:.3f},{target_camera[2]:.3f}) "
                f"std={std_norm:.3f} stable={self.stable_count}/{self.required_stable_frames}"
            )
            debug_payload = (corners, rvec, tvec, mean, target_camera)

            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
                self.last_publish_time_ns = now_ns
                self.pose_pub.publish(self.make_pose(rgb_msg, mean))
                self.lock_pub.publish(Bool(data=bool(locked)))

            self.get_logger().info(status + f" locked={locked}", throttle_duration_sec=0.5)
        else:
            self.reset_stability()

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
            self.last_publish_time_ns = now_ns
            self.lock_pub.publish(Bool(data=bool(locked)))

        if self.publish_debug:
            self.publish_debug_image(rgb_msg, rgb, debug_payload, status, locked, camera_matrix, dist_coeffs)

    # ------------------------------------------------------------------
    # Detection / pose helpers
    # ------------------------------------------------------------------
    def detect_marker_pose(self, rgb: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)

        if ids is None or len(ids) == 0:
            return None

        ids_flat = ids.flatten().astype(int)
        matches = np.where(ids_flat == self.marker_id)[0]
        if len(matches) == 0:
            self.get_logger().warn(
                f"ArUco marker는 보이지만 marker_id={self.marker_id}가 아님. visible ids={ids_flat.tolist()}",
                throttle_duration_sec=1.0,
            )
            return None
        i = int(matches[0])
        marker_corners = [corners[i]]
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            self.marker_size_m,
            camera_matrix,
            dist_coeffs,
        )
        return corners[i], rvecs[0].reshape(3), tvecs[0].reshape(3)

    def filtered_point_stats(self) -> Tuple[np.ndarray, float]:
        if not self.point_buffer:
            return np.zeros(3, dtype=np.float64), float("inf")
        pts = np.stack(list(self.point_buffer), axis=0)
        mean = np.mean(pts, axis=0)
        std_norm = float(np.linalg.norm(np.std(pts, axis=0)))
        return mean, std_norm

    @staticmethod
    def camera_calibration(info: CameraInfo):
        k = np.array(info.k, dtype=np.float64).reshape(3, 3)
        if abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9:
            return None, None
        d = np.array(info.d, dtype=np.float64).reshape(-1, 1) if info.d else np.zeros((5, 1), dtype=np.float64)
        return k, d

    def make_pose(self, image_msg: Image, point_camera: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = image_msg.header.stamp
        pose.header.frame_id = self.camera_info.header.frame_id or image_msg.header.frame_id or "sim_camera"
        pose.pose.position.x = float(point_camera[0])
        pose.pose.position.y = float(point_camera[1])
        pose.pose.position.z = float(point_camera[2])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    # ------------------------------------------------------------------
    # Image conversion / debug
    # ------------------------------------------------------------------
    def convert_rgb_image(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        if enc == "rgb8":
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if enc == "bgr8":
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if enc == "rgba8":
            rgba = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgba8")
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        if enc == "bgra8":
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgra8")
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray, payload, text: str, locked: bool,
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        debug = rgb.copy()
        color = (0, 255, 0) if locked else (255, 180, 0)
        if payload is not None:
            corners, rvec, tvec, mean, target_camera = payload
            cv2.aruco.drawDetectedMarkers(debug, [corners], np.array([[self.marker_id]], dtype=np.int32), color)
            if self.draw_axes:
                try:
                    cv2.drawFrameAxes(debug, camera_matrix, dist_coeffs, rvec.reshape(3, 1), tvec.reshape(3, 1), self.marker_size_m * 0.5)
                except Exception:
                    pass

            # target point projection
            try:
                projected, _ = cv2.projectPoints(
                    mean.reshape(1, 1, 3).astype(np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    camera_matrix,
                    dist_coeffs,
                )
                u, v = projected.reshape(2)
                cv2.circle(debug, (int(round(u)), int(round(v))), 7, (255, 255, 255), -1)
                cv2.circle(debug, (int(round(u)), int(round(v))), 9, color, 2)
            except Exception:
                pass
            cv2.putText(debug, f"ARUCO LOCK={locked} mode={self.current_mode}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug, text[:110], (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(debug, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoMarkerDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
