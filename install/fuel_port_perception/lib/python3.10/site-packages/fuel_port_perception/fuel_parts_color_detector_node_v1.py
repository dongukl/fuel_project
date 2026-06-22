#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuel_parts_color_detector_node_v1.py

M0609 automatic fueling project - multi-color fuel-part detector.

Input:
  /rgb          sensor_msgs/msg/Image
  /depth        sensor_msgs/msg/Image
  /camera_info  sensor_msgs/msg/CameraInfo

Targets:
  fuel_door       = yellow
  fuel_cap        = blue
  fuel_port_hole  = green

Output per target:
  /fuel_parts/<target>/pose_camera_raw       geometry_msgs/msg/PoseStamped
  /fuel_parts/<target>/pose_camera_filtered  geometry_msgs/msg/PoseStamped
  /fuel_parts/<target>/target_locked         std_msgs/msg/Bool

Legacy compatibility for Robot A / existing STA:
  fuel_port_hole filtered pose is also published to:
  /fuel_port/pose_camera_raw
  /fuel_port/pose_camera_filtered
  /fuel_port/pose_camera
  /fuel_port/target_locked

Debug:
  /fuel_parts/debug_image

Notes:
  - Pose coordinates are in the camera optical frame: +X right, +Y down, +Z forward.
  - This node only estimates center position from color + depth. It does not estimate orientation.
  - It is designed for the single wall camera setup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import message_filters


TARGET_NAMES = ("fuel_door", "fuel_cap", "fuel_port_hole")


@dataclass
class Detection:
    u: int
    v: int
    area: float
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray


class TargetRuntime:
    def __init__(self, buffer_size: int) -> None:
        self.point_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.stable_count: int = 0
        self.last_locked: bool = False
        self.last_raw_pose: Optional[PoseStamped] = None
        self.last_filtered_pose: Optional[PoseStamped] = None
        self.last_status: str = "NO TARGET"
        self.last_detection: Optional[Tuple[int, int, Tuple[int, int, int, int]]] = None

    def reset_stability(self, status: str = "NO TARGET") -> None:
        self.point_buffer.clear()
        self.stable_count = 0
        self.last_locked = False
        self.last_filtered_pose = None
        self.last_status = status

    def filtered_point_stats(self) -> Tuple[np.ndarray, float]:
        if not self.point_buffer:
            return np.zeros(3, dtype=np.float64), float("inf")
        pts = np.stack(list(self.point_buffer), axis=0)
        mean = np.mean(pts, axis=0)
        std_norm = float(np.linalg.norm(np.std(pts, axis=0)))
        return mean, std_norm


class FuelPartsColorDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("fuel_parts_color_detector_node")

        # ------------------------------------------------------------
        # Topic parameters
        # ------------------------------------------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("output_prefix", "/fuel_parts")
        self.declare_parameter("debug_image_topic", "/fuel_parts/debug_image")

        # Existing Robot A / single-target STA compatibility.
        self.declare_parameter("publish_legacy_fuel_port_topics", True)
        self.declare_parameter("legacy_raw_pose_topic", "/fuel_port/pose_camera_raw")
        self.declare_parameter("legacy_filtered_pose_topic", "/fuel_port/pose_camera_filtered")
        self.declare_parameter("legacy_pose_topic", "/fuel_port/pose_camera")
        self.declare_parameter("legacy_lock_topic", "/fuel_port/target_locked")

        # ------------------------------------------------------------
        # Detection / filtering parameters
        # ------------------------------------------------------------
        self.declare_parameter("depth_window", 9)
        self.declare_parameter("min_depth_m", 0.25)
        self.declare_parameter("max_depth_m", 3.00)
        self.declare_parameter("edge_margin_px", 20)
        self.declare_parameter("reject_edge_for_lock", False)
        self.declare_parameter("stable_buffer_size", 12)
        self.declare_parameter("required_stable_frames", 5)
        self.declare_parameter("stable_std_threshold_m", 0.030)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("morph_kernel_size", 5)

        # Area thresholds can differ by part because door/cap/hole appear with different sizes.
        self.declare_parameter("fuel_door_min_area", 80.0)
        self.declare_parameter("fuel_door_max_area", 80000.0)
        self.declare_parameter("fuel_cap_min_area", 40.0)
        self.declare_parameter("fuel_cap_max_area", 50000.0)
        self.declare_parameter("fuel_port_hole_min_area", 40.0)
        self.declare_parameter("fuel_port_hole_max_area", 50000.0)

        # HSV color thresholds.
        # Isaac Sim colors can vary by lighting, so start wide and tune with /fuel_parts/debug_image.
        self.declare_parameter("fuel_door_hsv_low", [20, 70, 70])
        self.declare_parameter("fuel_door_hsv_high", [38, 255, 255])       # yellow
        self.declare_parameter("fuel_cap_hsv_low", [90, 60, 50])
        self.declare_parameter("fuel_cap_hsv_high", [135, 255, 255])       # blue
        self.declare_parameter("fuel_port_hole_hsv_low", [35, 60, 50])
        self.declare_parameter("fuel_port_hole_hsv_high", [95, 255, 255])  # green

        # ------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.output_prefix = str(self.get_parameter("output_prefix").value).rstrip("/")
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)

        self.publish_legacy = bool(self.get_parameter("publish_legacy_fuel_port_topics").value)
        self.legacy_raw_pose_topic = str(self.get_parameter("legacy_raw_pose_topic").value)
        self.legacy_filtered_pose_topic = str(self.get_parameter("legacy_filtered_pose_topic").value)
        self.legacy_pose_topic = str(self.get_parameter("legacy_pose_topic").value)
        self.legacy_lock_topic = str(self.get_parameter("legacy_lock_topic").value)

        self.depth_window = int(self.get_parameter("depth_window").value)
        if self.depth_window % 2 == 0:
            self.depth_window += 1
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.edge_margin_px = int(self.get_parameter("edge_margin_px").value)
        self.reject_edge_for_lock = bool(self.get_parameter("reject_edge_for_lock").value)
        self.stable_buffer_size = int(self.get_parameter("stable_buffer_size").value)
        self.required_stable_frames = int(self.get_parameter("required_stable_frames").value)
        self.stable_std_threshold_m = float(self.get_parameter("stable_std_threshold_m").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.publish_period_ns = int(1e9 / max(self.publish_hz, 0.1))
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.morph_kernel_size = max(1, int(self.get_parameter("morph_kernel_size").value))
        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size += 1

        self.area_limits: Dict[str, Tuple[float, float]] = {
            name: (
                float(self.get_parameter(f"{name}_min_area").value),
                float(self.get_parameter(f"{name}_max_area").value),
            )
            for name in TARGET_NAMES
        }

        self.hsv_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            name: (
                np.array(self.get_parameter(f"{name}_hsv_low").value, dtype=np.uint8),
                np.array(self.get_parameter(f"{name}_hsv_high").value, dtype=np.uint8),
            )
            for name in TARGET_NAMES
        }

        self.draw_colors_rgb = {
            "fuel_door": (255, 255, 0),       # yellow
            "fuel_cap": (0, 128, 255),        # blue-ish for visibility on RGB debug
            "fuel_port_hole": (0, 255, 0),    # green
        }

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.targets: Dict[str, TargetRuntime] = {
            name: TargetRuntime(self.stable_buffer_size) for name in TARGET_NAMES
        }
        self.last_publish_time_ns = 0

        # ------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.10
        )
        self.sync.registerCallback(self.image_callback)

        self.raw_pose_pubs = {}
        self.filtered_pose_pubs = {}
        self.lock_pubs = {}
        for name in TARGET_NAMES:
            self.raw_pose_pubs[name] = self.create_publisher(
                PoseStamped, f"{self.output_prefix}/{name}/pose_camera_raw", 10
            )
            self.filtered_pose_pubs[name] = self.create_publisher(
                PoseStamped, f"{self.output_prefix}/{name}/pose_camera_filtered", 10
            )
            self.lock_pubs[name] = self.create_publisher(
                Bool, f"{self.output_prefix}/{name}/target_locked", 10
            )

        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        if self.publish_legacy:
            self.legacy_raw_pose_pub = self.create_publisher(PoseStamped, self.legacy_raw_pose_topic, 10)
            self.legacy_filtered_pose_pub = self.create_publisher(PoseStamped, self.legacy_filtered_pose_topic, 10)
            self.legacy_pose_pub = self.create_publisher(PoseStamped, self.legacy_pose_topic, 10)
            self.legacy_lock_pub = self.create_publisher(Bool, self.legacy_lock_topic, 10)
        else:
            self.legacy_raw_pose_pub = None
            self.legacy_filtered_pose_pub = None
            self.legacy_pose_pub = None
            self.legacy_lock_pub = None

        self.get_logger().info("FuelPartsColorDetectorNode started")
        self.get_logger().info(f"  rgb_topic         = {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic       = {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic = {self.camera_info_topic}")
        self.get_logger().info(f"  output_prefix     = {self.output_prefix}")
        self.get_logger().info(f"  debug_topic       = {self.debug_image_topic}")
        self.get_logger().info(f"  publish_legacy    = {self.publish_legacy}")
        for name in TARGET_NAMES:
            low, high = self.hsv_ranges[name]
            amin, amax = self.area_limits[name]
            self.get_logger().info(f"  {name:<15} hsv={low.tolist()}~{high.tolist()} area={amin:.1f}~{amax:.1f}")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self.camera_info is None:
            self.get_logger().warn("Waiting for /camera_info ...", throttle_duration_sec=2.0)
            return

        try:
            rgb = self.convert_rgb_image(rgb_msg)
            depth_m = self.convert_depth_image_to_meters(depth_msg)
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        locked_summary = {}

        for name in TARGET_NAMES:
            runtime = self.targets[name]
            detection = self.detect_color_target(hsv, name)
            runtime.last_detection = None

            if detection is None:
                runtime.reset_stability(status=f"NO {name}")
                locked_summary[name] = False
                continue

            u, v = detection.u, detection.v
            runtime.last_detection = (u, v, detection.bbox)
            edge_hit = self.is_near_edge(u, v, rgb.shape[1], rgb.shape[0])
            z = self.get_valid_depth(depth_m, u, v)
            if z is None:
                # Fallback: use valid depth values inside the colored mask/bbox.
                z = self.get_valid_depth_from_mask(depth_m, detection.mask, detection.bbox)

            if z is None:
                runtime.reset_stability(status=f"{name}: NO VALID DEPTH")
                locked_summary[name] = False
                continue

            point = self.pixel_depth_to_camera_point(u, v, z, self.camera_info)
            if point is None:
                runtime.reset_stability(status=f"{name}: BAD CAMERA INFO")
                locked_summary[name] = False
                continue

            point_np = np.array(point, dtype=np.float64)
            raw_pose = self.make_pose(rgb_msg, point_np)
            runtime.last_raw_pose = raw_pose
            self.raw_pose_pubs[name].publish(raw_pose)

            if self.reject_edge_for_lock and edge_hit:
                runtime.reset_stability(status=f"{name}: EDGE raw z={z:.2f}m")
                locked_summary[name] = False
                continue

            runtime.point_buffer.append(point_np)
            mean, std_norm = runtime.filtered_point_stats()
            if len(runtime.point_buffer) >= self.required_stable_frames and std_norm < self.stable_std_threshold_m:
                runtime.stable_count += 1
            else:
                runtime.stable_count = 0

            locked = runtime.stable_count >= 1
            runtime.last_locked = bool(locked)
            filtered_pose = self.make_pose(rgb_msg, mean)
            runtime.last_filtered_pose = filtered_pose
            edge_txt = " EDGE" if edge_hit else ""
            runtime.last_status = (
                f"{name}:{edge_txt} px=({u},{v}) area={detection.area:.0f} "
                f"raw=({point_np[0]:.3f},{point_np[1]:.3f},{point_np[2]:.3f}) "
                f"std={std_norm:.3f} stable={runtime.stable_count}/{self.required_stable_frames}"
            )
            locked_summary[name] = bool(locked)

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
            self.last_publish_time_ns = now_ns
            self.publish_all_filtered_and_locks()
            self.log_summary(locked_summary)

        if self.publish_debug:
            self.publish_debug_image(rgb_msg, rgb)

    # ------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------
    def detect_color_target(self, hsv: np.ndarray, name: str) -> Optional[Detection]:
        low, high = self.hsv_ranges[name]
        mask = cv2.inRange(hsv, low, high)

        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area, max_area = self.area_limits[name]
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((area, contour, (x, y, w, h)))

        if not candidates:
            return None

        area, contour, bbox = max(candidates, key=lambda item: item[0])
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            x, y, w, h = bbox
            u = int(x + w / 2)
            v = int(y + h / 2)
        else:
            u = int(m["m10"] / m["m00"])
            v = int(m["m01"] / m["m00"])

        return Detection(u=u, v=v, area=area, bbox=bbox, mask=mask)

    # ------------------------------------------------------------
    # Depth / camera model
    # ------------------------------------------------------------
    def get_valid_depth(self, depth_m: np.ndarray, u: int, v: int) -> Optional[float]:
        h, w = depth_m.shape[:2]
        half = self.depth_window // 2
        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)
        patch = depth_m[v0:v1, u0:u1].astype(np.float32)
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def get_valid_depth_from_mask(
        self,
        depth_m: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        h, w = depth_m.shape[:2]
        x, y, bw, bh = bbox
        x0 = max(0, x)
        x1 = min(w, x + bw)
        y0 = max(0, y)
        y1 = min(h, y + bh)
        if x0 >= x1 or y0 >= y1:
            return None

        depth_roi = depth_m[y0:y1, x0:x1].astype(np.float32)
        mask_roi = mask[y0:y1, x0:x1]
        valid = depth_roi[(mask_roi > 0) & np.isfinite(depth_roi)]
        valid = valid[(valid > self.min_depth_m) & (valid < self.max_depth_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def pixel_depth_to_camera_point(u: int, v: int, depth_m: float, info: CameraInfo) -> Optional[Tuple[float, float, float]]:
        k = info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            return None
        z = float(depth_m)
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return x, y, z

    # ------------------------------------------------------------
    # Publish / message helpers
    # ------------------------------------------------------------
    def publish_all_filtered_and_locks(self) -> None:
        for name in TARGET_NAMES:
            runtime = self.targets[name]
            locked = bool(runtime.last_locked)
            self.lock_pubs[name].publish(Bool(data=locked))
            if locked and runtime.last_filtered_pose is not None:
                self.filtered_pose_pubs[name].publish(runtime.last_filtered_pose)

        if self.publish_legacy:
            # Existing Robot A code expects the green hole pose on /fuel_port/*.
            hole = self.targets["fuel_port_hole"]
            hole_locked = bool(hole.last_locked)
            if hole.last_raw_pose is not None and self.legacy_raw_pose_pub is not None:
                self.legacy_raw_pose_pub.publish(hole.last_raw_pose)
            if hole_locked and hole.last_filtered_pose is not None:
                if self.legacy_filtered_pose_pub is not None:
                    self.legacy_filtered_pose_pub.publish(hole.last_filtered_pose)
                if self.legacy_pose_pub is not None:
                    self.legacy_pose_pub.publish(hole.last_filtered_pose)
            elif hole.last_raw_pose is not None and self.legacy_pose_pub is not None:
                # Keep legacy pose useful for debugging; controller should still use lock.
                self.legacy_pose_pub.publish(hole.last_raw_pose)
            if self.legacy_lock_pub is not None:
                self.legacy_lock_pub.publish(Bool(data=hole_locked))

    def make_pose(self, image_msg: Image, point_camera: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = image_msg.header.stamp
        pose.header.frame_id = self.camera_info.header.frame_id or "wall_camera"
        pose.pose.position.x = float(point_camera[0])
        pose.pose.position.y = float(point_camera[1])
        pose.pose.position.z = float(point_camera[2])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray) -> None:
        debug = rgb.copy()
        line_y = 24
        for name in TARGET_NAMES:
            runtime = self.targets[name]
            draw_color = self.draw_colors_rgb[name]
            if runtime.last_detection is not None:
                u, v, bbox = runtime.last_detection
                x, y, w, h = bbox
                cv2.rectangle(debug, (x, y), (x + w, y + h), draw_color, 2)
                cv2.circle(debug, (u, v), 5, (255, 255, 255), -1)
                cv2.putText(debug, name, (max(0, x), max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, draw_color, 1, cv2.LINE_AA)

            lock_txt = "LOCK" if runtime.last_locked else "----"
            txt = f"{name}: {lock_txt} {runtime.last_status[:90]}"
            cv2.putText(debug, txt, (12, line_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, draw_color, 1, cv2.LINE_AA)
            line_y += 20

        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)

    def log_summary(self, locked_summary: Dict[str, bool]) -> None:
        parts = []
        for name in TARGET_NAMES:
            runtime = self.targets[name]
            parts.append(f"{name}:locked={runtime.last_locked}")
        self.get_logger().info(" | ".join(parts), throttle_duration_sec=1.0)

    # ------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------
    def is_near_edge(self, u: int, v: int, width: int, height: int) -> bool:
        return (
            u < self.edge_margin_px
            or v < self.edge_margin_px
            or u > width - self.edge_margin_px
            or v > height - self.edge_margin_px
        )

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

    def convert_depth_image_to_meters(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if enc in ("32fc1", "32fc"):
            return depth.astype(np.float32)
        if enc in ("16uc1", "mono16"):
            return depth.astype(np.float32) * 0.001
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FuelPartsColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
