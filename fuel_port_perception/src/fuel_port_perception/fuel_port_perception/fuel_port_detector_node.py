#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuel_port_detector_node_v2.py

M0609 automatic fueling project - perception node v2.

Input:
  /rgb          sensor_msgs/msg/Image      (confirmed: rgb8, 640x640)
  /depth        sensor_msgs/msg/Image      (confirmed: 32FC1, meter, 640x640)
  /camera_info  sensor_msgs/msg/CameraInfo

Output:
  /fuel_port/pose_camera_raw       geometry_msgs/msg/PoseStamped
  /fuel_port/pose_camera_filtered  geometry_msgs/msg/PoseStamped
  /fuel_port/pose_camera           geometry_msgs/msg/PoseStamped  (legacy compatibility)
  /fuel_port/target_locked         std_msgs/msg/Bool
  /fuel_port/debug_image           sensor_msgs/msg/Image

What changed from v1:
  - raw pose and filtered pose are separated
  - depth/area/edge checks are added
  - pose is smoothed by a short moving buffer
  - target lock is published after N stable frames
  - pose publish rate is limited
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

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


class FuelPortDetectorNodeV2(Node):
    def __init__(self) -> None:
        super().__init__("fuel_port_detector_node_v2")

        # -----------------------------
        # Topic parameters
        # -----------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("raw_pose_topic", "/fuel_port/pose_camera_raw")
        self.declare_parameter("filtered_pose_topic", "/fuel_port/pose_camera_filtered")
        self.declare_parameter("legacy_pose_topic", "/fuel_port/pose_camera")
        self.declare_parameter("target_locked_topic", "/fuel_port/target_locked")
        self.declare_parameter("debug_image_topic", "/fuel_port/debug_image")

        # -----------------------------
        # Detection / filtering parameters
        # -----------------------------
        self.declare_parameter("min_area", 100.0)
        self.declare_parameter("max_area", 50000.0)
        self.declare_parameter("depth_window", 9)
        self.declare_parameter("min_depth_m", 0.25)
        self.declare_parameter("max_depth_m", 3)
        self.declare_parameter("edge_margin_px", 30)
        self.declare_parameter("reject_edge_for_lock", False)
        self.declare_parameter("stable_buffer_size", 12)
        self.declare_parameter("required_stable_frames", 8)
        self.declare_parameter("stable_std_threshold_m", 0.025)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("publish_debug", True)

        # HSV red threshold. Red wraps around hue=0, so we use two ranges.
        # self.declare_parameter("red_low1", [0, 80, 60])
        # self.declare_parameter("red_high1", [12, 255, 255])
        # self.declare_parameter("red_low2", [170, 80, 60])
        # self.declare_parameter("red_high2", [180, 255, 255])

        # 색 탐지 코드
        self.declare_parameter("green_high", [90, 255, 255])
        self.declare_parameter("green_low", [35, 80, 60])

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.raw_pose_topic = self.get_parameter("raw_pose_topic").value
        self.filtered_pose_topic = self.get_parameter("filtered_pose_topic").value
        self.legacy_pose_topic = self.get_parameter("legacy_pose_topic").value
        self.target_locked_topic = self.get_parameter("target_locked_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.min_area = float(self.get_parameter("min_area").value)
        self.max_area = float(self.get_parameter("max_area").value)
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

        # self.red_low1 = np.array(self.get_parameter("red_low1").value, dtype=np.uint8)
        # self.red_high1 = np.array(self.get_parameter("red_high1").value, dtype=np.uint8)
        # self.red_low2 = np.array(self.get_parameter("red_low2").value, dtype=np.uint8)
        # self.red_high2 = np.array(self.get_parameter("red_high2").value, dtype=np.uint8)

        self.green_low = np.array(self.get_parameter("green_low").value, dtype=np.uint8)
        self.green_high = np.array(self.get_parameter("green_high").value, dtype=np.uint8)
        
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.point_buffer: deque[np.ndarray] = deque(maxlen=self.stable_buffer_size)
        self.stable_count = 0
        self.last_publish_time_ns = 0
        self.last_locked = False

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

        self.raw_pose_pub = self.create_publisher(PoseStamped, self.raw_pose_topic, 10)
        self.filtered_pose_pub = self.create_publisher(PoseStamped, self.filtered_pose_topic, 10)
        # Legacy output: publish the filtered pose when locked, otherwise raw pose.
        self.legacy_pose_pub = self.create_publisher(PoseStamped, self.legacy_pose_topic, 10)
        self.lock_pub = self.create_publisher(Bool, self.target_locked_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info("FuelPortDetectorNodeV2 started")
        self.get_logger().info(f"  rgb_topic              = {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic            = {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic      = {self.camera_info_topic}")
        self.get_logger().info(f"  raw_pose_topic         = {self.raw_pose_topic}")
        self.get_logger().info(f"  filtered_pose_topic    = {self.filtered_pose_topic}")
        self.get_logger().info(f"  target_locked_topic    = {self.target_locked_topic}")
        self.get_logger().info(f"  publish_hz             = {self.publish_hz}")

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

        detection = self.detect_green_target(rgb)
        status = "NO GREEN TARGET"
        locked = False
        raw_pose = None
        filtered_pose = None
        debug_detection = None

        if detection is not None:
            u, v, area, bbox = detection
            debug_detection = (u, v, bbox)
            edge_hit = self.is_near_edge(u, v, rgb.shape[1], rgb.shape[0])
            z = self.get_valid_depth(depth_m, u, v)

            if z is None:
                status = "NO VALID DEPTH"
                self.reset_stability()
            else:
                point = self.pixel_depth_to_camera_point(u, v, z, self.camera_info)
                if point is None:
                    status = "BAD CAMERA INFO"
                    self.reset_stability()
                else:
                    point_np = np.array(point, dtype=np.float64)
                    raw_pose = self.make_pose(rgb_msg, point_np)
                    self.raw_pose_pub.publish(raw_pose)

                    if self.reject_edge_for_lock and edge_hit:
                        self.reset_stability()
                        status = f"EDGE WARN raw z={z:.2f}m"
                    else:
                        self.point_buffer.append(point_np)
                        mean, std_norm = self.filtered_point_stats()
                        if len(self.point_buffer) >= self.required_stable_frames and std_norm < self.stable_std_threshold_m:
                            self.stable_count += 1
                        else:
                            self.stable_count = 0

                        locked = self.stable_count >= 1
                        filtered_pose = self.make_pose(rgb_msg, mean)

                        edge_txt = " EDGE" if edge_hit else ""
                        status = (
                            f"{edge_txt} raw=({point_np[0]:.3f},{point_np[1]:.3f},{point_np[2]:.3f}) "
                            f"std={std_norm:.3f} stable={self.stable_count}/{self.required_stable_frames}"
                        )

                        now_ns = self.get_clock().now().nanoseconds
                        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
                            self.last_publish_time_ns = now_ns
                            if locked:
                                self.filtered_pose_pub.publish(filtered_pose)
                                self.legacy_pose_pub.publish(filtered_pose)
                            else:
                                # Legacy pose stays available for debugging, but controller should use filtered + lock.
                                self.legacy_pose_pub.publish(raw_pose)
                            self.lock_pub.publish(Bool(data=bool(locked)))

                        self.get_logger().info(
                            f"fuel_port v2: pixel=({u},{v}) depth={z:.3f}m "
                            f"area={area:.1f} locked={locked} {status}",
                            throttle_duration_sec=0.5,
                        )
        else:
            self.reset_stability()

        # Publish unlocked state occasionally even when no target is visible.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_publish_time_ns >= self.publish_period_ns:
            self.last_publish_time_ns = now_ns
            self.lock_pub.publish(Bool(data=bool(locked)))

        if self.publish_debug:
            self.publish_debug_image(rgb_msg, rgb, debug_detection, status, locked)

    def reset_stability(self) -> None:
        self.point_buffer.clear()
        self.stable_count = 0

    def filtered_point_stats(self) -> Tuple[np.ndarray, float]:
        if not self.point_buffer:
            return np.zeros(3, dtype=np.float64), float("inf")
        pts = np.stack(list(self.point_buffer), axis=0)
        mean = np.mean(pts, axis=0)
        std_norm = float(np.linalg.norm(np.std(pts, axis=0)))
        return mean, std_norm

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

    def detect_green_target(self, rgb: np.ndarray) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.green_low, self.green_high)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            candidates.append((area, c, (x, y, w, h)))
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
        return u, v, area, bbox

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

    def make_pose(self, image_msg: Image, point_camera: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = image_msg.header.stamp
        pose.header.frame_id = self.camera_info.header.frame_id or "Camera"
        pose.pose.position.x = float(point_camera[0])
        pose.pose.position.y = float(point_camera[1])
        pose.pose.position.z = float(point_camera[2])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray, detection, text: str, locked: bool) -> None:
        debug = rgb.copy()
        color = (0, 255, 0) if locked else (255, 180, 0)
        if detection is not None:
            u, v, bbox = detection
            x, y, w, h = bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            cv2.circle(debug, (u, v), 5, (255, 255, 255), -1)
            cv2.putText(debug, f"LOCK={locked} ({u},{v})", (max(0, x), max(20, y - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, text[:90], (max(0, x), max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(debug, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FuelPortDetectorNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
