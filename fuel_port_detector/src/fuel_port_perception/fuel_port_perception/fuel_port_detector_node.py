#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuel_port_detector_node.py

ROS 2 perception node for the M0609 automatic fueling simulation.

Input topics:
  /rgb          sensor_msgs/msg/Image
  /depth        sensor_msgs/msg/Image
  /camera_info  sensor_msgs/msg/CameraInfo

Output topics:
  /fuel_port/pose_camera       geometry_msgs/msg/PoseStamped
  /fuel_port/debug_image       sensor_msgs/msg/Image

Current detection method:
  Red-cylinder color segmentation -> centroid pixel -> median depth -> 3D point in Camera frame.

Notes:
  - This node publishes the target pose in the camera frame first.
  - World/base conversion should be added later using TF or Isaac Sim camera prim pose.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import message_filters


class FuelPortDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("fuel_port_detector_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("pose_topic", "/fuel_port/pose_camera")
        self.declare_parameter("debug_image_topic", "/fuel_port/debug_image")

        self.declare_parameter("min_area", 80.0)
        self.declare_parameter("max_area", 200000.0)
        self.declare_parameter("depth_window", 7)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("publish_debug", True)

        # HSV red threshold. Red wraps around hue=0, so we use two ranges.
        self.declare_parameter("red_low1", [0, 80, 60])
        self.declare_parameter("red_high1", [12, 255, 255])
        self.declare_parameter("red_low2", [170, 80, 60])
        self.declare_parameter("red_high2", [180, 255, 255])

        self.rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.debug_image_topic = self.get_parameter("debug_image_topic").get_parameter_value().string_value

        self.min_area = float(self.get_parameter("min_area").value)
        self.max_area = float(self.get_parameter("max_area").value)
        self.depth_window = int(self.get_parameter("depth_window").value)
        if self.depth_window % 2 == 0:
            self.depth_window += 1
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.red_low1 = np.array(self.get_parameter("red_low1").value, dtype=np.uint8)
        self.red_high1 = np.array(self.get_parameter("red_high1").value, dtype=np.uint8)
        self.red_low2 = np.array(self.get_parameter("red_low2").value, dtype=np.uint8)
        self.red_high2 = np.array(self.get_parameter("red_high2").value, dtype=np.uint8)

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None

        # Isaac Sim image topics often behave like sensor data, so BEST_EFFORT is usually safer.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )

        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.10,
        )
        self.sync.registerCallback(self.image_callback)

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.get_logger().info("FuelPortDetectorNode started")
        self.get_logger().info(f"  rgb_topic         = {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic       = {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic = {self.camera_info_topic}")
        self.get_logger().info(f"  pose_topic        = {self.pose_topic}")

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

        detection = self.detect_red_target(rgb)
        if detection is None:
            if self.publish_debug:
                self.publish_debug_image(rgb_msg, rgb, None, "NO RED TARGET")
            return

        u, v, area, bbox = detection
        z = self.get_valid_depth(depth_m, u, v)
        if z is None:
            if self.publish_debug:
                self.publish_debug_image(rgb_msg, rgb, (u, v, bbox), "NO VALID DEPTH")
            return

        point_camera = self.pixel_depth_to_camera_point(u, v, z, self.camera_info)
        if point_camera is None:
            return

        x, y, z = point_camera
        pose = PoseStamped()
        pose.header.stamp = rgb_msg.header.stamp
        # Publish in the camera frame for now.
        pose.header.frame_id = self.camera_info.header.frame_id or "Camera"
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        # Orientation is not estimated in this first version.
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        self.get_logger().info(
            f"fuel_port camera_frame: pixel=({u},{v}) depth={z:.3f}m "
            f"point=({x:.3f},{y:.3f},{z:.3f}) area={area:.1f}",
            throttle_duration_sec=0.25,
        )

        if self.publish_debug:
            self.publish_debug_image(rgb_msg, rgb, (u, v, bbox), f"z={z:.2f}m")

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
        # Fallback through cv_bridge conversion.
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def convert_depth_image_to_meters(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)

        if enc in ("32fc1", "32fc"):
            return depth.astype(np.float32)
        if enc in ("16uc1", "mono16"):
            # Common convention: uint16 depth is in millimeters.
            return depth.astype(np.float32) * 0.001

        # Isaac Sim sometimes reports unusual encodings. Try a safe interpretation.
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def detect_red_target(self, rgb: np.ndarray) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        # rgb -> hsv. OpenCV uses RGB2HSV when input is RGB.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, self.red_low1, self.red_high1)
        mask2 = cv2.inRange(hsv, self.red_low2, self.red_high2)
        mask = cv2.bitwise_or(mask1, mask2)

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

    def publish_debug_image(self, original_msg: Image, rgb: np.ndarray, detection, text: str) -> None:
        debug = rgb.copy()
        if detection is not None:
            u, v, bbox = detection
            x, y, w, h = bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug, (u, v), 5, (255, 255, 255), -1)
            cv2.putText(debug, f"({u},{v}) {text}", (max(0, x), max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(debug, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out = self.bridge.cv2_to_imgmsg(debug, encoding="rgb8")
        out.header = original_msg.header
        self.debug_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FuelPortDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
