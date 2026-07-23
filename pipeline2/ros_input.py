"""ROS 2 image-topic input for the lane detection pipeline."""

import cv2
import numpy as np


class RosImageSource:
    """Provide the latest ROS Image as an OpenCV BGR frame."""

    def __init__(self, topic):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = Node("pipeline2_scanline_image_subscriber")
        self._frame = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._subscription = self._node.create_subscription(
            Image, topic, self._on_image, qos
        )

    @staticmethod
    def _decode_image(msg):
        encoding = msg.encoding.lower()
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(encoding)
        if channels is None:
            raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")

        row_bytes = msg.step or msg.width * channels
        data = np.frombuffer(msg.data, dtype=np.uint8)
        required = msg.height * row_bytes
        if data.size < required:
            raise ValueError(
                f"Image data is too short: got {data.size} bytes, expected {required}"
            )

        rows = data[:required].reshape(msg.height, row_bytes)
        pixels = rows[:, :msg.width * channels]
        if channels == 1:
            image = pixels.reshape(msg.height, msg.width)
        else:
            image = pixels.reshape(msg.height, msg.width, channels)

        conversions = {
            "rgb8": cv2.COLOR_RGB2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR,
            "mono8": cv2.COLOR_GRAY2BGR,
        }
        conversion = conversions.get(encoding)
        if conversion is not None:
            return cv2.cvtColor(image, conversion)
        return image.copy()

    def _on_image(self, msg):
        try:
            self._frame = self._decode_image(msg)
        except ValueError as exc:
            self._node.get_logger().error(str(exc))

    def read(self):
        """Block until the next camera frame or ROS shuts down."""
        while self._rclpy.ok() and self._frame is None:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        if self._frame is None:
            return False, None
        frame = self._frame
        self._frame = None
        return True, frame

    def close(self):
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
