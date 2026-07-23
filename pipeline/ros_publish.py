"""Optional ROS 2 publishers for lane waypoints and the overlay image.

rclpy is only imported inside WaypointPublisher.__init__, so importing
this module (and running the rest of the pipeline) doesn't require a
ROS 2 install. Only constructing WaypointPublisher does.
"""

import config


class WaypointPublisher:
    """Publishes center waypoints and the processed lane-overlay image.

    Robot frame convention (REP-103): x = forward meters, y = left
    meters -- matching pipeline.waypoints.pixel_to_ground_m.
    """

    def __init__(
        self,
        topic=config.ROS_DEFAULT_TOPIC,
        frame_id=config.ROS_DEFAULT_FRAME_ID,
        image_topic=config.ROS_DEFAULT_IMAGE_TOPIC,
        image_frame_id=config.ROS_DEFAULT_IMAGE_FRAME_ID,
        debug_image_topic=None,
    ):
        import rclpy
        from nav_msgs.msg import Path
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._frame_id = frame_id
        self._image_frame_id = image_frame_id
        self._node = Node("lane_waypoint_publisher")
        self._publisher = self._node.create_publisher(Path, topic, 10)
        self._image_publisher = self._node.create_publisher(Image, image_topic, 10)
        self._debug_image_publisher = (
            self._node.create_publisher(Image, debug_image_topic, 10)
            if debug_image_topic
            else None
        )

    def _publish_bgr_image(self, publisher, image_bgr, frame_id, stamp):
        from sensor_msgs.msg import Image

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = frame_id
        image.height, image.width = image_bgr.shape[:2]
        image.encoding = "bgr8"
        image.is_bigendian = False
        image.step = image.width * 3
        image.data = image_bgr.tobytes()
        publisher.publish(image)

    def publish(self, coords, image_bgr=None, debug_image=None):
        """Publish the overlay image, the optional debug-steps image, and
        available ground waypoints.

        Frames with no locked lane (coords has no ground waypoints) are
        skipped rather than publishing an empty/stale Path. The images are
        still published so the detector output remains visible.
        """
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path

        stamp = self._node.get_clock().now().to_msg()

        if image_bgr is not None:
            self._publish_bgr_image(
                self._image_publisher, image_bgr, self._image_frame_id, stamp
            )

        if debug_image is not None and self._debug_image_publisher is not None:
            self._publish_bgr_image(
                self._debug_image_publisher, debug_image, self._image_frame_id, stamp
            )

        waypoints = coords.get("center_waypoints_m_approx") if coords else None
        if not waypoints:
            return

        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id

        for wp in waypoints:
            ground = wp.get("ground_m")
            if ground is None:
                continue
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = ground["x_forward_m"]
            pose.pose.position.y = ground["y_left_m"]
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        if msg.poses:
            self._publisher.publish(msg)

    def close(self):
        self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
