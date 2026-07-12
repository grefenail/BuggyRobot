"""Optional ROS 2 publisher for the lane centerline waypoints.

rclpy is only imported inside WaypointPublisher.__init__, so importing
this module (and running the rest of the pipeline) doesn't require a
ROS 2 install. Only constructing WaypointPublisher does.
"""

import config


class WaypointPublisher:
    """Publishes each frame's center waypoints as a nav_msgs/Path.

    Robot frame convention (REP-103): x = forward meters, y = left
    meters -- matching pipeline.waypoints.pixel_to_ground_m.
    """

    def __init__(self, topic=config.ROS_DEFAULT_TOPIC, frame_id=config.ROS_DEFAULT_FRAME_ID):
        import rclpy
        from nav_msgs.msg import Path
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._frame_id = frame_id
        self._node = Node("lane_waypoint_publisher")
        self._publisher = self._node.create_publisher(Path, topic, 10)

    def publish(self, coords):
        """Publish coords['center_waypoints_m_approx'] as a Path message.

        Frames with no locked lane (coords has no ground waypoints) are
        skipped rather than publishing an empty/stale Path.
        """
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path

        waypoints = coords.get("center_waypoints_m_approx")
        if not waypoints:
            return

        msg = Path()
        msg.header.stamp = self._node.get_clock().now().to_msg()
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
