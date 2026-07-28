"""ROS2 node: count parked cars the vehicle drives PAST, from a 2D lidar.

Handles message conversion and pub/sub only; the geometry lives in
autodrive_perception.core.scan_clusterer / pass_counter. Subscribes to a
LaserScan, clusters it, keeps the car-sized clusters inside the detection
zone (the ROI on the lidar's left), and counts passes: a car fills the zone
then clears it as we drive on -- one ENTER->EXIT cycle = one car. When
target_count cars have passed (the two flanking the empty slot) it flags
parking_ready.

Topics out:
  /perception/pass_count    std_msgs/Int32          running count of cars passed
  /perception/parking_ready std_msgs/Bool           True once count >= target_count
  /perception/zone_viz      visualization_msgs/MarkerArray  RViz overlay:
      - all raw scan points (grey) -- no separate /scan display needed
      - ROI box outline: cyan normally, GREEN while a car occupies the zone
      - green box on each car-sized cluster inside the zone
      - the running count as floating text
"""
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Int32
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_perception.core.scan_clusterer import cluster_scan, passes_vehicle_gate
from autodrive_perception.core.pass_counter import PassCounter


class ScanClusterNode(Node):
    """Subscribes to /scan, publishes the drive-past car count + ready flag."""

    def __init__(self) -> None:
        super().__init__('scan_cluster_node')

        self.declare_parameter('scan_topic', '/scan')

        # ROI box in the LASER frame (x fwd, y left, m) = the pass-count
        # DETECTION ZONE. A car counts as "in the zone" while a car-sized
        # cluster is inside this box, so it MUST be a small strip that's empty
        # when no car is passing -- e.g. the lidar's left within ~1 m (y in
        # [0, 1.0]). Leaving it wide-open makes walls permanently occupy the
        # zone, so the count never advances.
        self.declare_parameter('roi_x_min', -1.5)
        self.declare_parameter('roi_x_max', 1.5)
        self.declare_parameter('roi_y_min', 0.0)
        self.declare_parameter('roi_y_max', 1.0)

        # Safe-distance cutoff (m): ignore returns closer than this (drops the
        # rear-mounted lidar's view of the vehicle's own body).
        self.declare_parameter('min_range', 0.30)

        # Segmentation.
        self.declare_parameter('seg_dist_base', 0.15)
        self.declare_parameter('seg_dist_range_coeff', 0.08)
        self.declare_parameter('min_points', 4)
        self.declare_parameter('max_beam_gap', 6)

        # Car-size gate (m) -- LENGTH + point count (a 2D lidar sees a car as
        # a near-zero-thickness line/L, so width is left loose).
        self.declare_parameter('veh_min_length', 0.30)
        self.declare_parameter('veh_max_length', 1.20)
        self.declare_parameter('veh_min_width', 0.0)
        self.declare_parameter('veh_max_width', 1.0)
        self.declare_parameter('veh_min_points', 20)

        # Pass counting (debounced enter/exit; parking_ready at target_count).
        self.declare_parameter('enter_frames', 3)
        self.declare_parameter('exit_frames', 3)
        self.declare_parameter('target_count', 2)

        self._counter = PassCounter(
            enter_frames=self._p('enter_frames').integer_value,
            exit_frames=self._p('exit_frames').integer_value,
        )
        self._target_count = self._p('target_count').integer_value

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self._sub = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._count_pub = self.create_publisher(Int32, '/perception/pass_count', 10)
        self._ready_pub = self.create_publisher(Bool, '/perception/parking_ready', 10)
        self._viz_pub = self.create_publisher(MarkerArray, '/perception/zone_viz', 10)

        self.get_logger().info(f'scan_cluster_node started (pass counting on {scan_topic})')

    def _p(self, name):
        return self.get_parameter(name).get_parameter_value()

    def _on_scan(self, scan: LaserScan) -> None:
        roi = (
            self._p('roi_x_min').double_value, self._p('roi_x_max').double_value,
            self._p('roi_y_min').double_value, self._p('roi_y_max').double_value,
        )
        range_min = max(scan.range_min, self._p('min_range').double_value)
        clusters = cluster_scan(
            scan.angle_min, scan.angle_increment, scan.ranges,
            range_min, scan.range_max, roi=roi,
            seg_dist_base=self._p('seg_dist_base').double_value,
            seg_dist_range_coeff=self._p('seg_dist_range_coeff').double_value,
            min_points=self._p('min_points').integer_value,
            max_beam_gap=self._p('max_beam_gap').integer_value,
        )
        vehicles = [
            c for c in clusters if passes_vehicle_gate(
                c,
                min_length=self._p('veh_min_length').double_value,
                max_length=self._p('veh_max_length').double_value,
                min_width=self._p('veh_min_width').double_value,
                max_width=self._p('veh_max_width').double_value,
                min_points=self._p('veh_min_points').integer_value,
            )
        ]

        occupied = len(vehicles) > 0
        prev_count = self._counter.count
        count = self._counter.update(occupied)

        self._count_pub.publish(Int32(data=count))
        ready = count >= self._target_count
        self._ready_pub.publish(Bool(data=ready))

        if count != prev_count:
            self.get_logger().info(
                f'car passed -> count={count}/{self._target_count}'
                + ('  PARKING READY' if ready else ''))

        self._publish_viz(scan, roi, vehicles, occupied, count)

    def _publish_viz(self, scan, roi, vehicles, occupied, count):
        frame, stamp = scan.header.frame_id, scan.header.stamp
        x0, x1, y0, y1 = roi
        arr = MarkerArray()
        clear = Marker(); clear.header.frame_id = frame; clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        # Full raw scan as grey points (so /scan doesn't need its own display).
        pts = Marker()
        pts.header.frame_id = frame
        pts.header.stamp = stamp
        pts.ns = 'scan'
        pts.id = 0
        pts.type = Marker.POINTS
        pts.action = Marker.ADD
        pts.scale.x = pts.scale.y = 0.02  # point size (m)
        pts.color.r, pts.color.g, pts.color.b, pts.color.a = (0.6, 0.6, 0.6, 1.0)
        pts.pose.orientation.w = 1.0
        amin, ainc = scan.angle_min, scan.angle_increment
        rmax = scan.range_max
        for i, rng in enumerate(scan.ranges):
            if not (0.0 < rng <= rmax) or math.isinf(rng) or math.isnan(rng):
                continue
            a = amin + i * ainc
            pts.points.append(self._pt(rng * math.cos(a), rng * math.sin(a)))
        arr.markers.append(pts)

        # Lidar position: a red sphere at the origin + a label, so the scan
        # is easy to orient (everything is in the laser frame, so the lidar
        # itself sits at (0, 0)).
        lidar = Marker()
        lidar.header.frame_id = frame
        lidar.header.stamp = stamp
        lidar.ns = 'lidar'
        lidar.id = 0
        lidar.type = Marker.SPHERE
        lidar.action = Marker.ADD
        lidar.pose.orientation.w = 1.0
        lidar.scale.x = lidar.scale.y = lidar.scale.z = 0.12
        lidar.color.r, lidar.color.g, lidar.color.b, lidar.color.a = (1.0, 0.0, 0.0, 1.0)
        arr.markers.append(lidar)
        lidar_txt = Marker()
        lidar_txt.header.frame_id = frame
        lidar_txt.header.stamp = stamp
        lidar_txt.ns = 'lidar'
        lidar_txt.id = 1
        lidar_txt.type = Marker.TEXT_VIEW_FACING
        lidar_txt.action = Marker.ADD
        lidar_txt.pose.position.z = 0.15
        lidar_txt.pose.orientation.w = 1.0
        lidar_txt.scale.z = 0.12
        lidar_txt.color.r = lidar_txt.color.g = lidar_txt.color.b = lidar_txt.color.a = 1.0
        lidar_txt.text = 'LIDAR'
        arr.markers.append(lidar_txt)

        # ROI box outline -- green while occupied, cyan when clear.
        box = Marker()
        box.header.frame_id = frame
        box.header.stamp = stamp
        box.ns = 'zone'
        box.id = 0
        box.type = Marker.LINE_STRIP
        box.action = Marker.ADD
        box.scale.x = 0.02  # line width
        box.color.r, box.color.g, box.color.b, box.color.a = (
            (0.0, 1.0, 0.0, 1.0) if occupied else (0.0, 1.0, 1.0, 0.8))
        box.pose.orientation.w = 1.0
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        box.points = [self._pt(cx, cy) for cx, cy in corners]
        arr.markers.append(box)

        # a green filled box on each car-sized cluster in the zone
        for i, c in enumerate(vehicles):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'vehicles'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y = c.box_center
            half = c.yaw / 2.0
            m.pose.orientation.z = float(math.sin(half))
            m.pose.orientation.w = float(math.cos(half))
            m.scale.x = max(c.length, 0.05)
            m.scale.y = max(c.width, 0.05)
            m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 1.0, 0.0, 0.6)
            m.lifetime.nanosec = 300_000_000
            arr.markers.append(m)

        # the running count as floating text above the zone
        txt = Marker()
        txt.header.frame_id = frame
        txt.header.stamp = stamp
        txt.ns = 'count'
        txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = (x0 + x1) / 2.0
        txt.pose.position.y = (y0 + y1) / 2.0
        txt.pose.position.z = 0.3
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.2  # text height
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f'count {count}/{self._target_count}'
        arr.markers.append(txt)

        self._viz_pub.publish(arr)

    @staticmethod
    def _pt(x, y):
        p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
        return p


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ScanClusterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
