"""ROS2 node wrapping scan_clusterer for lidar vehicle/slot detection.

Handles message conversion and pub/sub only; the clustering geometry lives
in autodrive_perception.core.scan_clusterer. Subscribes to a 2D LaserScan,
clusters it, gates for parked-car-sized clusters, and finds the pair that
flanks an empty slot -- then publishes everything as a MarkerArray for RViz
so the parameters (ROI, segmentation thresholds, car-size gate, slot gap)
can be tuned against a live scan by eye.

Marker colors (namespace / color):
  clusters (white)      : every raw cluster's oriented box
  vehicles (green)      : clusters that passed the car-size gate
  slot     (cyan sphere): midpoint of the detected flanking pair, if any
"""
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_perception.core.scan_clusterer import (
    cluster_scan, passes_vehicle_gate, find_flanking_pair,
)


class ScanClusterNode(Node):
    """Subscribes to /scan, publishes cluster/vehicle/slot markers."""

    def __init__(self) -> None:
        super().__init__('scan_cluster_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/perception/clusters_viz')

        # ROI box in the LASER frame (x fwd, y left, m). Any of these can be
        # a very large magnitude to effectively disable that side. Default is
        # wide-open -- narrow it once the lidar mounting is fixed so only the
        # parking strip is considered.
        self.declare_parameter('roi_x_min', -12.0)
        self.declare_parameter('roi_x_max', 12.0)
        self.declare_parameter('roi_y_min', -12.0)
        self.declare_parameter('roi_y_max', 12.0)

        # Segmentation: adaptive gap threshold = base + coeff * range (m).
        self.declare_parameter('seg_dist_base', 0.06)
        self.declare_parameter('seg_dist_range_coeff', 0.05)
        self.declare_parameter('min_points', 4)

        # Car-size gate (m) -- LENGTH + point count only, not width (a 2D
        # lidar sees a car as a near-zero-thickness line/L).
        self.declare_parameter('veh_min_length', 0.12)
        self.declare_parameter('veh_max_length', 0.60)
        self.declare_parameter('veh_min_width', 0.0)
        self.declare_parameter('veh_max_width', 1.0)
        self.declare_parameter('veh_min_points', 6)

        # Flanking pair: centroid separation of the two cars bounding a slot.
        self.declare_parameter('slot_gap_min', 0.80)
        self.declare_parameter('slot_gap_max', 1.50)

        self._marker_frame = None  # taken from the scan header

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self._sub = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._pub = self.create_publisher(MarkerArray, marker_topic, 10)

        self.get_logger().info(f'scan_cluster_node started ({scan_topic} -> {marker_topic})')

    def _p(self, name):
        return self.get_parameter(name).get_parameter_value()

    def _on_scan(self, scan: LaserScan) -> None:
        roi = (
            self._p('roi_x_min').double_value, self._p('roi_x_max').double_value,
            self._p('roi_y_min').double_value, self._p('roi_y_max').double_value,
        )
        clusters = cluster_scan(
            scan.angle_min, scan.angle_increment, scan.ranges,
            scan.range_min, scan.range_max, roi=roi,
            seg_dist_base=self._p('seg_dist_base').double_value,
            seg_dist_range_coeff=self._p('seg_dist_range_coeff').double_value,
            min_points=self._p('min_points').integer_value,
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
        pair = find_flanking_pair(
            vehicles,
            gap_min=self._p('slot_gap_min').double_value,
            gap_max=self._p('slot_gap_max').double_value,
        )

        self._publish_markers(scan.header.frame_id, scan.header.stamp, clusters, vehicles, pair)

        if pair is not None:
            a, b = pair
            self.get_logger().info(
                f'2 vehicles + slot: center=({(a.centroid[0]+b.centroid[0])/2:.2f}, '
                f'{(a.centroid[1]+b.centroid[1])/2:.2f}) gap='
                f'{((a.centroid[0]-b.centroid[0])**2+(a.centroid[1]-b.centroid[1])**2)**0.5:.2f}m',
                throttle_duration_sec=1.0)

    def _box_marker(self, frame, stamp, ns, mid, cluster, rgba):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y = cluster.box_center
        m.pose.position.z = 0.0
        half = cluster.yaw / 2.0
        m.pose.orientation.z = float(math.sin(half))
        m.pose.orientation.w = float(math.cos(half))
        # a flat thin box; give it a small min size so near-1D clusters show
        m.scale.x = max(cluster.length, 0.02)
        m.scale.y = max(cluster.width, 0.02)
        m.scale.z = 0.02
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000  # 0.2s -- auto-clear stale markers
        return m

    def _publish_markers(self, frame, stamp, clusters, vehicles, pair):
        arr = MarkerArray()
        # a leading DELETEALL keeps counts from a busier frame from lingering
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for i, c in enumerate(clusters):
            arr.markers.append(self._box_marker(frame, stamp, 'clusters', i, c, (1.0, 1.0, 1.0, 0.5)))
        for i, c in enumerate(vehicles):
            arr.markers.append(self._box_marker(frame, stamp, 'vehicles', i, c, (0.0, 1.0, 0.0, 0.8)))

        if pair is not None:
            a, b = pair
            slot = Marker()
            slot.header.frame_id = frame
            slot.header.stamp = stamp
            slot.ns = 'slot'
            slot.id = 0
            slot.type = Marker.SPHERE
            slot.action = Marker.ADD
            slot.pose.position.x = (a.box_center[0] + b.box_center[0]) / 2.0
            slot.pose.position.y = (a.box_center[1] + b.box_center[1]) / 2.0
            slot.pose.position.z = 0.0
            slot.pose.orientation.w = 1.0
            slot.scale.x = slot.scale.y = slot.scale.z = 0.12
            slot.color.r, slot.color.g, slot.color.b, slot.color.a = (0.0, 1.0, 1.0, 1.0)
            slot.lifetime.nanosec = 200_000_000
            arr.markers.append(slot)

        self._pub.publish(arr)


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
