"""ROS2 node wrapping scan_clusterer for lidar vehicle detection + pass
counting.

Handles message conversion and pub/sub only; the geometry lives in
autodrive_perception.core.scan_clusterer / pass_counter. Subscribes to a 2D
LaserScan, clusters it, gates for parked-car-sized clusters, and:
  - publishes a MarkerArray for RViz tuning, and
  - counts cars PASSING the detection zone (the ROI) as the vehicle drives
    alongside the row -- each car that fills then clears the zone is one
    pass (see PassCounter). When the count reaches target_count (2, the two
    cars flanking the empty slot) it publishes a Bool trigger.

Marker colors (namespace / color):
  clusters (white)  : every raw cluster's oriented box
  vehicles (green)  : clusters that passed the car-size gate (= zone occupied)

Topics out:
  <marker_topic>            visualization_msgs/MarkerArray
  /perception/pass_count    std_msgs/Int32   (running count of cars passed)
  /perception/parking_ready std_msgs/Bool    (True once count >= target)
"""
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Int32
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_perception.core.scan_clusterer import cluster_scan, passes_vehicle_gate
from autodrive_perception.core.pass_counter import PassCounter


class ScanClusterNode(Node):
    """Subscribes to /scan, publishes cluster/vehicle/slot markers."""

    def __init__(self) -> None:
        super().__init__('scan_cluster_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/perception/clusters_viz')

        # ROI box in the LASER frame (x fwd, y left, m) -- this doubles as the
        # PASS-COUNT DETECTION ZONE: a car counts as "in the zone" when a
        # car-sized cluster is inside this box. For the drive-past mission set
        # it to the small strip on the lidar's left within ~1 m (e.g. y in
        # [0, 1.0]) so each passing car fills then clears it.
        self.declare_parameter('roi_x_min', -12.0)
        self.declare_parameter('roi_x_max', 12.0)
        self.declare_parameter('roi_y_min', -12.0)
        self.declare_parameter('roi_y_max', 12.0)

        # Safe-distance cutoff (m): ignore returns closer than this. The
        # rear-mounted lidar sees the vehicle's own body a few cm away, so a
        # min-range cutoff cleanly drops all self-returns without needing a
        # precise ROI box. Applied on top of the scan's own range_min.
        self.declare_parameter('min_range', 0.30)

        # Segmentation: adaptive gap threshold = base + coeff * range (m).
        self.declare_parameter('seg_dist_base', 0.06)
        self.declare_parameter('seg_dist_range_coeff', 0.05)
        self.declare_parameter('min_points', 4)
        # Allow up to this many dropped beams inside one cluster before
        # breaking -- a rounded/glossy car reflects nothing for a beam or
        # two mid-surface, which was splitting it into halves.
        self.declare_parameter('max_beam_gap', 6)

        # Car-size gate (m) -- LENGTH + point count only, not width (a 2D
        # lidar sees a car as a near-zero-thickness line/L).
        self.declare_parameter('veh_min_length', 0.12)
        self.declare_parameter('veh_max_length', 0.60)
        self.declare_parameter('veh_min_width', 0.0)
        self.declare_parameter('veh_max_width', 1.0)
        self.declare_parameter('veh_min_points', 6)

        # Pass counting: a car must occupy the zone for enter_frames in a row
        # before it counts as present, and be gone for exit_frames before it
        # counts as passed (debounce against edge flicker). parking_ready
        # fires when target_count cars have passed.
        self.declare_parameter('enter_frames', 3)
        self.declare_parameter('exit_frames', 3)
        self.declare_parameter('target_count', 2)

        self._counter = PassCounter(
            enter_frames=self._p('enter_frames').integer_value,
            exit_frames=self._p('exit_frames').integer_value,
        )
        self._target_count = self._p('target_count').integer_value

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self._sub = self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self._count_pub = self.create_publisher(Int32, '/perception/pass_count', 10)
        self._ready_pub = self.create_publisher(Bool, '/perception/parking_ready', 10)

        self.get_logger().info(f'scan_cluster_node started ({scan_topic} -> {marker_topic})')

    def _p(self, name):
        return self.get_parameter(name).get_parameter_value()

    def _on_scan(self, scan: LaserScan) -> None:
        roi = (
            self._p('roi_x_min').double_value, self._p('roi_x_max').double_value,
            self._p('roi_y_min').double_value, self._p('roi_y_max').double_value,
        )
        # Raise the scan's range_min to the safe-distance cutoff so the
        # vehicle's own body (a few cm behind the rear-mounted lidar) and
        # other near clutter are dropped before clustering.
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
        # A car occupies the detection zone whenever a car-sized cluster is
        # present (clusters are already ROI-restricted). Feed that to the
        # debounced pass counter.
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

        self._publish_markers(scan.header.frame_id, scan.header.stamp, clusters, vehicles)

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

    def _publish_markers(self, frame, stamp, clusters, vehicles):
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
