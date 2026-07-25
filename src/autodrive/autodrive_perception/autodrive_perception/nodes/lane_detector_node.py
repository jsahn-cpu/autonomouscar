"""ROS2 node wrapping LaneDetector.

Handles message conversion and pub/sub only; the actual masking/clustering
logic lives in autodrive_perception.core.lane_detector.
"""
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

from autodrive_perception.core.lane_detector import LaneDetector
from autodrive_perception.core.lane_tracker import LaneTracker
from autodrive_perception.core.reference_lane import ReferenceLaneBuilder


class LaneDetectorNode(Node):
    """Subscribes to the raw front camera feed and publishes a lane debug image."""

    def __init__(self) -> None:
        super().__init__('lane_detector_node')

        self.declare_parameter('adaptive_block_size', 151)
        self.declare_parameter('adaptive_c', -15.0)
        self.declare_parameter('blur_kernel_size', 5)
        self.declare_parameter('hough_threshold', 30)
        self.declare_parameter('hough_min_line_length', 20)
        self.declare_parameter('hough_max_line_gap', 8)
        self.declare_parameter('hough_scale', 0.75)
        self.declare_parameter('angle_tol_deg', 8.0)
        self.declare_parameter('collinear_dist_tol_px', 25.0)
        self.declare_parameter('max_segments', 500)
        self.declare_parameter('roi_top_ratio', 0.45)
        self.declare_parameter('roi_top_width_ratio', 0.5)
        self.declare_parameter('roi_bottom_width_ratio', 1.0)
        self.declare_parameter('roi_left_ratio', 0.0)
        self.declare_parameter('filter_outside_track_boundary', True)
        self.declare_parameter('min_boundary_span_ratio', 0.3)
        self.declare_parameter('min_line_aspect_ratio', 3.0)
        self.declare_parameter('min_line_length_px', 60.0)
        self.declare_parameter('corner_tol_px', 30.0)
        self.declare_parameter('smoothing_min_confirm_frames', 3)
        self.declare_parameter('smoothing_max_missed_frames', 3)
        self.declare_parameter('kalman_process_noise', 4.0)
        self.declare_parameter('kalman_measurement_noise', 25.0)
        self.declare_parameter('kalman_initial_velocity_variance', 400.0)
        self.declare_parameter('kalman_gating_mahalanobis', 3.0)
        self.declare_parameter('parallel_angle_tol_deg', 10.0)
        self.declare_parameter('spacing_tol_ratio', 0.35)
        self.declare_parameter('min_consistent_group_size', 2)
        self.declare_parameter('min_projection_slope', 0.15)
        self.declare_parameter('max_abs_x_ratio', 1.5)
        self.declare_parameter('lane_offset_px', 0.0)
        self.declare_parameter('reference_poly_degree', 2)
        self.declare_parameter('reference_num_samples', 20)
        self.declare_parameter('publish_pipeline_debug', False)
        self.declare_parameter('pipeline_debug_scale', 0.5)
        self.declare_parameter('crop_debug_to_roi', True)

        self._lane_detector = LaneDetector(
            adaptive_block_size=self.get_parameter(
                'adaptive_block_size').get_parameter_value().integer_value,
            adaptive_c=self.get_parameter('adaptive_c').get_parameter_value().double_value,
            blur_kernel_size=self.get_parameter('blur_kernel_size').get_parameter_value().integer_value,
            hough_threshold=self.get_parameter('hough_threshold').get_parameter_value().integer_value,
            hough_min_line_length=self.get_parameter('hough_min_line_length').get_parameter_value().integer_value,
            hough_max_line_gap=self.get_parameter('hough_max_line_gap').get_parameter_value().integer_value,
            hough_scale=self.get_parameter('hough_scale').get_parameter_value().double_value,
            angle_tol_deg=self.get_parameter('angle_tol_deg').get_parameter_value().double_value,
            collinear_dist_tol_px=self.get_parameter('collinear_dist_tol_px').get_parameter_value().double_value,
            max_segments=self.get_parameter('max_segments').get_parameter_value().integer_value,
            roi_top_ratio=self.get_parameter('roi_top_ratio').get_parameter_value().double_value,
            roi_top_width_ratio=self.get_parameter('roi_top_width_ratio').get_parameter_value().double_value,
            roi_bottom_width_ratio=self.get_parameter(
                'roi_bottom_width_ratio').get_parameter_value().double_value,
            roi_left_ratio=self.get_parameter('roi_left_ratio').get_parameter_value().double_value,
            filter_outside_track_boundary=self.get_parameter(
                'filter_outside_track_boundary').get_parameter_value().bool_value,
            min_boundary_span_ratio=self.get_parameter(
                'min_boundary_span_ratio').get_parameter_value().double_value,
            min_line_aspect_ratio=self.get_parameter(
                'min_line_aspect_ratio').get_parameter_value().double_value,
            min_line_length_px=self.get_parameter(
                'min_line_length_px').get_parameter_value().double_value,
            corner_tol_px=self.get_parameter('corner_tol_px').get_parameter_value().double_value,
        )
        # Smooths flicker and rejects one-off outliers: LaneDetector analyzes
        # each frame independently, so a real, continuous lane line can
        # still flicker frame to frame, and an unrelated marking can look
        # exactly as line-like in a single frame. LaneTracker Kalman-tracks
        # each line's position and only reports one once it has matched an
        # established (or newly forming) track for several frames -- see
        # lane_tracker.py for what this can and can't protect against.
        self._lane_tracker = LaneTracker(
            roi_top_ratio=self.get_parameter('roi_top_ratio').get_parameter_value().double_value,
            min_confirm_frames=self.get_parameter(
                'smoothing_min_confirm_frames').get_parameter_value().integer_value,
            max_missed_frames=self.get_parameter(
                'smoothing_max_missed_frames').get_parameter_value().integer_value,
            process_noise=self.get_parameter(
                'kalman_process_noise').get_parameter_value().double_value,
            measurement_noise=self.get_parameter(
                'kalman_measurement_noise').get_parameter_value().double_value,
            initial_velocity_variance=self.get_parameter(
                'kalman_initial_velocity_variance').get_parameter_value().double_value,
            gating_mahalanobis=self.get_parameter(
                'kalman_gating_mahalanobis').get_parameter_value().double_value,
            parallel_angle_tol_deg=self.get_parameter(
                'parallel_angle_tol_deg').get_parameter_value().double_value,
            spacing_tol_ratio=self.get_parameter(
                'spacing_tol_ratio').get_parameter_value().double_value,
            min_consistent_group_size=self.get_parameter(
                'min_consistent_group_size').get_parameter_value().integer_value,
            min_projection_slope=self.get_parameter(
                'min_projection_slope').get_parameter_value().double_value,
            max_abs_x_ratio=self.get_parameter(
                'max_abs_x_ratio').get_parameter_value().double_value,
        )
        # Right-boundary-relative target curve for lane keeping -- see
        # reference_lane.py for why "right boundary" is an assumption
        # (rightmost tracked line, no real solid/dashed label to check
        # against) and why lane_offset_px is a fixed pixel value rather than
        # a real metric distance (no camera calibration yet).
        self._reference_lane = ReferenceLaneBuilder(
            lane_offset_px=self.get_parameter('lane_offset_px').get_parameter_value().double_value,
            poly_degree=self.get_parameter(
                'reference_poly_degree').get_parameter_value().integer_value,
            num_samples=self.get_parameter(
                'reference_num_samples').get_parameter_value().integer_value,
        )
        self._publish_pipeline_debug: bool = self.get_parameter(
            'publish_pipeline_debug').get_parameter_value().bool_value
        self._pipeline_debug_scale: float = self.get_parameter(
            'pipeline_debug_scale').get_parameter_value().double_value
        self._crop_debug_to_roi: bool = self.get_parameter(
            'crop_debug_to_roi').get_parameter_value().bool_value
        self._bridge = CvBridge()

        # Depth 1 + drop-old-on-overflow: if processing falls behind the
        # camera's publish rate, always work on the latest frame instead of
        # queueing up stale ones, which would otherwise make the visible lag
        # grow over time instead of staying bounded.
        # Mono, not the color topic -- detect() converts to grayscale as its
        # very first step regardless, so subscribing to color here would
        # mean decoding a same-resolution color JPEG (~3x the color-decode +
        # BGR2GRAY cost, measured) just to immediately throw the color away.
        # camera_node only publishes this topic when its publish_mono
        # parameter is enabled (see camera.yaml).
        self._image_sub = self.create_subscription(
            CompressedImage, '/camera/front/image_mono/compressed', self._on_image, 1)
        # Published as CompressedImage (JPEG), not a raw sensor_msgs/Image --
        # an uncompressed 960x540 BGR frame is ~1.5MB vs a few hundred KB
        # compressed, and that gap in per-message serialization/transport
        # cost was the main reason this topic visibly lagged behind the raw
        # camera feed (which is already published compressed) even though
        # detect() itself is fast.
        self._debug_pub = self.create_publisher(CompressedImage, '/perception/lane_debug/image/compressed', 1)
        # PNG, not JPEG -- this is a binary (0/255) mask, and JPEG's lossy
        # compression puts ringing artifacts along the sharp edges that make
        # up the entire image, which would make it look less binary than it
        # actually is for what's purely a debug view.
        self._mask_pub = self.create_publisher(CompressedImage, '/perception/lane_mask/compressed', 1)
        # Flattened [x_near_0, x_far_0, x_near_1, x_far_1, ...], one pair per
        # confirmed line, sorted ascending by x_near (left to right in
        # image_width pixels at the ROI's near/far reference rows -- see
        # LaneTracker). This is NOT a lateral offset / lane-center estimate:
        # picking which of these lines actually bounds the vehicle's own
        # lane is a separate, still-unsolved problem (see lane_tracker.py's
        # _select_evenly_spaced limitations) -- a consumer needs to do that
        # selection itself for now rather than trust an index here.
        self._lines_pub = self.create_publisher(Float32MultiArray, '/perception/lane_lines', 1)
        # Flattened [x_0, y_0, x_1, y_1, ...], near to far, in IMAGE PIXEL
        # coordinates -- NOT the same thing as /planning/reference_path
        # (world-frame nav_msgs/Path from the full localization+planning
        # stack). This is a direct camera-space target curve, closer to
        # visual servoing than a planned path. See reference_lane.py.
        self._reference_pub = self.create_publisher(Float32MultiArray, '/perception/lane_reference', 1)
        # Off by default -- see LaneDetector.draw_pipeline_debug(). Meant for
        # re-tuning sessions (e.g. after the camera's mounting position
        # changes), not normal operation: composing+encoding a 2x2 tiled
        # image on top of everything else costs real CPU this node doesn't
        # otherwise spend.
        self._pipeline_debug_pub = (
            self.create_publisher(
                CompressedImage, '/perception/lane_pipeline_debug/image/compressed', 1)
            if self._publish_pipeline_debug else None
        )

        self.get_logger().info('lane_detector_node started')

    def _on_image(self, msg: CompressedImage) -> None:
        # Not desired_encoding='mono8': cv_bridge's compressed-image decoder
        # always assumes the JPEG it just decoded is 3-channel bgr8 and
        # unconditionally runs a bgr8->mono8 cvtColor on it, regardless of
        # what the message's format string says -- our mono JPEG decodes to
        # a genuinely 1-channel array, which that hardcoded conversion then
        # rejects ("Invalid number of channels ... scn is 1"). passthrough
        # skips that conversion and returns cv2.imdecode's result as-is,
        # which is already a plain 2D grayscale array for this topic.
        gray_image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        detections = self._lane_detector.detect(gray_image)

        mask = self._lane_detector.last_mask
        if mask is not None:
            mask_for_pub = self._lane_detector.crop_to_roi(mask) if self._crop_debug_to_roi else mask
            mask_msg = self._bridge.cv2_to_compressed_imgmsg(mask_for_pub, dst_format='png')
            mask_msg.header.stamp = msg.header.stamp
            mask_msg.header.frame_id = msg.header.frame_id
            self._mask_pub.publish(mask_msg)

        confirmed = self._lane_tracker.update(
            detections, image_height=gray_image.shape[0], image_width=gray_image.shape[1])
        # draw_debug() annotates in color (green lines, blue ROI outline),
        # so the mono frame needs a channel dimension back -- this doesn't
        # recover any color information, it just gives cv2.line() somewhere
        # to put non-gray pixels.
        debug_base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        debug_image = self._lane_detector.draw_debug(debug_base, confirmed)

        reference = self._reference_lane.build(confirmed)
        if reference is not None:
            pts = reference.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(debug_image, [pts], isClosed=False, color=(0, 0, 255), thickness=3)

        # Crop-and-zoom to the ROI, not a perspective/BEV transform (that
        # was dropped for this project -- see LaneDetector.crop_to_roi()).
        # Applied last, after all annotations are drawn, so the crop+resize
        # carries them along instead of needing separately-adjusted
        # coordinates for a cropped canvas.
        if self._crop_debug_to_roi:
            debug_image = self._lane_detector.crop_to_roi(debug_image)

        debug_msg = self._bridge.cv2_to_compressed_imgmsg(debug_image, dst_format='jpg')
        debug_msg.header.stamp = msg.header.stamp
        debug_msg.header.frame_id = msg.header.frame_id
        self._debug_pub.publish(debug_msg)

        lines_by_x = sorted(confirmed, key=lambda det: det['x_near'])
        lines_msg = Float32MultiArray()
        for det in lines_by_x:
            lines_msg.data.append(det['x_near'])
            lines_msg.data.append(det['x_far'])
        self._lines_pub.publish(lines_msg)

        reference_msg = Float32MultiArray()
        if reference is not None:
            reference_msg.data = reference.flatten().tolist()
        self._reference_pub.publish(reference_msg)

        if self._pipeline_debug_pub is not None:
            pipeline_debug = self._lane_detector.draw_pipeline_debug(
                debug_base, confirmed, crop_to_roi=self._crop_debug_to_roi)
            if self._pipeline_debug_scale != 1.0:
                pipeline_debug = cv2.resize(
                    pipeline_debug, None,
                    fx=self._pipeline_debug_scale, fy=self._pipeline_debug_scale,
                    interpolation=cv2.INTER_AREA)
            pipeline_msg = self._bridge.cv2_to_compressed_imgmsg(pipeline_debug, dst_format='jpg')
            pipeline_msg.header.stamp = msg.header.stamp
            pipeline_msg.header.frame_id = msg.header.frame_id
            self._pipeline_debug_pub.publish(pipeline_msg)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
