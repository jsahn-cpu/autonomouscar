"""Fit continuous lane curves from the segmentation class map and publish the
2nd-lane centre path (the target trajectory for the pulse controller).

Pipeline stage after lane_seg_node:

  camera -> lane_seg_node (/perception/lane_seg/class_map) -> THIS NODE
         -> /perception/lane/center_path -> (future) pulse steering controller

For each lane class the model fires only on actual paint (dashes, fragments);
this node fits x=f(y) through the per-row centroids to bridge the gaps into one
continuous curve (see core/lane_curves.py), smooths it across frames, then
takes the midpoint of the dashed (class 2, left boundary of the 2nd lane) and
the right_solid (class 3, right boundary) as the lane centre.

Built to extend: the controller only needs /perception/lane/center_path; adding
left_solid handling, IPM, or a Kalman tracker changes only this node.

Publishes:
  /perception/lane/center_path  std_msgs/Float32MultiArray -- flat [y0,x0,
      y1,x1,...] image-pixel points of the 2nd-lane centre, bottom row first.
      Empty when no lane is found (controller should hold / slow).
  /perception/lane/viz/compressed  sensor_msgs/CompressedImage (jpeg) -- the
      colourised class map + fitted continuous curves + centre path, for debug.
"""
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

from autodrive_perception.core.lane_curves import LaneCurve, fit_curve, lane_center_x

# class id -> (BGR colour for raw pixels, BGR for the fitted continuous line)
_STYLE = {
    1: ((160, 60, 60), (255, 120, 120)),   # left_solid  (blue)
    2: ((60, 160, 60), (120, 255, 120)),    # center_dashed (green)
    3: ((60, 60, 160), (120, 120, 255)),    # right_solid (red)
}


class LaneCurveNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_curve_node')

        self.declare_parameter('class_map_topic', '/perception/lane_seg/class_map/compressed')
        self.declare_parameter('dashed_class', 2)   # left boundary of 2nd lane
        self.declare_parameter('right_class', 3)     # right boundary
        self.declare_parameter('left_class', 1)      # not used for centre, drawn for debug
        self.declare_parameter('degree', 2)
        self.declare_parameter('min_rows', 6)
        self.declare_parameter('roi_top_ratio', 0.35)  # ignore rows above this frac (far/sky)
        self.declare_parameter('ema_alpha', 0.5)        # 1=no smoothing, ->0 = heavier
        self.declare_parameter('max_hold_frames', 5)    # keep last curve this many missed frames
        self.declare_parameter('half_lane_px', 0.0)     # single-boundary fallback (0=off)
        self.declare_parameter('sample_step', 20)       # centre-path row spacing (px)
        self.declare_parameter('publish_viz', True)

        self._dashed = self.get_parameter('dashed_class').get_parameter_value().integer_value
        self._right = self.get_parameter('right_class').get_parameter_value().integer_value
        self._left = self.get_parameter('left_class').get_parameter_value().integer_value
        self._degree = self.get_parameter('degree').get_parameter_value().integer_value
        self._min_rows = self.get_parameter('min_rows').get_parameter_value().integer_value
        self._roi_top_ratio = self.get_parameter('roi_top_ratio').get_parameter_value().double_value
        self._alpha = self.get_parameter('ema_alpha').get_parameter_value().double_value
        self._max_hold = self.get_parameter('max_hold_frames').get_parameter_value().integer_value
        self._half_lane_px = self.get_parameter('half_lane_px').get_parameter_value().double_value
        self._sample_step = self.get_parameter('sample_step').get_parameter_value().integer_value
        self._viz_on = self.get_parameter('publish_viz').get_parameter_value().bool_value
        topic = self.get_parameter('class_map_topic').get_parameter_value().string_value

        # per-class temporal state: smoothed curve + consecutive-miss count
        self._smooth = {}
        self._miss = {}

        self._path_pub = self.create_publisher(Float32MultiArray, '/perception/lane/center_path', 5)
        self._viz_pub = self.create_publisher(
            CompressedImage, '/perception/lane/viz/compressed', 5) if self._viz_on else None
        self._sub = self.create_subscription(CompressedImage, topic, self._on_class_map, 5)

        self.get_logger().info('lane_curve_node started')

    def _update_smooth(self, cid: int, fit: Optional[LaneCurve]) -> Optional[LaneCurve]:
        """EMA-blend a fresh fit with the previous curve; hold the previous one
        for up to max_hold_frames when this frame has no fit, so a single very
        fragmented frame doesn't drop the line."""
        prev = self._smooth.get(cid)
        if fit is None:
            self._miss[cid] = self._miss.get(cid, 0) + 1
            if prev is not None and self._miss[cid] <= self._max_hold:
                return prev
            self._smooth[cid] = None
            return None
        self._miss[cid] = 0
        if prev is not None and len(prev.poly) == len(fit.poly) and 0.0 < self._alpha < 1.0:
            blended = self._alpha * fit.poly + (1.0 - self._alpha) * prev.poly
            fit = LaneCurve(poly=blended, y_min=fit.y_min, y_max=fit.y_max, n_rows=fit.n_rows)
        self._smooth[cid] = fit
        return fit

    def _on_class_map(self, msg: CompressedImage) -> None:
        cm = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if cm is None:
            return
        if cm.ndim == 3:
            cm = cm[:, :, 0]
        h, w = cm.shape
        roi_top = int(round(h * self._roi_top_ratio))

        curves = {}
        for cid in (self._left, self._dashed, self._right):
            mask = cm == cid
            mask[:roi_top] = False
            curves[cid] = self._update_smooth(cid, fit_curve(
                mask, degree=self._degree, min_rows=self._min_rows))

        # 2nd-lane centre = midpoint of dashed (left) and right_solid (right).
        ys = np.arange(h - 1, roi_top, -self._sample_step, dtype=np.float64)
        xs, mode = lane_center_x(
            curves.get(self._dashed), curves.get(self._right), ys,
            half_lane_px=(self._half_lane_px or None))

        path = Float32MultiArray()
        if xs is not None:
            valid = (xs >= 0) & (xs < w)
            pts = np.stack([ys[valid], xs[valid]], axis=1).reshape(-1)
            path.data = [float(v) for v in pts]
        self._path_pub.publish(path)

        if self._viz_pub is not None:
            self._publish_viz(msg.header, cm, curves, ys, xs, mode)

    def _publish_viz(self, header, cm, curves, ys, xs, mode) -> None:
        h, w = cm.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, (raw_c, _) in _STYLE.items():
            canvas[cm == cid] = raw_c
        # fitted continuous curves on top
        for cid, (_, line_c) in _STYLE.items():
            c = curves.get(cid)
            if c is None:
                continue
            yy = np.arange(c.y_min, c.y_max + 1, dtype=np.float64)
            xx = c.x_at(yy)
            pts = np.stack([xx, yy], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, line_c, 2)
        # centre path (yellow)
        if xs is not None:
            v = (xs >= 0) & (xs < w)
            cpts = np.stack([xs[v], ys[v]], axis=1).astype(np.int32)
            cv2.polylines(canvas, [cpts], False, (0, 255, 255), 3)
        cv2.putText(canvas, f'center: {mode}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        ok, buf = cv2.imencode('.jpg', canvas)
        if ok:
            out = CompressedImage()
            out.header = header
            out.format = 'jpeg'
            out.data = buf.tobytes()
            self._viz_pub.publish(out)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LaneCurveNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
