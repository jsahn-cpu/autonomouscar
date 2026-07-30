"""Real-time lane-line segmentation inference (trained PIDNetLite).

Runs the model trained on label/lane_dataset (ml/training) on the front
camera feed and publishes a per-pixel class map:

    0 background   1 left_solid   2 center_dashed   3 right_solid

Downstream (the 2nd-lane follower) uses classes 2 (dashed = left boundary of
the 2nd lane) and 3 (right_solid = right boundary) to build the target path.

Preprocessing MUST match ml/training/dataset.py exactly or the model sees a
different distribution than it trained on:
  - image kept **BGR** (dataset.py reads cv2 BGR and does NOT convert to RGB)
  - resize to image_size (H,W) with INTER_AREA
  - float / 255.0, CHW, no mean/std normalisation
The model in eval mode returns only main_logits; argmax over the class axis
is the class map, upsampled back to the frame size (INTER_NEAREST).

Requires PyTorch in the ROS Python env (rclpy's interpreter) -- it is NOT a
rosdep dependency. Install torch for that interpreter (CPU build is fine if
the GPU is unavailable; the model is tiny). Device defaults to CUDA when
available, else CPU.

Publishes:
  /perception/lane_seg/class_map/compressed  CompressedImage (png, single
      channel, pixel value = class id 0-3) -- the machine-readable output.
  /perception/lane_seg/overlay/compressed    CompressedImage (jpeg) -- the
      class map colour-blended on the frame, for RViz/rqt debugging only.
"""
import sys
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import torch

from autodrive_perception.core.pidnet import PIDNetLite

# class id -> BGR colour for the debug overlay
_CLASS_BGR = {
    1: (255, 0, 0),    # left_solid  -> blue
    2: (0, 255, 0),    # center_dashed -> green
    3: (0, 0, 255),    # right_solid -> red
}


class LaneSegNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_seg_node')

        self.declare_parameter('checkpoint', '')  # path to best.pt (required)
        self.declare_parameter('input_topic', '/camera/front/image/compressed')
        self.declare_parameter('image_size', [288, 512])   # [H, W] -- MUST match train.yaml
        self.declare_parameter('num_classes', 4)
        self.declare_parameter('base_channels', 32)
        self.declare_parameter('crop_bottom_fraction', 1.0)  # match train.yaml
        self.declare_parameter('device', 'auto')             # auto|cuda|cpu
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('overlay_alpha', 0.5)

        ckpt_path = self.get_parameter('checkpoint').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        size = list(self.get_parameter('image_size').get_parameter_value().integer_array_value)
        self._h, self._w = int(size[0]), int(size[1])
        num_classes = self.get_parameter('num_classes').get_parameter_value().integer_value
        base_channels = self.get_parameter('base_channels').get_parameter_value().integer_value
        self._crop_bottom = self.get_parameter('crop_bottom_fraction').get_parameter_value().double_value
        dev = self.get_parameter('device').get_parameter_value().string_value
        self._publish_overlay = self.get_parameter('publish_overlay').get_parameter_value().bool_value
        self._alpha = self.get_parameter('overlay_alpha').get_parameter_value().double_value

        if not ckpt_path:
            self.get_logger().error('checkpoint param is empty -- set it to the trained best.pt')

        self._device = torch.device(
            ('cuda' if torch.cuda.is_available() else 'cpu') if dev == 'auto' else dev)

        self._model = PIDNetLite(num_classes=num_classes, base_channels=base_channels)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self._device)
            state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(state)
        self._model.eval().to(self._device)

        self._class_pub = self.create_publisher(
            CompressedImage, '/perception/lane_seg/class_map/compressed', 5)
        self._overlay_pub = self.create_publisher(
            CompressedImage, '/perception/lane_seg/overlay/compressed', 5) if self._publish_overlay else None

        self._sub = self.create_subscription(CompressedImage, input_topic, self._on_image, 5)

        self.get_logger().info(
            f'lane_seg_node started (device={self._device}, in={self._h}x{self._w}, '
            f'ckpt={"set" if ckpt_path else "MISSING"})')

    def _preprocess(self, bgr: np.ndarray):
        """Frame -> (batched CHW tensor, crop_top row). Matches dataset.py."""
        crop_top = 0
        if self._crop_bottom < 1.0:
            crop_top = int(round(bgr.shape[0] * (1.0 - self._crop_bottom)))
            bgr = bgr[crop_top:, :]
        resized = cv2.resize(bgr, (self._w, self._h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0  # BGR, (3,H,W)
        return t.unsqueeze(0).to(self._device), crop_top

    @torch.no_grad()
    def _on_image(self, msg: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if bgr is None:
            return
        full_h, full_w = bgr.shape[:2]
        x, crop_top = self._preprocess(bgr)

        logits = self._model(x)                       # eval -> main_logits (1,C,h,w)
        cls_small = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()  # (h,w)

        # Back to full frame size. If bottom-cropped, the cropped-off top rows
        # are background (0), so pad them back to keep coords aligned to the frame.
        crop_h = full_h - crop_top
        cls_crop = cv2.resize(cls_small, (full_w, crop_h), interpolation=cv2.INTER_NEAREST)
        class_map = np.zeros((full_h, full_w), dtype=np.uint8)
        class_map[crop_top:, :] = cls_crop

        ok, buf = cv2.imencode('.png', class_map)
        if ok:
            out = CompressedImage()
            out.header = msg.header
            out.format = 'png'
            out.data = buf.tobytes()
            self._class_pub.publish(out)

        if self._overlay_pub is not None:
            overlay = bgr.copy()
            color = np.zeros_like(bgr)
            for cid, bgr_c in _CLASS_BGR.items():
                color[class_map == cid] = bgr_c
            m = class_map > 0
            overlay[m] = (self._alpha * color[m] + (1 - self._alpha) * overlay[m]).astype(np.uint8)
            ok, buf = cv2.imencode('.jpg', overlay)
            if ok:
                o = CompressedImage()
                o.header = msg.header
                o.format = 'jpeg'
                o.data = buf.tobytes()
                self._overlay_pub.publish(o)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LaneSegNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
