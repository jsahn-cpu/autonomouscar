"""ROS2 node wrapping a single camera driver (front or rear).

Publishes the raw camera stream. No perception logic lives here -- see
autodrive_perception for BEV/local-map processing.

The vehicle has two cameras (front, rear); this same node/executable runs as
two separate instances (see camera.yaml's camera_front_node/camera_rear_node
sections and core.launch.py), distinguished by the `camera_name` parameter,
which picks both the topic namespace and the default frame_id.
"""
import os
from typing import Optional

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage

from autodrive_sensors.drivers.camera_driver import CameraDriver


def _load_camera_info(path: str, frame_id: str) -> Optional[CameraInfo]:
    """Parse a ROS-style camera_info YAML file into a CameraInfo message."""
    if not path or not os.path.isfile(path):
        return None

    with open(path, 'r') as f:
        calib = yaml.safe_load(f)

    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = calib['image_width']
    info.height = calib['image_height']
    info.distortion_model = calib['distortion_model']
    info.d = [float(v) for v in calib['distortion_coefficients']['data']]
    info.k = [float(v) for v in calib['camera_matrix']['data']]
    info.r = [float(v) for v in calib['rectification_matrix']['data']]
    info.p = [float(v) for v in calib['projection_matrix']['data']]
    return info


class CameraNode(Node):
    """Publishes /camera/{camera_name}/image/compressed and
    /camera/{camera_name}/camera_info, where camera_name is "front" or "rear".

    Optionally also publishes /camera/{camera_name}/image_mono/compressed
    (grayscale) alongside the color stream -- consumers that don't need
    color (e.g. lane_detector_node, which converts to grayscale as its very
    first step anyway) can subscribe to the mono topic instead and skip a
    same-sized color JPEG decode + BGR2GRAY conversion. This is additive:
    the color topic keeps publishing unchanged for consumers that do need
    color (e.g. the still-unimplemented traffic light detector)."""

    def __init__(self) -> None:
        super().__init__('camera_node')

        self.declare_parameter('camera_name', 'front')
        self.declare_parameter('device', '')
        self.declare_parameter('frame_id', 'camera_front_optical_frame')
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)
        self.declare_parameter('camera_info_file', '')
        self.declare_parameter('publish_mono', False)
        # Republish the camera's MJPG frames straight to the compressed topic
        # with no decode+re-encode (see CameraDriver's passthrough). Much
        # faster (a real 30fps MJPG stream drops to ~10fps through the
        # decode/encode path) and no double-JPEG quality loss. Off by default
        # since it makes NO decoded BGR frame available -- so publish_mono
        # can't work with it, and any in-process consumer needing raw pixels
        # would have to decode the compressed topic itself.
        self.declare_parameter('passthrough', False)

        camera_name: str = self.get_parameter('camera_name').get_parameter_value().string_value
        self._frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value
        self._publish_mono: bool = self.get_parameter('publish_mono').get_parameter_value().bool_value
        self._passthrough: bool = self.get_parameter('passthrough').get_parameter_value().bool_value
        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value

        if self._passthrough and self._publish_mono:
            self.get_logger().warn(
                'passthrough=true has no decoded frame to build the mono topic from '
                '-- disabling publish_mono.')
            self._publish_mono = False

        self._device: str = self.get_parameter('device').get_parameter_value().string_value
        self._driver = CameraDriver(
            device=self._device or None, width=width, height=height, passthrough=self._passthrough)
        self._open_and_check_driver()

        # Consecutive read_frame() failures (USB disconnect/protocol error,
        # not just an occasional dropped frame) trigger a close+reopen --
        # without this, a camera that drops off the bus (see camera
        # stability notes: xHCI errors -> UVC probe -110/-71 -> USB
        # disconnect) never recovers even after the OS re-enumerates it,
        # since read_frame() on a dead cv2.VideoCapture just keeps
        # returning None forever.
        self._consecutive_failures = 0
        self._reconnect_failure_threshold = 10

        camera_info_file: str = self.get_parameter('camera_info_file').get_parameter_value().string_value
        camera_info_path = ''
        if camera_info_file:
            bringup_share = get_package_share_directory('autodrive_bringup')
            camera_info_path = os.path.join(bringup_share, 'config', camera_info_file)
        self._camera_info = _load_camera_info(camera_info_path, self._frame_id)
        if self._camera_info is None:
            self.get_logger().warn(f'No camera_info loaded from "{camera_info_path}"')

        self._image_pub = self.create_publisher(
            CompressedImage, f'/camera/{camera_name}/image/compressed', 10)
        self._camera_info_pub = self.create_publisher(
            CameraInfo, f'/camera/{camera_name}/camera_info', 10)
        self._mono_pub = (
            self.create_publisher(CompressedImage, f'/camera/{camera_name}/image_mono/compressed', 10)
            if self._publish_mono else None
        )

        # Fire the read timer FASTER than the camera's frame rate (2x) so the
        # blocking cap.read() is what actually paces the loop. A timer exactly
        # AT the frame rate leaves no margin for the callback's own work
        # (publish, etc.), so the driver buffer slowly backs up and the
        # observed rate drifts down over time (measured: starts ~30, decays).
        # read() blocks until the next frame regardless, so the extra timer
        # wakeups are cheap and just keep the buffer drained.
        read_rate_hz = publish_rate_hz * 2.0 if publish_rate_hz > 0.0 else 60.0
        self._timer = self.create_timer(1.0 / read_rate_hz, self._on_timer)

        self.get_logger().info(f'camera_node started (camera_name={camera_name})')

    def _open_and_check_driver(self) -> None:
        if not self._driver.open():
            self.get_logger().error(f'Failed to open camera device: "{self._device}"')
            return
        if not self._driver.format_ok:
            self.get_logger().error(
                f'Camera "{self._device}" did not accept the requested MJPG format -- it is likely '
                'the wrong device or does not support MJPG at this resolution, and will fall '
                'back to a raw format at a much lower frame rate than publish_rate_hz.')
        if not self._driver.exposure_fix_ok:
            self.get_logger().warn(
                f'Camera "{self._device}": could not disable exposure_dynamic_framerate (v4l2-ctl '
                'missing, wrong device, or camera does not support the control) -- frame rate may '
                'silently drop below publish_rate_hz in dim lighting.')

    def _handle_read_failure(self) -> None:
        """Count a failed read; after enough in a row, close+reopen the
        device (recovers from a USB drop the OS has since re-enumerated)."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._reconnect_failure_threshold:
            self.get_logger().error(
                f'Camera "{self._device}": {self._consecutive_failures} consecutive read '
                'failures -- reopening device.')
            self._driver.close()
            self._open_and_check_driver()
            self._consecutive_failures = 0

    def _on_timer(self) -> None:
        """Capture a frame and publish it as a CompressedImage + CameraInfo."""
        stamp = self.get_clock().now().to_msg()

        if self._passthrough:
            # Republish the camera's own JPEG bytes directly -- no decode, no
            # re-encode, no mono (see the passthrough param).
            jpeg = self._driver.read_jpeg()
            if jpeg is None:
                self._handle_read_failure()
                return
            self._consecutive_failures = 0
            image_msg = CompressedImage()
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self._frame_id
            image_msg.format = 'jpeg'
            image_msg.data = jpeg
            self._image_pub.publish(image_msg)
            if self._camera_info is not None:
                self._camera_info.header.stamp = stamp
                self._camera_info_pub.publish(self._camera_info)
            return

        frame = self._driver.read_frame()
        if frame is None:
            self._handle_read_failure()
            return
        self._consecutive_failures = 0

        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            return

        image_msg = CompressedImage()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self._frame_id
        image_msg.format = 'jpeg'
        image_msg.data = encoded.tobytes()
        self._image_pub.publish(image_msg)

        if self._mono_pub is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, encoded_gray = cv2.imencode('.jpg', gray)
            if ok:
                mono_msg = CompressedImage()
                mono_msg.header.stamp = stamp
                mono_msg.header.frame_id = self._frame_id
                mono_msg.format = 'mono8; jpeg compressed'
                mono_msg.data = encoded_gray.tobytes()
                self._mono_pub.publish(mono_msg)

        if self._camera_info is not None:
            self._camera_info.header.stamp = stamp
            self._camera_info_pub.publish(self._camera_info)

    def destroy_node(self) -> bool:
        self._driver.close()
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
