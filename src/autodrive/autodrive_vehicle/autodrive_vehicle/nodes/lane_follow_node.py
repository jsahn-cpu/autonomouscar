"""Vision lane-following: /perception/lane/center_path -> steering PULSES.

Closes the loop on the CAMERA (no steering-angle sensor -- the pot broke), so
each control tick it reads the 2nd-lane centre's lateral error at a lookahead
row and fires one open-loop steering pulse (ST) toward it, plus a held
throttle (M). The next camera frame sees the effect and corrects.

Talks to the Arduino directly over serial (same as keyboard_teleop_node) and
so is subject to the same rule: ONLY ONE node may own the port -- run this
INSTEAD OF keyboard_teleop_node / arduino_bridge_node. Requires the open-loop
firmware (mega_motor_controller.ino).

SAFETY: starts DISABLED. Publish std_msgs/Bool true on /lane_follow/enable to
arm; false (or a stale/empty path) stops the drive motor. throttle_pwm
defaults to 0 (steering-only) so you can verify steering before it drives.

  camera -> lane_seg_node -> lane_curve_node (/perception/lane/center_path)
         -> THIS NODE -> Arduino (ST pulses + M throttle)
"""
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray

from autodrive_sensors.drivers.serial_driver import SerialDriver
from autodrive_vehicle.core.serial_protocol import DriveCommand, SerialProtocol, SteerPulseCommand
from autodrive_vehicle.core.lane_follow import compute_steer


class LaneFollowNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_follow_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)
        self.declare_parameter('setpoint_x', -1.0)      # <0 -> image_width/2 (calibrate!)
        self.declare_parameter('lookahead_ratio', 0.75)  # lookahead row = ratio * height
        self.declare_parameter('deadband_px', 40.0)
        self.declare_parameter('steer_pwm', 150)
        self.declare_parameter('steer_min_duration_ms', 80)
        self.declare_parameter('steer_max_duration_ms', 250)
        self.declare_parameter('duration_gain', 0.5)      # ms per px of error beyond deadband
        self.declare_parameter('right_is_negative', True)  # right turn -> negative ST pwm (a=left/+ in teleop)
        self.declare_parameter('throttle_pwm', 0)         # 0 = steering-only (safe default)
        self.declare_parameter('control_rate_hz', 15.0)
        self.declare_parameter('path_timeout_sec', 0.5)

        gp = self.get_parameter
        serial_port = gp('serial_port').get_parameter_value().string_value
        baudrate = gp('baudrate').get_parameter_value().integer_value
        self._w = gp('image_width').get_parameter_value().integer_value
        self._h = gp('image_height').get_parameter_value().integer_value
        sp = gp('setpoint_x').get_parameter_value().double_value
        self._setpoint_x = sp if sp >= 0 else self._w / 2.0
        self._lookahead_y = gp('lookahead_ratio').get_parameter_value().double_value * self._h
        self._deadband = gp('deadband_px').get_parameter_value().double_value
        self._steer_pwm = gp('steer_pwm').get_parameter_value().integer_value
        self._min_dur = gp('steer_min_duration_ms').get_parameter_value().integer_value
        self._max_dur = gp('steer_max_duration_ms').get_parameter_value().integer_value
        self._dur_gain = gp('duration_gain').get_parameter_value().double_value
        self._right_neg = gp('right_is_negative').get_parameter_value().bool_value
        self._throttle = gp('throttle_pwm').get_parameter_value().integer_value
        rate = gp('control_rate_hz').get_parameter_value().double_value
        self._path_timeout = gp('path_timeout_sec').get_parameter_value().double_value

        self._protocol = SerialProtocol()
        self._driver = SerialDriver(port=serial_port or None, baudrate=baudrate)
        if not self._driver.connect():
            self.get_logger().error(
                f'Failed to open Arduino serial port "{serial_port}" -- '
                'commands computed but not sent until it connects.')

        self._enabled = False
        self._points: Optional[np.ndarray] = None
        self._points_stamp = 0.0
        self._steer_busy_until = 0.0   # don't fire a new pulse until the last one finishes

        self.create_subscription(Float32MultiArray, '/perception/lane/center_path', self._on_path, 5)
        self.create_subscription(Bool, '/lane_follow/enable', self._on_enable, 5)
        self._ey_pub = self.create_publisher(Float32, '/control/lane_follow/e_y', 5)

        self._timer = self.create_timer(1.0 / rate if rate > 0 else 0.067, self._on_timer)
        self.get_logger().info(
            f'lane_follow_node started (DISABLED; setpoint_x={self._setpoint_x:.0f}, '
            f'lookahead_y={self._lookahead_y:.0f}, throttle={self._throttle})')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_enable(self, msg: Bool) -> None:
        if msg.data != self._enabled:
            self.get_logger().info(f'lane_follow {"ENABLED" if msg.data else "disabled"}')
        self._enabled = msg.data
        if not self._enabled:
            self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=0)))

    def _on_path(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data, dtype=np.float64)
        self._points = data.reshape(-1, 2) if data.size >= 2 else None
        self._points_stamp = self._now()

    def _on_timer(self) -> None:
        # always drain acks so firmware ERR/OK are visible
        while True:
            raw = self._driver.read()
            if raw is None:
                break
            parsed = self._protocol.decode(raw)
            if parsed and not parsed['ok']:
                self.get_logger().warn(f"[arduino] {parsed['raw']}")

        if not self._enabled:
            self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=0)))
            return

        now = self._now()
        fresh = self._points is not None and (now - self._points_stamp) <= self._path_timeout
        if not fresh:
            # lost the lane -> stop the drive motor, hold steering (no pulse)
            self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=0)))
            return

        # held throttle
        self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=self._throttle)))

        cmd = compute_steer(
            self._points, self._setpoint_x, self._lookahead_y, self._deadband,
            self._steer_pwm, self._min_dur, self._max_dur, self._dur_gain, self._right_neg)

        ey = Float32(); ey.data = float(cmd.e_y if cmd.valid else 0.0)
        self._ey_pub.publish(ey)

        # one pulse at a time: wait for the previous to finish before firing again
        if cmd.valid and cmd.steer_pwm != 0 and now >= self._steer_busy_until:
            self._driver.write(self._protocol.encode_steer_pulse(
                SteerPulseCommand(steer_pwm=cmd.steer_pwm, duration_ms=cmd.duration_ms)))
            self._steer_busy_until = now + cmd.duration_ms / 1000.0

    def destroy_node(self) -> bool:
        self._driver.write(self._protocol.encode_stop())
        self._driver.disconnect()
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LaneFollowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
