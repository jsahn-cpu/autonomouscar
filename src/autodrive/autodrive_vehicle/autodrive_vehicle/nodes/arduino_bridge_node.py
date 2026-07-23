"""ROS2 node bridging vehicle commands to the Arduino Mega over serial.

This is the ONLY node allowed to talk to the Arduino for driving commands.
It subscribes to /safety/command (throttle) and /vehicle/steering_pwm
(steering, computed open-loop by steering_pid_node until a real steering
feedback sensor exists -- see that node's docstring). It must never
subscribe to /control/command directly, so that autodrive_safety remains the
single arbiter/watchdog gate in front of the actuators.

Drive and steering are sent as independent serial commands as soon as their
respective topic updates, not bundled into one write -- see
autodrive_vehicle.core.serial_protocol for why re-sending an unchanged
steering pulse alongside every throttle update would be wrong (it would
double-count the Arduino's software position estimate for a pulse that
never actually repeated).
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

from autodrive_sensors.drivers.serial_driver import SerialDriver
from autodrive_vehicle.core.serial_protocol import DriveCommand, SerialProtocol, SteerPulseCommand


class ArduinoBridgeNode(Node):
    """Subscribes to /safety/command + /vehicle/steering_pwm, writes to the Arduino Mega."""

    def __init__(self) -> None:
        super().__init__('arduino_bridge_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('speed_to_pwm_gain', 0.0)
        self.declare_parameter('max_throttle_pwm', 0)
        # Must match steering_pid_node's value -- see module docstring on
        # why the pulse duration isn't carried on /vehicle/steering_pwm
        # itself.
        self.declare_parameter('steer_pulse_duration_ms', 0)

        self._serial_port: str = self.get_parameter('serial_port').get_parameter_value().string_value
        self._baudrate: int = self.get_parameter('baudrate').get_parameter_value().integer_value
        self._speed_to_pwm_gain: float = self.get_parameter(
            'speed_to_pwm_gain').get_parameter_value().double_value
        self._max_throttle_pwm: int = self.get_parameter(
            'max_throttle_pwm').get_parameter_value().integer_value
        self._steer_pulse_duration_ms: int = self.get_parameter(
            'steer_pulse_duration_ms').get_parameter_value().integer_value

        self._protocol = SerialProtocol()
        self._driver = SerialDriver(port=self._serial_port or None, baudrate=self._baudrate)
        if not self._driver.connect():
            self.get_logger().error(
                f'Failed to open Arduino serial port "{self._serial_port}" -- '
                'commands will be computed but not sent until it connects.')

        self._safety_command_sub = self.create_subscription(
            AckermannDriveStamped, '/safety/command', self._on_safety_command, 10)
        self._steering_pwm_sub = self.create_subscription(
            Float32, '/vehicle/steering_pwm', self._on_steering_pwm, 10)

        self.get_logger().info('arduino_bridge_node started')

    def _on_safety_command(self, msg: AckermannDriveStamped) -> None:
        """Convert speed (m/s) to throttle PWM and write the drive command.

        TODO: speed_to_pwm_gain/max_throttle_pwm are placeholders (0) until
        the drive motor is characterized against real speed measurements.
        """
        pwm = msg.drive.speed * self._speed_to_pwm_gain
        pwm = max(-self._max_throttle_pwm, min(self._max_throttle_pwm, pwm))
        command = DriveCommand(throttle_pwm=int(pwm))
        self._driver.write(self._protocol.encode_drive(command))

    def _on_steering_pwm(self, msg: Float32) -> None:
        """Write the already-computed steering pulse (see steering_pid_node)."""
        command = SteerPulseCommand(
            steer_pwm=int(msg.data), duration_ms=self._steer_pulse_duration_ms)
        self._driver.write(self._protocol.encode_steer_pulse(command))

    def destroy_node(self) -> bool:
        self._driver.write(self._protocol.encode_stop())
        self._driver.disconnect()
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ArduinoBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
