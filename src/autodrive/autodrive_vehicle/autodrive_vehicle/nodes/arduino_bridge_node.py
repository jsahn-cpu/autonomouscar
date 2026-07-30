"""ROS2 node bridging vehicle commands to the Arduino Mega over serial.

This is the ONLY node allowed to talk to the Arduino for driving commands in
the autonomous stack. It subscribes to /safety/command (throttle + steering
angle) so autodrive_safety stays the single arbiter/watchdog gate in front of
the actuators -- it must never subscribe to /control/command directly.

Steering is CLOSED-LOOP in the firmware now (mega_steer_closed_loop.ino, POT
on A6): this node maps the desired steering ANGLE (rad, from LQR via safety)
to a target POT reading with SteeringPot and sends it as `SA`. It no longer
receives a pre-computed steering PWM -- steering_pid_node / the /vehicle/
steering_pwm topic are obsolete (the loop moved into the firmware).

Because ONLY ONE node may hold the Arduino serial port, this node -- the port
owner while the autonomous stack drives -- is also what reads the firmware's
FB telemetry and republishes the measured steering angle on
/vehicle/steering_feedback (the old standalone steering_feedback_node can't
open the same port and is deprecated).
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

from autodrive_sensors.drivers.serial_driver import SerialDriver
from autodrive_vehicle.core.serial_protocol import DriveCommand, SerialProtocol, SteerAngleCommand
from autodrive_vehicle.core.steering_pot import SteeringPot


class ArduinoBridgeNode(Node):
    """Subscribes to /safety/command, writes drive + closed-loop steering to the Arduino."""

    def __init__(self) -> None:
        super().__init__('arduino_bridge_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('speed_to_pwm_gain', 0.0)
        self.declare_parameter('max_throttle_pwm', 0)
        # Steering-pot calibration (mirror of the Arduino EEPROM / vehicle.yaml)
        self.declare_parameter('steer_adc_min', 210)
        self.declare_parameter('steer_adc_max', 710)
        self.declare_parameter('steer_adc_center', 460)
        self.declare_parameter('steer_max_angle_rad', 0.35)
        self.declare_parameter('steer_increase_adc_is_left', True)

        self._serial_port: str = self.get_parameter('serial_port').get_parameter_value().string_value
        self._baudrate: int = self.get_parameter('baudrate').get_parameter_value().integer_value
        self._speed_to_pwm_gain: float = self.get_parameter(
            'speed_to_pwm_gain').get_parameter_value().double_value
        self._max_throttle_pwm: int = self.get_parameter(
            'max_throttle_pwm').get_parameter_value().integer_value

        self._pot = SteeringPot(
            adc_min=self.get_parameter('steer_adc_min').get_parameter_value().integer_value,
            adc_max=self.get_parameter('steer_adc_max').get_parameter_value().integer_value,
            adc_center=self.get_parameter('steer_adc_center').get_parameter_value().integer_value,
            max_angle_rad=self.get_parameter('steer_max_angle_rad').get_parameter_value().double_value,
            increase_adc_is_left=self.get_parameter(
                'steer_increase_adc_is_left').get_parameter_value().bool_value,
        )

        self._protocol = SerialProtocol()
        self._driver = SerialDriver(port=self._serial_port or None, baudrate=self._baudrate)
        if not self._driver.connect():
            self.get_logger().error(
                f'Failed to open Arduino serial port "{self._serial_port}" -- '
                'commands will be computed but not sent until it connects.')

        self._safety_command_sub = self.create_subscription(
            AckermannDriveStamped, '/safety/command', self._on_safety_command, 10)
        self._feedback_pub = self.create_publisher(Float32, '/vehicle/steering_feedback', 10)

        # Drain the firmware's ~50 Hz FB telemetry and republish measured angle.
        self._serial_timer = self.create_timer(0.01, self._drain_serial)

        self.get_logger().info('arduino_bridge_node started (closed-loop steering)')

    def _on_safety_command(self, msg: AckermannDriveStamped) -> None:
        """Write the drive command (speed->PWM) and the closed-loop steering
        target (angle->ADC).

        TODO: speed_to_pwm_gain/max_throttle_pwm are placeholders (0) until
        the drive motor is characterized against real speed measurements.
        """
        pwm = msg.drive.speed * self._speed_to_pwm_gain
        pwm = max(-self._max_throttle_pwm, min(self._max_throttle_pwm, pwm))
        self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=int(pwm))))

        target_adc = self._pot.angle_to_adc(msg.drive.steering_angle)
        self._driver.write(self._protocol.encode_steer_angle(SteerAngleCommand(target_adc=target_adc)))

    def _drain_serial(self) -> None:
        """Republish FB telemetry as the measured steering angle (rad)."""
        while True:
            raw = self._driver.read()
            if raw is None:
                return
            parsed = self._protocol.decode(raw)
            if parsed and parsed.get('fb'):
                out = Float32()
                out.data = float(self._pot.adc_to_angle(parsed['pot']))
                self._feedback_pub.publish(out)

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
