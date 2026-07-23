"""ROS2 node running low-level steering control.

LQR (autodrive_control) computes the desired steering angle; this node
drives the physical steering motor toward that angle. It only reads its
setpoint from /safety/command (never /control/command directly) so the
safety arbiter/watchdog stays in the loop.

There is no real steering angle sensor installed yet, so this currently runs
SteeringOpenLoop (angle -> pulse, no feedback term) instead of SteeringPID.
/vehicle/steering_feedback is still subscribed and tracked so the wiring is
already in place -- once a real sensor exists, swap in SteeringPID here
(same topics, same node) rather than rebuilding this node's interface.

Publishes /vehicle/steering_pwm (std_msgs/Float32, provisional -- not yet
part of the finalized topic list) for arduino_bridge_node to combine with
the throttle command into serial writes. The pulse duration itself is not
carried on this topic (Float32 has nowhere to put it); both this node and
arduino_bridge_node read the same steer_pulse_duration_ms parameter from
vehicle.yaml so they agree on it without a custom message type.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

from autodrive_vehicle.core.steering_open_loop import SteeringOpenLoop
from autodrive_vehicle.core.steering_pid import SteeringPID


class SteeringPidNode(Node):
    """Subscribes to /safety/command + /vehicle/steering_feedback, drives steering PWM."""

    def __init__(self) -> None:
        super().__init__('steering_pid_node')

        self.declare_parameter('kp', 0.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('angle_to_pwm_gain', 0.0)
        self.declare_parameter('max_steer_pwm', 0)
        self.declare_parameter('steer_pulse_duration_ms', 0)

        kp: float = self.get_parameter('kp').get_parameter_value().double_value
        ki: float = self.get_parameter('ki').get_parameter_value().double_value
        kd: float = self.get_parameter('kd').get_parameter_value().double_value
        angle_to_pwm_gain: float = self.get_parameter(
            'angle_to_pwm_gain').get_parameter_value().double_value
        max_steer_pwm: int = self.get_parameter('max_steer_pwm').get_parameter_value().integer_value
        steer_pulse_duration_ms: int = self.get_parameter(
            'steer_pulse_duration_ms').get_parameter_value().integer_value

        # TODO: not used until /vehicle/steering_feedback carries a real
        # measurement -- kept constructed (rather than added later) so the
        # eventual swap is a one-line change in _on_feedback, not a new node.
        self._pid = SteeringPID(kp=kp, ki=ki, kd=kd)
        self._open_loop = SteeringOpenLoop(
            angle_to_pwm_gain=angle_to_pwm_gain,
            max_steer_pwm=max_steer_pwm,
            pulse_duration_ms=steer_pulse_duration_ms,
        )
        self._desired_angle: Optional[float] = None
        self._measured_angle: Optional[float] = None

        self._safety_command_sub = self.create_subscription(
            AckermannDriveStamped, '/safety/command', self._on_safety_command, 10)
        self._feedback_sub = self.create_subscription(
            Float32, '/vehicle/steering_feedback', self._on_feedback, 10)
        # TODO: provisional output topic, see module docstring.
        self._pwm_pub = self.create_publisher(Float32, '/vehicle/steering_pwm', 10)

        self.get_logger().info('steering_pid_node started (open-loop, no steering feedback sensor yet)')

    def _on_safety_command(self, msg: AckermannDriveStamped) -> None:
        """Update the desired angle and publish the open-loop pulse PWM.

        TODO: once /vehicle/steering_feedback is real, gate this on having a
        recent self._measured_angle and call self._pid.compute(...) instead.
        """
        self._desired_angle = msg.drive.steering_angle
        pulse = self._open_loop.compute(self._desired_angle)
        out = Float32()
        out.data = float(pulse.steer_pwm)
        self._pwm_pub.publish(out)

    def _on_feedback(self, msg: Float32) -> None:
        """Track the measured angle for when SteeringPID replaces the
        open-loop mapping above.

        TODO: unused for control until a real steering feedback sensor is
        installed -- see module docstring.
        """
        self._measured_angle = msg.data


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SteeringPidNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
