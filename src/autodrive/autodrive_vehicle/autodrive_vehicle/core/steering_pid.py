"""Low-level steering angle PID core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

This is distinct from the high-level LQR in autodrive_control: LQR
computes the desired steering angle, this PID drives the physical
steering motor toward that angle.
"""


class SteeringPID:
    """PID controller: (desired angle, measured angle) -> motor PWM.

    TODO: implement the actual PID math (proportional/integral/derivative
    terms, anti-windup, output clamping).
    """

    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd

    def compute(self, desired_angle: float, measured_angle: float, dt: float) -> float:
        """Return the steering motor PWM command.

        TODO: implement using self._kp/_ki/_kd. Returns 0.0 until implemented.
        """
        return 0.0

    def reset(self) -> None:
        """Reset internal PID state (integral term, previous error).

        TODO: implement once internal state is added.
        """
        pass
