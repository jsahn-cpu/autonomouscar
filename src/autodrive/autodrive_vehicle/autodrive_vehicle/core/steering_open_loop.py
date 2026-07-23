"""Open-loop steering angle -> motor pulse mapping.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

There is no real steering angle sensor installed yet, so closed-loop PID
(see steering_pid.py) has no measurement to run on. This maps a desired
angle straight to a steering pulse instead, with no feedback term. It is a
placeholder for SteeringPID, not a replacement for it -- once a real sensor
exists, only the computation swaps (see steering_pid_node.py); the topics
and node wiring stay the same.
"""
from autodrive_vehicle.core.serial_protocol import SteerPulseCommand


class SteeringOpenLoop:
    """Maps a desired steering angle to a fixed-duration steering pulse."""

    def __init__(
        self,
        angle_to_pwm_gain: float = 0.0,
        max_steer_pwm: int = 0,
        pulse_duration_ms: int = 0,
    ) -> None:
        self._angle_to_pwm_gain = angle_to_pwm_gain
        self._max_steer_pwm = max_steer_pwm
        self._pulse_duration_ms = pulse_duration_ms

    def compute(self, desired_angle: float) -> SteerPulseCommand:
        """Return the steering pulse for one desired-angle update.

        TODO: proportional mapping only (steer_pwm = angle * gain, a fixed
        pulse duration); all three constants default to 0 until the
        steering motor is characterized (max safe PWM, angle moved per
        ms-at-a-given-pwm) -- tune once real driving tests are possible.
        """
        pwm = desired_angle * self._angle_to_pwm_gain
        pwm = max(-self._max_steer_pwm, min(self._max_steer_pwm, pwm))
        return SteerPulseCommand(steer_pwm=int(pwm), duration_ms=self._pulse_duration_ms)
