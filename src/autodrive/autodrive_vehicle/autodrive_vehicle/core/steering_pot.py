"""Steering potentiometer calibration: map between the A6 POT reading (ADC
counts) and a steering angle in radians.

No rclpy dependency -- pure math, unit-testable. Shared by:
  - the closed-loop steering command path (desired angle -> target ADC to
    send as `SA`), and
  - the feedback path (measured POT from the firmware's `FB` telemetry ->
    /vehicle/steering_feedback in radians).

Calibration comes from the firmware's CAL sweep (persisted in the Arduino's
EEPROM) and is mirrored here as parameters (see vehicle.yaml):
  - adc_min / adc_max : the two mechanical end-stops (measured: right=210,
    left=710)
  - adc_center        : POT reading with the wheels physically straight
  - max_angle_rad     : steering angle magnitude at an end-stop (PHYSICAL,
    hard to measure exactly -- user estimate ~20 deg (0.35 rad), so radians
    are approximate; tune the parameter later if a better number is found)
  - increase_adc_is_left : whether a HIGHER ADC means a left (positive,
    ROS/CCW) steering angle (measured: yes -- left=710 > right=210)
"""
from dataclasses import dataclass


@dataclass
class SteeringPot:
    adc_min: int = 210
    adc_max: int = 710
    adc_center: int = 460
    max_angle_rad: float = 0.35         # ~20 deg (user estimate, approximate)
    increase_adc_is_left: bool = True

    def _half_span(self) -> float:
        # Symmetric scale about center; adc_center defaults to the geometric
        # mid but is kept separate so a mechanically-offset straight-ahead
        # can be honoured without rescaling the ends.
        return max((self.adc_max - self.adc_min) / 2.0, 1.0)

    def adc_to_angle(self, adc: float) -> float:
        """Measured POT reading -> steering angle (rad), + = left (CCW)."""
        frac = (adc - self.adc_center) / self._half_span()
        angle = frac * self.max_angle_rad
        return angle if self.increase_adc_is_left else -angle

    def angle_to_adc(self, angle: float) -> int:
        """Desired steering angle (rad, + = left) -> target POT reading,
        clamped to the calibrated travel so a too-large request can't drive
        the steering into its mechanical end-stop."""
        signed = angle if self.increase_adc_is_left else -angle
        frac = signed / self.max_angle_rad if self.max_angle_rad else 0.0
        adc = self.adc_center + frac * self._half_span()
        return int(round(self.clamp_adc(adc)))

    def clamp_adc(self, adc: float) -> float:
        return max(self.adc_min, min(self.adc_max, adc))
