"""Vision lane-following control law: 2nd-lane centre path -> steering pulse.

Steering is open-loop pulses (the potentiometer broke), so there is NO
steering-angle feedback -- the CAMERA is the feedback. Each control tick we
read the lateral error e_y of the lane centre at a lookahead row and fire ONE
steering pulse toward it; the next frame sees the effect and corrects. A
deadband keeps it from twitching when already centred.

No rclpy dependency -- pure logic, unit-testable. The node (lane_follow_node)
wraps this and sends the pulse over serial.

e_y sign: lane-centre-x minus the setpoint (where the lane centre sits in the
image when the vehicle is correctly positioned; default = image centre, but
calibrate it -- camera mount is not perfectly centred). e_y > 0 => lane centre
is to the RIGHT of the setpoint => steer RIGHT.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SteerCommand:
    valid: bool          # False if no usable path point
    e_y: float           # lateral error at the lookahead row (px)
    steer_pwm: int       # signed pulse PWM; 0 when inside the deadband
    duration_ms: int     # pulse duration (0 when no pulse)


def lookahead_x(points: np.ndarray, lookahead_y: float) -> Optional[float]:
    """x of the centre path at the row nearest lookahead_y. points: (N,2) [y,x]
    (as published on /perception/lane/center_path). None if empty."""
    if points is None or len(points) == 0:
        return None
    ys = points[:, 0]
    return float(points[np.argmin(np.abs(ys - lookahead_y)), 1])


def compute_steer(
    points: np.ndarray,
    setpoint_x: float,
    lookahead_y: float,
    deadband_px: float,
    steer_pwm: int,
    min_duration_ms: int,
    max_duration_ms: int,
    duration_gain: float,      # ms added per px of error beyond the deadband
    right_is_negative: bool,   # does a RIGHT turn need a negative ST pwm?
) -> SteerCommand:
    """One control tick: lateral error -> a single steering pulse (or none)."""
    x = lookahead_x(points, lookahead_y)
    if x is None:
        return SteerCommand(valid=False, e_y=0.0, steer_pwm=0, duration_ms=0)

    e_y = x - setpoint_x
    if abs(e_y) <= deadband_px:
        return SteerCommand(valid=True, e_y=e_y, steer_pwm=0, duration_ms=0)

    # direction: e_y>0 -> steer right. right pulse sign is -1 if right_is_negative.
    right_sign = -1 if right_is_negative else 1
    pwm = int(right_sign * (1 if e_y > 0 else -1) * abs(steer_pwm))

    over = abs(e_y) - deadband_px
    dur = int(min(max_duration_ms, max(min_duration_ms, min_duration_ms + duration_gain * over)))
    return SteerCommand(valid=True, e_y=e_y, steer_pwm=pwm, duration_ms=dur)
