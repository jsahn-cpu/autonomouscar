"""Wire protocol between ROS and the Arduino Mega.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Line-based ASCII protocol (baud rate is a deployment parameter, not part of
the protocol itself), adapted from a prior single-driver prototype's proven
Arduino-side command set (M/ST/SC/STOP) to this vehicle's Ackermann layout --
one throttle value instead of independent left/right drive motors:

    M <throttle_pwm>              -- drive motor PWM, signed
    ST <steer_pwm> <duration_ms>  -- pulse the steering motor at steer_pwm
                                     for duration_ms, then stop
    SC                            -- reset the Arduino's software-estimated
                                     steering center to 0
    STOP                          -- stop both motors immediately

Drive and steering are sent as separate commands, on their own topics'
schedules (see arduino_bridge_node) -- they are physically independent
motors, and bundling them into one message would mean re-sending a steering
pulse every time an unrelated throttle update arrives, double-counting the
Arduino's software position estimate for a pulse that was never actually
repeated.

There is no real steering angle sensor yet, so "ST" is an open-loop timed
pulse: the Arduino tracks an internal software ESTIMATE of steering position
(not a measurement) purely to clamp against a soft travel limit. See
autodrive_vehicle.core.steering_open_loop for the ROS-side angle->pulse
mapping that replaces closed-loop PID until real feedback exists.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriveCommand:
    """`M` command: signed drive motor PWM."""

    throttle_pwm: int = 0


@dataclass
class SteerPulseCommand:
    """`ST` command: pulse the steering motor at steer_pwm for duration_ms."""

    steer_pwm: int = 0
    duration_ms: int = 0


class SerialProtocol:
    """Encodes ROS-side commands and decodes the Arduino's ack lines."""

    def encode_drive(self, command: DriveCommand) -> bytes:
        return f'M {int(command.throttle_pwm)}\n'.encode('ascii')

    def encode_steer_pulse(self, command: SteerPulseCommand) -> bytes:
        return f'ST {int(command.steer_pwm)} {int(command.duration_ms)}\n'.encode('ascii')

    def encode_center_reset(self) -> bytes:
        return b'SC\n'

    def encode_stop(self) -> bytes:
        return b'STOP\n'

    def decode(self, raw: bytes) -> Optional[dict]:
        """Parse one ack line from the Arduino into a dict, or None if it
        doesn't match a recognized reply.

        Replies observed from the firmware: "OK M <l> <r>", "OK ST <pwm>
        <ms> EST <est>", "OK STEER DONE EST <est>", "OK SC 0", "OK STOP",
        "ERR ...". Returns {'ok': bool, 'raw': str} plus any parsed fields
        (e.g. 'estimate') that are present.
        """
        try:
            text = raw.decode('ascii', errors='ignore').strip()
        except (UnicodeDecodeError, AttributeError):
            return None
        if not text:
            return None

        parsed: dict = {'raw': text, 'ok': text.startswith('OK')}

        if 'EST' in text:
            tail = text.split('EST', 1)[1].strip().split()
            if tail:
                try:
                    parsed['estimate'] = int(tail[0])
                except ValueError:
                    pass

        return parsed
