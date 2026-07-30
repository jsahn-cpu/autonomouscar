"""Wire protocol between ROS and the Arduino Mega.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Line-based ASCII protocol (baud rate is a deployment parameter, not part of
the protocol itself). Matches mega_steer_closed_loop.ino -- keep the two in
sync if either changes.

    M <throttle_pwm>     -- drive motor PWM, signed (both drive motors)
    SA <target_adc>      -- steer to a target POT reading (CLOSED LOOP: the
                            Arduino holds this setpoint via its own P loop
                            on the A6 potentiometer, so the host sends a
                            target, not a PWM)
    SC                   -- steer to center
    SH                   -- release steering (motor off, loop disengaged)
    CAL                  -- auto-sweep both mechanical ends, store the
                            calibration (min/max/sign) in the Arduino's EEPROM
    STOP                 -- stop everything + release steering

Steering used to be an open-loop timed pulse ("ST <pwm> <ms>") because no
angle sensor existed; a potentiometer on A6 now measures the real steering
angle, so the loop closed and moved INTO the firmware. The host no longer
computes steering PWM -- it just names a target (see steering_pot.py for the
ROS-side angle<->ADC mapping). encode_steer_pulse is kept only for the legacy
open-loop firmware (mega_motor_controller.ino).

Telemetry: the closed-loop firmware streams "FB <pot> <target> <steer_pwm>
<engaged>" at ~50 Hz. decode() parses that (and the OK/EVENT/SUMMARY/CFG
ack lines) so whichever node owns the serial port can republish the measured
POT as /vehicle/steering_feedback.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriveCommand:
    """`M` command: signed drive motor PWM."""

    throttle_pwm: int = 0


@dataclass
class SteerAngleCommand:
    """`SA` command: closed-loop steering setpoint as a target POT reading."""

    target_adc: int = 0


@dataclass
class SteerPulseCommand:
    """`ST` command (LEGACY open-loop firmware only): pulse the steering
    motor at steer_pwm for duration_ms. Unused by the closed-loop path."""

    steer_pwm: int = 0
    duration_ms: int = 0


class SerialProtocol:
    """Encodes ROS-side commands and decodes the Arduino's reply/telemetry lines."""

    def encode_drive(self, command: DriveCommand) -> bytes:
        return f'M {int(command.throttle_pwm)}\n'.encode('ascii')

    def encode_steer_angle(self, command: SteerAngleCommand) -> bytes:
        return f'SA {int(command.target_adc)}\n'.encode('ascii')

    def encode_steer_center(self) -> bytes:
        return b'SC\n'

    def encode_steer_hold_off(self) -> bytes:
        return b'SH\n'

    def encode_calibrate(self) -> bytes:
        return b'CAL\n'

    def encode_steer_pulse(self, command: SteerPulseCommand) -> bytes:
        """LEGACY open-loop firmware only (mega_motor_controller.ino)."""
        return f'ST {int(command.steer_pwm)} {int(command.duration_ms)}\n'.encode('ascii')

    def encode_stop(self) -> bytes:
        return b'STOP\n'

    def decode(self, raw: bytes) -> Optional[dict]:
        """Parse one line from the Arduino into a dict, or None if empty.

        Closed-loop firmware streams telemetry "FB <pot> <target> <pwm>
        <engaged>"; also emits OK.../ERR.../EVENT.../SUMMARY.../CFG... lines.
        Returns {'raw': str, 'ok': bool} plus, for an FB line, the parsed
        fields under 'fb' (True) with 'pot'/'target'/'steer_pwm'/'engaged'.
        Legacy 'EST <n>' is still parsed into 'estimate'.
        """
        try:
            text = raw.decode('ascii', errors='ignore').strip()
        except (UnicodeDecodeError, AttributeError):
            return None
        if not text:
            return None

        parsed: dict = {'raw': text, 'ok': text.startswith('OK')}

        if text.startswith('FB '):
            fields = text.split()
            if len(fields) >= 5:
                try:
                    parsed.update(
                        fb=True,
                        pot=int(fields[1]),
                        target=int(fields[2]),
                        steer_pwm=int(fields[3]),
                        engaged=bool(int(fields[4])),
                    )
                except ValueError:
                    pass
            return parsed

        if 'EST' in text:  # legacy open-loop firmware ack
            tail = text.split('EST', 1)[1].strip().split()
            if tail:
                try:
                    parsed['estimate'] = int(tail[0])
                except ValueError:
                    pass

        return parsed
