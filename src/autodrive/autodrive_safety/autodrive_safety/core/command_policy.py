"""Command arbitration policy core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Priority (highest first):
  1. Emergency stop
  2. Sensor / localization failure stop
  3. Normal controller command
"""
from dataclasses import dataclass


@dataclass
class VehicleCommand:
    """Minimal vehicle command used by the safety core (mirrors AckermannDriveStamped.drive)."""

    steering_angle: float = 0.0
    speed: float = 0.0


class CommandPolicy:
    """Arbitrates between emergency stop, failure stop, and the controller command.

    TODO: implement the actual stop command (e.g. zero speed, hold
    steering) and any hysteresis/latching behavior for each state.
    """

    def arbitrate(
        self,
        controller_command: VehicleCommand,
        emergency_stop: bool,
        failure_stop: bool,
    ) -> VehicleCommand:
        """Return the arbitrated command to forward to /safety/command.

        TODO: implement the real stop-command construction. Currently
        passes the controller command through unchanged regardless of
        the flags.
        """
        if emergency_stop:
            pass  # TODO: return an emergency-stop command.
        if failure_stop:
            pass  # TODO: return a failure-stop command.
        return controller_command
