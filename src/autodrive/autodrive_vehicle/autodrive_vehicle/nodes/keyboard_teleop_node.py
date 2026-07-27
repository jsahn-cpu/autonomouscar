"""Manual keyboard driving -- talks to the Arduino Mega directly over the
same serial link/protocol as arduino_bridge_node (SerialDriver +
SerialProtocol), commanding raw PWM instead of going through
AckermannDriveStamped/the control-safety-vehicle topic chain.

Bypasses that chain on purpose, not just for convenience: every gain
between physical units (m/s, rad) and PWM is still an unset 0.0 placeholder
in vehicle.yaml (see arduino_bridge_node.py/steering_pid_node.py's TODOs --
the drive/steering motors haven't been characterized yet), so a command
expressed in m/s or radians couldn't produce any real motion anyway. Raw
PWM is what the firmware actually understands, and driving it directly is
also how those gains will eventually get characterized in the first place.

  g : arm (WASD starts affecting the vehicle)
  x : stop and quit (also disarms and sends STOP before disconnecting)
  w : throttle_pwm += throttle_step   s : throttle_pwm -= throttle_step
  a : one steer pulse left            d : one steer pulse right
      (steer_pwm sign for "left" is a guess -- see below -- flip
      steer_step's sign here if the vehicle turns the wrong way)

Throttle and steering are NOT symmetric here, matching serial_protocol's
actual wire semantics: throttle (`M`) is a held absolute PWM value, safe
and expected to be re-sent unchanged every tick (see the timer below), so
w/s adjust a persistent throttle_pwm that's continuously written. Steering
(`ST`) is a ONE-SHOT timed pulse against the Arduino's internal software
position estimate -- re-sending an unchanged pulse would double-count a
movement that never actually repeated (see serial_protocol.py's
docstring), so a/d each fire exactly one pulse per keypress instead of
setting a held state.

ONLY ONE node should hold the Arduino serial port at a time -- run this
INSTEAD OF arduino_bridge_node for manual testing/driving, not alongside
it (both opening the same port will conflict). Swap back to
arduino_bridge_node once ready to drive from the autonomous stack again.

Reads raw keypresses without requiring Enter (termios/tty, POSIX only --
same technique ROS's own teleop_twist_keyboard uses), on a background
thread so the main thread can rclpy.spin() normally. Must run in a real
terminal (not piped/redirected stdin).
"""
import sys
import termios
import threading
import tty
from typing import Optional

import rclpy
from rclpy.node import Node

from autodrive_sensors.drivers.serial_driver import SerialDriver
from autodrive_vehicle.core.serial_protocol import DriveCommand, SerialProtocol, SteerPulseCommand

_INSTRUCTIONS = """\r
keyboard_teleop_node (direct Arduino serial -- see arduino_bridge_node)\r
  g : arm/disarm toggle\r
  x : stop and quit\r
  w/s : throttle +/- step (held)\r
  a/d : one steer pulse left/right\r
(terminal is in raw mode -- no need to press Enter)\r
"""


class KeyboardTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__('keyboard_teleop_node')

        # Same parameter names as arduino_bridge_node's serial_port/baudrate
        # on purpose -- copy the values straight out of vehicle.yaml so this
        # connects to the same Arduino.
        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('throttle_step', 20)          # PWM units per w/s press
        self.declare_parameter('max_throttle_pwm', 60)         # conservative manual-test ceiling
        self.declare_parameter('steer_step', 40)               # PWM units per a/d pulse
        self.declare_parameter('steer_pulse_duration_ms', 150)  # per pulse

        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._throttle_step = self.get_parameter('throttle_step').get_parameter_value().integer_value
        self._max_throttle_pwm = self.get_parameter('max_throttle_pwm').get_parameter_value().integer_value
        self._steer_step = self.get_parameter('steer_step').get_parameter_value().integer_value
        self._steer_pulse_duration_ms = self.get_parameter(
            'steer_pulse_duration_ms').get_parameter_value().integer_value

        self._protocol = SerialProtocol()
        self._driver = SerialDriver(port=serial_port or None, baudrate=baudrate)
        if not self._driver.connect():
            self.get_logger().error(
                f'Failed to open Arduino serial port "{serial_port}" -- '
                'commands will be computed but not sent until it connects.')

        self._lock = threading.Lock()
        self._armed = False
        self._throttle_pwm = 0
        self._quit = False
        self._last_ack = ''

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 0.05
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('keyboard_teleop_node started')

    def _drain_acks_locked(self) -> None:
        """Print anything the Arduino wrote back, so it's possible to tell
        'command reached the firmware but the motor didn't move' (a wiring/
        power problem) apart from 'command never reached the firmware' (a
        serial/port problem) -- writes alone can't distinguish those.
        Dedups identical consecutive OK lines so the throttle ack (sent
        every timer tick) doesn't spam the terminal; ERR lines always print.
        Caller must already hold self._lock (see its call sites) -- all
        self._driver I/O is serialized through that lock since the timer
        (main thread) and handle_key (key-reading thread) would otherwise
        touch the same pyserial object concurrently."""
        while True:
            raw = self._driver.read()
            if raw is None:
                return
            parsed = self._protocol.decode(raw)
            if parsed is None:
                continue
            if not parsed['ok'] or parsed['raw'] != self._last_ack:
                print(f"\r[arduino] {parsed['raw']}\r")
            self._last_ack = parsed['raw']

    def _on_timer(self) -> None:
        with self._lock:
            throttle_pwm = self._throttle_pwm if self._armed else 0
            self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=throttle_pwm)))
            self._drain_acks_locked()

    def handle_key(self, key: str) -> None:
        with self._lock:
            if key == 'g':
                self._armed = not self._armed
                if not self._armed:
                    self._throttle_pwm = 0
                state = 'ARMED' if self._armed else 'disarmed'
                print(f'\r{state} (throttle_pwm={self._throttle_pwm})\r')
                return
            if key == 'x':
                self._armed = False
                self._throttle_pwm = 0
                self._quit = True
                print('\rstopping, quitting...\r')
                return
            if key == 'w':
                self._throttle_pwm = min(self._throttle_pwm + self._throttle_step, self._max_throttle_pwm)
            elif key == 's':
                self._throttle_pwm = max(self._throttle_pwm - self._throttle_step, -self._max_throttle_pwm)
            elif key in ('a', 'd'):
                if not self._armed:
                    print('\rnot armed -- press g first\r')
                    return
                sign = -1 if key == 'a' else 1
                pulse = SteerPulseCommand(
                    steer_pwm=sign * self._steer_step, duration_ms=self._steer_pulse_duration_ms)
                self._driver.write(self._protocol.encode_steer_pulse(pulse))
                print(f'\rsteer pulse {pulse.steer_pwm} for {pulse.duration_ms}ms\r')
                self._drain_acks_locked()
                return
            else:
                return
            armed_note = '' if self._armed else ' (not armed -- press g first)'
            print(f'\rthrottle_pwm={self._throttle_pwm}{armed_note}\r')

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def destroy_node(self) -> bool:
        self._driver.write(self._protocol.encode_stop())
        self._driver.disconnect()
        return super().destroy_node()


def _read_keys_loop(node: KeyboardTeleopNode) -> None:
    """Runs on a background thread -- blocking single-char reads off stdin
    in cbreak mode, forwarded to handle_key() until 'x' or the node itself
    is asked to quit."""
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not node.should_quit:
            key = sys.stdin.read(1)
            node.handle_key(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    print(_INSTRUCTIONS)

    key_thread = threading.Thread(target=_read_keys_loop, args=(node,), daemon=True)
    key_thread.start()

    try:
        while rclpy.ok() and not node.should_quit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # sends STOP + disconnects, see override above
        rclpy.shutdown()
        key_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
