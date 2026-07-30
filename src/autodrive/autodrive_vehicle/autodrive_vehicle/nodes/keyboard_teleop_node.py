"""Manual keyboard driving -- talks to the Arduino Mega directly over the
same serial link/protocol as arduino_bridge_node (SerialDriver +
SerialProtocol), commanding raw drive PWM and a CLOSED-LOOP steering target.

Steering is now closed-loop (mega_steer_closed_loop.ino): a potentiometer on
A6 measures the real steering angle and the Arduino holds whatever target it
was last given. So a/d no longer fire one-shot pulses against a drifting
software estimate -- they nudge a HELD target ADC that the firmware servos
to. That also means the "current steering command" is a real, loggable value
(the target) alongside the measured feedback, which the earlier pulse scheme
could not provide.

Throttle is still a held raw PWM (see serial_protocol's DriveCommand):
uncharacterized speed->PWM gains make m/s meaningless, and raw PWM is what
the firmware understands and how those gains will eventually be measured.

  g : arm/disarm toggle (w/s throttle only affects the vehicle while armed)
  x : stop -- disarms, zeroes throttle, releases steering (SH); NOT quit
  Ctrl+C : quit (sends STOP and disconnects before exiting)
  w : throttle_pwm += throttle_step   s : throttle_pwm -= throttle_step
  a : steer target += steer_step (toward LEFT, higher ADC)
  d : steer target -= steer_step (toward RIGHT, lower ADC)
      (a=left/d=right holds only if increase_adc_is_left; confirm on vehicle)
  f : steer to center (SC)
  k : run steering CAL sweep (auto-measures end-stops into EEPROM)

Publishes /vehicle/steering_feedback (std_msgs/Float32, radians) from the
firmware's FB telemetry -- since ONLY ONE node may hold the Arduino serial
port at a time, the port owner (this node while manually driving) is what
republishes the measured steering angle, not a separate feedback node.

ONLY ONE node should hold the Arduino serial port at a time -- run this
INSTEAD OF arduino_bridge_node for manual testing/driving, not alongside it.

Reads raw keypresses without requiring Enter (termios/tty, POSIX only) on a
background thread so the main thread can rclpy.spin(). Must run in a real
terminal (not piped/redirected stdin).
"""
import sys
import termios
import threading
import tty
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from autodrive_sensors.drivers.serial_driver import SerialDriver
from autodrive_vehicle.core.serial_protocol import DriveCommand, SerialProtocol, SteerAngleCommand
from autodrive_vehicle.core.steering_pot import SteeringPot

_INSTRUCTIONS = """\r
keyboard_teleop_node (direct Arduino serial -- CLOSED-LOOP steering)\r
  g : arm/disarm toggle\r
  x : stop (disarm + release steering) -- g to resume\r
  Ctrl+C : quit\r
  w/s : throttle +/- step (held)\r
  a/d : steer target toward left/right (held, servo'd by firmware)\r
  f : steer to center      k : run CAL sweep (measure end-stops)\r
(terminal is in raw mode -- no need to press Enter)\r
"""


class KeyboardTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__('keyboard_teleop_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('throttle_step', 20)      # PWM units per w/s press
        self.declare_parameter('max_throttle_pwm', 60)   # conservative manual-test ceiling
        self.declare_parameter('steer_step_adc', 25)     # ADC counts per a/d press
        # Steering-pot calibration (mirror of the Arduino EEPROM / vehicle.yaml)
        self.declare_parameter('steer_adc_min', 210)
        self.declare_parameter('steer_adc_max', 710)
        self.declare_parameter('steer_adc_center', 460)
        self.declare_parameter('steer_max_angle_rad', 0.35)
        self.declare_parameter('steer_increase_adc_is_left', True)

        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        publish_rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._throttle_step = self.get_parameter('throttle_step').get_parameter_value().integer_value
        self._max_throttle_pwm = self.get_parameter('max_throttle_pwm').get_parameter_value().integer_value
        self._steer_step = self.get_parameter('steer_step_adc').get_parameter_value().integer_value

        self._pot = SteeringPot(
            adc_min=self.get_parameter('steer_adc_min').get_parameter_value().integer_value,
            adc_max=self.get_parameter('steer_adc_max').get_parameter_value().integer_value,
            adc_center=self.get_parameter('steer_adc_center').get_parameter_value().integer_value,
            max_angle_rad=self.get_parameter('steer_max_angle_rad').get_parameter_value().double_value,
            increase_adc_is_left=self.get_parameter(
                'steer_increase_adc_is_left').get_parameter_value().bool_value,
        )

        self._protocol = SerialProtocol()
        self._driver = SerialDriver(port=serial_port or None, baudrate=baudrate)
        if not self._driver.connect():
            self.get_logger().error(
                f'Failed to open Arduino serial port "{serial_port}" -- '
                'commands will be computed but not sent until it connects.')

        self._feedback_pub = self.create_publisher(Float32, '/vehicle/steering_feedback', 10)

        self._lock = threading.Lock()
        self._armed = False
        self._throttle_pwm = 0
        self._steer_target = self._pot.adc_center  # held closed-loop setpoint
        self._last_ack = ''

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 0.05
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('keyboard_teleop_node started (closed-loop steering)')

    def _drain_serial_locked(self) -> None:
        """Read everything the Arduino wrote: republish FB telemetry as the
        measured steering angle, and print non-FB replies (deduped OK spam)
        so firmware events/errors are visible. Caller holds self._lock."""
        while True:
            raw = self._driver.read()
            if raw is None:
                return
            parsed = self._protocol.decode(raw)
            if parsed is None:
                continue
            if parsed.get('fb'):
                msg = Float32()
                msg.data = float(self._pot.adc_to_angle(parsed['pot']))
                self._feedback_pub.publish(msg)
                continue
            if not parsed['ok'] or parsed['raw'] != self._last_ack:
                print(f"\r[arduino] {parsed['raw']}\r")
            self._last_ack = parsed['raw']

    def _on_timer(self) -> None:
        with self._lock:
            throttle_pwm = self._throttle_pwm if self._armed else 0
            self._driver.write(self._protocol.encode_drive(DriveCommand(throttle_pwm=throttle_pwm)))
            self._drain_serial_locked()

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
                self._driver.write(self._protocol.encode_steer_hold_off())  # release steering
                print('\rstopped (disarmed + steering released) -- g to resume\r')
                return
            if key == 'w':
                self._throttle_pwm = min(self._throttle_pwm + self._throttle_step, self._max_throttle_pwm)
            elif key == 's':
                self._throttle_pwm = max(self._throttle_pwm - self._throttle_step, -self._max_throttle_pwm)
            elif key in ('a', 'd'):
                # a = toward LEFT (higher ADC when increase_adc_is_left), d = right
                step = self._steer_step if key == 'a' else -self._steer_step
                if not self._pot.increase_adc_is_left:
                    step = -step
                self._steer_target = int(self._pot.clamp_adc(self._steer_target + step))
                self._driver.write(self._protocol.encode_steer_angle(
                    SteerAngleCommand(target_adc=self._steer_target)))
                angle_deg = self._pot.adc_to_angle(self._steer_target) * 57.2958
                print(f'\rsteer target={self._steer_target} (~{angle_deg:+.1f} deg)\r')
                self._drain_serial_locked()
                return
            elif key == 'f':
                self._steer_target = self._pot.adc_center
                self._driver.write(self._protocol.encode_steer_center())
                print('\rsteer -> center\r')
                self._drain_serial_locked()
                return
            elif key == 'k':
                self._driver.write(self._protocol.encode_calibrate())
                print('\rCAL sweep started (watch [arduino] SUMMARY)\r')
                self._drain_serial_locked()
                return
            else:
                return
            armed_note = '' if self._armed else ' (not armed -- press g first)'
            print(f'\rthrottle_pwm={self._throttle_pwm}{armed_note}\r')

    def destroy_node(self) -> bool:
        self._driver.write(self._protocol.encode_stop())
        self._driver.disconnect()
        return super().destroy_node()


def _read_keys_loop(node: KeyboardTeleopNode) -> None:
    """Background daemon thread: blocking single-char reads in cbreak mode,
    forwarded to handle_key() for the life of the process (no quit key -- it
    dies with the process on Ctrl+C)."""
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
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
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # sends STOP + disconnects, see override above
        rclpy.shutdown()
        key_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
