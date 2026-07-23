"""Serial link driver used to talk to the Arduino Mega.

This class intentionally has no ROS dependency so it can be unit tested
or reused outside of rclpy.
"""
import time
from typing import Optional

import serial


class SerialDriver:
    """Thin wrapper around a pyserial connection to the Arduino Mega."""

    def __init__(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: Optional[serial.Serial] = None

    def connect(self) -> bool:
        """Open the serial connection.

        Returns False without raising if `port` is unset or the port can't
        be opened (e.g. not plugged in yet) -- callers are expected to keep
        running and retry rather than crash the node.
        """
        if not self._port:
            return False
        try:
            self._serial = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=0.05)
        except serial.SerialException:
            self._serial = None
            return False

        # The Mega resets on serial open; give its bootloader/setup() time to
        # finish before sending anything, and drop whatever noise it wrote
        # during reset so the first real read isn't stale garbage.
        time.sleep(2.0)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        return True

    def write(self, data: bytes) -> None:
        """Write raw bytes to the serial link. No-op if not connected."""
        if self._serial is None or not self._serial.is_open:
            return
        self._serial.write(data)
        self._serial.flush()

    def read(self) -> Optional[bytes]:
        """Read one newline-terminated line from the serial link, or None if
        nothing is available (the wire protocol is line-based -- see
        autodrive_vehicle.core.serial_protocol)."""
        if self._serial is None or not self._serial.is_open:
            return None
        if self._serial.in_waiting == 0:
            return None
        line = self._serial.readline()
        return line if line else None

    def disconnect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open
