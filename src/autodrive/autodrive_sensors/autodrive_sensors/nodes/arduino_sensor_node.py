"""ROS2 node publishing generic Arduino-side sensor/status data.

Note on message choice: /vehicle/status is not in the standard message
table, so this skeleton uses std_msgs/String as a placeholder status
payload. Revisit once the real status fields are defined.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from autodrive_sensors.drivers.serial_driver import SerialDriver


class ArduinoSensorNode(Node):
    """Publishes /vehicle/status (std_msgs/String, placeholder)."""

    def __init__(self) -> None:
        super().__init__('arduino_sensor_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate_hz', 10.0)

        serial_port: str = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate: int = self.get_parameter('baudrate').get_parameter_value().integer_value
        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self._driver = SerialDriver(port=serial_port or None, baudrate=baudrate)

        self._status_pub = self.create_publisher(String, '/vehicle/status', 10)

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('arduino_sensor_node started')

    def _on_timer(self) -> None:
        """Read Arduino sensor/status data and publish it.

        TODO: parse the actual Arduino sensor payload via self._driver and
        publish a meaningful vehicle status message.
        """
        raw: Optional[bytes] = self._driver.read()
        if raw is None:
            return
        # TODO: decode raw payload and publish String.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ArduinoSensorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
