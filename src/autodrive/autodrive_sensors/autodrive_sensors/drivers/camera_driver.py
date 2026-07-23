"""Hardware driver for the front camera.

This class intentionally has no ROS dependency so it can be unit tested
or reused outside of rclpy.
"""
from typing import Any, Optional

import cv2


class CameraDriver:
    """Thin wrapper around a V4L2 camera device via OpenCV VideoCapture."""

    def __init__(
        self,
        device: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._requested_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self._format_ok = False

    def open(self) -> bool:
        """Open the camera device."""
        if not self._device:
            self._is_open = False
            return False

        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, self._requested_fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        self._is_open = cap.isOpened()
        self._cap = cap if self._is_open else None

        # cap.set() can silently no-op (e.g. wrong device, or a mode the
        # camera doesn't support), leaving the driver on its raw fallback
        # format at a fraction of the requested frame rate. Reading back the
        # negotiated fourcc is the only way to tell MJPG actually took.
        self._format_ok = self._is_open and int(cap.get(cv2.CAP_PROP_FOURCC)) == self._requested_fourcc
        return self._is_open

    @property
    def format_ok(self) -> bool:
        """Whether the camera confirmed the requested MJPG fourcc."""
        return self._format_ok

    def read_frame(self) -> Optional[Any]:
        """Read a single BGR frame from the camera, or None on failure."""
        if not self._is_open or self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open
