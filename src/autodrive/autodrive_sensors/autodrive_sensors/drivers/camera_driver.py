"""Hardware driver for the front camera.

This class intentionally has no ROS dependency so it can be unit tested
or reused outside of rclpy.
"""
import subprocess
from typing import Any, Optional

import cv2


class CameraDriver:
    """Thin wrapper around a V4L2 camera device via OpenCV VideoCapture."""

    def __init__(
        self,
        device: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        buffer_size: int = 4,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        # OpenCV's own default is already 4 on most backends, but that's
        # undocumented/backend-dependent -- set it explicitly rather than
        # rely on it, since a driver buffer of 1 leaves no slot to receive
        # the next USB frame while the current one is still being JPEG-
        # decoded at 1080p, which silently halves the observed frame rate
        # after the stream has been running for a while.
        self._buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._requested_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self._format_ok = False
        self._exposure_fix_ok = False

    def open(self) -> bool:
        """Open the camera device."""
        if not self._device:
            self._is_open = False
            return False

        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, self._requested_fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        self._is_open = cap.isOpened()
        self._cap = cap if self._is_open else None

        # cap.set() can silently no-op (e.g. wrong device, or a mode the
        # camera doesn't support), leaving the driver on its raw fallback
        # format at a fraction of the requested frame rate. Reading back the
        # negotiated fourcc is the only way to tell MJPG actually took.
        self._format_ok = self._is_open and int(cap.get(cv2.CAP_PROP_FOURCC)) == self._requested_fourcc

        if self._is_open:
            self._exposure_fix_ok = self._disable_exposure_dynamic_framerate()

        return self._is_open

    def _disable_exposure_dynamic_framerate(self) -> bool:
        """Best-effort: turn off the C920's exposure_dynamic_framerate
        control, which silently lowers the actual frame rate below
        publish_rate_hz in dim lighting (the camera lengthens its shutter
        instead of raising gain) -- easy to mistake for a USB bandwidth
        problem. Not exposed via any cv2.VideoCapture property, so this
        shells out to v4l2-ctl. Ships ON (=1) from the factory and resets
        to 1 on every USB replug, which is also handled by the udev rule
        in autodrive_bringup/config/udev/ -- this call additionally covers
        the case where that rule isn't installed, or the device was never
        unplugged since boot."""
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', self._device, '--set-ctrl', 'exposure_dynamic_framerate=0'],
                capture_output=True, timeout=2.0,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @property
    def format_ok(self) -> bool:
        """Whether the camera confirmed the requested MJPG fourcc."""
        return self._format_ok

    @property
    def exposure_fix_ok(self) -> bool:
        """Whether exposure_dynamic_framerate=0 was successfully applied."""
        return self._exposure_fix_ok

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
