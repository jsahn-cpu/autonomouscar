"""Counts objects that pass a detection zone, by ENTER-then-EXIT cycles.

For the parking mission the vehicle drives forward alongside the row of
parked cars with the lidar watching a small zone on its left. Rather than
trying to see both cars + the slot at once, we just count passes: a car
fills the zone (ENTER), then clears it as we drive on (EXIT) -- one full
ENTER->EXIT cycle = one car. Two cars counted means we've driven past the
two that flank the empty slot, i.e. we're positioned to start reversing.

No rclpy dependency -- pure state machine, unit-testable. The caller decides
each frame whether the zone is occupied (e.g. any car-sized cluster inside
the ROI) and feeds that boolean in; this debounces it and counts.
"""


class PassCounter:
    """Debounced enter/exit pass counter.

    Debounce: a candidate presence must hold for enter_frames consecutive
    frames before the zone is considered OCCUPIED, and an absence for
    exit_frames before it's considered CLEAR again -- so an object
    flickering at the zone edge (one noisy frame) can't add a phantom
    count. The count increments on each OCCUPIED->CLEAR transition (the
    moment a car has fully passed)."""

    def __init__(self, enter_frames: int = 3, exit_frames: int = 3) -> None:
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.count = 0
        self.occupied = False
        self._present_streak = 0
        self._absent_streak = 0

    def update(self, detected: bool) -> int:
        """Feed this frame's zone-occupied boolean; returns the pass count."""
        if detected:
            self._present_streak += 1
            self._absent_streak = 0
        else:
            self._absent_streak += 1
            self._present_streak = 0

        if not self.occupied and self._present_streak >= self.enter_frames:
            self.occupied = True
        elif self.occupied and self._absent_streak >= self.exit_frames:
            self.occupied = False
            self.count += 1  # a car has fully entered then exited the zone
        return self.count

    def reset(self) -> None:
        self.count = 0
        self.occupied = False
        self._present_streak = 0
        self._absent_streak = 0
