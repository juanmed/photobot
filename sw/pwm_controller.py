"""
PWMController — Centralized hardware PWM control for Raspberry Pi 5.

Manages all available hardware PWM channels (0–3) on a configured chip.
Designed for BLDC motor control in a robot arm system.

API contract:
- External scheduler owns the control loop and calls update() each tick.
- Thread-safe for concurrent command submission from multiple threads
  in a single process. Cross-process safety requires the caller to route
  commands through an IPC mechanism (e.g., multiprocessing.Queue).
- Safe state: duty cycle 0% then PWM channel disabled (motor coast).
- Runtime frequency changes are disabled by default due to a momentary
  0% duty glitch in the sysfs backend (see allow_runtime_freq_change).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Per-channel configuration for a single hardware PWM output.

    Attributes:
        channel_id: Unique string identifier for this channel (e.g. "motor_0").
        pwm_channel: Hardware PWM channel number (0–3 on Raspberry Pi 5).
        chip: PWM chip number (configurable; verify against target hardware).
        freq_hz: Initial PWM carrier frequency in Hz.
        duty_pct: Initial duty cycle in percent (0.0–100.0).
        freq_min_hz: Minimum allowed frequency in Hz (inclusive).
        freq_max_hz: Maximum allowed frequency in Hz (inclusive).
        duty_min_pct: Minimum allowed duty cycle in percent (inclusive).
        duty_max_pct: Maximum allowed duty cycle in percent (inclusive).
        allow_runtime_freq_change: If True, set_frequency() is permitted after
            start(). WARNING: the sysfs backend momentarily zeroes duty cycle
            during a frequency change, which may disturb the motor. False by default.
        enabled: Whether this channel participates in update() ticks.
            Channels start disabled; call enable_channel() to activate.
    """

    channel_id: str
    pwm_channel: int
    chip: int
    freq_hz: float
    duty_pct: float
    freq_min_hz: float
    freq_max_hz: float
    duty_min_pct: float
    duty_max_pct: float
    allow_runtime_freq_change: bool = False
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.freq_min_hz <= 0:
            raise ValueError(f"freq_min_hz must be > 0, got {self.freq_min_hz}")
        if self.freq_max_hz < self.freq_min_hz:
            raise ValueError(
                f"freq_max_hz ({self.freq_max_hz}) must be >= freq_min_hz ({self.freq_min_hz})"
            )
        if not (0.0 <= self.duty_min_pct <= 100.0):
            raise ValueError(f"duty_min_pct must be in [0, 100], got {self.duty_min_pct}")
        if not (0.0 <= self.duty_max_pct <= 100.0):
            raise ValueError(f"duty_max_pct must be in [0, 100], got {self.duty_max_pct}")
        if self.duty_max_pct < self.duty_min_pct:
            raise ValueError(
                f"duty_max_pct ({self.duty_max_pct}) must be >= duty_min_pct ({self.duty_min_pct})"
            )
        if self.pwm_channel not in range(4):
            raise ValueError(f"pwm_channel must be 0–3, got {self.pwm_channel}")


@dataclass
class _PendingCommand:
    """Staged command values waiting to be applied on the next update() tick."""

    duty_pct: Optional[float] = None
    freq_hz: Optional[float] = None


class PWMUpdateError(Exception):
    """Raised when a hardware write fails during update().

    Attributes:
        channel_id: The channel that caused the failure.
        cause: The underlying exception.
    """

    def __init__(self, channel_id: str, cause: Exception) -> None:
        self.channel_id = channel_id
        self.cause = cause
        super().__init__(f"PWM update failed on channel '{channel_id}': {cause}")


class PWMController:
    """Centralized hardware PWM controller for Raspberry Pi 5.

    Manages up to 4 hardware PWM channels (0–3) on a single chip.
    All channels are updated sequentially in registration order on each
    update() call; update() is driven by an external scheduler.

    Thread safety:
        Thread-safe for concurrent command submission (set_duty_cycle,
        set_frequency, enable_channel, disable_channel) from multiple threads
        in a single process. Cross-process safety requires external IPC.

    Safe state:
        fail_safe() sets duty cycle to 0% on all channels and marks them
        disabled. It does NOT call pwm.stop() — the sysfs entry is preserved
        for clean teardown. stop() calls fail_safe() then pwm.stop() on each
        channel for final teardown.
    """

    def __init__(self, channels: list[ChannelConfig]) -> None:
        """Initialize the controller with a list of channel configurations.

        Args:
            channels: One ChannelConfig per PWM channel to manage. Must not
                contain duplicate channel_id or pwm_channel values.

        Raises:
            ValueError: On duplicate channel_id or pwm_channel values, or
                empty channel list.
        """
        if not channels:
            raise ValueError("At least one channel must be provided.")

        self._channels: dict[str, ChannelConfig] = {}
        seen_pwm_channels: set[int] = set()

        for cfg in channels:
            if cfg.channel_id in self._channels:
                raise ValueError(f"Duplicate channel_id: '{cfg.channel_id}'")
            if cfg.pwm_channel in seen_pwm_channels:
                raise ValueError(f"Duplicate pwm_channel: {cfg.pwm_channel}")
            self._channels[cfg.channel_id] = cfg
            seen_pwm_channels.add(cfg.pwm_channel)

        self._pending: dict[str, _PendingCommand] = {
            cid: _PendingCommand() for cid in self._channels
        }
        self._hwpwm: dict[str, object] = {}  # channel_id -> HardwarePWM instance
        self._timings: deque[float] = deque(maxlen=10_000)
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_frequency(self, channel_id: str, freq_hz: float) -> None:
        """Raise ValueError if freq_hz is outside the channel's configured range."""
        cfg = self._channels.get(channel_id)
        if cfg is None:
            raise ValueError(f"Unknown channel_id: '{channel_id}'")
        if not (cfg.freq_min_hz <= freq_hz <= cfg.freq_max_hz):
            raise ValueError(
                f"freq_hz {freq_hz} out of range [{cfg.freq_min_hz}, {cfg.freq_max_hz}] "
                f"for channel '{channel_id}'"
            )

    def _validate_duty(self, channel_id: str, duty_pct: float) -> None:
        """Raise ValueError if duty_pct is outside the channel's configured range."""
        cfg = self._channels.get(channel_id)
        if cfg is None:
            raise ValueError(f"Unknown channel_id: '{channel_id}'")
        if not (0.0 <= duty_pct <= 100.0):
            raise ValueError(
                f"duty_pct {duty_pct} out of global range [0.0, 100.0]"
            )
        if not (cfg.duty_min_pct <= duty_pct <= cfg.duty_max_pct):
            raise ValueError(
                f"duty_pct {duty_pct} out of range [{cfg.duty_min_pct}, {cfg.duty_max_pct}] "
                f"for channel '{channel_id}'"
            )

    def _require_started(self) -> None:
        """Raise RuntimeError if the controller has not been started."""
        if not self._running:
            raise RuntimeError(
                "Controller not started. Call start() before issuing commands."
            )

    # ------------------------------------------------------------------
    # Public command interface (stubs — fully implemented in later phases)
    # ------------------------------------------------------------------

    def set_duty_cycle(self, channel_id: str, duty_pct: float) -> None:
        """Stage a duty-cycle change for the given channel.

        Args:
            channel_id: The channel to update.
            duty_pct: New duty cycle in percent (0.0–100.0, within channel limits).

        Raises:
            RuntimeError: If the controller has not been started.
            ValueError: If channel_id is unknown or duty_pct is out of range.
        """
        self._require_started()
