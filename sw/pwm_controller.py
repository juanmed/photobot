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
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from rpi_hardware_pwm import HardwarePWM  # type: ignore[import]
except ImportError:  # not available on non-Pi hardware
    HardwarePWM = None  # type: ignore[assignment,misc]

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
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize hardware PWM channels and transition to running state.

        Creates one HardwarePWM instance per configured channel, initializes
        each at 0% duty cycle. All channels start disabled; call
        enable_channel() to activate.

        Raises:
            RuntimeError: If the controller is already started.
        """
        if self._running:
            raise RuntimeError("Controller is already started.")

        initialized: list[str] = []
        try:
            for cid, cfg in self._channels.items():
                cfg.enabled = False  # channels must start disabled
                pwm = HardwarePWM(pwm_channel=cfg.pwm_channel, hz=cfg.freq_hz, chip=cfg.chip)
                pwm.start(0)
                self._hwpwm[cid] = pwm
                initialized.append(cid)
        except Exception:
            # Clean up already-initialized channels before re-raising.
            self.fail_safe()
            raise

        self._running = True

    def fail_safe(self) -> None:
        """Drive all channels to safe state (duty=0, disabled) without teardown.

        Sets duty cycle to 0% on every initialized HardwarePWM channel and
        marks all channels as disabled. Does NOT call pwm.stop() — the sysfs
        entry is preserved for continued use or clean teardown via stop().

        This method is always safe to call, even during exception handling.
        It logs but does not re-raise per-channel errors.
        """
        for cid, pwm in self._hwpwm.items():
            try:
                pwm.change_duty_cycle(0)
                self._channels[cid].enabled = False
            except Exception as exc:
                logger.error("fail_safe: error on channel '%s': %s", cid, exc)
        self._running = False

    def stop(self) -> None:
        """Drive all channels to safe state and tear down hardware PWM instances.

        Calls fail_safe() first (duty=0, all disabled), then calls pwm.stop()
        on each channel to release the sysfs registration.
        """
        self.fail_safe()
        for cid, pwm in list(self._hwpwm.items()):
            try:
                pwm.stop()
            except Exception as exc:
                logger.error("stop: error stopping channel '%s': %s", cid, exc)
        self._hwpwm.clear()

    def enable_channel(self, channel_id: str) -> None:
        """Mark a channel as active so update() will apply commands to it.

        Args:
            channel_id: The channel to enable.

        Raises:
            RuntimeError: If the controller has not been started.
            ValueError: If channel_id is unknown.
        """
        self._require_started()
        if channel_id not in self._channels:
            raise ValueError(f"Unknown channel_id: '{channel_id}'")
        with self._lock:
            self._channels[channel_id].enabled = True

    def disable_channel(self, channel_id: str) -> None:
        """Write duty=0 immediately and mark the channel inactive.

        The HardwarePWM instance is not torn down; re-enabling and calling
        update() will resume driving the channel.

        Args:
            channel_id: The channel to disable.

        Raises:
            RuntimeError: If the controller has not been started.
            ValueError: If channel_id is unknown.
        """
        self._require_started()
        if channel_id not in self._channels:
            raise ValueError(f"Unknown channel_id: '{channel_id}'")
        pwm = self._hwpwm.get(channel_id)
        if pwm is not None:
            pwm.change_duty_cycle(0)
        with self._lock:
            self._channels[channel_id].enabled = False

    # ------------------------------------------------------------------
    # Public command interface (stubs — fully implemented in later phases)
    # ------------------------------------------------------------------

    def set_duty_cycle(self, channel_id: str, duty_pct: float) -> None:
        """Stage a duty-cycle change for the given channel.

        The change is applied on the next update() call.

        Args:
            channel_id: The channel to update.
            duty_pct: New duty cycle in percent (0.0–100.0, within channel limits).

        Raises:
            RuntimeError: If the controller has not been started.
            ValueError: If channel_id is unknown or duty_pct is out of range.
        """
        self._require_started()
        self._validate_duty(channel_id, duty_pct)
        with self._lock:
            self._pending[channel_id].duty_pct = duty_pct

    def set_frequency(self, channel_id: str, freq_hz: float) -> None:
        """Stage a frequency change for the given channel.

        Frequency changes are only permitted when the channel's
        ``allow_runtime_freq_change`` flag is True.

        WARNING: The sysfs backend momentarily zeroes duty cycle during a
        frequency change, which may disturb BLDC motors. Enable only when
        the application can tolerate this transient glitch.

        Args:
            channel_id: The channel to update.
            freq_hz: New PWM carrier frequency in Hz (within channel limits).

        Raises:
            RuntimeError: If the controller has not been started, or if
                ``allow_runtime_freq_change`` is False for the channel.
            ValueError: If channel_id is unknown or freq_hz is out of range.
        """
        self._require_started()
        self._validate_frequency(channel_id, freq_hz)
        cfg = self._channels[channel_id]
        if not cfg.allow_runtime_freq_change:
            raise RuntimeError(
                f"Runtime frequency changes not allowed for channel '{channel_id}'. "
                "Set allow_runtime_freq_change=True in ChannelConfig to enable."
            )
        logger.warning(
            "set_frequency: channel '%s' frequency change to %.1f Hz will cause a "
            "momentary 0%% duty-cycle glitch on the sysfs backend.",
            channel_id,
            freq_hz,
        )
        with self._lock:
            self._pending[channel_id].freq_hz = freq_hz

    def update(self) -> None:
        """Apply all pending commands to enabled channels sequentially.

        Snapshots and clears the pending command state under the lock, then
        performs hardware I/O outside the lock so that command-staging threads
        are not blocked during sysfs writes.

        Iterates channels in registration order and records total elapsed time.

        On any hardware write failure, immediately invokes fail_safe() and
        raises PWMUpdateError.

        Raises:
            RuntimeError: If the controller has not been started.
            PWMUpdateError: If a hardware write fails on any channel.
        """
        self._require_started()
        t_start = time.monotonic()

        # Snapshot pending commands, enabled flags, and hwpwm references under
        # the lock, then release it before any sysfs I/O.
        with self._lock:
            snapshot: list[tuple[str, _PendingCommand, object]] = []
            for cid, pending in self._pending.items():
                cfg = self._channels[cid]
                if not cfg.enabled:
                    continue
                pwm = self._hwpwm.get(cid)
                if pwm is None:
                    continue
                cmd = _PendingCommand(duty_pct=pending.duty_pct, freq_hz=pending.freq_hz)
                pending.duty_pct = None
                pending.freq_hz = None
                snapshot.append((cid, cmd, pwm))

        # Apply commands to hardware outside the lock.
        for cid, cmd, pwm in snapshot:
            try:
                if cmd.duty_pct is not None:
                    pwm.change_duty_cycle(cmd.duty_pct)
                if cmd.freq_hz is not None:
                    pwm.change_frequency(cmd.freq_hz)
            except Exception as exc:
                self.fail_safe()
                raise PWMUpdateError(cid, exc) from exc

        t_end = time.monotonic()
        with self._lock:
            self._timings.append(t_end - t_start)

    def get_timing_stats(self) -> dict:
        """Return latency statistics for update() calls.

        Returns:
            A dict with keys ``p50_ms``, ``p95_ms``, ``p99_ms``, ``count``.
            Returns ``{"count": 0}`` if no update() calls have been recorded.
        """
        if not self._timings:
            return {"count": 0}
        arr = np.array(self._timings) * 1000.0  # convert seconds → ms
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "count": len(self._timings),
        }
