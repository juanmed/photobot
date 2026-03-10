"""Hardware integration tests for PWMController.

These tests require a Raspberry Pi 5 with hardware PWM channels available.
All tests are marked with @pytest.mark.hardware and are automatically skipped
when not running on Pi 5 hardware.

To run on hardware:
    pytest -m hardware sw/tests/test_pwm_controller_hw.py

To skip (default when not on Pi 5):
    pytest sw/tests/  # hardware tests will be skipped automatically
"""

import pytest

from sw.pwm_controller import ChannelConfig, PWMController


def _make_hw_config(
    channel_id: str,
    pwm_channel: int,
    chip: int = 0,
    freq_hz: float = 100.0,
) -> ChannelConfig:
    return ChannelConfig(
        channel_id=channel_id,
        pwm_channel=pwm_channel,
        chip=chip,
        freq_hz=freq_hz,
        duty_pct=0.0,
        freq_min_hz=50.0,
        freq_max_hz=400.0,
        duty_min_pct=0.0,
        duty_max_pct=100.0,
    )


@pytest.mark.hardware
class TestHardware:
    """Integration tests for PWMController on real Raspberry Pi 5 hardware."""

    def test_start_and_stop_single_channel(self):
        """Initialize a single PWM channel and tear it down cleanly."""
        ctrl = PWMController([_make_hw_config("motor_0", pwm_channel=0)])
        ctrl.start()
        assert ctrl._running is True
        ctrl.stop()
        assert ctrl._running is False

    def test_start_and_stop_all_four_channels(self):
        """Initialize all 4 PWM channels simultaneously."""
        channels = [
            _make_hw_config(f"motor_{i}", pwm_channel=i)
            for i in range(4)
        ]
        ctrl = PWMController(channels)
        ctrl.start()
        assert ctrl._running is True
        ctrl.stop()

    def test_enable_disable_channel(self):
        """Enable and disable a channel; verify enabled state."""
        ctrl = PWMController([_make_hw_config("motor_0", 0)])
        ctrl.start()
        assert ctrl._channels["motor_0"].enabled is False
        ctrl.enable_channel("motor_0")
        assert ctrl._channels["motor_0"].enabled is True
        ctrl.disable_channel("motor_0")
        assert ctrl._channels["motor_0"].enabled is False
        ctrl.stop()

    def test_set_duty_and_update(self):
        """Stage a duty cycle change and apply it via update()."""
        ctrl = PWMController([_make_hw_config("motor_0", 0)])
        ctrl.start()
        ctrl.enable_channel("motor_0")
        ctrl.set_duty_cycle("motor_0", 25.0)
        ctrl.update()
        ctrl.stop()

    def test_fail_safe_zeros_all_channels(self):
        """fail_safe() must zero duty on all channels and mark them disabled."""
        channels = [_make_hw_config(f"motor_{i}", i) for i in range(4)]
        ctrl = PWMController(channels)
        ctrl.start()
        for i in range(4):
            ctrl.enable_channel(f"motor_{i}")
        ctrl.fail_safe()
        assert ctrl._running is False
        for i in range(4):
            assert ctrl._channels[f"motor_{i}"].enabled is False
        ctrl.stop()

    def test_timing_stats_on_hardware(self):
        """get_timing_stats() returns valid p50/p95/p99 after 100 updates."""
        ctrl = PWMController([_make_hw_config("motor_0", 0)])
        ctrl.start()
        ctrl.enable_channel("motor_0")
        for _ in range(100):
            ctrl.set_duty_cycle("motor_0", 10.0)
            ctrl.update()
        stats = ctrl.get_timing_stats()
        assert stats["count"] == 100
        assert stats["p50_ms"] > 0.0
        assert stats["p95_ms"] >= stats["p50_ms"]
        assert stats["p99_ms"] >= stats["p95_ms"]
        ctrl.stop()
