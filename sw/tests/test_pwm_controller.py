"""Tests for PWMController — built incrementally across implementation phases."""

import pytest

from sw.pwm_controller import ChannelConfig, PWMController, PWMUpdateError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(
    channel_id: str = "ch0",
    pwm_channel: int = 0,
    chip: int = 0,
    freq_hz: float = 100.0,
    duty_pct: float = 0.0,
    freq_min_hz: float = 50.0,
    freq_max_hz: float = 400.0,
    duty_min_pct: float = 0.0,
    duty_max_pct: float = 100.0,
    allow_runtime_freq_change: bool = False,
    enabled: bool = False,
) -> ChannelConfig:
    return ChannelConfig(
        channel_id=channel_id,
        pwm_channel=pwm_channel,
        chip=chip,
        freq_hz=freq_hz,
        duty_pct=duty_pct,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        duty_min_pct=duty_min_pct,
        duty_max_pct=duty_max_pct,
        allow_runtime_freq_change=allow_runtime_freq_change,
        enabled=enabled,
    )


def make_four_channels() -> list[ChannelConfig]:
    return [
        make_config(channel_id=f"motor_{i}", pwm_channel=i, chip=0)
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# Phase 1: Core Data Model and Class Skeleton
# ---------------------------------------------------------------------------


class TestPhase1:

    # --- ChannelConfig construction ---

    def test_channel_config_basic(self):
        cfg = make_config()
        assert cfg.channel_id == "ch0"
        assert cfg.pwm_channel == 0
        assert cfg.chip == 0
        assert cfg.freq_hz == 100.0
        assert cfg.duty_pct == 0.0
        assert cfg.freq_min_hz == 50.0
        assert cfg.freq_max_hz == 400.0
        assert cfg.duty_min_pct == 0.0
        assert cfg.duty_max_pct == 100.0
        assert cfg.allow_runtime_freq_change is False
        assert cfg.enabled is False

    def test_channel_config_allow_runtime_freq(self):
        cfg = make_config(allow_runtime_freq_change=True)
        assert cfg.allow_runtime_freq_change is True

    def test_channel_config_invalid_freq_range(self):
        with pytest.raises(ValueError, match="freq_max_hz"):
            make_config(freq_min_hz=400.0, freq_max_hz=50.0)

    def test_channel_config_invalid_freq_min_zero(self):
        with pytest.raises(ValueError, match="freq_min_hz"):
            make_config(freq_min_hz=0.0)

    def test_channel_config_invalid_duty_range(self):
        with pytest.raises(ValueError, match="duty_max_pct"):
            make_config(duty_min_pct=50.0, duty_max_pct=10.0)

    def test_channel_config_duty_out_of_0_100(self):
        with pytest.raises(ValueError):
            make_config(duty_min_pct=-1.0)
        with pytest.raises(ValueError):
            make_config(duty_max_pct=101.0)

    def test_channel_config_invalid_pwm_channel(self):
        with pytest.raises(ValueError, match="pwm_channel"):
            make_config(pwm_channel=4)
        with pytest.raises(ValueError, match="pwm_channel"):
            make_config(pwm_channel=-1)

    # --- PWMController.__init__ ---

    def test_controller_init_single_channel(self):
        ctrl = PWMController([make_config()])
        assert not ctrl._running

    def test_controller_init_all_four_channels(self):
        ctrl = PWMController(make_four_channels())
        assert len(ctrl._channels) == 4

    def test_controller_init_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            PWMController([])

    def test_controller_init_duplicate_channel_id_raises(self):
        with pytest.raises(ValueError, match="Duplicate channel_id"):
            PWMController([make_config("ch0", 0), make_config("ch0", 1)])

    def test_controller_init_duplicate_pwm_channel_raises(self):
        with pytest.raises(ValueError, match="Duplicate pwm_channel"):
            PWMController([make_config("ch0", 0), make_config("ch1", 0)])

    # --- _validate_frequency ---

    def test_validate_frequency_in_range(self):
        ctrl = PWMController([make_config(freq_min_hz=50.0, freq_max_hz=400.0)])
        ctrl._validate_frequency("ch0", 100.0)  # no exception

    def test_validate_frequency_at_bounds(self):
        ctrl = PWMController([make_config(freq_min_hz=50.0, freq_max_hz=400.0)])
        ctrl._validate_frequency("ch0", 50.0)
        ctrl._validate_frequency("ch0", 400.0)

    def test_validate_frequency_below_min_raises(self):
        ctrl = PWMController([make_config(freq_min_hz=50.0, freq_max_hz=400.0)])
        with pytest.raises(ValueError, match="out of range"):
            ctrl._validate_frequency("ch0", 49.9)

    def test_validate_frequency_above_max_raises(self):
        ctrl = PWMController([make_config(freq_min_hz=50.0, freq_max_hz=400.0)])
        with pytest.raises(ValueError, match="out of range"):
            ctrl._validate_frequency("ch0", 400.1)

    def test_validate_frequency_unknown_channel_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(ValueError, match="Unknown channel_id"):
            ctrl._validate_frequency("nonexistent", 100.0)

    # --- _validate_duty ---

    def test_validate_duty_in_range(self):
        ctrl = PWMController([make_config(duty_min_pct=0.0, duty_max_pct=100.0)])
        ctrl._validate_duty("ch0", 50.0)  # no exception

    def test_validate_duty_at_bounds(self):
        ctrl = PWMController([make_config(duty_min_pct=5.0, duty_max_pct=95.0)])
        ctrl._validate_duty("ch0", 5.0)
        ctrl._validate_duty("ch0", 95.0)

    def test_validate_duty_below_min_raises(self):
        ctrl = PWMController([make_config(duty_min_pct=5.0, duty_max_pct=95.0)])
        with pytest.raises(ValueError, match="out of range"):
            ctrl._validate_duty("ch0", 4.9)

    def test_validate_duty_above_max_raises(self):
        ctrl = PWMController([make_config(duty_min_pct=5.0, duty_max_pct=95.0)])
        with pytest.raises(ValueError, match="out of range"):
            ctrl._validate_duty("ch0", 95.1)

    def test_validate_duty_negative_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(ValueError, match="out of global range"):
            ctrl._validate_duty("ch0", -1.0)

    def test_validate_duty_over_100_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(ValueError, match="out of global range"):
            ctrl._validate_duty("ch0", 100.1)

    def test_validate_duty_unknown_channel_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(ValueError, match="Unknown channel_id"):
            ctrl._validate_duty("nonexistent", 50.0)

    # --- _require_started ---

    def test_require_started_raises_before_start(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(RuntimeError, match="not started"):
            ctrl._require_started()

    def test_no_hardware_pwm_imported_or_instantiated(self):
        """Phase 1 must not exercise any HardwarePWM calls."""
        ctrl = PWMController(make_four_channels())
        assert ctrl._hwpwm == {}
