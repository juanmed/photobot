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

    # --- set_duty_cycle before start ---

    def test_set_duty_cycle_before_start_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(RuntimeError, match="not started"):
            ctrl.set_duty_cycle("ch0", 50.0)

    def test_no_hardware_pwm_imported_or_instantiated(self):
        """Phase 1 must not exercise any HardwarePWM calls."""
        ctrl = PWMController(make_four_channels())
        assert ctrl._hwpwm == {}


# ---------------------------------------------------------------------------
# Phase 2: Hardware Lifecycle and Fail-Safe
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_hwpwm(monkeypatch):
    """Patch HardwarePWM with a mock that records calls."""
    from unittest.mock import MagicMock, patch

    instances: dict[int, MagicMock] = {}

    def factory(pwm_channel, hz, chip):
        inst = MagicMock()
        inst.pwm_channel = pwm_channel
        instances[pwm_channel] = inst
        return inst

    with patch("sw.pwm_controller.HardwarePWM", side_effect=factory) as mock_cls:
        yield mock_cls, instances


class TestPhase2:

    def test_start_calls_start_zero_on_each_channel(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController(make_four_channels())
        ctrl.start()
        for i in range(4):
            instances[i].start.assert_called_once_with(0)

    def test_start_sets_running(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        assert ctrl._running is True

    def test_start_forces_channels_disabled(self, mock_hwpwm):
        """start() must force all channels to disabled even if config has enabled=True."""
        ctrl = PWMController([make_config(enabled=True)])
        ctrl.start()
        assert ctrl._channels["ch0"].enabled is False

    def test_start_twice_raises(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        with pytest.raises(RuntimeError, match="already started"):
            ctrl.start()

    def test_start_partial_failure_calls_fail_safe(self, mock_hwpwm):
        """If HardwarePWM init fails mid-way, fail_safe is called on prior instances."""
        from unittest.mock import MagicMock, patch

        call_count = 0

        def failing_factory(pwm_channel, hz, chip):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("hardware error")
            inst = MagicMock()
            return inst

        with patch("sw.pwm_controller.HardwarePWM", side_effect=failing_factory):
            ctrl = PWMController(
                [make_config("ch0", 0), make_config("ch1", 1)]
            )
            with pytest.raises(RuntimeError, match="hardware error"):
                ctrl.start()
            assert ctrl._running is False

    def test_fail_safe_calls_change_duty_cycle_zero(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController(make_four_channels())
        ctrl.start()
        ctrl.fail_safe()
        for i in range(4):
            instances[i].change_duty_cycle.assert_called_with(0)

    def test_fail_safe_does_not_call_stop(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.fail_safe()
        instances[0].stop.assert_not_called()

    def test_fail_safe_sets_running_false(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        assert ctrl._running is True
        ctrl.fail_safe()
        assert ctrl._running is False

    def test_fail_safe_does_not_raise_on_channel_error(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        instances[0].change_duty_cycle.side_effect = OSError("sysfs error")
        ctrl.fail_safe()  # must not raise

    def test_fail_safe_marks_channels_disabled(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.enable_channel("ch0")
        assert ctrl._channels["ch0"].enabled is True
        ctrl.fail_safe()
        assert ctrl._channels["ch0"].enabled is False

    def test_stop_calls_pwm_stop(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.stop()
        instances[0].stop.assert_called_once()

    def test_stop_clears_hwpwm_references(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.stop()
        assert ctrl._hwpwm == {}

    def test_enable_channel_before_start_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(RuntimeError, match="not started"):
            ctrl.enable_channel("ch0")

    def test_enable_channel_unknown_raises(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        with pytest.raises(ValueError, match="Unknown channel_id"):
            ctrl.enable_channel("nonexistent")

    def test_enable_channel_sets_enabled(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        assert ctrl._channels["ch0"].enabled is False
        ctrl.enable_channel("ch0")
        assert ctrl._channels["ch0"].enabled is True

    def test_disable_channel_before_start_raises(self):
        ctrl = PWMController([make_config()])
        with pytest.raises(RuntimeError, match="not started"):
            ctrl.disable_channel("ch0")

    def test_disable_channel_unknown_raises(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        with pytest.raises(ValueError, match="Unknown channel_id"):
            ctrl.disable_channel("nonexistent")

    def test_disable_channel_writes_zero_duty_immediately(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.enable_channel("ch0")
        ctrl.disable_channel("ch0")
        instances[0].change_duty_cycle.assert_called_with(0)
        assert ctrl._channels["ch0"].enabled is False


# ---------------------------------------------------------------------------
# Phase 3: Synchronized Update with Timing Instrumentation
# ---------------------------------------------------------------------------


class TestPhase3:

    def test_set_duty_cycle_stages_pending(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.set_duty_cycle("ch0", 75.0)
        assert ctrl._pending["ch0"].duty_pct == 75.0

    def test_set_duty_cycle_invalid_raises(self, mock_hwpwm):
        ctrl = PWMController([make_config(duty_min_pct=5.0, duty_max_pct=95.0)])
        ctrl.start()
        with pytest.raises(ValueError):
            ctrl.set_duty_cycle("ch0", 99.0)

    def test_set_frequency_blocked_by_flag(self, mock_hwpwm):
        ctrl = PWMController([make_config(allow_runtime_freq_change=False)])
        ctrl.start()
        with pytest.raises(RuntimeError, match="not allowed"):
            ctrl.set_frequency("ch0", 100.0)

    def test_set_frequency_allowed_when_flag_true(self, mock_hwpwm):
        ctrl = PWMController([make_config(allow_runtime_freq_change=True)])
        ctrl.start()
        ctrl.set_frequency("ch0", 200.0)
        assert ctrl._pending["ch0"].freq_hz == 200.0

    def test_set_frequency_invalid_range_raises(self, mock_hwpwm):
        ctrl = PWMController([make_config(allow_runtime_freq_change=True)])
        ctrl.start()
        with pytest.raises(ValueError):
            ctrl.set_frequency("ch0", 1000.0)  # above max 400 Hz

    def test_update_applies_pending_duty(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.enable_channel("ch0")
        ctrl.set_duty_cycle("ch0", 50.0)
        ctrl.update()
        instances[0].change_duty_cycle.assert_called_with(50.0)

    def test_update_applies_pending_freq(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config(allow_runtime_freq_change=True)])
        ctrl.start()
        ctrl.enable_channel("ch0")
        ctrl.set_frequency("ch0", 200.0)
        ctrl.update()
        instances[0].change_frequency.assert_called_with(200.0)

    def test_update_skips_disabled_channels(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        # channel disabled (default); stage a command
        ctrl._pending["ch0"].duty_pct = 50.0
        ctrl.update()
        instances[0].change_duty_cycle.assert_not_called()

    def test_update_clears_pending_after_apply(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.enable_channel("ch0")
        ctrl.set_duty_cycle("ch0", 50.0)
        ctrl.update()
        assert ctrl._pending["ch0"].duty_pct is None

    def test_update_on_failure_calls_fail_safe_and_raises(self, mock_hwpwm):
        _, instances = mock_hwpwm
        ctrl = PWMController([make_config()])
        ctrl.start()
        # instances populated after start()
        instances[0].change_duty_cycle.side_effect = OSError("sysfs error")
        ctrl.enable_channel("ch0")
        ctrl.set_duty_cycle("ch0", 50.0)
        from sw.pwm_controller import PWMUpdateError
        with pytest.raises(PWMUpdateError):
            ctrl.update()
        assert ctrl._running is False  # fail_safe was called

    def test_update_records_timing(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        ctrl.update()
        assert len(ctrl._timings) == 1

    def test_get_timing_stats_empty(self):
        ctrl = PWMController([make_config()])
        stats = ctrl.get_timing_stats()
        assert stats == {"count": 0}

    def test_get_timing_stats_after_updates(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        for _ in range(10):
            ctrl.update()
        stats = ctrl.get_timing_stats()
        assert stats["count"] == 10
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats

    def test_timing_buffer_bounded(self, mock_hwpwm):
        ctrl = PWMController([make_config()])
        ctrl.start()
        for _ in range(15000):
            ctrl.update()
        assert len(ctrl._timings) == 10000  # deque maxlen

    def test_update_registration_order(self, mock_hwpwm):
        """Channels updated in insertion (registration) order."""
        _, instances = mock_hwpwm
        channels = [
            make_config("ch0", 0),
            make_config("ch1", 1),
            make_config("ch2", 2),
        ]
        ctrl = PWMController(channels)
        ctrl.start()
        for cid in ("ch0", "ch1", "ch2"):
            ctrl.enable_channel(cid)
            ctrl.set_duty_cycle(cid, 30.0)
        ctrl.update()
        for i in range(3):
            instances[i].change_duty_cycle.assert_called_with(30.0)
