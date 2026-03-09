# Plan Review Rebuttal — Iteration 1

**Project**: 1-implement-initial-pwm-class
**Phase**: plan
**Iteration**: 1
**Date**: 2026-03-09

Gemini returned COMMENT, Codex returned REQUEST_CHANGES, Claude returned COMMENT. All feedback was valid. The plan has been updated. This rebuttal documents what was changed and why.

---

## Gemini Feedback

### Issue 1: Spec Divergence — Frequency Change Opt-In
**ACCEPTED. Changed.**
`ChannelConfig` now includes `allow_runtime_freq_change: bool = False`. `set_frequency()` checks this flag; if False, it raises `RuntimeError` with a descriptive message. If True, it proceeds with a WARNING log. This enforces the spec's explicit safety requirement.

### Issue 2: Incremental Test Creation
**ACCEPTED. Changed.**
The plan now explicitly states in Phase 1 that the test file is created there and expanded incrementally each phase. Phase 5 is now "Test Coverage and Stress Tests" (consolidation, gap fill, thread-safety stress test, latency simulation). Test class names (`TestPhase1` through `TestPhase5`) are specified per phase.

---

## Codex Feedback

### Issue 1: Multi-Process Safety Out of Scope
**PARTIALLY ACCEPTED. Documented.**
The plan now explicitly explains the architectural boundary decision: `PWMController` is thread-safe within a single process. Cross-process coordination (e.g., `multiprocessing.Queue`, shared memory, Unix socket) is the responsibility of the higher-level system and is out of scope for this low-level hardware interface class. The API docstring will explicitly state this. This is the appropriate design boundary — the spec's cross-process requirement is acknowledged and delegated to the system integration layer, not the hardware class itself.

### Issue 2: Frequency Change Opt-In Missing
**ACCEPTED. Changed.** (Same as Gemini Issue 1.)

### Issue 3: "Disable" Semantics Ambiguous
**ACCEPTED. Changed.**
The plan now has a dedicated "disabled semantics" section in Phase 2:
- `disabled` = `update()` skips the channel. HardwarePWM instance stays alive.
- `disable_channel()` writes duty=0 immediately to hardware.
- `fail_safe()` = `change_duty_cycle(0)` only (does NOT call `pwm.stop()`); marks all disabled.
- `stop()` = `fail_safe()` then `pwm.stop()` on each channel (final teardown).
This fully resolves the ambiguity between "skip in update" vs "hardware output disabled."

### Issue 4: Commands Before `start()` Unspecified
**ACCEPTED. Changed.**
A `_require_started()` helper is now defined in Phase 1. It raises `RuntimeError("Controller not started")` if `_running` is False. All command methods (`set_duty_cycle()`, `set_frequency()`, `enable_channel()`, `disable_channel()`, `update()`) call it at the top.

### Issue 5: Mock Lacks Sysfs Latency Simulation
**ACCEPTED. Changed.**
Phase 5 now includes a `MockHardwarePWMWithLatency` fixture (called `mock_hwpwm_slow`) that adds a configurable `time.sleep(0.0001)` per call to simulate sysfs latency. A test verifies `get_timing_stats()` reports p50 > 0 under this fixture.

---

## Claude Feedback

### Issue 1: `sw/` Not a Python Package
**ACCEPTED. Changed.**
Phase 1 deliverables now include `sw/__init__.py` and `sw/tests/__init__.py`. The monkeypatch path `sw.pwm_controller.HardwarePWM` will resolve correctly once `sw/` is a package.

### Issue 2: Ambiguous Test Timing (Per-Phase vs Phase 5)
**ACCEPTED. Changed.** (Same as Gemini Issue 2.)

### Issue 3: `fail_safe()` Using `pwm.stop()` — Risk of Destroying sysfs Entry
**ACCEPTED. Changed.**
The decision is now explicit and documented in Phase 2: `fail_safe()` uses `change_duty_cycle(0)` ONLY. It does NOT call `pwm.stop()`. `pwm.stop()` is reserved for final teardown in `stop()`. This avoids destroying the sysfs entry during fault recovery.

### Issue 4: Wrong Spec Path in Metadata
**ACCEPTED. Changed.**
Metadata now references `codev/specs/1-implement-initial-pwm-class.md`.

### Issue 5: `start()` Partial Failure Not Addressed
**ACCEPTED. Changed.**
Phase 2 now specifies: on any `HardwarePWM.__init__` failure during `start()`, call `fail_safe()` on already-created instances and re-raise the exception. No orphaned hardware state is left.

### Minor: Python Version
**ACCEPTED.** Resource Requirements updated to Python 3.12+ (per `pyproject.toml`).

### Minor: numpy Already Available
**ACCEPTED.** Removed the risk about numpy not being present. `get_timing_stats()` uses `numpy.percentile` directly.

---

## Summary of Changes

All feedback accepted. Plan updated with:
- `allow_runtime_freq_change` field + enforcement in `set_frequency()`
- Explicit multi-process scope boundary documented
- Disambiguated disable/fail_safe/stop semantics
- `_require_started()` guard for pre-start command protection
- `mock_hwpwm_slow` fixture for latency simulation in Phase 5
- `sw/__init__.py` and `sw/tests/__init__.py` as Phase 1 deliverables
- Incremental test cadence clarified (test file created in Phase 1, expanded per phase)
- `fail_safe()` decided: `change_duty_cycle(0)` only, no `pwm.stop()`
- `start()` partial failure handling added
- Spec path fixed in metadata
- Python version and numpy dependency corrected
