# Implementation Plan: Raspberry Pi 5 HardwarePWM BLDC Multi-Motor Control Class

## Metadata
- **ID**: plan-2026-03-09-pwm-control-class
- **Status**: draft
- **Specification**: `codev/specs/0001_pwm_control_class.md`
- **Created**: 2026-03-09

## Executive Summary

Implement a centralized `PWMController` class (Approach 1 from spec) that manages 2–4 hardware PWM channels on Raspberry Pi 5 for BLDC motor control. The class exposes an `update()` method called by an external scheduler; it does not own a control loop. Thread-safe command submission is supported via a lock. Fail-safe behavior (duty=0 then disable for all channels) is enforced on any partial update failure. The implementation uses the existing `rpi_hardware_pwm` library (`HardwarePWM`) already present in the codebase.

## Success Metrics
- [ ] All spec success criteria met
- [ ] Unit test coverage ≥90% (excluding hardware-only paths)
- [ ] Tests run cleanly with mock `HardwarePWM` backend (no real hardware required)
- [ ] `update()` sequential write window measured and logged (p50/p95/p99)
- [ ] Fail-safe verified: duty=0 then disable on any channel fault
- [ ] Thread-safe command submission smoke-tested

## Phases (Machine Readable)

<!-- REQUIRED: porch uses this JSON to track phase progress. -->

```json
{
  "phases": [
    {"id": "phase_1", "title": "Core Data Model and Class Skeleton"},
    {"id": "phase_2", "title": "Hardware Lifecycle and Fail-Safe"},
    {"id": "phase_3", "title": "Synchronized Update with Timing Instrumentation"},
    {"id": "phase_4", "title": "Thread-Safe Command Interface"},
    {"id": "phase_5", "title": "Test Suite"}
  ]
}
```

## Phase Breakdown

### Phase 1: Core Data Model and Class Skeleton
**Dependencies**: None

#### Objectives
- Define `ChannelConfig` dataclass holding per-channel parameters and valid ranges.
- Define `PWMController` class with `__init__`, channel registration, and input validation helpers.
- No hardware I/O in this phase — all HardwarePWM calls stubbed.

#### Deliverables
- [ ] `sw/pwm_controller.py` — `ChannelConfig` dataclass and `PWMController` class skeleton
- [ ] Input validation logic for frequency and duty cycle against per-channel configured ranges
- [ ] `ValueError` raised on out-of-range inputs or unknown channel IDs

#### Implementation Details

**File**: `sw/pwm_controller.py`

```
ChannelConfig:
    channel_id: str
    pwm_channel: int          # HardwarePWM channel number (0-3)
    chip: int                 # configurable chip number (0 or 1)
    freq_hz: float            # initial/current frequency
    duty_pct: float           # initial/current duty cycle (0.0-100.0)
    freq_min_hz: float        # per-channel min frequency
    freq_max_hz: float        # per-channel max frequency
    duty_min_pct: float       # per-channel min duty cycle
    duty_max_pct: float       # per-channel max duty cycle
    enabled: bool = False     # whether channel is active

PWMController:
    __init__(channels: list[ChannelConfig]) -> None
        - Stores channel configs keyed by channel_id
        - Validates no duplicate channel_ids or pwm_channel numbers
        - Sets running=False, _lock=threading.Lock()

    _validate_frequency(channel_id, freq_hz) -> None
        - Raises ValueError if out of configured range

    _validate_duty(channel_id, duty_pct) -> None
        - Raises ValueError if out of range [0.0, 100.0] and channel range
```

#### Acceptance Criteria
- [ ] `ChannelConfig` can be instantiated with all required fields
- [ ] `PWMController.__init__` raises `ValueError` on duplicate channel IDs
- [ ] `_validate_frequency` raises `ValueError` for out-of-range values
- [ ] `_validate_duty` raises `ValueError` for negative or >100% values
- [ ] No `HardwarePWM` imports exercised in this phase

#### Test Plan
- **Unit Tests**: Test `ChannelConfig` creation; test `__init__` with valid/invalid channel lists; test `_validate_frequency` and `_validate_duty` boundary conditions.
- **No hardware**: HardwarePWM not needed.

#### Rollback Strategy
Delete `sw/pwm_controller.py` — no side effects on existing code.

#### Risks
- **Risk**: `ChannelConfig` fields may need adjustment in later phases.
  - **Mitigation**: Fields are immutable at construction; mutable state is separate.

---

### Phase 2: Hardware Lifecycle and Fail-Safe
**Dependencies**: Phase 1

#### Objectives
- Implement `start()`: initialize all `HardwarePWM` instances, set all channels to 0% duty and disabled state.
- Implement `fail_safe()` / `stop()`: set duty=0 then disable all channels.
- Implement `enable_channel()` and `disable_channel()`.

#### Deliverables
- [ ] `start()` method in `sw/pwm_controller.py`
- [ ] `fail_safe()` method (callable at any time, always safe)
- [ ] `stop()` method (calls `fail_safe()` then cleans up HardwarePWM objects)
- [ ] `enable_channel(channel_id)` / `disable_channel(channel_id)` methods

#### Implementation Details

```
start() -> None:
    - For each ChannelConfig, create HardwarePWM(pwm_channel=..., hz=freq_hz, chip=chip)
    - Call pwm.start(0) to initialize at 0% duty
    - Immediately disable the channel (pwm.stop() is not the same as disable — use
      duty=0 and keep the PWM instance; "disabled" means we skip writing it in update())
    - Set self._running = True
    - Raises RuntimeError if already started

fail_safe() -> None:
    - For each HardwarePWM instance (best-effort, catches exceptions per channel):
        1. pwm.change_duty_cycle(0)
        2. pwm.stop()
    - Sets self._running = False
    - Logs each channel result (success/failure)
    - Does NOT raise — must be callable during error handling

stop() -> None:
    - Calls fail_safe()
    - Cleans up HardwarePWM instances

enable_channel(channel_id: str) -> None:
    - Sets config.enabled = True

disable_channel(channel_id: str) -> None:
    - Sets config.enabled = False
    - Also sets duty to 0 on hardware immediately
```

**Note on "disabled" state**: A channel being "disabled" means `update()` skips it. It does NOT mean the HardwarePWM hardware output is stopped between ticks; the hardware holds the last written value. Calling `disable_channel()` writes 0% duty immediately to ensure the motor is not driven.

#### Acceptance Criteria
- [ ] `start()` with mock `HardwarePWM` calls `start(0)` on every channel
- [ ] `fail_safe()` calls `change_duty_cycle(0)` then `stop()` on every channel even if one raises
- [ ] `fail_safe()` does not raise even when `HardwarePWM.stop()` throws
- [ ] `stop()` leaves `_running = False`

#### Test Plan
- **Unit Tests**: Mock `HardwarePWM`; verify call order (duty=0 then stop); verify fail_safe() is exception-safe; test enable/disable state.
- **Integration**: Manually verify on hardware that all channels go to 0% on fail_safe().

#### Rollback Strategy
Phase 2 adds methods to `pwm_controller.py` — revert the file to Phase 1 state.

#### Risks
- **Risk**: `rpi_hardware_pwm` `stop()` behavior may differ from expectation (it may unregister the sysfs file, requiring `start()` again before next use).
  - **Mitigation**: Check library source in `sw/` or installed package. If `stop()` tears down sysfs entry, use `change_duty_cycle(0)` only for fail_safe and `stop()` only on final cleanup.

---

### Phase 3: Synchronized Update with Timing Instrumentation
**Dependencies**: Phase 2

#### Objectives
- Implement `update(commands: dict[str, tuple[float, float]])` that applies pending duty/frequency to all enabled channels sequentially.
- Record per-call timing and accumulate p50/p95/p99 latency statistics.
- On any channel write failure: invoke `fail_safe()` and raise `PWMUpdateError`.

#### Deliverables
- [ ] `update()` method in `sw/pwm_controller.py`
- [ ] `PWMUpdateError` exception class
- [ ] `get_timing_stats()` method returning p50/p95/p99 of `update()` durations
- [ ] `set_duty_cycle(channel_id, duty_pct)` and `set_frequency(channel_id, freq_hz)` command-staging methods

#### Implementation Details

```
set_duty_cycle(channel_id: str, duty_pct: float) -> None:
    - Validates duty_pct against channel range
    - Stages pending duty change: self._pending[channel_id].duty = duty_pct

set_frequency(channel_id: str, freq_hz: float) -> None:
    - Validates freq_hz against channel range
    - Stages pending frequency change
    - Documents: change_frequency() zeroes duty momentarily (sysfs glitch)

update() -> None:
    - Called by external scheduler each tick
    - t_start = time.monotonic()
    - For each enabled channel in deterministic order (insertion order):
        - Apply staged duty/freq from _pending (if any)
        - If duty changed: pwm.change_duty_cycle(duty)
        - If freq changed: pwm.change_frequency(freq)  [sysfs glitch caveat applies]
        - On exception: fail_safe(); raise PWMUpdateError(channel_id, cause)
    - t_end = time.monotonic()
    - self._timings.append(t_end - t_start)  [bounded circular buffer, e.g. last 10000]

get_timing_stats() -> dict:
    - Returns {"p50": ..., "p95": ..., "p99": ..., "count": ...} in milliseconds
    - Uses numpy or statistics module; falls back to empty if no samples
```

**Deterministic ordering**: channels are iterated in the order they were registered (insertion order of dict, guaranteed in Python 3.7+).

**Pending state**: `_pending` is a dict mapping channel_id → dataclass with optional duty/freq fields. `update()` consumes and clears the pending state atomically before writing to hardware.

#### Acceptance Criteria
- [ ] `update()` applies pending duty/freq to all enabled channels in registration order
- [ ] `update()` skips disabled channels
- [ ] On channel write failure, `fail_safe()` is called and `PWMUpdateError` is raised
- [ ] `get_timing_stats()` returns p50/p95/p99 after at least one `update()` call
- [ ] Timing buffer is bounded (does not grow unboundedly)

#### Test Plan
- **Unit Tests**: Mock `HardwarePWM`; verify correct call order for duty and freq; verify fail_safe triggered on exception; verify timing stats populated; verify disabled channels skipped.
- **Non-functional**: Run 10,000 `update()` calls with mock backend and verify `get_timing_stats()` returns values (cannot test actual sysfs latency without hardware).

#### Rollback Strategy
Revert `pwm_controller.py` to Phase 2 state.

#### Risks
- **Risk**: `change_frequency()` triggers duty glitch on BLDC motors — undesirable during runtime.
  - **Mitigation**: Document in API docstring. For initial implementation, allow frequency change but emit a warning log. A future improvement may restrict frequency changes to a pre-arm configuration window.
- **Risk**: numpy dependency for percentile calculation may not be present.
  - **Mitigation**: Use `statistics.quantiles` (stdlib, Python 3.8+) as primary; fallback to simple sorted-list approach.

---

### Phase 4: Thread-Safe Command Interface
**Dependencies**: Phase 3

#### Objectives
- Wrap all command-staging methods and `update()` with a `threading.Lock` to ensure safe concurrent access from multiple threads.
- Document that cross-process safety requires an external IPC layer (out of scope for this class).

#### Deliverables
- [ ] `threading.Lock` protecting `_pending`, `_channels`, `_timings` from concurrent mutation
- [ ] Lock acquired in: `set_duty_cycle()`, `set_frequency()`, `enable_channel()`, `disable_channel()`, `update()`
- [ ] Docstring on `PWMController` documenting thread-safety guarantees and cross-process limitation

#### Implementation Details

The lock (initialized in `__init__` as `self._lock = threading.Lock()`) is held briefly:
- In `set_duty_cycle()` / `set_frequency()`: hold lock only while writing to `_pending`.
- In `update()`: hold lock while reading+clearing `_pending` into a local snapshot, then release before making hardware I/O calls (to avoid holding the lock across slow sysfs writes).

```
update() with lock:
    with self._lock:
        snapshot = copy of _pending
        _pending = {} (clear)
    # hardware writes outside the lock
    for channel in enabled_channels:
        apply snapshot[channel] to hardware
        ...
```

This prevents `set_duty_cycle()` calls from blocking behind sysfs I/O while still protecting shared state.

#### Acceptance Criteria
- [ ] Concurrent `set_duty_cycle()` from two threads does not corrupt `_pending`
- [ ] `update()` does not hold the lock during `change_duty_cycle()` or `change_frequency()` calls
- [ ] Docstring documents that cross-process safety requires external IPC

#### Test Plan
- **Thread-safety smoke test**: Spawn 10 threads each calling `set_duty_cycle()` 1000 times concurrently while `update()` runs in a separate thread; assert no exceptions and final state is consistent.

#### Rollback Strategy
Remove lock acquisition from all methods and `_lock` initialization.

#### Risks
- **Risk**: GIL may mask race conditions in CPython testing, giving false confidence.
  - **Mitigation**: Document that tests cover logical races; real deterministic guarantees under GIL are limited. Cross-process safety is explicitly out of scope for this class.

---

### Phase 5: Test Suite
**Dependencies**: Phases 1–4

#### Objectives
- Write a comprehensive pytest-based test suite covering all acceptance criteria from Phases 1–4.
- All tests run without real hardware using a mock `HardwarePWM` backend.

#### Deliverables
- [ ] `sw/tests/test_pwm_controller.py` — full unit test suite
- [ ] Mock `HardwarePWM` fixture capturing all calls
- [ ] Tests grouped by: validation, lifecycle, update, fail_safe, timing, thread-safety

#### Implementation Details

**Mock fixture**:
```python
@pytest.fixture
def mock_hwpwm(monkeypatch):
    calls = defaultdict(list)
    class MockHardwarePWM:
        def __init__(self, pwm_channel, hz, chip): ...
        def start(self, duty): calls['start'].append(duty)
        def change_duty_cycle(self, duty): calls['change_duty_cycle'].append(duty)
        def change_frequency(self, hz): calls['change_frequency'].append(hz)
        def stop(self): calls['stop'].append(True)
    monkeypatch.setattr('sw.pwm_controller.HardwarePWM', MockHardwarePWM)
    return calls
```

**Test groups**:
1. Validation: out-of-range duty, out-of-range frequency, unknown channel ID, duplicate channel registration.
2. Lifecycle: `start()` initializes all channels at 0%; `stop()` calls fail_safe; double-start raises.
3. Fail-safe: `fail_safe()` called on all channels even if one raises; does not propagate exception.
4. Update: duty applied; freq applied; disabled channels skipped; insertion-order preserved; `PWMUpdateError` on mock hardware exception.
5. Timing: `get_timing_stats()` returns p50/p95/p99 after multiple `update()` calls.
6. Thread-safety: concurrent `set_duty_cycle()` calls do not corrupt state.

#### Acceptance Criteria
- [ ] `pytest sw/tests/test_pwm_controller.py` passes with no hardware attached
- [ ] Coverage ≥90% on `sw/pwm_controller.py`
- [ ] No test relies on real `HardwarePWM` or sysfs

#### Test Plan
- All unit; no real hardware needed.
- Coverage measured with `pytest --cov=sw.pwm_controller`.

#### Rollback Strategy
Delete `sw/tests/test_pwm_controller.py`.

#### Risks
- **Risk**: Mock may not accurately reflect real `HardwarePWM` behavior (e.g., sysfs errors, init order).
  - **Mitigation**: Tests validate logical behavior only; hardware validation is manual on Raspberry Pi 5.

---

## Dependency Map
```
Phase 1 (Data Model) ──→ Phase 2 (Lifecycle) ──→ Phase 3 (Update + Timing) ──→ Phase 4 (Thread Safety) ──→ Phase 5 (Tests)
```

## Resource Requirements
### Development Resources
- **Environment**: Python 3.8+, `rpi_hardware_pwm` installed (already in project), `pytest`, `pytest-cov`

### Infrastructure
- No new services or database changes
- Tests run on any machine (mock backend — no Raspberry Pi required for unit tests)

## Integration Points
### Internal Systems
- **`rpi_hardware_pwm.HardwarePWM`**: Used in Phase 2–4. Imported in `sw/pwm_controller.py`.
- **External scheduler**: Not implemented here. Caller must invoke `pwm.update()` at each control tick.

## Risk Analysis
### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| sysfs latency prevents 1 kHz control rate | Medium | High | Expose timing metrics; treat 1 kHz as aspirational; benchmark on hardware |
| `change_frequency()` duty glitch causes motor disturbance | Medium | Medium | Document in API; restrict to pre-arm configuration by default |
| `rpi_hardware_pwm.stop()` teardown prevents reuse | Low | Medium | Validate in Phase 2; use duty=0 instead of stop() for fail_safe if needed |
| GIL inadequate for multi-process safety | Low | Medium | Document cross-process IPC as out of scope; use external coordinator |

## Validation Checkpoints
1. **After Phase 1**: Validation logic tests pass cleanly
2. **After Phase 2**: Lifecycle tests pass; fail_safe() confirmed exception-safe
3. **After Phase 3**: update() sequence verified; timing stats populated
4. **After Phase 4**: Thread-safety smoke test passes
5. **After Phase 5**: Full test suite passes; coverage ≥90%

## Documentation Updates Required
- [ ] Docstrings on all public methods in `sw/pwm_controller.py`
- [ ] Module-level docstring describing API contract, sysfs latency caveats, thread-safety guarantees, and safe state definition

## Expert Review
*(To be completed after 3-way consultation)*

## Approval
- [ ] Technical Lead Review
- [ ] Expert AI Consultation Complete

## Change Log
| Date | Change | Reason |
|------|--------|--------|
| 2026-03-09 | Initial plan draft | SPIR plan phase |
