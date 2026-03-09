# Implementation Plan: Raspberry Pi 5 HardwarePWM BLDC Multi-Motor Control Class

## Metadata
- **ID**: plan-2026-03-09-pwm-control-class
- **Status**: draft
- **Specification**: `codev/specs/1-implement-initial-pwm-class.md`
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
- [ ] `sw/__init__.py` — makes `sw` a Python package (enables `from sw.pwm_controller import ...`)
- [ ] `sw/tests/__init__.py` — makes `sw/tests` a package (enables pytest discovery and monkeypatching)
- [ ] `sw/pwm_controller.py` — `ChannelConfig` dataclass and `PWMController` class skeleton
- [ ] `sw/tests/test_pwm_controller.py` — skeleton test file with Phase 1 validation tests
- [ ] Input validation logic for frequency and duty cycle against per-channel configured ranges
- [ ] `ValueError` raised on out-of-range inputs or unknown channel IDs

**Note on test cadence**: The test file is created in Phase 1 and expanded incrementally each phase. Phase 5 consolidates, fills coverage gaps, and adds the thread-safety stress test. Each phase's acceptance criteria must be verified by passing tests before moving on.

#### Implementation Details

**File**: `sw/pwm_controller.py`

```
ChannelConfig:
    channel_id: str
    pwm_channel: int                      # HardwarePWM channel number (0-3)
    chip: int                             # configurable chip number (0 or 1)
    freq_hz: float                        # initial/current frequency
    duty_pct: float                       # initial/current duty cycle (0.0-100.0)
    freq_min_hz: float                    # per-channel min frequency
    freq_max_hz: float                    # per-channel max frequency
    duty_min_pct: float                   # per-channel min duty cycle
    duty_max_pct: float                   # per-channel max duty cycle
    allow_runtime_freq_change: bool = False   # opt-in required for runtime frequency changes
    enabled: bool = False                 # whether channel is active

PWMController:
    __init__(channels: list[ChannelConfig]) -> None
        - Stores channel configs keyed by channel_id
        - Validates no duplicate channel_ids or pwm_channel numbers
        - Sets _running=False, _lock=threading.Lock()

    _validate_frequency(channel_id, freq_hz) -> None
        - Raises ValueError if out of configured range

    _validate_duty(channel_id, duty_pct) -> None
        - Raises ValueError if out of range [0.0, 100.0] and channel range

    _require_started() -> None
        - Raises RuntimeError("Controller not started") if _running is False
        - Called at the top of set_duty_cycle(), set_frequency(), enable_channel(),
          disable_channel(), and update()
```

#### Acceptance Criteria
- [ ] `ChannelConfig` can be instantiated with all required fields, including `allow_runtime_freq_change`
- [ ] `PWMController.__init__` raises `ValueError` on duplicate channel IDs
- [ ] `_validate_frequency` raises `ValueError` for out-of-range values
- [ ] `_validate_duty` raises `ValueError` for negative or >100% values
- [ ] No `HardwarePWM` imports exercised in this phase
- [ ] Phase 1 tests pass via `pytest sw/tests/test_pwm_controller.py::TestPhase1`

#### Test Plan
- **Unit Tests** (in `sw/tests/test_pwm_controller.py::TestPhase1`): Test `ChannelConfig` creation; test `__init__` with valid/invalid channel lists; test `_validate_frequency` and `_validate_duty` boundary conditions; test `set_duty_cycle()` before `start()` raises `RuntimeError`.
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
- [ ] `stop()` method (drives channels safe then calls `pwm.stop()` for teardown)
- [ ] `enable_channel(channel_id)` / `disable_channel(channel_id)` methods
- [ ] Tests for Phase 2 added to `sw/tests/test_pwm_controller.py::TestPhase2`

#### Implementation Details

```
start() -> None:
    - Raises RuntimeError if already started (_running == True)
    - Creates HardwarePWM instances one by one; on any failure, calls fail_safe()
      on already-created instances and re-raises (no orphaned hardware state)
    - For each ChannelConfig: HardwarePWM(pwm_channel=..., hz=freq_hz, chip=chip)
    - Calls pwm.start(0) — initializes at 0% duty (hardware holds this value)
    - Channel starts as disabled (enabled=False); update() skips it until enabled
    - Sets self._running = True only after ALL channels are initialized successfully

fail_safe() -> None:
    DECISION: fail_safe() uses change_duty_cycle(0) ONLY — does NOT call pwm.stop().
    This preserves the HardwarePWM sysfs registration so the channel can continue to
    be used or cleanly torn down. pwm.stop() is reserved for final stop() teardown.
    - For each HardwarePWM instance (best-effort, catches exceptions per channel):
        1. pwm.change_duty_cycle(0)
        2. Mark channel as disabled in internal state
    - Sets self._running = False
    - Logs each channel result (success/failure) via Python logging module
    - Does NOT raise — must be callable during exception handling paths

stop() -> None:
    - Calls fail_safe() first (duty=0, all disabled)
    - Then for each HardwarePWM instance: pwm.stop() (teardown sysfs entry)
    - Cleans up internal HardwarePWM instance references

enable_channel(channel_id: str) -> None:
    - Calls _require_started()
    - Sets config.enabled = True

disable_channel(channel_id: str) -> None:
    - Calls _require_started()
    - Writes duty=0 to hardware immediately (not just on next tick)
    - Sets config.enabled = False
```

**"Disabled" semantics**: A disabled channel is skipped by `update()`. Calling `disable_channel()` also writes duty=0 immediately so the motor is not driven. The HardwarePWM instance remains valid and registered — re-enabling just sets `enabled=True` and the next `update()` tick will apply commands.

**Safe state**: `fail_safe()` = duty=0 on all channels + mark all disabled. No `pwm.stop()` in fail_safe — hardware stays configured. `stop()` additionally tears down sysfs via `pwm.stop()`.

#### Acceptance Criteria
- [ ] `start()` with mock `HardwarePWM` calls `start(0)` on every channel
- [ ] `start()` on partial `HardwarePWM` init failure calls fail_safe and re-raises
- [ ] `fail_safe()` calls only `change_duty_cycle(0)` per channel (NOT `pwm.stop()`)
- [ ] `fail_safe()` does not raise even when `change_duty_cycle()` throws
- [ ] `stop()` calls `pwm.stop()` on all channels after fail_safe
- [ ] `disable_channel()` writes duty=0 immediately to hardware

#### Test Plan
- **Unit Tests** (`TestPhase2`): Mock `HardwarePWM`; verify `start(0)` call; verify fail_safe call sequence (duty=0, no stop); verify stop() calls pwm.stop(); verify exception safety; test partial init failure.
- **Integration**: Manually verify on hardware that all channels go to 0% on fail_safe().

#### Rollback Strategy
Phase 2 adds methods to `pwm_controller.py` — revert the file to Phase 1 state.

#### Risks
- **Risk**: `rpi_hardware_pwm` `start()` called twice on the same channel may raise or silently corrupt state.
  - **Mitigation**: Guard with `_running` check; `start()` raises if already started.

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
- [ ] Tests for Phase 3 added to `sw/tests/test_pwm_controller.py::TestPhase3`

#### Implementation Details

```
set_duty_cycle(channel_id: str, duty_pct: float) -> None:
    - Calls _require_started()
    - Validates duty_pct against channel range (raises ValueError)
    - Stages pending duty change: self._pending[channel_id].duty = duty_pct

set_frequency(channel_id: str, freq_hz: float) -> None:
    - Calls _require_started()
    - Validates freq_hz against channel range (raises ValueError)
    - Checks ChannelConfig.allow_runtime_freq_change:
        - If False: raises RuntimeError("Runtime frequency changes not allowed for channel X.
          Set allow_runtime_freq_change=True in ChannelConfig to enable.")
        - If True: stages pending frequency change and emits a WARNING log about sysfs glitch

update() -> None:
    - Calls _require_started()
    - t_start = time.monotonic()
    - For each enabled channel in deterministic order (insertion order):
        - Apply staged duty/freq from _pending (if any)
        - If duty changed: pwm.change_duty_cycle(duty)
        - If freq changed: pwm.change_frequency(freq)
        - On exception: fail_safe(); raise PWMUpdateError(channel_id, cause)
    - t_end = time.monotonic()
    - self._timings.append(t_end - t_start)  [deque(maxlen=10000) — bounded]

get_timing_stats() -> dict:
    - Returns {"p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "count": ...}
    - Uses numpy.percentile (numpy is a project dependency in pyproject.toml)
    - Returns {"count": 0} if no samples yet
```

**Deterministic ordering**: channels iterated in registration order (Python 3.7+ dict insertion order).

**Pending state**: `_pending` maps channel_id → `_PendingCommand(duty=None, freq=None)`. `update()` snapshots+clears `_pending` atomically (Phase 4 adds the lock; Phase 3 snapshot is single-threaded for now).

#### Acceptance Criteria
- [ ] `set_frequency()` raises `RuntimeError` when `allow_runtime_freq_change=False`
- [ ] `set_frequency()` proceeds (with WARNING log) when `allow_runtime_freq_change=True`
- [ ] `update()` applies pending duty/freq to all enabled channels in registration order
- [ ] `update()` skips disabled channels
- [ ] On channel write failure, `fail_safe()` is called and `PWMUpdateError` is raised
- [ ] `get_timing_stats()` returns p50/p95/p99 after at least one `update()` call
- [ ] Timing buffer is bounded (deque maxlen=10000)

#### Test Plan
- **Unit Tests** (`TestPhase3`): Mock `HardwarePWM`; verify set_frequency blocks/allows by flag; verify correct call order; verify fail_safe triggered on exception; verify timing stats; verify disabled channels skipped.
- **Non-functional**: Run 10,000 `update()` calls with mock backend; verify `get_timing_stats()` returns valid values.

#### Rollback Strategy
Revert `pwm_controller.py` to Phase 2 state.

#### Risks
- **Risk**: `change_frequency()` triggers duty glitch on BLDC motors — undesirable during runtime.
  - **Mitigation**: Enforced via `allow_runtime_freq_change` flag; false by default. Glitch is documented in API docstring and emits a WARNING when triggered.

---

### Phase 4: Thread-Safe Command Interface
**Dependencies**: Phase 3

#### Objectives
- Wrap all command-staging methods and `update()` with a `threading.Lock` to ensure safe concurrent access from multiple threads.
- Document clearly that cross-process (multi-process) safety requires an external IPC layer and is out of scope for this class. The spec requires cross-process safety but the class API contract (single process, multiple threads) is the implementation boundary; the broader system integration must provide IPC.

#### Deliverables
- [ ] `threading.Lock` protecting `_pending`, `_channels`, `_timings` from concurrent mutation
- [ ] Lock acquired in: `set_duty_cycle()`, `set_frequency()`, `enable_channel()`, `disable_channel()`, `update()`
- [ ] Docstring on `PWMController` documenting thread-safety guarantees and cross-process limitation
- [ ] Tests for Phase 4 added to `sw/tests/test_pwm_controller.py::TestPhase4`

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
- **Unit Tests** (`TestPhase4`): Verify lock is released before sysfs I/O in `update()`.
- **Thread-safety smoke test** (in Phase 5): Spawn 10 threads each calling `set_duty_cycle()` 1000 times concurrently while `update()` runs in a separate thread; assert no exceptions and final state is consistent.

#### Rollback Strategy
Remove lock acquisition from all methods and `_lock` initialization.

#### Risks
- **Risk**: GIL may mask race conditions in CPython testing, giving false confidence.
  - **Mitigation**: Document that tests cover logical races; real deterministic guarantees under GIL are limited. Cross-process safety is explicitly out of scope for this class.
- **Risk**: Spec says class must support concurrent access from multiple processes; plan limits to multi-thread only.
  - **Decision**: Cross-process coordination (e.g., multiprocessing.Queue, shared memory, Unix socket) is the responsibility of the higher-level system. The class API docstring will explicitly state: "Thread-safe for multiple threads in a single process. Cross-process safety requires the caller to route commands through an IPC mechanism." This is the correct architectural boundary for a low-level hardware interface class.

---

### Phase 5: Test Coverage and Stress Tests
**Dependencies**: Phases 1–4

#### Objectives
- Consolidate all incremental tests added in Phases 1–4 into a final passing suite.
- Fill any coverage gaps (target ≥90% on `sw/pwm_controller.py`).
- Add thread-safety stress test and optional latency simulation test.

#### Deliverables
- [ ] `sw/tests/test_pwm_controller.py` — complete test suite passing at ≥90% coverage
- [ ] Thread-safety stress test (10 threads × 1000 concurrent `set_duty_cycle()` calls)
- [ ] Optional: `MockHardwarePWMWithLatency` fixture that adds configurable `time.sleep` per call to validate timing instrumentation under simulated sysfs delays

#### Implementation Details

**Test file structure** (built incrementally across phases):
```
TestPhase1: validation tests
TestPhase2: lifecycle tests (start, stop, fail_safe, enable/disable)
TestPhase3: update, timing stats, frequency opt-in enforcement
TestPhase4: lock behavior (lock not held during I/O)
TestPhase5: coverage gap fill, stress test, latency simulation
```

**Mock fixture** (defined in conftest.py or top of test file):
```python
@pytest.fixture
def mock_hwpwm(monkeypatch):
    """Captures all HardwarePWM calls by channel for assertion."""
    instances = {}
    class MockHardwarePWM:
        def __init__(self, pwm_channel, hz, chip):
            self.pwm_channel = pwm_channel
            self.calls = defaultdict(list)
            instances[pwm_channel] = self
        def start(self, duty): self.calls['start'].append(duty)
        def change_duty_cycle(self, duty): self.calls['duty'].append(duty)
        def change_frequency(self, hz): self.calls['freq'].append(hz)
        def stop(self): self.calls['stop'].append(True)
    monkeypatch.setattr('sw.pwm_controller.HardwarePWM', MockHardwarePWM)
    return instances

@pytest.fixture
def mock_hwpwm_slow(monkeypatch):
    """Simulates sysfs latency for timing instrumentation tests."""
    import time
    class SlowMockHardwarePWM:
        def __init__(self, pwm_channel, hz, chip): pass
        def start(self, duty): pass
        def change_duty_cycle(self, duty): time.sleep(0.0001)  # 0.1ms per call
        def change_frequency(self, hz): time.sleep(0.0001)
        def stop(self): pass
    monkeypatch.setattr('sw.pwm_controller.HardwarePWM', SlowMockHardwarePWM)
```

**Thread-safety stress test**:
```python
def test_concurrent_set_duty_cycle(mock_hwpwm):
    # 10 threads × 1000 calls; verify no exceptions and timing stats are consistent
    ...
```

#### Acceptance Criteria
- [ ] `pytest sw/tests/test_pwm_controller.py` passes with no hardware attached
- [ ] Coverage ≥90% on `sw/pwm_controller.py` (`pytest --cov=sw.pwm_controller`)
- [ ] No test relies on real `HardwarePWM` or sysfs
- [ ] Thread-safety stress test completes without exception
- [ ] Latency simulation test verifies `get_timing_stats()` reports p50 > 0

#### Test Plan
- All unit; no real hardware needed.
- Coverage measured with `pytest --cov=sw.pwm_controller`.

#### Rollback Strategy
Remove stress test and latency simulation tests; tests from Phases 1–4 remain.

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
- **Environment**: Python 3.12+ (per `pyproject.toml`), `rpi_hardware_pwm` installed (already in project), `numpy` installed (project dependency), `pytest`, `pytest-cov`

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
| 2026-03-09 | Plan revision after 3-way review (Amendment 1) | Addressed Gemini COMMENT, Codex REQUEST_CHANGES, Claude COMMENT — see 1-plan-iter1-rebuttals.md |
