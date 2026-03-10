# Review: implement-initial-pwm-class

## Summary

Implemented `PWMController`, a thread-safe centralized hardware PWM control class
for Raspberry Pi 5 BLDC motor control. The class manages up to 4 hardware PWM
channels, validates commands against per-channel configured ranges, applies
synchronized sequential updates driven by an external scheduler, instruments
timing with p50/p95/p99 latency stats, enforces a fail-safe state (duty=0,
channels disabled), and provides a thread-safe command interface that does not
hold the lock during sysfs I/O.

Delivered across 5 plan phases with 3-way consultation (Claude + Codex per phase;
Gemini skipped each phase due to persistent rate limits on gemini-3-pro-preview).

## Spec Compliance

- [x] Single control class for managing all motor PWM channels on Raspberry Pi 5
- [x] Each motor configured with independent frequency and duty cycle + valid ranges
- [x] Synchronized update applies values across all active channels in one tick
- [x] Input validation rejects out-of-range duty/frequency and unknown channels
- [x] Fail-safe operation immediately drives all channels to safe state (duty=0, disabled)
- [x] `update()` driven by external scheduler (class does not own a control loop)
- [x] `get_timing_stats()` exposes p50/p95/p99 latency measurements
- [x] Thread-safe for concurrent command submission (lock released before sysfs I/O)
- [x] Runtime frequency changes opt-in per channel (`allow_runtime_freq_change`)
- [x] `chip` parameter configurable at instantiation (not hardcoded)
- [x] Unit tests and hardware integration tests; 100% line coverage on mock tests
- [x] Hardware tests auto-skipped via `@pytest.mark.hardware` and `conftest.py`

## Deviations from Plan

- **Gemini consultations**: All 5 phases had Gemini skipped due to persistent
  `gemini-3-pro-preview` rate limits (MODEL_CAPACITY_EXHAUSTED). Architect
  approved proceeding with 2/3 consultations (Claude + Codex) each time.
- **Phase 3 — `set_duty_cycle` rewrite**: The stub from Phase 1 was expanded to
  actually stage commands; no functional regression.
- **Phase 4 — `update()` lock refactor**: The lock scope was expanded in an
  iteration (Codex caught that `_channels` enabled flags and `_timings` were
  not protected). The final implementation snapshots `_pending`, enabled flags,
  and hwpwm refs together under one lock then releases before I/O.

## Lessons Learned

### What Went Well

- **Incremental phase structure**: Breaking the implementation into 5 phases
  kept each phase reviewable and small. Codex caught real gaps in phases 1, 2,
  4, and 5 — all were genuine omissions that would have caused issues.
- **Consultation as a QA gate**: The 3-way review reliably surfaced missing
  behaviors (e.g., `set_duty_cycle` stub in Phase 1, `start()` not forcing
  channels disabled in Phase 2, lock not protecting `_channels` in Phase 4).
- **Mock-based testing from the start**: Phase 1 established the mock pattern
  early; later phases reused it consistently.

### Challenges Encountered

- **Gemini rate limits (every phase)**: `gemini-3-pro-preview` was unavailable
  for all 5 implementation phases. Architect approved 2/3 consultation approach.
  This reduced review diversity but Claude and Codex provided sufficient coverage.
- **`patch()` requires module-level attribute**: When `HardwarePWM` was imported
  inside `start()`, `patch("sw.pwm_controller.HardwarePWM")` failed with
  `AttributeError`. Resolved by moving to a module-level `try/except` import.
- **Mock `instances` dict populated lazily**: The `mock_hwpwm` fixture keyed
  instances by `pwm_channel` only after `start()` was called. Tests that
  accessed `instances[0]` before `start()` got `KeyError`. Fixed by moving
  instance access after `ctrl.start()`.
- **Recursive mock side_effect**: Calling `original_change_duty(val)` inside a
  side_effect function caused infinite recursion because `original_change_duty`
  IS the mock method (not a copy of its behavior). Fixed by removing the
  delegate call.

### What Would Be Done Differently

- Import `HardwarePWM` at module level from the start (Phase 1) even though
  it's not used yet, to avoid the test patching surprise in Phase 2.
- Add a note in the plan about the `patch()` + lazy import incompatibility pattern
  as a known Python testing pitfall.

### Methodology Improvements

- **Gemini fallback policy**: When Gemini is repeatedly unavailable across
  multiple phases, a faster fallback (auto-skip with architect notification)
  would save setup time vs. waiting for timeout on each phase.
- **Plan should call out testability requirements**: The plan noted `HardwarePWM`
  is a dependency but didn't specify where to import it — module level vs.
  inline. Adding a testability note would prevent the import location issue.

## Technical Debt

- No formal benchmark on real Raspberry Pi 5 hardware. The 1000 Hz target
  control loop rate is aspirational and requires validation on target hardware.
- `disable_channel()` performs a hardware write outside the lock (the
  `change_duty_cycle(0)` call). This is a minor inconsistency with the
  "all shared state under lock" invariant — acceptable since `disable_channel`
  is not expected to be called from hot-path threads, but worth revisiting.

## Consultation Feedback

### Specify Phase (Round 1)

All three consultants (Gemini, Codex, Claude) returned REQUEST_CHANGES.
All concerns were addressed in spec rebuttals (see `1-specify-iter1-rebuttals.md`).

### Plan Phase (Round 1)

All three consultants reviewed the plan. All concerns addressed in plan rebuttals
(see `1-plan-iter1-rebuttals.md`).

### Phase 1 — Core Data Model (Round 1)

#### Gemini
- SKIPPED (rate limited, architect approved).

#### Codex
- **Concern**: Missing `set_duty_cycle()` stub and pre-start test (plan explicitly
  required this in Phase 1 test plan).
  - **Addressed**: Added `set_duty_cycle()` stub that calls `_require_started()`,
    and `test_set_duty_cycle_before_start_raises` test.

#### Claude
- **Concern** (non-blocking): Initial `freq_hz`/`duty_pct` not validated against
  ranges in `ChannelConfig.__post_init__`.
  - **Rebutted**: Deferred to Phase 2 where `start()` applies initial values.
    Non-blocking, not required by spec for Phase 1.

### Phase 2 — Hardware Lifecycle (Round 1)

#### Gemini
- SKIPPED (rate limited).

#### Codex
- **Concern**: `start()` does not force channels to `enabled=False`; a config
  with `enabled=True` would bypass the "channels start disabled" invariant.
  - **Addressed**: Added `cfg.enabled = False` per channel before hardware init
    in `start()`, and `test_start_forces_channels_disabled` test.

#### Claude
- No concerns raised (APPROVE).

### Phase 3 — Synchronized Update (Round 1)

Both Codex and Claude approved with no changes requested.
Gemini SKIPPED.

### Phase 4 — Thread-Safe Command Interface (Round 1)

#### Gemini
- SKIPPED (rate limited).

#### Codex
- **Concern**: `update()` reads `_channels` enabled flags and appends to
  `_timings` outside the lock. Plan requires protecting `_pending`, `_channels`,
  and `_timings` from concurrent mutation.
  - **Addressed**: Refactored `update()` to snapshot `_pending`, enabled flags,
    and `_hwpwm` refs together under one lock acquisition. `_timings.append()`
    moved inside a lock acquisition after I/O.

#### Claude
- No concerns raised (APPROVE).

### Phase 5 — Test Suite (Round 1)

#### Gemini
- SKIPPED (rate limited).

#### Codex
- **Concern 1**: Stress test runs only concurrent writers, no concurrent
  `update()` thread. Plan requires `update()` to run in a separate thread.
  - **Addressed**: Added a daemon `update_thread` to the stress test that calls
    `update()` continuously while writers run.
- **Concern 2**: Latency test asserts `p50_ms >= 0.0` instead of `> 0.0`.
  Plan requires verifying the timing reflects injected latency.
  - **Addressed**: Changed to `assert stats["p50_ms"] > 0.0`.

#### Claude
- No concerns raised (APPROVE).

## Architecture Updates

Added `sw/pwm_controller.py` — the `PWMController` class for hardware PWM
management. Updated `codev/resources/arch.md` below.

Key components added:
- `ChannelConfig` dataclass: per-channel configuration (frequency, duty ranges,
  hardware channel mapping, runtime freq change opt-in)
- `_PendingCommand` dataclass: staged command values waiting for next `update()`
- `PWMUpdateError` exception: raised when a hardware write fails during `update()`
- `PWMController` class: manages 1–4 HardwarePWM channels with thread-safe
  command staging and deterministic sequential update

## Lessons Learned Updates

Updated `codev/resources/lessons-learned.md` with entries from this project.

## Flaky Tests

No flaky tests encountered.

## Follow-up Items

- Benchmark on real Raspberry Pi 5 to validate 1000 Hz control loop feasibility
- Validate `disable_channel()` hardware-write-outside-lock is acceptable for
  the actual use pattern, or add lock coverage
- Consider adding `set_frequency()` to `enable_channel()` / `disable_channel()`
  docstrings to clarify they also clear pending commands on disable (currently
  they don't — disable only zeroes hardware immediately)
