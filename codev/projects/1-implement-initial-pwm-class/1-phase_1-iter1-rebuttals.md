# Phase 1 Iteration 1 — Rebuttal

## Codex: REQUEST_CHANGES — Missing `set_duty_cycle()` stub and test

**Issue**: `set_duty_cycle()` method not present in `sw/pwm_controller.py`, and
no corresponding test in `TestPhase1` for the pre-start `RuntimeError`.

**Resolution**: ACCEPTED and FIXED.

Added `set_duty_cycle()` stub to `PWMController` that calls `_require_started()`
immediately, raising `RuntimeError` before `start()` has been called.
Added `test_set_duty_cycle_before_start_raises` to `TestPhase1`.

All 27 Phase 1 tests now pass.

## Claude: APPROVE — Non-blocking suggestion

**Suggestion**: Validate initial `freq_hz` / `duty_pct` against configured ranges
in `ChannelConfig.__post_init__`.

**Resolution**: Deferred to Phase 2. The `start()` method (Phase 2) applies
initial values through hardware validation. Adding this in Phase 1 is acceptable
but not required by the spec or plan for this phase.

## Gemini: SKIPPED

Gemini consultation skipped per architect instruction — `gemini-3-pro-preview`
was unavailable due to rate limiting. Proceeding with 2/3 approvals (Claude + Codex).
