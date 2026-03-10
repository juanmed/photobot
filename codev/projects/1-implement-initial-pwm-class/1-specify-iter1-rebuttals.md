# Spec Review Rebuttal — Iteration 1

**Project**: 1-implement-initial-pwm-class
**Phase**: specify
**Iteration**: 1
**Date**: 2026-03-09

All three reviewers (Gemini, Codex, Claude) returned REQUEST_CHANGES with a HIGH confidence verdict. The feedback was consistent and valid. Changes have been made to the spec (Amendment 3). This rebuttal documents what was changed and why.

---

## Gemini Feedback

### Issue 1: Solution Approach Undecided
**ACCEPTED. Changed.**
Approach 1 (Centralized Synchronized PWM Controller Class) is now explicitly selected in the spec with rationale. Approach 2 is retained for reference but marked "Not Selected."

### Issue 2: Conflicting Hardware Mapping Context
**ACCEPTED. Changed.**
The `chip` parameter is now explicitly required to be configurable at instantiation (not hardcoded), documented as a Technical Constraint. The provisional channel-count open question acknowledges the `chip=0` vs `chip=1` discrepancy observed in existing code.

### Issue 3: Synchronization Definition
**ACCEPTED. Changed.**
"Synchronized" is now defined as best-effort sequential software writes completing within a bounded window (target: ≤1 ms across all channels), with an explicit disclaimer that hardware-atomic simultaneous latching is not possible via sysfs and is not required.

---

## Codex Feedback

### Issue 1: Critical Open Questions Still Blocking
**ACCEPTED. Changed.**
All three Critical open questions now have provisional values:
- Frequency/duty ranges: configurable with safe defaults (50–400 Hz, 0–100%).
- Channel count: 2–4 (Pi 5 has channels 0–3); chip number must be configurable.
- Jitter budget: TBD pending hardware benchmark; 1000 Hz target qualified as aspirational.

### Issue 2: Process-safe Concurrency Model Underspecified
**PARTIALLY ACCEPTED.**
The threading model is already resolved (Amendment 1): thread-safe command submission from multiple threads/processes, with deterministic updates on the control thread. The concurrency model (locks, shared state, IPC) is deliberately deferred to the plan phase — this is appropriate for a spec which defines WHAT, not HOW. No change made to concurrency model beyond existing constraint.

### Issue 3: Jitter Budget Undefined
**ACCEPTED. Changed.**
The spec now qualifies 1000 Hz as aspirational, requires timing instrumentation exposing per-cycle latency metrics, and notes that the concrete threshold is TBD pending hardware benchmarking. Test scenarios updated accordingly.

### Issue 4: Fail-Safe Behavior Underspecified
**ACCEPTED. Changed.**
Safe state is now explicitly defined: set duty cycle to 0%, then disable PWM channel for each managed channel (produces motor coast). Partial failure behavior during `update()` is now defined: invoke full fail-safe stop on any channel failure, then raise exception.

### Issue 5: HardwarePWM Backend Constraints Not Grounded
**ACCEPTED. Changed.**
Documented provisional channel count (2–4) and noted that per-channel frequency support and chip constraints must be validated on target hardware. Added configurable `chip` constraint.

---

## Claude Feedback

### Issue 1: 1000 Hz + sysfs Feasibility
**ACCEPTED. Changed.**
The 1000 Hz target is now qualified as aspirational pending benchmarking. The sysfs I/O overhead (multiple file opens/writes per channel per tick) is acknowledged in the spec via the timing instrumentation requirement and caveat. The spec does not promise 1 kHz is achievable.

### Issue 2: "Synchronized" Definition
**ACCEPTED. Changed.** (Same as Gemini Issue 3 above.)

### Issue 3: `change_frequency()` Momentary Glitch
**ACCEPTED. Changed.**
Added a new Important open question documenting the sysfs backend's transient 0% duty cycle glitch during frequency changes. A decision is required (and tracked) on whether runtime frequency changes are permitted after initial configuration, and what safety constraints apply.

### Issue 4: No Solution Approach Selected
**ACCEPTED. Changed.** (Same as Gemini Issue 1 above.)

### Issue 5: Safe State Undefined
**ACCEPTED. Changed.** (Same as Codex Issue 4 above.)

### Issue 6: Control Loop Ownership Unspecified
**ACCEPTED. Changed.**
The spec now explicitly states the class does not own a control loop. It exposes an `update()` method called by an external scheduler. Added to both Desired State and Technical Constraints.

### Issue 7: Critical Open Questions Should Be Resolved or Given Provisional Values
**ACCEPTED. Changed.** (Same as Codex Issue 1 above.)

---

## Summary of Changes Made

All major issues accepted. The spec (Amendment 3) now:
- Explicitly selects Approach 1
- Defines safe state (duty=0 + disabled = motor coast)
- Defines control loop ownership (external scheduler calls `update()`)
- Qualifies 1000 Hz as aspirational with sysfs caveat
- Defines "synchronized" with measurable bound (≤1 ms sequential window)
- Provides provisional values for all Critical open questions
- Defines partial failure behavior (full fail-safe + exception)
- Defines startup/shutdown sequencing
- Documents `change_frequency()` glitch hazard as tracked open question
- Adds configurable chip parameter as technical constraint
- Updates test scenarios with mock harness note and provisional threshold caveat

No review feedback was rejected.
