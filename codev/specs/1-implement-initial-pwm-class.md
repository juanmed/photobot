# Specification: Raspberry Pi 5 HardwarePWM BLDC Multi-Motor Control Class

## Metadata
- **ID**: spec-2026-03-09-pwm-control-class
- **Status**: draft
- **Created**: 2026-03-09

## Clarifying Questions Asked
- No additional clarifying questions were asked in this phase.
- User-provided requirements:
  - Control multiple BLDC motor drivers from Raspberry Pi 5 with hardware PWM.
  - Each motor can require different frequency and duty cycle.
  - All motors must be controlled simultaneously/in parallel.
  - Operation must be real-time and time synchronized for smooth and safe robot arm motion.
  - Current request scope is limited to specification and single-agent review only.
  - Implementation language is Python; code must follow Pythonic conventions and Python best practices for real-time systems.

## Problem Statement
Robot arm control quality and safety degrade when PWM outputs are not coordinated with deterministic timing. The system needs a dedicated low-level control class that manages multiple hardware PWM channels on Raspberry Pi 5, supports per-motor PWM parameters, applies synchronized updates, and enforces safety behavior under invalid input or timing faults.

## Current State
- No completed specification currently defines the low-level PWM control behavior for this project.
- There is no documented contract for:
  - Multi-motor synchronized PWM updates.
  - Runtime safety constraints (range validation, emergency stop behavior).
  - Real-time update timing expectations.
- Without a clear spec, implementation risks non-deterministic behavior and unsafe motor commands.

## Desired State
A production-ready specification defines a hardware PWM control class that:
- Models each motor channel independently (frequency + duty cycle + enable state), where each channel is configured with explicit valid ranges for frequency and duty cycle; all calls that modify frequency or duty cycle validate inputs against these ranges before any hardware write.
- Applies command updates to all configured motors in a deterministic synchronized control cycle. "Synchronized" is defined as best-effort sequential software writes completing within a bounded window (target: ≤1 ms total across all channels); true hardware-atomic simultaneous latching is not possible via sysfs and is not required.
- Supports periodic command updates at a **target** control loop rate of 1000 Hz. This is aspirational and subject to benchmarking: the `rpi-hardware-pwm` library operates via Linux sysfs file I/O, which incurs multiple syscalls per channel update. Actual achievable rate depends on channel count and hardware; the class design must not assume 1 kHz is always achievable and must expose a measured timing interface. The minimum acceptable rate is TBD (requires hardware benchmarking) and must be established before acceptance tests can be finalized. Note: this control loop rate is independent of the PWM carrier frequency, which may range from a few Hz to MHz depending on the motor driver.
- Does **not** own or run an internal control loop; instead, the class exposes an `update()` method that an external scheduler calls each tick. This keeps loop ownership with the system orchestrator and simplifies timing and threading concerns.
- Enforces fail-safe constraints and safe fallback states when commands or timing are invalid. **Safe state is defined as**: duty cycle set to 0% followed by PWM channel disabled for each managed channel; this behavior produces motor coast on the expected BLDC drivers.
- Is testable via deterministic functional and timing-oriented acceptance criteria.

## Stakeholders
- **Primary Users**: Robotics software developers writing low-level motor control logic.
- **Secondary Users**: System integrators and test engineers validating arm smoothness/safety.
- **Technical Team**: Maintainers of hardware abstraction and robot-control software.
- **Business Owners**: Project owner responsible for robot performance and operational safety.

## Success Criteria
- [ ] A single control class interface exists for managing all motor PWM channels on Raspberry Pi 5.
- [ ] Each motor can be configured with independent target frequency and duty cycle.
- [ ] A synchronized update operation applies new PWM values across all active motors within one control tick.
- [ ] Input validation rejects unsafe PWM values (out-of-range duty/frequency, unknown channel/motor).
- [ ] A fail-safe operation immediately drives all managed motors to a safe state.
- [ ] Control loop jitter stays within documented limits needed for smooth arm control.
- [ ] Unit/integration tests validate normal operation, synchronization behavior, and safety behavior.
- [ ] Documentation describes API contract, timing guarantees, and safety assumptions.

## Constraints
### Technical Constraints
- Must target Raspberry Pi 5 hardware PWM capabilities.
- Must use project-approved `HardwarePWM` interface/library for actual PWM output.
- Must support control of 2–4 BLDC driver channels (configurable channel count).
- Must avoid non-deterministic update ordering that can desynchronize motor outputs.
- Must be thread-safe for concurrent command submission from multiple threads and processes, while preserving deterministic synchronized updates on the dedicated control thread.
- Must be implemented in Python, following Pythonic conventions and Python best practices for real-time systems (e.g., avoiding the GIL for timing-critical paths, preferring `threading` or `multiprocessing` primitives with care, minimizing allocations in hot loops).
- The class must not run its own internal control loop or timing thread. It exposes an `update()` method called by an external scheduler.
- The `chip` parameter for `HardwarePWM` must be configurable at instantiation; do not hardcode chip number.
- Frequency changes at runtime (after initial configuration) are permitted only if the caller explicitly opts in per-channel; they must be documented as potentially causing a momentary 0% duty glitch due to the sysfs backend.
- The safe state (fail-safe) is: for each channel, set duty cycle to 0%, then disable the PWM channel. This produces motor coast behavior.

### Business Constraints
- Safety behavior must be explicit and auditable.
- Scope should stay limited to low-level PWM control class responsibilities.
- Current phase output must stop at spec + single-agent review.

## Assumptions
- BLDC motor drivers accept PWM frequency and duty-cycle signals as control input.
- Frequency and duty-cycle valid ranges are known from hardware documentation and can be codified.
- The software runtime environment can schedule a periodic control loop with stable timing.
- Higher-level kinematics/planning components are out of scope for this spec.

## Solution Approaches

### Selected Approach: Approach 1 — Centralized Synchronized PWM Controller Class

**Rationale**: Approach 1 is selected. A single class owning all PWM channels minimizes coordination complexity, provides a clear enforcement point for safety invariants, and is easier to test deterministically. Approach 2's added modularity does not justify its higher risk of update skew for this use case.

**Description**: One class owns all PWM channels, stores per-motor targets, validates inputs, and commits all channel changes in a deterministic sequential update step (called `update()`). An external scheduler owns and drives the control loop; the class does not spawn or manage any internal timer or loop thread.

**Pros**:
- Clear single source of truth for motor command state.
- Straightforward enforcement of global safety and synchronization invariants.
- Easier to test deterministic behavior.

**Cons**:
- Requires careful internal locking/state management if accessed concurrently.
- Class becomes critical-path component and must be highly robust.

**Estimated Complexity**: Medium
**Risk Level**: Medium

### Approach 2: Per-Motor Controllers with External Synchronizer (Not Selected)
**Description**: Separate controller object per motor, coordinated by an additional sync manager that triggers aligned updates.

**Pros**:
- Better modularity per motor channel.
- Potentially easier extension for heterogeneous motor types.

**Cons**:
- More coordination complexity between objects.
- Higher risk of update skew if synchronization contract is weak.

**Estimated Complexity**: Medium
**Risk Level**: Medium-High

## Open Questions

### Critical (Blocks Progress)
- [ ] Exact motor/driver frequency and duty-cycle ranges per channel. **Provisional**: Until hardware documentation is confirmed, implementation shall use configurable per-channel ranges with safe defaults of 50–400 Hz frequency, 0–100% duty cycle.
- [ ] Required control loop frequency and maximum acceptable jitter for this robot arm. **Provisional**: Target 1000 Hz; minimum acceptable rate TBD pending hardware benchmark. Jitter budget TBD; implementation must expose per-cycle timing metrics to enable threshold validation once hardware data is available.
- [ ] Hardware channel count and mapping constraints on Raspberry Pi 5 for selected `HardwarePWM` backend. **Provisional**: Design for 2–4 channels (Pi 5 has 4 PWM channels: 0–3). The `chip` parameter must be configurable at instantiation (not hardcoded) to decouple from hardware mapping. Existing code uses both `chip=0` and `chip=1`; this must be verified on target hardware.

### Important (Affects Design)
- [x] Threading model expectation (single control thread vs multi-threaded command producers). **Decision**: The class must support concurrent access from multiple threads and processes. It is one component of a larger system that will include higher-level controllers, planners, computer vision, navigation, and localization modules. The design must account for thread-safe command submission while preserving deterministic synchronized updates on the control thread.
- [x] Required behavior when one channel update fails during synchronized commit. **Decision**: On any channel update failure during a synchronized commit, immediately invoke the full fail-safe stop (set duty 0 and disable all channels), log the fault with channel identity and error detail, and raise an exception to the caller. Partial committed state is not acceptable.
- [x] Startup/shutdown sequencing requirements for safe motor arming/disarming. **Decision**: On startup, all channels must be initialized to 0% duty and disabled before any command is accepted. On shutdown or fail-safe, all channels are driven to 0% duty then disabled. Explicit `arm()` / `disarm()` methods may be added; their semantics will be defined in the plan phase.

### Important (Affects Design — sysfs backend)
- [ ] `change_frequency()` in the `rpi-hardware-pwm` library zeroes duty cycle momentarily during frequency changes (writes period=new, then restores duty). This creates a transient 0% output glitch per frequency change. Implementation must document this limitation and specify whether frequency changes are permitted at runtime (vs only at channel configuration time) for BLDC motor safety.

### Nice-to-Know (Optimization)
- [ ] Need for runtime telemetry (per-cycle timing stats, dropped/late update counters).
- [ ] Need for ramping/slew-rate constraints in this class vs higher control layer.

## Performance Requirements
- **Response Time**: Command submission to applied synchronized PWM update in next control tick.
- **Throughput**: Sustain configured control loop rate while updating all active channels every tick.
- **Resource Usage**: Stable memory footprint and bounded CPU usage appropriate for Raspberry Pi 5 real-time control tasks.
- **Availability**: Control class must remain operational throughout active motion sessions with deterministic degradation to safe state on fault.

## Security Considerations
- Restrict command source to trusted in-process components.
- Validate all command values before hardware output.
- Record safety faults and rejected commands for audit/debug purposes.
- Ensure emergency stop path cannot be bypassed by stale queued commands.

## Test Scenarios
### Functional Tests
1. Configure multiple motors with distinct frequency/duty targets and verify correct per-channel application.
2. Submit synchronized multi-motor update and verify all outputs reflect same control-cycle commit.
3. Trigger invalid parameter inputs and verify rejection without unsafe hardware state change.
4. Invoke fail-safe stop and verify all managed outputs transition to safe state immediately.

### Non-Functional Tests
1. Measure control loop update latency across extended runtime using timing instrumentation; capture p50/p95/p99 latencies. Comparison against a concrete threshold is deferred until hardware benchmarks establish a realistic jitter budget.
2. Stress update rate with full channel set and verify no desynchronization or missed safety checks.
3. Fault-injection test for channel update failure and verify all channels immediately transition to safe state (duty=0, disabled).

### Test Harness Notes
- Tests must run on both real hardware and a mock/stub `HardwarePWM` backend. The mock must simulate per-call sysfs latency to validate timing estimates.
- Acceptance criteria for timing-sensitive tests are provisional until the jitter budget is defined from hardware measurements.

## Dependencies
- **External Services**: None.
- **Internal Systems**: Robot control runtime that provides periodic control-cycle scheduling.
- **Libraries/Frameworks**: Raspberry Pi 5 compatible `HardwarePWM` library/interface.

## References
- `codev/protocols/tick/protocol.md`
- Raspberry Pi 5 hardware PWM documentation (to be linked in implementation phase).
- BLDC driver datasheets used by this robot arm (to be linked in implementation phase).

## Risks and Mitigation
| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| PWM update skew between channels causes arm instability | Medium | High | Define synchronized commit semantics and test timing behavior under load |
| Unsafe command values produce dangerous motor behavior | Medium | High | Strict input validation and fail-safe fallback before hardware write |
| Real-time jitter exceeds smooth-control tolerance | Medium | High | Establish measurable jitter budget and include timing regression tests |
| HardwarePWM backend limitations differ from assumptions | Medium | Medium | Validate channel/frequency constraints early against target hardware |

## Expert Consultation
Not yet completed in this document; single-agent consultation result is captured in the paired review artifact for this checkpoint.

## Approval
- [ ] Technical Lead Review
- [ ] Product Owner Review
- [ ] Stakeholder Sign-off
- [ ] Expert AI Consultation Complete

## Notes
- This repository currently has no integrated parent SPIR spec for this topic; strict TICK precondition is therefore not fully satisfied yet.
- This specification serves as the initial feature baseline requested by the user for review before branching and further work.

---

## Amendments

This section tracks all TICK amendments to this specification. TICKs are lightweight changes that refine an existing spec rather than creating a new one.

### Amendment 3 — 2026-03-09: 3-way review feedback (SPIR iteration 1)

Addressed review feedback from Gemini, Codex, and Claude (all REQUEST_CHANGES):

1. **Solution approach selected** (Solution Approaches): Explicitly selected Approach 1 (Centralized Synchronized PWM Controller Class). Added rationale and marked Approach 2 as not selected.
2. **Control loop ownership** (Desired State + Technical Constraints): Clarified the class does not own a control loop. It exposes `update()` for external scheduler invocation.
3. **"Synchronized" definition** (Desired State): Defined as best-effort sequential software writes within a bounded window (≤1 ms target); explicitly disclaimed hardware-atomic simultaneity.
4. **1000 Hz feasibility caveat** (Desired State): Qualified 1000 Hz as aspirational target pending hardware benchmarking with sysfs backend. Added required timing instrumentation.
5. **Safe state defined** (Desired State + Technical Constraints): Defined safe state as duty=0 then channel disabled, producing motor coast behavior.
6. **Critical open questions given provisional values** (Open Questions): Added provisional defaults for frequency/duty ranges, channel count, and chip parameter. Elevated chip configurability to a technical constraint.
7. **Partial failure behavior defined** (Open Questions): On any channel failure during `update()`, invoke full fail-safe stop and raise exception.
8. **Startup/shutdown sequencing** (Open Questions): Defined: all channels initialized to 0%/disabled before accepting commands; disarm follows same sequence.
9. **`change_frequency()` glitch hazard** (Open Questions): Added new Important open question documenting the sysfs backend's transient 0% duty glitch on frequency changes, with decision required on runtime frequency change policy.
10. **Test harness notes** (Test Scenarios): Added mock/stub requirement and provisional acceptance criteria caveat.
11. **Channel count** (Technical Constraints): Changed "simultaneous" to "2–4 configurable channels".

---

### Amendment 2 — 2026-03-09: Architect annotation review (language constraint)

Addressed one `REVIEW(@architect)` annotation:

1. **Implementation language** (Clarifying Questions + Technical Constraints): Specified that implementation must be in Python, following Pythonic conventions and Python best practices for real-time systems. Added to user-provided requirements and as a Technical Constraint with guidance on GIL awareness, concurrency primitives, and minimizing hot-loop allocations.

---

### Amendment 1 — 2026-03-09: Architect annotation review

Addressed three `REVIEW(@architect)` annotations:

1. **Input range validation** (Desired State): Clarified that each channel is configured with explicit valid ranges, and that all frequency/duty-cycle modification calls validate against those ranges before any hardware write.
2. **Control loop rate** (Desired State): Specified the real-time periodic update rate as ≥1000 Hz, and explicitly distinguished it from the PWM carrier frequency (few Hz to MHz range).
3. **Threading model** (Open Questions): Resolved as multi-threaded/multi-process. The class must support concurrent command submission from multiple threads and processes while preserving deterministic synchronized updates on the control thread. Added corresponding Technical Constraint.
