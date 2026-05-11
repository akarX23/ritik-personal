<!--
SYNC IMPACT REPORT
- Version change: 1.0.0 -> 2.0.0
- Modified principles:
  - I. Test-First Development (NON-NEGOTIABLE) -> I. Testing Standards (NON-NEGOTIABLE)
  - II. Code Quality Standards -> II. Code Quality Standards
  - III. User Experience & API Design -> III. User Experience Consistency
  - IV. Performance & Efficiency Requirements -> IV. Performance Requirements
  - V. Reproducibility & Tracking -> V. User Approval Before Any Commit
- Added sections: None
- Removed sections: None
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/tasks-template.md
  - ⚠ pending: .specify/templates/commands/*.md (directory not present in this repository)
- Follow-up TODOs: None
-->

# MNIST SDD Constitution

## Core Principles

### I. Testing Standards (NON-NEGOTIABLE)

All changes MUST follow a test-first workflow.
- Tests MUST be written before implementation for new behavior.
- The Red-Green-Refactor cycle MUST be used for feature and bug-fix work.
- Unit tests MUST cover all new functions and branches.
- Integration tests MUST cover end-to-end user flows that span multiple modules.
- Coverage for changed code MUST be at least 80%.
- A pull request MUST NOT be merged if tests fail.

Rationale: Mandatory test discipline prevents regressions and makes expected behavior explicit.

### II. Code Quality Standards

Code MUST be readable, maintainable, and verifiable.
- All production code MUST pass linting and static analysis.
- Type annotations (or equivalent type contracts) MUST be provided for public interfaces.
- Functions SHOULD remain small and focused; exceptions MUST be justified in review.
- Complex logic MUST be decomposed into testable units.
- TODO or FIXME markers MUST include an issue reference.
- Documentation for externally visible behavior changes MUST be updated in the same change.

Rationale: High-quality code reduces maintenance cost and enables safer iteration.

### III. User Experience Consistency

User-facing behavior MUST be predictable and consistent across interfaces.
- Error messages MUST be actionable and use consistent structure.
- Input validation MUST be explicit, fail-fast, and user-readable.
- Output formats MUST remain stable unless a documented version change is made.
- CLI and API behavior MUST follow shared naming and response conventions.
- Any user-facing change MUST include updated examples in docs or help text.

Rationale: Consistent UX reduces user confusion and lowers support overhead.

### IV. Performance Requirements

Performance MUST be treated as a release quality gate, not a post-release concern.
- Performance budgets MUST be defined in specs for latency and throughput where relevant.
- New performance-critical paths MUST include baseline measurements.
- Changes MUST NOT introduce unexplained regressions against agreed baselines.
- Resource usage (CPU, memory, I/O) MUST be monitored for core workflows.
- Performance trade-offs MUST be documented when accepting slower behavior.

Rationale: Performance requirements preserve reliability and usability at scale.

### V. User Approval Before Any Commit

No code change MAY be committed without explicit user approval.
- The implementer MUST present the change summary and validation results before commit.
- The user MUST explicitly approve the commit action in the current session.
- Auto-commit hooks MUST remain disabled unless the user requests them for that action.
- If approval is not provided, work MAY remain staged or unstaged but MUST NOT be committed.

Rationale: Commit authority remains with the user and ensures intentional history creation.

## Code Quality Gates

The following MUST be satisfied before merge:

- **Linting**: Zero linter violations in changed files
- **Testing**: All relevant unit and integration tests pass
- **Coverage**: Changed code coverage >= 80%
- **Types**: No static type errors in changed scope
- **UX Consistency**: User-facing error messages and outputs follow project conventions
- **Performance**: No unexplained regression against documented baselines
- **Commit Control**: Any commit action has explicit user approval

## Development Workflow & Review Process

1. **Specify**: Define behavior, test expectations, UX constraints, and performance targets.
2. **Test First**: Add failing tests for the intended behavior.
3. **Implement**: Make the minimal change to pass tests while preserving style and contracts.
4. **Validate**: Run lint, tests, type checks, and performance checks as applicable.
5. **Review**: Verify constitutional compliance, including UX consistency and performance impact.
6. **Request Commit Approval**: Present change summary and ask user approval before any commit.
7. **Commit and Merge**: Commit only after approval, then merge via standard review controls.

## Governance

**Constitution Supersession**: This constitution overrides conflicting project practices.

**Amendment Process**:
- Amendments MUST be submitted with rationale and impact analysis.
- Maintainer approval is required before adoption.
- Versioning policy for this constitution:
  - MAJOR: incompatible principle removals or redefinitions
  - MINOR: new principle or materially expanded requirement
  - PATCH: wording clarifications without policy change

**Compliance Verification**:
- Every pull request MUST include a constitution compliance check.
- Reviewers MUST block merge on violations of testing, code quality, UX consistency, or performance requirements.
- Any commit performed through assisted workflows MUST have explicit user approval recorded in-session.

**Runtime Guidance**: Implementation details belong in templates and workflow docs; this document defines mandatory policy.

**Version**: 2.0.0 | **Ratified**: 2026-05-08 | **Last Amended**: 2026-05-11
