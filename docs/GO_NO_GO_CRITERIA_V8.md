# Laplace v8 go/no-go criteria

## Mandatory non-GPU gates

All of the following must be PASS:

1. exact certified base ancestry, dedicated v8 branch, clean implementation
   commit, unchanged clean stable checkout, and no production state/process
   mutation;
2. full Python fixture suite, browser suite, Ruff, strict mypy, Bandit,
   compileall, and `git diff --check`;
3. v5/v6/v7 copied-state migration dry-run, backup, apply, integrity, restart,
   rollback, reapply, idempotency, and interruption recovery across 11 stores;
4. local CI contract validation and the Linux/Windows Python 3.11/3.12 matrix,
   either remotely PASS or exactly
   `NOT_EXECUTED_ENVIRONMENT_LIMITATION`;
5. reproducible wheel/sdist, dependency-empty import and console smoke, safe
   members, exact version/revision, offline dependency/SBOM/license inventory;
6. operational, personal-corpus, Agent-worktree, governance, remote-access,
   desktop-sync, documentation, security, offline-evaluation, and CPU-soak
   evidence;
7. zero open P0/P1 defects and no undocumented P2 defect affecting the release
   contract;
8. sanitized manifest-verified certification archive.

Any failed mandatory gate yields `NO_GO_DEFECTS_REMAIN`.

## Conditional live-GPU gate

Live certification is eligible only after the mandatory CPU gates pass and a
fresh ownership snapshot resolves every compute PID. SpecDec or a protected
association yields `BLOCKED_BY_SPECDEC_ACTIVE`; incomplete ownership evidence
yields `BLOCKED_BY_UNCERTAIN_GPU_OWNERSHIP`. Neither status is a GPU failure.

Before and after each bounded Laplace model group, ownership is checked once.
If SpecDec appears after Laplace starts, only Laplace-owned process groups are
released and the result is `YIELDED_TO_SPECDEC`; the GPU is not reacquired.

The pre-live decision is `GO_FOR_CONTROLLED_LIVE_GPU_CERTIFICATION`. The final
decision may be `GO_FOR_RELEASE_REVIEW_AFTER_LIVE_CERTIFICATION` only when every
live check and safe-shutdown check is PASS. A blocked conditional gate does not
permit that final release-review decision.

