# Laplace v8 defect register

## RCV8-001 — P0 — GPU ownership observation

- Evidence: `observe_gpu()` accepted a successful GPU summary when the separate
  compute-process query failed, returning an empty PID tuple.
- Reproduction: inject a successful GPU query followed by return code 9 for the
  compute query.
- Expected: uncertain process ownership must block admission.
- Observed: admission could treat the GPU as having no compute processes.
- Security/data-loss impact: a Laplace model could contend with or disrupt a
  protected SpecDec workload.
- Minimal fix: reject failed or malformed compute queries and classify resolved
  PIDs through command, parent, and working-directory evidence.
- Regression test: `tests/test_tiered_serving.py` GPU-observation cases and
  `tests/test_gpu_coordination_v8.py`.
- Status: FIXED.
- Commit: `6b21736`.

## RCV8-002 — P1 — desktop patch apply

- Evidence: a fixture target with an unrelated tracked edit accepted an incoming
  patch and retained both edits.
- Reproduction: modify `other.py` in the target, then apply a valid patch for
  `module.py` at the same HEAD.
- Expected: conflict before any write.
- Observed: `APPLIED`, with both files dirty.
- Security/data-loss impact: unrelated local changes could be silently combined
  with a remote change set.
- Minimal fix: require an entirely clean target worktree before `git apply
  --check`.
- Regression test:
  `test_patch_application_rejects_dirty_target_without_combining_changes`.
- Status: FIXED.
- Commit: `6dab923`.

## RCV8-003 — P1 — package entry-point certification

- Evidence: v7 clean-install smoke imported only the package. In the same
  dependency-empty environment, installed `laplace --version` failed while
  importing PyYAML.
- Reproduction: install the v7 wheel with `--no-deps --no-index`, unset
  `PYTHONPATH`, and run `bin/laplace --version`.
- Expected: the certified version entry point runs and reports exact package/build
  identity.
- Observed: `ModuleNotFoundError: yaml`.
- Security/data-loss impact: none; release usability and false-positive package
  certification.
- Minimal fix: dependency-light version dispatch, package readme/license metadata,
  wheel symlink rejection, per-run build cache, and actual entry-point execution.
- Regression test: `test_dependency_light_packaged_version_entrypoint` plus
  `scripts/package_release.py`.
- Status: FIXED.
- Commit: `71358b2`.

## RCV8-004 — P1 — desktop sync plan integrity

- Evidence: `DesktopSyncClient.prepare()` checked only SHA-256, accepting a plan
  with a false byte count or changed-path list.
- Reproduction: copy a valid plan and change `patch_size_bytes` or
  `changed_paths`.
- Expected: client rejects a plan/patch mismatch before durable staging.
- Observed: the operation was staged with inconsistent progress and audit
  metadata.
- Security/data-loss impact: misleading approval and transfer evidence.
- Minimal fix: re-run bounded patch validation and require exact size and path
  equality.
- Regression test: `test_prepare_revalidates_patch_size_and_path_plan`.
- Status: FIXED.
- Commit: `e725854`.

## RCV8-005 — P2 — desktop branch binding

- Evidence: patch apply checked HEAD but not branch; two branches at the same
  commit were treated as equivalent.
- Reproduction: clone the source, create `other-branch` at the same HEAD, and
  apply a plan created on the source branch.
- Expected: explicit branch conflict.
- Observed: patch accepted on the unintended branch.
- Security/data-loss impact: wrong-branch modification; no force push.
- Minimal fix: bind apply to the exact branch or explicit `DETACHED` state.
- Regression test:
  `test_patch_application_rejects_same_head_on_different_branch`.
- Status: FIXED.
- Commit: `e725854`.

## RCV8-006 — P1 — release-candidate CI entry point

- Evidence: the manual release workflow invoked the v7 certification command and
  hard-coded the former GPU-unavailable status.
- Reproduction: inspect `.github/workflows/release-candidate.yml` at the v7 base.
- Expected: v8 workflow invokes the v8 CPU/fixture certifier and permits only the
  dedicated v8 branch for review validation.
- Observed: a remote run could not produce the v8 evidence contract.
- Security/data-loss impact: false release evidence; no production mutation.
- Minimal fix: install the v8 certifier, update workflow and validator, and keep
  permissions read-only.
- Regression test: `scripts/validate_ci.py` and remote workflow validation.
- Status: FIXED in the certification/CI commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-007 — P1 — certification evidence sanitation

- Evidence: the registered-GUI fixture serialized a real-looking personal email
  address into operational JSON.
- Reproduction: run `scripts/run_registered_gui_fixture_smoke.py` at the v7 base
  and inspect `account.email`.
- Expected: certification uses only clearly synthetic identities.
- Observed: personal identifier-shaped fixture data was retained.
- Security/data-loss impact: private-identifier disclosure in a shareable archive.
- Minimal fix: replace the account and user ID with invalid-domain synthetic
  fixture identifiers.
- Regression test: rerun the registered GUI fixture and scan output.
- Status: FIXED.
- Commit: `b01b871`.
