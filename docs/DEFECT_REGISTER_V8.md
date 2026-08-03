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

## RCV8-008 — P1 — remote package build

- Evidence: the first remote package job invoked `uv build` on a runner where
  `uv` had not been installed.
- Reproduction: run the package workflow at `82e1dcc`.
- Expected: the package job installs the exact audited builder before use.
- Observed: the job exited before producing package evidence.
- Security/data-loss impact: false package readiness; no production mutation.
- Minimal fix: pin and install `uv==0.11.28` in the read-only package job.
- Regression test: remote package-build workflow at the repaired commit.
- Status: FIXED.
- Commit: `fce6952`.

## RCV8-009 — P1 — clean-clone CI isolation

- Evidence: Linux remote tests read ignored local model paths, historical GPU
  probes, a repository-local virtual environment, and optional EDA binaries.
- Reproduction: run the non-browser suite from a clean clone.
- Expected: unit/fixture tests are independent of models and optional host tools.
- Observed: clean-clone assertions failed although the same tests passed in the
  prepared implementation worktree.
- Security/data-loss impact: false-positive local certification and
  non-reproducible CI.
- Minimal fix: use synthetic GPU evidence, avoid requiring model availability,
  inject the control-plane interpreter, and skip only the explicit EDA
  integration when its native tools are absent.
- Regression test: clean-clone non-browser suite and the remote Linux matrix.
- Status: FIXED in the final CI repair commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-010 — P1 — Windows import portability

- Evidence: Windows CI collection failed on unconditional imports of `fcntl` and
  `resource`.
- Reproduction: collect the suite on Windows/Python 3.11 or 3.12 at `fce6952`.
- Expected: portable control-plane and corpus modules import on supported
  platforms; process locks use native primitives.
- Observed: `ModuleNotFoundError` prevented test collection.
- Security/data-loss impact: Windows package unusable for supported client-side
  and fixture operations.
- Minimal fix: host-native advisory locks, POSIX directory `fsync` only on POSIX,
  and optional parser address-space limits while retaining process timeout and
  upload quotas everywhere.
- Regression test: Windows/Python 3.11 and 3.12 remote jobs.
- Status: FIXED in the final CI repair commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-011 — P2 — CI action runtime

- Evidence: GitHub annotated every job because the pinned checkout and
  setup-python revisions used the deprecated Node.js 20 action runtime.
- Reproduction: inspect remote annotations at `fce6952`.
- Expected: immutable current action pins without runtime-deprecation warnings.
- Observed: GitHub forced the actions onto Node.js 24.
- Security/data-loss impact: future CI breakage risk; no application data impact.
- Minimal fix: resolve official v5/v6 tags and pin their exact commits.
- Regression test: static CI validation and the final remote workflow set.
- Status: FIXED in the final CI repair commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-012 — P1 — live evidence sanitation

- Evidence: the guarded live runner imported a personal identifier-shaped admin
  account into JSON and screenshots.
- Reproduction: inspect `ADMIN_EMAIL` used by the live production certifier.
- Expected: shareable certification evidence contains synthetic invalid-domain
  identities only.
- Observed: a real-looking personal identifier would be retained.
- Security/data-loss impact: private-identifier disclosure in the final archive.
- Minimal fix: override both live accounts with explicit `example.test` fixture
  identities.
- Regression test: static evidence scan plus any conditional live run.
- Status: FIXED in the final CI repair commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-013 — P1 — live-result decision integrity

- Evidence: an input live result with status `FAIL` was normalized to an allowed
  deferred status but did not make a CPU gate fail.
- Reproduction: pass a JSON object with `"status": "FAIL"` to the v8 certifier.
- Expected: an invalid supplied result produces `NO_GO_DEFECTS_REMAIN`.
- Observed: the report could recommend proceeding to controlled live
  certification.
- Security/data-loss impact: false release evidence.
- Minimal fix: retain an explicit supplied-result-validity gate while serializing
  only the bounded status vocabulary.
- Regression test: invalid-result certifier fixture and final integrated run.
- Status: FIXED in the final CI repair commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-014 — P1 — live GPU capability coverage

- Evidence: the initial v8 live runner could return PASS after three GUI views,
  while omitting required personal retrieval, citations, Python verification,
  real SystemVerilog verification, cancellation, provider failure, cross-user
  isolation, and Basic capability enforcement.
- Reproduction: inspect the PASS check set before the final live-gate repair.
- Expected: every Phase 9 capability has direct bounded evidence or an explicit
  supported/not-supported classification.
- Observed: patch preflight text was treated as language-tool verification.
- Security/data-loss impact: false live-release evidence.
- Minimal fix: add real vLLM SSE evidence, owner-private retrieval and citations,
  two model-backed Agent patches, deterministic pytest/Ruff and
  Verilator/Icarus/Yosys verification, isolation/capability checks, cancellation,
  provider-failure handling, and between-group SpecDec checks.
- Regression test: CPU helper tests plus the conditional guarded live gate.
- Status: FIXED in the live-gate completeness commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-015 — P1 — live evidence bundle integrity

- Evidence: live result JSON listed screenshots relative to its external run
  directory, but the final archive copied only operational fixture screenshots.
- Reproduction: inspect the pre-fix archive manifest after supplying a live PASS.
- Expected: every referenced live screenshot is hash-verified against the live
  manifest and included in the final certification archive.
- Observed: referenced evidence was absent from the archive.
- Security/data-loss impact: unverifiable live certification.
- Minimal fix: verify source containment, type, and SHA-256, then copy live
  screenshots into the final bundle; fail the live-result-validity gate on any
  mismatch.
- Regression test: CPU duplicate/path-escape/hash-tamper fixtures plus final
  integrated archive verification.
- Status: FIXED in the live-gate completeness commit.
- Commit: recorded in final `defect_register.json`.

## RCV8-016 — P1 — Windows logical and declared paths

- Evidence: the first collecting Windows matrix interpreted POSIX-declared local
  model paths as relative and serialized governed logical paths with backslashes.
- Reproduction: run the full fixture suite on Windows at `de58c46`.
- Expected: protocol/logical paths remain POSIX canonical, while configuration
  accepts explicit native or POSIX absolute declarations without accessing the
  declared model.
- Observed: model-profile construction failed and governed reference paths
  drifted by host separator.
- Security/data-loss impact: Windows client/fixture incompatibility and
  non-portable provenance identifiers.
- Minimal fix: use `PurePosixPath` for declarations and `.as_posix()` for
  protocol metadata; keep actual file resolution host-native.
- Regression test: Windows 3.11/3.12 matrix plus Linux fixture suite.
- Status: FIXED in the final Windows CI repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-017 — P1 — Windows migration rollback

- Evidence: interrupted migration, recovery, and rollback failed with WinError 5
  while atomically replacing SQLite stores.
- Reproduction: run `tests/test_migrations_v7.py` on Windows at `de58c46`.
- Expected: every read/write SQLite handle closes before restore replacement.
- Observed: `sqlite3.Connection` transaction context managers did not close the
  underlying Windows handles.
- Security/data-loss impact: rollback/recovery unavailable after interruption.
- Minimal fix: wrap every migration SQLite connection in `contextlib.closing`
  while retaining explicit commit/rollback behavior.
- Regression test: migration suite on both Windows matrix versions and local
  v5/v6/v7 rehearsal.
- Status: FIXED in the final Windows CI repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-018 — P1 — CPU-soak determinism

- Evidence: the full suite intermittently failed only the disk-pressure
  scenario, while its isolated rerun passed.
- Reproduction: run the full CPU suite while temporary files are being removed
  on the same filesystem.
- Expected: the synthetic pressure threshold is impossible to satisfy regardless
  of concurrent free-space changes.
- Observed: setting the threshold to the observed free bytes plus one allowed a
  concurrent cleanup to make the admission succeed.
- Security/data-loss impact: false negative reliability certification.
- Minimal fix: set the fixture threshold above total filesystem capacity.
- Regression test: full suite and 64-iteration final CPU soak.
- Status: FIXED in the final reliability repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-019 — P1 — Windows private-state and filesystem identity portability

- Evidence: Windows 3.11/3.12 run `30794147886` rejected registry and artifact
  fixtures because Windows does not expose POSIX mode bits, while unsigned
  64-bit file IDs overflowed SQLite's signed INTEGER adapter.
- Reproduction: run the unit/integration matrix on Windows at `2f1753c`.
- Expected: POSIX permission checks remain strict on POSIX hosts; Windows uses
  its native access controls without interpreting synthetic `st_mode` bits, and
  file identity remains exact in SQLite.
- Observed: 0600/0700 checks saw synthetic 0666 bits and repository registration
  raised `OverflowError`.
- Security/data-loss impact: all registered-account and Agent/worktree flows were
  unavailable on Windows.
- Minimal fix: gate POSIX-bit validation by host and bijectively map unsigned
  64-bit device/inode values into SQLite's signed 64-bit range.
- Regression test: signed-boundary fixture, POSIX permission tests, full local
  suite, and Windows 3.11/3.12 matrix.
- Status: FIXED in the final Windows CI repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-020 — P1 — Windows byte-exact and Linux-process assumptions

- Evidence: Windows run `30794147886` changed hash-bound schema/licence line
  endings, treated regular fixture executables as non-executable, and attempted
  to enumerate `/proc` and probe Unix sanitizers.
- Reproduction: run the unit/integration matrix on Windows at `2f1753c`.
- Expected: byte-pinned repository inputs checkout identically; executable and
  optional-tool checks use host semantics; Linux process inspection is bounded
  to hosts with `/proc`.
- Observed: corpus/schema hashes drifted, serving validation failed, and model
  release tests raised WinError 3/32.
- Security/data-loss impact: false CI failures and unavailable safe lifecycle
  validation on Windows clients.
- Minimal fix: mark pinned inputs `-text`, use Windows file semantics, preserve a
  minimal Windows verifier environment, and guard `/proc`/sanitizer probes.
- Regression test: full local suite and Windows 3.11/3.12 matrix.
- Status: FIXED in the final Windows CI repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-021 — P1 — Windows governance database cleanup

- Evidence: the CPU soak passed its assertions but `TemporaryDirectory.cleanup`
  failed with WinError 32 on `governance.sqlite3` in run `30794147886`.
- Reproduction: run `tests/test_reliability_v7.py` on Windows at `2f1753c`.
- Expected: every governance SQLite connection commits or rolls back and closes
  before fixture cleanup.
- Observed: transaction context managers finalized transactions without
  explicitly closing their Windows file handles.
- Security/data-loss impact: false reliability certification and leaked fixture
  resources.
- Minimal fix: combine transaction contexts with `contextlib.closing` for every
  governance connection.
- Regression test: reliability suite, 64-iteration CPU soak, and Windows matrix.
- Status: FIXED in the final Windows CI repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-022 — P2 — deprecated CI action runtime

- Evidence: successful exact-commit CI run `30795868597` emitted a GitHub warning
  that pinned `actions/upload-artifact` v4.6.2 targets deprecated Node 20 and was
  being forced onto Node 24.
- Reproduction: inspect annotations for package job `91629196539` at `748da1f`.
- Expected: official actions are immutable-pinned to a supported declared runtime.
- Observed: CI succeeded only through runner compatibility forcing.
- Security/data-loss impact: future artifact loss if the compatibility shim is
  removed; no current package contents were exposed.
- Minimal fix: pin the official v7.0.1 commit, whose `action.yml` declares
  `node24`, in both artifact-producing workflows.
- Regression test: static CI validator plus final remote workflow annotations.
- Status: FIXED in the final CI-governance repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-023 — P1 — HTTP concurrency fixture timing race

- Evidence: at exact commit `988fd1f`, Windows 3.12 and both Ubuntu jobs passed,
  while Windows 3.11 job `91631416883` alone observed maximum concurrency one;
  the unchanged fixture used a 50 ms sleep as its only overlap window.
- Reproduction: inspect run `30796582269` and compare it with the all-green
  `748da1f` matrix for the same application/test implementation.
- Expected: the fixture deterministically proves that at least two blocking model
  calls enter the thread pool together.
- Observed: scheduler timing could let every 50 ms window close before another
  request entered, producing a false failure.
- Security/data-loss impact: false release rejection; no production request was
  executed or serialized by the failure.
- Minimal fix: use a bounded two-call event rendezvous; a truly serialized server
  still times out and fails with maximum concurrency one.
- Regression test: repeated local concurrency fixture and final four-job matrix.
- Status: FIXED in the final fixture-determinism repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-024 — P1 — concurrent Windows artifact-root resolution

- Evidence: Windows 3.12 job `91633116806` alone raised
  `artifact_path_escape` during 32 concurrent artifact creations at `dd7ff45`;
  the same test passed on the other three matrix jobs and prior Windows runs.
- Reproduction: run the concurrent artifact consistency fixture on Windows while
  the owner/repository directory does not yet exist.
- Expected: all creators resolve the same owner-private root and retain the
  containment check.
- Observed: resolution occurred before directory creation and raced with
  concurrent `mkdir`, allowing inconsistent Windows canonical forms.
- Security/data-loss impact: safe false rejection with no path escape or partial
  registry row; concurrent artifact publication was unavailable.
- Minimal fix: validate the lexical components, then create and strict-resolve
  the root under the registry's existing re-entrant lock before the canonical
  containment check.
- Regression test: 32-way local fixture and final four-job matrix.
- Status: FIXED in the final artifact-concurrency repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-025 — P1 — live admin independent capabilities

- Evidence: guarded live run `live_gpu_v8_20260803T083911Z` timed out waiting for
  the Knowledge button immediately after successful synthetic-admin activation.
- Reproduction: bootstrap an Operator-tier account without explicit independent
  capabilities and inspect the navigation surface.
- Expected: the synthetic live administrator has the exact Agent, personal
  corpus, and administrative capabilities required by Phase 9.
- Observed: secure Operator defaults intentionally exclude Agent and personal
  corpus, so the new live checks were unreachable.
- Security/data-loss impact: correct fail-closed capability behavior, but false
  live certification failure; no cross-user data was exposed.
- Minimal fix: explicitly enumerate all nine required synthetic-admin
  capabilities in the bootstrap command.
- Regression test: exact capability-set fixture plus guarded live rerun.
- Status: FIXED in the final live-runner repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-026 — P1 — unexpected live-failure evidence

- Evidence: the same failed live run cleaned up its owned P1 process and closed
  both model ports, but left only server logs and no terminal result/manifest.
- Reproduction: inject a Playwright exception after P1 startup.
- Expected: every unexpected failure preserves a bounded category, safe-shutdown
  evidence, GPU ownership result, and hash manifest after cleanup.
- Observed: `main` handled only explicit coordination blocks.
- Security/data-loss impact: shutdown occurred, but its evidence was incomplete.
- Minimal fix: after the runner's ownership-aware `finally`, record a sanitized
  FAIL result and manifest only within an existing safe output root.
- Regression test: CPU exception fixture plus guarded live rerun.
- Status: FIXED in the final live-runner repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-027 — P1 — live chat terminal-state case mismatch

- Evidence: guarded live run `live_gpu_v8_20260803T085749Z` completed both P1
  inference requests and returned `POST /api/v1/chat` with HTTP 200, but the
  browser waited for the full 300-second bound before safe shutdown.
- Reproduction: submit either live chat through the production GUI and compare
  the UI's uppercase `COMPLETE`/`FAILED` state with the runner's title-case
  predicate.
- Expected: the runner observes the production UI terminal-state contract and
  continues immediately after the successful response.
- Observed: the runner waited for unreachable `Complete`/`Failed` prefixes.
- Security/data-loss impact: false live-certification failure and avoidable GPU
  residency; ownership-aware cleanup still closed both endpoints.
- Minimal fix: share the exact uppercase chat terminal-state tuple across both
  live chat waits and their failure checks.
- Regression test: exact terminal-state contract fixture plus guarded live rerun.
- Status: FIXED in the final live-runner synchronization repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-028 — P1 — Verilator timing-option compatibility

- Evidence: guarded live run `live_gpu_v8_d94b4e6_20260803T0924Z` passed
  Icarus compilation/simulation and Yosys synthesis, but installed Verilator
  4.028 rejected the version-5-only `--timing` option.
- Reproduction: run the isolated SystemVerilog verifier with Verilator 4.028.
- Expected: the verifier selects timing arguments supported by the installed
  deterministic tool while retaining all four verification gates.
- Observed: unconditional `--timing` produced `Invalid Option: --timing`.
- Security/data-loss impact: false live-certification failure only; the isolated
  worktree and source checkout were preserved.
- Minimal fix: retain a fail-closed version probe and lint the synthesizable DUT
  without a version-specific timing option; Icarus/vvp independently validates
  timed testbench behavior.
- Regression test: exact design-only Verilator command fixture plus live rerun.
- Status: FIXED in the final live-verifier compatibility repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-029 — P2 — expected provider-failure console classification

- Evidence: the same guarded run recorded one browser console error while its
  intentional provider-failure request correctly returned HTTP 403 and the UI
  reached `FAILED`.
- Reproduction: stop the owned CodeV endpoint and submit the bounded negative
  request through Chromium.
- Expected: the induced HTTP failure is retained as expected negative-path
  evidence while any earlier or unrelated console error still fails the gate.
- Observed: all console errors were treated identically, making the required
  negative-path check incompatible with the no-unexpected-error gate.
- Security/data-loss impact: false certification failure; fail-closed provider
  handling worked and no response or credential was exposed.
- Minimal fix: establish the provider-failure boundary and exclude only known
  403 resource-load console messages after it from the unexpected-error count.
- Regression test: before/expected/after console partition fixture plus live rerun.
- Status: FIXED in the final live-browser evidence repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-030 — P1 — governance disk-pressure fixture race

- Evidence: exact-commit remote run `30801998477` failed only Windows 2025 /
  Python 3.11 because the governance pressure scenario did not raise; the other
  three unit-matrix jobs passed.
- Reproduction: compute the synthetic threshold from current free bytes while
  another process releases temporary filesystem space before admission.
- Expected: the pressure fixture remains impossible to admit on every runner.
- Observed: the threshold was only one byte above a transient free-space sample.
- Security/data-loss impact: false negative CI certification; production quota
  enforcement was not changed.
- Minimal fix: set the fixture threshold one byte above total filesystem
  capacity, which cannot be satisfied by concurrent cleanup.
- Regression test: the corrected governance scenario and full four-job matrix.
- Status: FIXED in the final governance-fixture repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-031 — P1 — legacy Verilator timed-testbench warning

- Evidence: guarded live run `live_gpu_v8_520e92e_20260803T0952Z` proved the
  option-selection repair but Verilator 4.028 then failed on two `STMTDLY`
  warnings from the timed testbench; Icarus simulation and Yosys still passed.
- Reproduction: lint `rtl/tb_example.sv` with Verilator 4.028 after removing the
  unsupported version-5 timing option.
- Expected: Verilator independently lints synthesizable Agent output while the
  timed behavioral testbench is executed by Icarus/vvp.
- Observed: legacy Verilator ignores testbench delays and exits nonzero because
  those warnings are fatal by default.
- Security/data-loss impact: false live-certification failure only; all isolated
  state and ownership-aware cleanup checks passed.
- Minimal fix: restrict Verilator lint to `rtl/example.sv`; retain Icarus compile
  and simulation of both files plus Yosys synthesis of the DUT.
- Regression test: exact design-only Verilator argv fixture plus live rerun.
- Status: FIXED in the final legacy-toolchain compatibility repair.
- Commit: recorded in final `defect_register.json`.

## RCV8-032 — P1 — Basic capability UI synchronization

- Evidence: guarded live run `live_gpu_v8_eb8a9d3_20260803T1010Z` passed every
  model, retrieval, Agent, browser, and toolchain check but intermittently saw
  the prior account's privileged navigation immediately after Basic activation.
- Reproduction: inspect navigation counts as soon as the authentication dialog
  closes; `acceptSession` hides it before asynchronous workspace initialization
  fetches capabilities and rebuilds navigation.
- Expected: capability enforcement is asserted only after the UI displays the
  activated Basic tier, while API enforcement remains fail-closed throughout.
- Observed: a DOM timing race produced a false negative; earlier runs passed.
- Security/data-loss impact: certification-only race. Server-side capability
  checks remained authoritative and no Basic request reached a privileged API.
- Minimal fix: wait up to 30 seconds for `#account-tier` to show `basic` before
  asserting the absence of Agent and Knowledge navigation.
- Regression test: exact account-tier synchronization predicate fixture plus live rerun.
- Status: FIXED in the final live-account synchronization repair.
- Commit: recorded in final `defect_register.json`.
