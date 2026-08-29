import pytest
from research_workspace.candidate_assurance import AssuranceState, CandidateAssurance


def test_mutation_after_verification_becomes_unverified() -> None:
    state = CandidateAssurance(base_revision="abc")
    state = state.observe_candidate("diff-1").start_verification("v1").verification_passed("v1")
    assert state.promotion_eligible
    state = state.observe_candidate("diff-2")
    assert state.state is AssuranceState.UNVERIFIED_CANDIDATE
    assert not state.is_currently_verified
    assert state.mutation_epoch == 2
    assert state.verified_epoch == 1


def test_failed_verification_does_not_prevent_later_mutation() -> None:
    state = CandidateAssurance(base_revision="abc").observe_candidate("diff-1")
    state = state.start_verification("v1").verification_failed()
    state = state.observe_candidate("diff-2")
    assert state.mutation_epoch == 2
    assert state.state is AssuranceState.UNVERIFIED_CANDIDATE


def test_same_fingerprint_does_not_increment_epoch() -> None:
    state = CandidateAssurance(base_revision="abc").observe_candidate("diff-1")
    assert state.observe_candidate("diff-1") == state


def test_cannot_promote_stale_candidate() -> None:
    state = CandidateAssurance(base_revision="abc").observe_candidate("diff-1")
    with pytest.raises(ValueError, match="candidate_not_promotion_eligible"):
        state.promoted()
