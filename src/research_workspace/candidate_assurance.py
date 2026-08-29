"""Pure candidate-assurance state for long-running repository development.

This module intentionally contains no filesystem, SQLite, worktree, or process
logic.  ``agent_sandbox`` remains authoritative for repository isolation and
must persist/drive this state explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class AssuranceState(StrEnum):
    CLEAN = "clean"
    LEGACY_UNVERIFIED = "legacy_unverified"
    OBSERVATION_FAILED = "observation_failed"
    UNVERIFIED_CANDIDATE = "unverified_candidate"
    VERIFYING = "verifying"
    VERIFICATION_FAILED = "verification_failed"
    VERIFIED_CANDIDATE = "verified_candidate"
    PROMOTED = "promoted"


@dataclass(frozen=True, slots=True)
class VerificationBinding:
    base_revision: str
    mutation_epoch: int
    candidate_fingerprint: str
    verifier_digest: str


@dataclass(frozen=True, slots=True)
class CandidateAssurance:
    base_revision: str
    mutation_epoch: int = 0
    verified_epoch: int = 0
    candidate_fingerprint: str = ""
    verified_fingerprint: str = ""
    verifier_digest: str = ""
    state: AssuranceState = AssuranceState.CLEAN

    def __post_init__(self) -> None:
        if not self.base_revision:
            raise ValueError("base_revision_required")
        if self.mutation_epoch < 0 or self.verified_epoch < 0:
            raise ValueError("epoch_negative")
        if self.verified_epoch > self.mutation_epoch:
            raise ValueError("verified_epoch_ahead")
        if self.state is AssuranceState.VERIFIED_CANDIDATE and not self.is_currently_verified:
            raise ValueError("verified_state_stale")
        if self.state is AssuranceState.PROMOTED and not self.is_currently_verified:
            raise ValueError("promoted_state_unverified")

    @property
    def is_currently_verified(self) -> bool:
        return (
            self.mutation_epoch == self.verified_epoch
            and bool(self.candidate_fingerprint)
            and self.candidate_fingerprint == self.verified_fingerprint
        )

    @property
    def promotion_eligible(self) -> bool:
        return self.state is AssuranceState.VERIFIED_CANDIDATE and self.is_currently_verified

    def observe_candidate(self, fingerprint: str) -> "CandidateAssurance":
        """Record a materially new candidate exactly once per fingerprint."""
        if not fingerprint:
            raise ValueError("candidate_fingerprint_required")
        if fingerprint == self.candidate_fingerprint:
            return self
        return replace(
            self,
            mutation_epoch=self.mutation_epoch + 1,
            candidate_fingerprint=fingerprint,
            state=AssuranceState.UNVERIFIED_CANDIDATE,
        )

    def observe_clean(self) -> "CandidateAssurance":
        """Clear the current candidate after a worktree returns to its base state."""

        if self.state is AssuranceState.CLEAN and not self.candidate_fingerprint:
            return self
        return replace(self, candidate_fingerprint="", state=AssuranceState.CLEAN)

    def observation_failed(self) -> "CandidateAssurance":
        """Preserve the last candidate when Git can no longer be observed."""

        return replace(self, state=AssuranceState.OBSERVATION_FAILED)

    def start_verification(self, verifier_digest: str) -> "CandidateAssurance":
        if not self.candidate_fingerprint:
            raise ValueError("candidate_required")
        if not verifier_digest:
            raise ValueError("verifier_digest_required")
        return replace(self, verifier_digest=verifier_digest, state=AssuranceState.VERIFYING)

    def verification_failed(self) -> "CandidateAssurance":
        if self.state is not AssuranceState.VERIFYING:
            raise ValueError("verification_not_running")
        return replace(self, state=AssuranceState.VERIFICATION_FAILED)

    def verification_passed(self, verifier_digest: str) -> "CandidateAssurance":
        if self.state is not AssuranceState.VERIFYING:
            raise ValueError("verification_not_running")
        if verifier_digest != self.verifier_digest:
            raise ValueError("verifier_digest_changed")
        return replace(
            self,
            verified_epoch=self.mutation_epoch,
            verified_fingerprint=self.candidate_fingerprint,
            state=AssuranceState.VERIFIED_CANDIDATE,
        )

    def binding(self) -> VerificationBinding:
        if not self.is_currently_verified or not self.verifier_digest:
            raise ValueError("candidate_not_verified")
        return VerificationBinding(
            base_revision=self.base_revision,
            mutation_epoch=self.verified_epoch,
            candidate_fingerprint=self.verified_fingerprint,
            verifier_digest=self.verifier_digest,
        )

    def promoted(self) -> "CandidateAssurance":
        if not self.promotion_eligible:
            raise ValueError("candidate_not_promotion_eligible")
        return replace(self, state=AssuranceState.PROMOTED)
