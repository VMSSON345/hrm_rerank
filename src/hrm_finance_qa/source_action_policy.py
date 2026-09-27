from __future__ import annotations

from typing import Sequence


def classify_source_action(
    *,
    missing_periods: Sequence[str],
    verification_state: str,
    required_states: Sequence[str],
    mapping_resolution: str,
    abstained: bool,
) -> str:
    """Classify the public action from proof state alone.

    Confidence calibration decides the abstained input before this function is
    called; it cannot bypass missing evidence, invalid proof, or non-unique
    physical mapping here.
    """

    if missing_periods:
        return "ABSTAIN"
    if (
        verification_state == "INVALID"
        or "INVALID" in required_states
        or mapping_resolution == "CONFLICT"
    ):
        return "REJECT"
    if (
        mapping_resolution != "UNIQUE"
        or abstained
        or not required_states
        or any(state != "VALID" for state in required_states)
    ):
        return "ABSTAIN"
    return "ANSWER"

from dataclasses import dataclass, replace
from .calibration_policy import CalibrationPolicy, CalibrationPolicyError

CALIBRATED_ACTION_POLICY = "v14.2-policy-calibrated-public-answer-v1"
EVIDENCE_ACTION_POLICY = "v14.2-policy-evidence-public-answer-experimental-v1"
PUBLICATION_POLICIES = (CALIBRATED_ACTION_POLICY, EVIDENCE_ACTION_POLICY)


def validate_publication_policy(policy_id: str) -> str:
    if policy_id not in PUBLICATION_POLICIES:
        raise ValueError(f"Unknown publication policy: {policy_id!r}")
    return policy_id


@dataclass(frozen=True)
class AnswerDecision:
    policy_id: str
    proof_eligible: bool
    calibration_status: str
    calibration_accepted: bool
    execution_status: str = "NOT_RUN"
    certificate_status: str = "NOT_RUN"
    presentation_status: str = "NOT_RUN"
    final_action: str | None = None
    acceptance_basis: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def policy_accepted(self) -> bool:
        if not self.proof_eligible:
            return False
        if self.policy_id == EVIDENCE_ACTION_POLICY:
            return True
        return (self.policy_id == CALIBRATED_ACTION_POLICY
                and self.calibration_status == "CALIBRATED" and self.calibration_accepted)


def decide_answer(source_action: str, policy_payload: object, confidence: float,
                  *, reason_codes: Sequence[str] = (),
                  policy_id: str = CALIBRATED_ACTION_POLICY) -> AnswerDecision:
    """Admit to execution under an explicit policy, never directly to publication.

    B retains genuine calibration diagnostics without using them as a veto.
    Executor/certificate/presentation must still finish the decision.
    """
    validate_publication_policy(policy_id)
    status, accepted = "UNCALIBRATED", False
    if isinstance(policy_payload, dict):
        try:
            policy = CalibrationPolicy.from_dict(policy_payload)
            status, accepted = policy.status.value, policy.accepts(confidence)
        except (CalibrationPolicyError, ValueError, TypeError, KeyError):
            pass
    proof = source_action == "ANSWER"
    reasons = list(reason_codes)
    if not proof:
        reasons.append("public_source_gate:" + source_action)
    if not accepted and policy_id == CALIBRATED_ACTION_POLICY:
        reasons.append("calibration:" + status)
    policy_allowed = proof and (policy_id == EVIDENCE_ACTION_POLICY or accepted)
    return AnswerDecision(
        policy_id=policy_id, proof_eligible=proof,
        calibration_status=status, calibration_accepted=accepted,
        final_action=("REJECT" if source_action == "REJECT" else "ABSTAIN")
        if not policy_allowed else None,
        acceptance_basis=(policy_id,) if policy_allowed else (),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def finish_answer_decision(decision: AnswerDecision, action: str, *, execution_failed: bool = False,
                           reason_codes: Sequence[str] = ()) -> AnswerDecision:
    """Called only after the envelope's independent execution/certificate validation."""
    if action == "ANSWER":
        if not decision.policy_accepted:
            raise ValueError("Cannot promote a policy-rejected decision")
        return replace(decision, execution_status="VALID", certificate_status="VALID",
                       presentation_status="VALID", final_action="ANSWER")
    return replace(decision, final_action=action,
                   execution_status=("INVALID" if execution_failed else "UNKNOWN") if decision.policy_accepted else "NOT_RUN",
                   certificate_status=("NOT_RUN" if execution_failed else "UNKNOWN") if decision.policy_accepted else "NOT_RUN",
                   presentation_status="INVALID" if decision.policy_accepted else "NOT_RUN",
                   reason_codes=tuple(dict.fromkeys((*decision.reason_codes, *reason_codes,
                                                    "final_envelope:" + action))))


def aggregate_output_actions(actions: Sequence[str]) -> str:
    if actions and all(action == "ANSWER" for action in actions):
        return "ANSWER"
    if "ANSWER" in actions:
        return "PARTIAL"
    return "REJECT" if "REJECT" in actions else "ABSTAIN"
