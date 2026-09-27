"""Bounded exact execution and proof replay for multi-step finance programs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

from .exact_executor import (
    DEFAULT_DECIMAL_PRECISION,
    ExactExecutionError,
    ExactExecutionResult,
    ExactOperand,
    InvalidProofCertificateError,
    LeanVerificationResult,
    ProofCertificate,
    UnitSpec,
    _fraction,
    _fraction_from_payload,
    _lean_fraction,
    _safe_evidence_ref,
    evidence_reference_from_candidate,
    execute_decimal_operation,
    execute_financial_program,
    parse_unit,
    verify_proof_certificate,
)


DAG_CERTIFICATE_SCHEMA_VERSION = "finhrm-exact-dag-proof-v1"
DEFAULT_MAX_DAG_NODES = 32


@dataclass(frozen=True)
class ExactDAGInput:
    kind: str
    reference: int | str


@dataclass(frozen=True)
class ExactDAGNode:
    node_id: str
    operation: str
    inputs: tuple[ExactDAGInput, ...]


@dataclass(frozen=True)
class DAGProofCertificate:
    schema_version: str
    nodes: tuple[dict[str, object], ...]
    output_node: str
    leaf_evidence_refs: tuple[dict[str, object], ...]
    decimal_precision: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "nodes": [dict(item) for item in self.nodes],
            "output_node": self.output_node,
            "leaf_evidence_refs": [dict(item) for item in self.leaf_evidence_refs],
            "decimal_precision": self.decimal_precision,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DAGProofCertificate":
        try:
            return cls(
                schema_version=str(payload["schema_version"]),
                nodes=tuple(dict(item) for item in payload["nodes"]),  # type: ignore[arg-type]
                output_node=str(payload["output_node"]),
                leaf_evidence_refs=tuple(
                    dict(item) for item in payload["leaf_evidence_refs"]  # type: ignore[arg-type]
                ),
                decimal_precision=int(payload["decimal_precision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidProofCertificateError(
                "Malformed exact DAG proof certificate"
            ) from exc


@dataclass(frozen=True)
class ExactDAGExecutionResult:
    output_node: str
    output: ExactExecutionResult
    node_results: tuple[tuple[str, ExactExecutionResult], ...]
    certificate: DAGProofCertificate

    @property
    def decimal_value(self) -> Decimal:
        return self.output.decimal_value


def _adapt_node(raw_node: object) -> ExactDAGNode:
    inputs: list[ExactDAGInput] = []
    for raw_input in tuple(getattr(raw_node, "inputs")):
        if isinstance(raw_input, tuple) and len(raw_input) == 2:
            kind, reference = raw_input
        else:
            kind = getattr(raw_input, "kind", "")
            reference = getattr(raw_input, "reference", None)
        if str(kind) == "candidate" and isinstance(reference, int):
            inputs.append(ExactDAGInput("candidate", reference))
        elif str(kind) == "evidence" and isinstance(reference, str):
            inputs.append(ExactDAGInput("evidence", reference))
        elif str(kind) == "node" and isinstance(reference, str):
            inputs.append(ExactDAGInput("node", reference))
        else:
            raise ExactExecutionError(f"Invalid DAG input: {kind!r}/{reference!r}")
    return ExactDAGNode(
        node_id=str(getattr(raw_node, "node_id")),
        operation=str(getattr(raw_node, "operation")).upper(),
        inputs=tuple(inputs),
    )


def _deduplicate_references(
    references: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for reference in references:
        key = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return tuple(output)


def _execute_nodes(
    nodes: Sequence[ExactDAGNode],
    output_node: str,
    *,
    candidate_operands: Mapping[int, ExactOperand] | None = None,
    evidence_operands: Mapping[str, ExactOperand] | None = None,
    output_unit: str | UnitSpec | None = None,
    require_units: bool = True,
    require_evidence: bool = True,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
    max_nodes: int = DEFAULT_MAX_DAG_NODES,
) -> ExactDAGExecutionResult:
    if not nodes:
        raise ExactExecutionError("A DAG program must contain at least one node")
    if not 1 <= max_nodes <= DEFAULT_MAX_DAG_NODES:
        raise ExactExecutionError(
            f"max_nodes must be between 1 and {DEFAULT_MAX_DAG_NODES}"
        )
    if len(nodes) > max_nodes:
        raise ExactExecutionError(
            f"DAG has {len(nodes)} nodes, exceeding the bound of {max_nodes}"
        )
    candidates = {} if candidate_operands is None else dict(candidate_operands)
    evidence = {} if evidence_operands is None else dict(evidence_operands)
    results: dict[str, ExactExecutionResult] = {}
    records: list[dict[str, object]] = []
    leaf_refs: list[dict[str, object]] = []

    for node_index, node in enumerate(nodes):
        if not node.node_id or node.node_id in results:
            raise ExactExecutionError("DAG node IDs must be non-empty and unique")
        operands: list[ExactOperand] = []
        input_records: list[dict[str, object]] = []
        for slot, item in enumerate(node.inputs):
            if item.kind == "candidate" and isinstance(item.reference, int):
                operand = candidates.get(item.reference)
                if operand is None:
                    raise ExactExecutionError(
                        "DAG candidate input is outside the candidate list"
                    )
                evidence_ref = _safe_evidence_ref(operand.evidence_ref)
                leaf_refs.append(evidence_ref)
                input_records.append(
                    {
                        "kind": "candidate",
                        "reference": item.reference,
                        "evidence_ref": evidence_ref,
                    }
                )
            elif item.kind == "evidence" and isinstance(item.reference, str):
                operand = evidence.get(item.reference)
                if operand is None:
                    raise ExactExecutionError(
                        f"Unknown DAG evidence input: {item.reference}"
                    )
                evidence_ref = _safe_evidence_ref(operand.evidence_ref)
                leaf_refs.append(evidence_ref)
                input_records.append(
                    {
                        "kind": "evidence",
                        "reference": item.reference,
                        "evidence_ref": evidence_ref,
                    }
                )
            elif item.kind == "node" and isinstance(item.reference, str):
                prior = results.get(item.reference)
                if prior is None:
                    raise ExactExecutionError(
                        "DAG node inputs must reference an earlier node"
                    )
                operand = ExactOperand(
                    value=prior.exact_value,
                    unit=prior.unit,
                    evidence_ref={"node_id": item.reference},
                    operand_id=f"node:{item.reference}",
                )
                input_records.append({"kind": "node", "reference": item.reference})
            else:
                raise ExactExecutionError(f"Unknown DAG input kind: {item.kind}")
            if not operand.operand_id:
                operand = ExactOperand(
                    value=operand.value,
                    unit=operand.unit,
                    evidence_ref=operand.evidence_ref,
                    operand_id=f"{node.node_id}:input_{slot + 1}",
                )
            operands.append(operand)

        node_output_unit = output_unit if node.node_id == output_node else None
        result = execute_financial_program(
            node.operation,
            operands,
            output_unit=node_output_unit,
            require_units=require_units,
            require_evidence=require_evidence,
            decimal_precision=decimal_precision,
        )
        results[node.node_id] = result
        records.append(
            {
                "node_id": node.node_id,
                "node_index": node_index,
                "operation": node.operation,
                "inputs": input_records,
                "proof": result.certificate.to_dict(),
            }
        )

    if output_node not in results:
        raise ExactExecutionError("DAG output node is missing")
    certificate = DAGProofCertificate(
        schema_version=DAG_CERTIFICATE_SCHEMA_VERSION,
        nodes=tuple(records),
        output_node=output_node,
        leaf_evidence_refs=_deduplicate_references(leaf_refs),
        decimal_precision=decimal_precision,
    )
    return ExactDAGExecutionResult(
        output_node=output_node,
        output=results[output_node],
        node_results=tuple(results.items()),
        certificate=certificate,
    )


def execute_reasoning_dag(
    program: object,
    candidates: Sequence[object],
    *,
    output_unit: str | UnitSpec | None = None,
    require_units: bool = True,
    require_evidence: bool = True,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
    max_nodes: int = DEFAULT_MAX_DAG_NODES,
) -> ExactDAGExecutionResult:
    nodes = tuple(_adapt_node(node) for node in tuple(getattr(program, "nodes")))
    candidate_operands = {
        index: ExactOperand(
            value=getattr(candidate, "value"),
            unit=getattr(candidate, "unit", None),
            evidence_ref=evidence_reference_from_candidate(candidate),
            operand_id=f"candidate:{index}",
        )
        for index, candidate in enumerate(candidates)
    }
    return _execute_nodes(
        nodes,
        str(getattr(program, "output_node")),
        candidate_operands=candidate_operands,
        output_unit=output_unit,
        require_units=require_units,
        require_evidence=require_evidence,
        decimal_precision=decimal_precision,
        max_nodes=max_nodes,
    )


def _presentation_leaf_value(operand: ExactOperand) -> Fraction:
    """Mirror the frozen envelope's currency-scale semantics exactly."""
    raw_value = _fraction(operand.value)
    if operand.unit is None or not str(operand.unit).strip():
        return raw_value
    unit = parse_unit(operand.unit)
    has_currency = any(name == "currency" for name, _ in unit.dimensions)
    return raw_value * unit.scale_to_canonical if has_currency else raw_value

def execute_program_records(
    records: Sequence[tuple[str, str, Sequence[str], str] | tuple[str, str, Sequence[str]]],
    operands_by_ref: Mapping[str, ExactOperand],
    *,
    output_node: str,
    output_unit: str | UnitSpec | None = None,
    require_units: bool = True,
    require_evidence: bool = True,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
) -> ExactDAGExecutionResult:
    """Replay ``VerifiedAnswerEnvelope.program`` records from cited inputs."""

    produced: set[str] = set()
    nodes: list[ExactDAGNode] = []
    presentation_results: dict[str, Fraction] = {}
    for record in records:
        raw_node_id, raw_operation, raw_inputs = record[:3]
        node_id = str(raw_node_id)
        operation = str(raw_operation).upper()
        inputs: list[ExactDAGInput] = []
        presentation_inputs: list[Fraction] = []
        for reference in raw_inputs:
            if reference in produced:
                inputs.append(ExactDAGInput("node", reference))
                presentation_inputs.append(presentation_results[str(reference)])
            elif reference in operands_by_ref:
                inputs.append(ExactDAGInput("evidence", reference))
                presentation_inputs.append(_presentation_leaf_value(operands_by_ref[str(reference)]))
            else:
                raise ExactExecutionError(f"Unknown program-record input: {reference}")
        computed = Fraction(execute_decimal_operation(
            operation, presentation_inputs, decimal_precision=decimal_precision
        ))
        if len(record) >= 4 and str(record[3]).strip():
            try:
                recorded = Fraction(Decimal(str(record[3])))
            except Exception as exc:
                raise ExactExecutionError(f"Invalid recorded result for DAG node {node_id}") from exc
            if computed != recorded:
                raise ExactExecutionError(f"Recorded result mismatch for DAG node {node_id}")
        presentation_results[node_id] = computed
        nodes.append(ExactDAGNode(node_id, operation, tuple(inputs)))
        produced.add(node_id)
    return _execute_nodes(
        nodes,
        output_node,
        evidence_operands=operands_by_ref,
        output_unit=output_unit,
        require_units=require_units,
        require_evidence=require_evidence,
        decimal_precision=decimal_precision,
    )


def verify_dag_proof_certificate(
    certificate: DAGProofCertificate | Mapping[str, object],
) -> bool:
    try:
        parsed = certificate if isinstance(certificate, DAGProofCertificate) else (
            DAGProofCertificate.from_dict(certificate)
        )
        if parsed.schema_version != DAG_CERTIFICATE_SCHEMA_VERSION:
            return False
        if not 1 <= len(parsed.nodes) <= DEFAULT_MAX_DAG_NODES:
            return False
        results: dict[str, ProofCertificate] = {}
        leaf_refs: list[dict[str, object]] = []
        for expected_index, record in enumerate(parsed.nodes):
            node_id = str(record["node_id"])
            if not node_id or node_id in results or int(record["node_index"]) != expected_index:
                return False
            proof = ProofCertificate.from_dict(record["proof"])  # type: ignore[arg-type]
            if proof.decimal_precision != parsed.decimal_precision:
                return False
            inputs = tuple(record["inputs"])  # type: ignore[arg-type]
            if str(record["operation"]) != proof.operation or len(inputs) != len(
                proof.normalized_inputs
            ):
                return False
            if not verify_proof_certificate(proof):
                return False
            for input_record, normalized in zip(inputs, proof.normalized_inputs):
                kind = str(input_record["kind"])
                if kind == "node":
                    reference = str(input_record["reference"])
                    prior = results.get(reference)
                    if prior is None:
                        return False
                    if normalized["canonical_value"] != prior.result["canonical_value"]:
                        return False
                    if normalized["canonical_unit"] != prior.result["canonical_unit"]:
                        return False
                    if normalized["evidence_ref"] != {"node_id": reference}:
                        return False
                elif kind in {"candidate", "evidence"}:
                    reference = dict(input_record["evidence_ref"])
                    if not reference or normalized["evidence_ref"] != reference:
                        return False
                    leaf_refs.append(reference)
                else:
                    return False
            results[node_id] = proof
        if parsed.output_node not in results:
            return False
        return _deduplicate_references(leaf_refs) == parsed.leaf_evidence_refs
    except (ExactExecutionError, KeyError, TypeError, ValueError):
        return False


def _lean_node_expression(operation: str, inputs: Sequence[str]) -> str:
    if operation == "SELECT":
        return inputs[0]
    if operation in {"ADD", "SUM3", "SUM4", "SUM_N"}:
        return "(" + " + ".join(inputs) + ")"
    if operation == "SUBTRACT":
        return f"({inputs[0]} - {inputs[1]})"
    if operation == "MULTIPLY":
        return f"({inputs[0]} * {inputs[1]})"
    if operation in {"DIVIDE", "RATIO", "RATIO_PERCENT", "SHARE_OF_TOTAL"}:
        return f"({inputs[0]} / {inputs[1]})"
    if operation == "PERCENT_CHANGE":
        return f"(({inputs[0]} - {inputs[1]}) / {inputs[1]})"
    if operation == "AVERAGE":
        return f"(({inputs[0]} + {inputs[1]}) / 2)"
    raise ExactExecutionError(f"Unsupported Lean DAG operation: {operation}")


def export_lean4_dag(
    certificate: DAGProofCertificate | Mapping[str, object],
    *,
    theorem_name: str = "finhrm_exact_dag_result",
) -> str:
    parsed = certificate if isinstance(certificate, DAGProofCertificate) else (
        DAGProofCertificate.from_dict(certificate)
    )
    if not verify_dag_proof_certificate(parsed):
        raise InvalidProofCertificateError("Refusing to export an invalid DAG certificate")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", theorem_name):
        raise InvalidProofCertificateError("Unsafe Lean theorem name")
    expressions: dict[str, str] = {}
    output_value = Fraction(0)
    for record in parsed.nodes:
        proof = ProofCertificate.from_dict(record["proof"])  # type: ignore[arg-type]
        inputs: list[str] = []
        for input_record, normalized in zip(record["inputs"], proof.normalized_inputs):  # type: ignore[arg-type]
            if input_record["kind"] == "node":
                inputs.append(expressions[str(input_record["reference"])])
            else:
                inputs.append(
                    _lean_fraction(_fraction_from_payload(normalized["canonical_value"]))
                )
        node_id = str(record["node_id"])
        expressions[node_id] = _lean_node_expression(proof.operation, inputs)
        if node_id == parsed.output_node:
            output_value = _fraction_from_payload(proof.result["canonical_value"])
    certificate_json = json.dumps(
        parsed.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return (
        "import Mathlib\n\n"
        f"-- FinHRM DAG certificate: {certificate_json}\n"
        f"theorem {theorem_name} : {expressions[parsed.output_node]} = "
        f"{_lean_fraction(output_value)} := by\n"
        "  norm_num\n"
    )


def verify_lean4_dag(
    certificate: DAGProofCertificate | Mapping[str, object],
    *,
    command: Sequence[str] | None = None,
    cwd: str | Path | None = None,
    timeout_seconds: int = 30,
) -> LeanVerificationResult:
    source = export_lean4_dag(certificate)
    resolved = tuple(command) if command is not None else (
        ("lake", "env", "lean") if shutil.which("lake") else (
            ("lean",) if shutil.which("lean") else ()
        )
    )
    if not resolved or not shutil.which(resolved[0]):
        return LeanVerificationResult(
            "UNAVAILABLE", False, None, source,
            stderr="Lean 4 executable not found; exact runtime result remains valid",
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", encoding="utf-8", delete=False
        ) as handle:
            handle.write(source)
            temporary_path = Path(handle.name)
        completed = subprocess.run(
            [*resolved, str(temporary_path)],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return LeanVerificationResult(
            "VERIFIED" if completed.returncode == 0 else "FAILED",
            completed.returncode == 0,
            " ".join(resolved),
            source,
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return LeanVerificationResult(
            "FAILED", False, " ".join(resolved), source, stderr=str(exc)
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

