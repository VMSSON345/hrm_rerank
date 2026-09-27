"""Paired numeric evaluation keeps source gold separate from verifier outcomes."""
from __future__ import annotations

from collections import Counter
from random import Random
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from math import comb
import json
from pathlib import Path

from .metric_equivalence import diagnostic_fact_signature

SCHEMA = "numeric-program-value-eval-v1"


def decimal(value, name):
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"{name} must be decimal") from None
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return parsed


def fact_signature(item):
    if item.get("evidence_kind") == "pdf_visual":
        return ("PDF_VISUAL", item["source_name"], item["page_number"],
                item["raw_value"], item["metric"], item["period"],
                item["role"], item["scope"], item.get("unit_state", ""))
    citation = item.get("citation", item)
    return (citation["source_name"], citation["line_number"],
            tuple(item.get("source_token_span") or citation["source_span"]),
            item["metric"], item["period"],
            item["role"], item["scope"], item.get("unit_state", ""))


def physical_citations(response):
    return tuple((c["source_name"], c["line_number"], tuple(c["source_span"]))
                 for c in response.get("citations") or []
                 if not c.get("operand_ref", "").startswith("crosscheck_"))


def selected_program(response):
    program = response.get("program") or {}
    pool = (response.get("program_trace") or {}).get("fact_pool") or []
    indices = program.get("operands") or []
    if not program.get("operation") or not pool or not indices:
        return None
    try:
        return (program["operation"],
                tuple(fact_signature(pool[i]) for i in indices))
    except (IndexError, KeyError, TypeError):
        return None


def validate_runtime_freeze(manifest, *, base_directory=Path(".")):
    """Check the pre-inference code/source freeze, not just its JSON filename."""
    binding = manifest.get("runtime_freeze") or {}
    name, expected = binding.get("path"), binding.get("sha256")
    if not name or not expected:
        raise ValueError("Independent holdout requires runtime freeze path and hash")
    path = Path(name)
    if not path.is_absolute():
        path = base_directory / path
    if not path.is_file():
        raise ValueError("Runtime freeze missing")
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError("Runtime freeze hash changed")
    record = json.loads(raw)
    files = record.get("files") or []
    if not files or record.get("model_inference_runs_on_selected_sources", 0):
        raise ValueError("Runtime freeze incomplete or already used for inference")
    for item in files:
        artifact = Path(item["path"])
        if not artifact.is_absolute():
            artifact = base_directory / artifact
        if not artifact.is_file() or sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Frozen runtime file changed: " + item["path"])
    return digest


def validate_manifest(manifest, *, base_directory=Path(".")):
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unknown evaluation schema")
    scope = manifest.get("evaluation_scope")
    if scope not in {"development_replay", "independent_holdout"}:
        raise ValueError("Unknown evaluation scope")
    if scope == "independent_holdout" and not manifest.get("locked_before_inference"):
        raise ValueError("Holdout must be locked before inference")
    if scope == "independent_holdout":
        validate_runtime_freeze(manifest, base_directory=base_directory)
    sources, cases = manifest.get("sources") or [], manifest.get("cases") or []
    if not sources or not cases:
        raise ValueError("Missing sources or cases")
    paths = {}
    source_meta = {}
    for source in sources:
        key = source["id"]
        if key in paths:
            raise ValueError("Duplicate source id")
        exposure = source.get("exposure")
        if exposure not in {"previously_used", "new_unseen"}:
            raise ValueError("Source exposure missing")
        if scope == "independent_holdout" and exposure != "new_unseen":
            raise ValueError("Previously used source cannot be a fresh holdout")
        if scope == "independent_holdout" and not source.get("layout_tags"):
            raise ValueError("Holdout source needs reviewed layout tags")
        path = Path(source["artifact"])
        if not path.is_absolute():
            path = base_directory / path
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source missing or hash changed: " + key)
        if scope == "independent_holdout":
            pdf_name = source.get("pdf_path")
            pdf_hash = source.get("pdf_sha256")
            if not pdf_name or not pdf_hash:
                raise ValueError("Holdout requires original PDF path and hash")
            pdf = Path(pdf_name)
            if not pdf.is_absolute():
                pdf = base_directory / pdf
            if (pdf.suffix.lower() != ".pdf" or not pdf.is_file()
                    or sha256(pdf.read_bytes()).hexdigest() != pdf_hash):
                raise ValueError("Original PDF missing or hash changed: " + key)
            if not isinstance(source.get("pdf_pages"), int) or source["pdf_pages"] < 1:
                raise ValueError("Holdout source requires reviewed PDFium page count")
        paths[key] = path
        source_meta[key] = source
    seen = set()
    for case in cases:
        if case["id"] in seen or case["source_id"] not in paths or not case.get("question"):
            raise ValueError("Duplicate, sourceless, or empty case")
        seen.add(case["id"])
        answer = case.get("answer_gold")
        if answer is not None:
            decimal(answer["value"], "answer_gold.value")
            if decimal(answer["tolerance"], "answer_gold.tolerance") < 0:
                raise ValueError("Negative tolerance")
        choice = case.get("choice_class_gold")
        if choice not in {None, "single", "multiple_competing",
                          "equivalent_alternatives", "no_valid_program", "outside_supported_dsl"}:
            raise ValueError("Unknown gold choice class")
        intent = case.get("intent_allowed_operations")
        if intent is not None and (not isinstance(intent, list) or not intent
                                   or any(not isinstance(x, str) for x in intent)):
            raise ValueError("Intent labels must be a nonempty operation list")
        sufficiency = case.get("source_sufficiency_gold")
        if sufficiency not in (None, True, False):
            raise ValueError("Unknown source sufficiency")
        lines = paths[case["source_id"]].read_text(encoding="utf-8-sig").splitlines()
        source = source_meta[case["source_id"]]
        for program in case.get("gold_programs") or []:
            if not program.get("operation") or not program.get("facts"):
                raise ValueError("Incomplete gold program")
            for fact in program["facts"]:
                if fact.get("evidence_kind") == "pdf_visual":
                    if scope != "independent_holdout":
                        raise ValueError("PDF-only gold is reserved for reviewed holdout")
                    if (Path(fact["source_name"]).name !=
                            Path(source["pdf_path"]).name
                            or fact.get("pdf_sha256") != source["pdf_sha256"]
                            or not isinstance(fact.get("page_number"), int)
                            or not 1 <= fact["page_number"] <= source["pdf_pages"]
                            or not fact.get("raw_value")
                            or not fact.get("unit_state")
                            or not (fact.get("visual_review") or {}).get(
                                "reviewed_pdf_page")):
                        raise ValueError("PDF-only gold lacks physical source review")
                    fact_signature(fact)
                    continue
                if Path(fact["source_name"]).name != paths[case["source_id"]].name:
                    raise ValueError("Gold fact belongs to another source")
                n, (start, end) = fact["line_number"], fact["source_span"]
                if not (1 <= n <= len(lines) and 0 <= start < end <= len(lines[n-1])):
                    raise ValueError("Invalid gold fact span")
                if scope == "independent_holdout" and not (
                        fact.get("source_line_sha256") and fact.get("raw_value")
                        and fact.get("unit_state")):
                    raise ValueError(
                        "Holdout fact needs line hash, exact raw token, and unit state")
                if fact.get("source_line_sha256") and sha256(
                        lines[n-1].encode()).hexdigest() != fact["source_line_sha256"]:
                    raise ValueError("Gold fact line changed")
                if fact.get("raw_value") and lines[n-1][start:end] != fact["raw_value"]:
                    raise ValueError("Gold fact raw token and span differ")
                fact_signature(fact)
        if scope == "independent_holdout":
            review = case.get("gold_review") or {}
            positive = answer is not None and bool(case.get("gold_programs"))
            if positive and not answer.get("unit"):
                raise ValueError("Holdout answer needs canonical unit")
            outside_dsl = (answer is not None and not case.get("gold_programs")
                           and choice == "outside_supported_dsl"
                           and sufficiency is True)
            negative = (answer is None and choice == "no_valid_program"
                        and sufficiency is False)
            if not (choice is not None and sufficiency is not None and
                    intent is not None and (positive or outside_dsl or negative) and
                    review.get("reviewer_id") and review.get("reviewed_before_inference")):
                raise ValueError("Holdout case lacks locked independent gold")
            if (positive and sufficiency is not True) or (
                    negative and case.get("gold_programs")):
                raise ValueError("Gold answer and source sufficiency conflict")
            if positive and not any(program["operation"] in intent
                                    for program in case["gold_programs"]):
                raise ValueError("No gold program satisfies reviewed intent")
            if (choice == "equivalent_alternatives"
                    and len(case.get("gold_programs") or []) < 2):
                raise ValueError("Equivalent alternatives need two gold programs")
    return paths


def verify_holdout_lock(manifest_path, manifest_bytes, checkpoint_sha256):
    path = Path(manifest_path).with_suffix(".lock.json")
    if not path.is_file():
        raise ValueError("Independent holdout lacks a pre-inference lock file")
    lock = json.loads(path.read_text(encoding="utf8"))
    manifest = json.loads(manifest_bytes)
    runtime_digest = validate_runtime_freeze(manifest, base_directory=Path("."))
    if (lock.get("manifest_sha256") != sha256(manifest_bytes).hexdigest()
            or lock.get("checkpoint_sha256") != checkpoint_sha256
            or lock.get("runtime_freeze_sha256") != runtime_digest
            or not lock.get("locked_at_utc")):
        raise ValueError("Holdout lock differs from manifest, runtime, or checkpoint")
    return lock


def score_response(case, response, *, selector=None):
    status = response.get("status", "ERROR")
    answer = case.get("answer_gold")
    negative_gold = (
        answer is None and case.get("choice_class_gold") == "no_valid_program"
        and case.get("source_sufficiency_gold") is False)
    correct = None
    if answer is not None:
        value = response.get("canonical_value")
        numeric_ok = (
            status == "ACCEPTED" and value is not None
            and abs(decimal(value, "canonical_value")
                    - decimal(answer["value"], "answer_gold.value"))
                <= decimal(answer["tolerance"], "answer_gold.tolerance"))
        unit_ok = (
            not answer.get("unit")
            or response.get("unit") == answer["unit"])
        primary = [
            citation for citation in response.get("citations") or []
            if not citation.get("operand_ref", "").startswith("crosscheck_")]
        scope_ok = (
            not answer.get("scope")
            or bool(primary) and all(
                citation.get("entity_scope") == answer["scope"]
                for citation in primary))
        correct = bool(numeric_ok and unit_ok and scope_ok)
    program = selected_program(response)
    gold = tuple((p["operation"], tuple(fact_signature(f) for f in p["facts"]))
                 for p in case.get("gold_programs") or [])
    trace = response.get("program_trace") or {}
    retrieval = trace.get("retrieval") or {}
    raw_pool = trace.get("fact_pool") or []
    pool = tuple(fact_signature(f) for f in raw_pool)
    pool_sha256 = (sha256(json.dumps(raw_pool, ensure_ascii=False,
                                   sort_keys=True, separators=(",", ":"))
                          .encode("utf8")).hexdigest()
                   if raw_pool else None)
    recall = semantic_recall = op_ok = pair_ok = first_ok = None
    pool_observed = bool(raw_pool)
    if gold and pool_observed:
        recall = any(all(f in pool for f in facts) for _, facts in gold)
        semantic_pool = {diagnostic_fact_signature(f) for f in pool}
        semantic_recall = any(
            all(diagnostic_fact_signature(f) in semantic_pool for f in facts)
            for _, facts in gold)
        op_ok = bool(program and any(program[0] == op for op, _ in gold))
        pair_ok = program in gold
        proposals = trace.get("proposals") or []
        first_ok = False
        if proposals:
            first = proposals[0]
            try:
                signature = (first["operation"],
                             tuple(fact_signature(trace["fact_pool"][i])
                                   for i in first["operands"]))
                first_ok = signature in gold
            except (IndexError, KeyError, TypeError):
                first_ok = False
    operation = (response.get("program") or {}).get("operation")
    physical = physical_citations(response)
    intent = case.get("intent_allowed_operations")
    if negative_gold:
        diagnostic_stage = (
            "false_accept_no_valid_program" if status == "ACCEPTED"
            else "no_valid_program_not_accepted")
    elif answer is None:
        diagnostic_stage = None
    elif case.get("choice_class_gold") == "outside_supported_dsl":
        diagnostic_stage = "source_sufficient_program_outside_supported_dsl"
    elif status == "ACCEPTED" and correct:
        diagnostic_stage = "answer_correct"
    elif gold and not pool_observed:
        diagnostic_stage = "pool_not_observed"
    elif recall is False:
        diagnostic_stage = "gold_fact_absent_from_pool"
    elif status == "ACCEPTED":
        diagnostic_stage = (
            "gold_program_selected_but_answer_wrong" if pair_ok is True
            else "wrong_answer_program_or_source_review")
    elif pair_ok is True:
        diagnostic_stage = "gold_program_selected_but_not_published"
    elif recall is True:
        diagnostic_stage = "gold_fact_present_program_not_selected"
    else:
        diagnostic_stage = "halt_without_program_gold"
    return {
        "status": status, "canonical_value": response.get("canonical_value"),
        "correct": correct, "diagnostic_stage": diagnostic_stage,
        "wrong_accepted": bool(
            status == "ACCEPTED" and (
                (answer is not None and not correct) or negative_gold)),
        "negative_gold_scored": negative_gold,
        "negative_correct": (
            status != "ACCEPTED" if negative_gold else None),
        "abstain_despite_sufficient_source": bool(
            status == "ABSTAIN" and case.get("source_sufficiency_gold") is True),
        "halt_despite_sufficient_source": bool(
            status != "ACCEPTED" and case.get("source_sufficiency_gold") is True),
        "program": [program[0], [list(f) for f in program[1]]] if program else None,
        "physical_program": [operation, [list(f) for f in physical]]
                            if operation and physical else None,
        "candidate_recall": recall,
        "semantic_candidate_recall": semantic_recall,
        "operation_correct": op_ok,
        "operands_correct": pair_ok, "first_proposal_correct": first_ok,
        "pool_observed": pool_observed,
        "program_gate_trace": response.get("program_gate_trace"),
        "intent_alignment": operation in intent if intent is not None else None,
        "verifier_choice_class": trace.get("choice_class"),
        "candidate_count": trace.get("candidate_count"),
        "pool_sha256": pool_sha256,
        "tried_proposals": [
            {"operation": item.get("operation"),
             "fact_ids": item.get("fact_ids"),
             "reason": item.get("reason")}
            for item in trace.get("proposals") or []],
        "retrieval": ({
            "mode": retrieval.get("mode"),
            "dense": retrieval.get("dense"),
            "pool_selection": retrieval.get("pool_selection"),
            "catalog_candidates": retrieval.get("catalog_candidates"),
            "bm25_candidates": retrieval.get("bm25_candidates"),
            "selected_candidates": retrieval.get("selected_candidates"),
            "graph_edges": retrieval.get("graph_edges"),
            "graph_added_selected": len(retrieval.get("graph_added_selected") or []),
            "eligible_fact_count": retrieval.get("eligible_fact_count"),
            "retained_typed_fact_count": retrieval.get("retained_typed_fact_count"),
            "slots": retrieval.get("slots"),
        } if retrieval else None),
        "neural_operation": trace.get("neural_operation"),
        "first_proposal_reason": ((trace.get("proposals") or [{}])[0].get("reason")),
        "blocking_codes": [check["code"] for check in
                           (response.get("validation") or {}).get("checks", [])
                           if check.get("blocking")],
    }


def _binomial_upper_95(successes, trials):
    """One-sided exact binomial upper bound; descriptive for source clusters."""
    if trials <= 0:
        return None
    if successes >= trials:
        return 1.0
    low, high = successes / trials, 1.0
    for _ in range(70):
        p = (low + high) / 2
        cdf = sum(comb(trials, k) * p**k * (1 - p)**(trials - k)
                  for k in range(successes + 1))
        if cdf > 0.05:
            low = p
        else:
            high = p
    return round(high, 6)


def paired_answer_effect(rows, left="baseline", right="hrm"):
    """Paired answer delta; source-cluster bootstrap is descriptive, not a gate."""
    scored = [
        row for row in rows
        if row["modes"][left]["correct"] is not None
        and row["modes"][right]["correct"] is not None]
    if not scored:
        return None
    left_only = sum(
        row["modes"][left]["correct"] is True
        and row["modes"][right]["correct"] is False for row in scored)
    right_only = sum(
        row["modes"][right]["correct"] is True
        and row["modes"][left]["correct"] is False for row in scored)
    both = sum(
        row["modes"][left]["correct"] is True
        and row["modes"][right]["correct"] is True for row in scored)
    groups = {}
    for row in scored:
        groups.setdefault(row.get("source_id") or row["id"], []).append(row)
    discordant_clusters = sum(any(
        row["modes"][left]["correct"] != row["modes"][right]["correct"]
        for row in group) for group in groups.values())
    interval = None
    if len(groups) >= 2:
        random = Random(20260925)
        keys = sorted(groups)
        draws = []
        for _ in range(2000):
            sample = [
                row for _ in keys
                for row in groups[random.choice(keys)]]
            draws.append(sum(
                int(row["modes"][right]["correct"])
                - int(row["modes"][left]["correct"])
                for row in sample) / len(sample))
        draws.sort()
        interval = [round(draws[49], 6), round(draws[1949], 6)]
    return {
        "positive_gold_cases": len(scored),
        "source_clusters": len(groups),
        "both_correct": both,
        left + "_only_correct": left_only,
        right + "_only_correct": right_only,
        "neither_correct": len(scored) - both - left_only - right_only,
        "delta_accuracy": round((right_only - left_only) / len(scored), 6),
        "source_cluster_bootstrap_95_descriptive": interval,
        "bootstrap_samples": 2000 if interval is not None else 0,
        "discordant_source_clusters": discordant_clusters,
        "source_discordance_rate_upper_95_one_sided": _binomial_upper_95(
            discordant_clusters, len(groups)),
        "uncertainty_note": (
            "Bootstrap is conditional on observed sources; [0,0] when all "
            "observed deltas tie does not establish equality on new sources. "
            "The source-level discordance bound assumes exchangeable sources."),
    }


def summarize(rows):
    """Status matrix is descriptive; overlap is not full same-candidate parity."""
    result = {"case_count": len(rows), "modes": {}, "decision_matrices": {},
              "observed_program_overlap": {}, "by_gold_choice_class": {}}
    for mode in ("off", "baseline", "hrm"):
        outcomes = [row["modes"][mode] for row in rows]
        scored = [o for o in outcomes if o["correct"] is not None]
        program_scored = sum(o["operands_correct"] is not None for o in outcomes)
        pool_observed = sum(o.get("pool_observed", False) for o in outcomes)
        recalled = [o for o in outcomes if o["candidate_recall"] is True]
        intent_scored = sum(o["intent_alignment"] is not None for o in outcomes)
        sufficiency_scored = sum(row.get("source_sufficiency_gold") is not None
                                 for row in rows)
        result["modes"][mode] = {
            "statuses": dict(Counter(o["status"] for o in outcomes)),
            "diagnostic_stages": dict(Counter(
                o.get("diagnostic_stage") for o in outcomes
                if o.get("diagnostic_stage") is not None)),
            "verifier_choice_classes_diagnostic": dict(Counter(
                o.get("verifier_choice_class") for o in outcomes
                if o.get("verifier_choice_class") is not None)),
            "answer_scored": len(scored),
            "answer_correct": sum(o["correct"] is True for o in scored),
            "wrong_accepted": sum(o["wrong_accepted"] for o in outcomes),
            "negative_gold_scored": sum(o.get("negative_gold_scored", False)
                                        for o in outcomes),
            "negative_accepted": sum(
                o.get("negative_gold_scored", False) and o["status"] == "ACCEPTED"
                for o in outcomes),
            "source_sufficiency_scored": sufficiency_scored,
            "abstain_despite_sufficient_source": (
                sum(o["abstain_despite_sufficient_source"] for o in outcomes)
                if sufficiency_scored else None),
            "halt_despite_sufficient_source": (
                sum(o.get("halt_despite_sufficient_source", False)
                    for o in outcomes) if sufficiency_scored else None),
            "program_gold_scored": program_scored,
            "pool_observed": pool_observed,
            "pool_unobserved": len(outcomes) - pool_observed,
            "no_pool_stop_reasons": dict(Counter(
                (o.get("program_gate_trace") or {}).get("stop_reason") or
                "route_without_program_pool"
                for o in outcomes if not o.get("pool_observed"))),
            "multiple_valid_program_pools": sum(
                o.get("verifier_choice_class") == "multiple_valid_programs"
                for o in outcomes),
            "candidate_recall": (sum(o["candidate_recall"] is True for o in outcomes)
                                 if program_scored else None),
            "semantic_candidate_recall": (
                sum(o.get("semantic_candidate_recall") is True for o in outcomes)
                if program_scored else None),
            "operation_correct": (sum(o["operation_correct"] is True for o in outcomes)
                                  if program_scored else None),
            "operands_correct": (sum(o["operands_correct"] is True for o in outcomes)
                                 if program_scored else None),
            "first_proposal_correct": (
                sum(o["first_proposal_correct"] is True for o in outcomes)
                if program_scored else None),
            "program_given_candidate_recall": {
                "cases": len(recalled),
                "operation_correct": sum(
                    o["operation_correct"] is True for o in recalled),
                "operands_correct": sum(
                    o["operands_correct"] is True for o in recalled),
                "first_proposal_correct": sum(
                    o["first_proposal_correct"] is True for o in recalled),
            } if program_scored else None,
            "intent_scored": intent_scored,
            "intent_aligned": (sum(o["intent_alignment"] is True for o in outcomes)
                               if intent_scored else None),
            "seconds": round(sum(row["seconds"][mode] for row in rows), 3),
        }
    paired_multiple = [row for row in rows
        if row["modes"]["baseline"].get("pool_sha256")
        and row["modes"]["baseline"]["pool_sha256"] ==
            row["modes"]["hrm"].get("pool_sha256")
        and row["modes"]["baseline"].get("verifier_choice_class") ==
            row["modes"]["hrm"].get("verifier_choice_class") ==
            "multiple_valid_programs"]
    result["paired_multiple_valid_programs"] = {
        "cases": len(paired_multiple),
        "case_ids": [row["id"] for row in paired_multiple],
        "baseline_correct": sum(row["modes"]["baseline"]["correct"] is True
                                for row in paired_multiple),
        "hrm_correct": sum(row["modes"]["hrm"]["correct"] is True
                           for row in paired_multiple),
        "baseline_wrong_accepted": sum(row["modes"]["baseline"]["wrong_accepted"]
                                       for row in paired_multiple),
        "hrm_wrong_accepted": sum(row["modes"]["hrm"]["wrong_accepted"]
                                  for row in paired_multiple),
        "scope": "Only identical real fact pools with at least two distinct "
                 "program signatures accepted by the same verifier; gold "
                 "quality and program equivalence require separate review",
    }
    for left, right in (("off", "hrm"), ("baseline", "hrm")):
        matrix = Counter()
        overlap = []
        for row in rows:
            a, b = row["modes"][left], row["modes"][right]
            matrix[("ACCEPTED" if a["status"] == "ACCEPTED" else "NOT_ACCEPTED",
                    "ACCEPTED" if b["status"] == "ACCEPTED" else "NOT_ACCEPTED")] += 1
            if (a["physical_program"] and b["physical_program"] and
                    a["physical_program"] == b["physical_program"]):
                overlap.append((row["id"], a["status"], b["status"]))
        key = left + "_vs_" + right
        result["decision_matrices"][key] = {
            a + "/" + b: matrix[(a, b)]
            for a in ("ACCEPTED", "NOT_ACCEPTED")
            for b in ("ACCEPTED", "NOT_ACCEPTED")}
        different = [row["id"] for row in rows
                     if (row["modes"][left]["status"] == "ACCEPTED"
                         and row["modes"][right]["status"] == "ACCEPTED"
                         and row["modes"][left]["physical_program"]
                         and row["modes"][right]["physical_program"]
                         and row["modes"][left]["physical_program"]
                             != row["modes"][right]["physical_program"])]
        result["observed_program_overlap"][key] = {
            "count": len(overlap),
            "status_disagreements": [name for name, a, b in overlap if a != b],
            "different_accepted_programs": different,
            "scope": "observed final physical programs only; not full verifier parity",
        }
    for choice in ("single", "multiple_competing", "equivalent_alternatives",
                   "outside_supported_dsl", "no_valid_program", "unlabelled"):
        subset = [row for row in rows
                  if (row.get("choice_class_gold") or "unlabelled") == choice]
        if subset:
            result["by_gold_choice_class"][choice] = {
                "cases": len(subset),
                "answer_scored": sum(r["modes"]["baseline"]["correct"] is not None
                                     for r in subset),
                "baseline_correct": sum(r["modes"]["baseline"]["correct"] is True
                                        for r in subset),
                "hrm_correct": sum(r["modes"]["hrm"]["correct"] is True
                                   for r in subset),
                "baseline_wrong_accepted": sum(
                    r["modes"]["baseline"]["wrong_accepted"] for r in subset),
                "hrm_wrong_accepted": sum(
                    r["modes"]["hrm"]["wrong_accepted"] for r in subset),
                "paired_answer_effect": paired_answer_effect(subset),
            }
    result["paired_answer_effects"] = {
        "baseline_vs_hrm": paired_answer_effect(rows),
        "off_vs_hrm": paired_answer_effect(rows, "off", "hrm"),
    }
    return result
