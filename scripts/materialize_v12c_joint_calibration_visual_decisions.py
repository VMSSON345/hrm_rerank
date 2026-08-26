#!/usr/bin/env python3
"""Materialize the audited visual decisions for the fresh V12-C joint cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_input(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    normalized = path_text.replace("\\", "/")
    marker = "/reasoning_model/"
    if marker in normalized:
        candidate = root / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return root / path.name


POSITIVE = {
    "CTF_24CN_BCTN.pdf": ("consolidated", "VND", "31/12/2024", "01/01/2024", ("2041682704872", "361590089085", "3917631338920"), ("1858337969859", "329495133189", "3657456636415")),
    "GHC_24CN_BCTN.pdf": ("standalone", "VND", "31/12/2024", "01/01/2024", ("13800000000", "378725000000", "1493446150007"), ("19800000000", "450025000000", "1496786272944")),
    "HHV_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("1004158155482", "19279986194751", "36780154788883"), ("872603704822", "19780099074104", "35653232484507")),
    "HPT_23CN_BCTN.pdf": ("standalone", "VND", "31/03/2024", "01/04/2023", ("26068272823", "274684657", "400074178435"), ("20996523017", "844819813", "374207695299")),
    "KDH_23CN_BCTN.pdf": ("consolidated", "thousand_vnd", "31/12/2023", "31/12/2022", ("1444346120", "4901499893", "26480719916"), ("1194723968", "5576290560", "21539250652")),
    "NBB_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("1101012692962", "2539396000000", "6910132131013"), ("971948604185", "2158200685697", "6387256322627")),
    "OIL_24CN_BCTN.pdf": ("consolidated", "VND", "31/12/2024", "01/01/2024", ("7423447631211", "110600366921", "41734469508904"), ("6966663180400", "87810540210", "38839172567714")),
    "PDB_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("23373403117", "790000000", "221544896797"), ("32775879959", "3282932600", "222465700988")),
    "PDR_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("814680676017", "2290488404619", "21067822665509"), ("2667966838466", "1771912500000", "22843295507162")),
    "PSP_23CN_BCTN.pdf": ("standalone", "VND", "31/12/2023", "31/12/2022", ("14954040000", "60813277306", "573146050995"), ("11813995000", "50570924773", "544369510081")),
    "PVD_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("553352813440", "2787681809360", "21632848207600"), ("835317024600", "2999752062520", "20704046691470")),
    "RAL_23CN_BCTN.pdf": ("standalone", "VND", "31/12/2023", "01/01/2023", ("3102786916832", "837486394", "7751688239828"), ("2297652485048", "837486394", "6716483094288")),
    "SAB_24CN_BCTN.pdf": ("consolidated", "VND", "31/12/2024", "01/01/2024", ("245383983749", "169444430406", "33439406364651"), ("530129396183", "170828414155", "34056624839705")),
    "SBT_23CN_BCTN.pdf": ("separate", "VND", "30/06/2024", "01/07/2023", ("8184611316006", "3705864179586", "33611176346405"), ("8027777854111", "684004974773", "30065839577791")),
    "SMC_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "01/01/2023", ("2685470160731", "331830155408", "6178691063726"), ("3004102523373", "571369707623", "8329093955830")),
    "VHC_23CN_BCTN.pdf": ("consolidated", "VND", "31/12/2023", "31/12/2022", ("2157386405831", "101863000000", "11942549471150"), ("2213546608238", "174909254819", "11583289702919")),
    "VPL_24CN_BCTN.pdf": ("consolidated", "million_vnd", "31/12/2024", "01/01/2024", ("3459311", "9414227", "76483141"), ("13856491", "91600", "44072730")),
    "VTP_24CN_BCTN.pdf": ("consolidated", "VND", "31/12/2024", "01/01/2024", ("1375517858200", "72502613634", "6363556558123"), ("1670707830732", "40790712116", "6439394156721")),
}

POSITIVE_REJECTED = {
    "DXG_24CN_BCTN.pdf": "management overview is not a unique same-table proof triplet",
    "POW_24CN_BCTN.pdf": "management overview is not a unique same-table proof triplet",
}

NEGATIVE = {
    "BSR_23CN_BCTN.pdf": (38, "management_summary", "billion_vnd", "ambiguous_duplicate_role", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings", "total_capital"], {
        "31/12/2023": [("aggregate_borrowings_unknown_term", "Tổng vay ngân hàng", "23110"), ("total_assets", "Tổng tài sản", "86595"), ("owners_equity", "Vốn chủ sở hữu", "57269")]
    }),
    "BCE_23CN_BCTN.pdf": (28, "management_summary", "billion_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "353.50"), ("long_term_liabilities", "Nợ dài hạn", "0.731"), ("total_capital", "Tổng nguồn vốn", "687.64")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "312.81"), ("long_term_liabilities", "Nợ dài hạn", "2.89"), ("total_capital", "Tổng nguồn vốn", "648.33")]
    }),
    "BLI_23CN_BCTN.pdf": (40, "standalone_insurer", "VND", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "1488477527130"), ("long_term_liabilities", "Nợ dài hạn", "3616899333"), ("total_capital", "Tổng nguồn vốn", "2328999152861")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "1771027506804"), ("long_term_liabilities", "Nợ dài hạn", "3626399333"), ("total_capital", "Tổng nguồn vốn", "2522621226360")]
    }),
    "BWE_23CN_BCTN.pdf": (107, "consolidated_summary", "billion_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "2375.96"), ("long_term_liabilities", "Nợ dài hạn", "4773.19"), ("total_capital", "Tổng nguồn vốn", "12112.91")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "1989.02"), ("long_term_liabilities", "Nợ dài hạn", "3460.43"), ("total_capital", "Tổng nguồn vốn", "9987.29")]
    }),
    "CSV_24CN_BCTN.pdf": (40, "management_summary", "million_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2024": [("current_liabilities", "Nợ ngắn hạn", "304982"), ("long_term_liabilities", "Nợ dài hạn", "0"), ("total_capital", "Tổng nguồn vốn", "1858998")],
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "285728"), ("long_term_liabilities", "Nợ dài hạn", "0"), ("total_capital", "Tổng nguồn vốn", "1721955")]
    }),
    "GVR_23CN_BCTN.pdf": (47, "consolidated_summary", "billion_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "12289"), ("long_term_liabilities", "Nợ dài hạn", "10796"), ("total_capital", "Tổng nguồn vốn", "75055")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "12673"), ("long_term_liabilities", "Nợ dài hạn", "11271"), ("total_capital", "Tổng nguồn vốn", "78263")]
    }),
    "MSN_23CN_BCTN.pdf": (70, "consolidated", "million_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "50422500"), ("long_term_liabilities", "Nợ dài hạn", "58723731"), ("total_capital", "Tổng nguồn vốn", "147383472")],
        "01/01/2023": [("current_liabilities", "Nợ ngắn hạn", "65320877"), ("long_term_liabilities", "Nợ dài hạn", "39385199"), ("total_capital", "Tổng nguồn vốn", "141342815")]
    }),
    "PGI_23CN_BCTN.pdf": (67, "standalone_insurer", "VND", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "5532379449811"), ("long_term_liabilities", "Nợ dài hạn", "51718536072"), ("total_capital", "Tổng nguồn vốn", "7448720197305")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "5043586995356"), ("long_term_liabilities", "Nợ dài hạn", "37911203358"), ("total_capital", "Tổng nguồn vốn", "6761973416591")]
    }),
    "SFG_23CN_BCTN.pdf": (34, "management_summary", "million_vnd", "missing_required_raw_role", "INCOMPLETE_REQUIRED_ROLE_ABSENT", ["short_term_borrowings", "long_term_borrowings", "total_capital"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "590138"), ("long_term_liabilities", "Nợ dài hạn", "268"), ("total_assets", "Tổng tài sản", "1243947")],
        "01/01/2023": [("current_liabilities", "Nợ ngắn hạn", "469098"), ("long_term_liabilities", "Nợ dài hạn", "372"), ("total_assets", "Tổng tài sản", "1090148")]
    }),
    "SIP_24CN_BCTN.pdf": (28, "consolidated_summary", "million_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings", "total_capital"], {
        "31/12/2024": [("current_liabilities", "Nợ ngắn hạn", "4143521"), ("long_term_liabilities", "Nợ dài hạn", "16041538"), ("total_assets", "Tổng tài sản", "25052881")],
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "3061004"), ("long_term_liabilities", "Nợ dài hạn", "13971373"), ("total_assets", "Tổng tài sản", "21060427")]
    }),
    "SPD_24CN_BCTN.pdf": (28, "management_summary", "million_vnd", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings", "total_capital"], {
        "31/12/2024": [("current_liabilities", "Nợ ngắn hạn", "272248"), ("long_term_liabilities", "Nợ dài hạn", "6396"), ("total_assets", "Tổng tài sản", "410829")],
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "259522"), ("long_term_liabilities", "Nợ dài hạn", "8627"), ("total_assets", "Tổng tài sản", "391985")]
    }),
    "TPC_23CN_BCTN.pdf": (27, "standalone_summary", "VND", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "174752762325"), ("long_term_liabilities", "Nợ dài hạn", "0"), ("total_capital", "Tổng nguồn vốn", "461911203713")],
        "31/12/2022": [("current_liabilities", "Nợ ngắn hạn", "305627839531"), ("long_term_liabilities", "Nợ dài hạn", "0"), ("total_capital", "Tổng nguồn vốn", "641562410234")]
    }),
    "TRS_24CN_BCTN.pdf": (31, "management_summary", "VND", "invalid_borrowing_role_semantics", "INVALID_ROLE_SEMANTICS", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2024": [("current_liabilities", "Nợ ngắn hạn", "171554054437"), ("long_term_liabilities", "Nợ dài hạn", "17010821018"), ("total_capital", "Tổng nguồn vốn", "452796099539")],
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "156387656891"), ("long_term_liabilities", "Nợ dài hạn", "4528923134"), ("total_capital", "Tổng nguồn vốn", "394809306827")]
    }),
    "VLC_23CN_BCTN.pdf": (52, "consolidated", "VND", "missing_required_raw_role", "INCOMPLETE_REQUIRED_ROLE_ABSENT", ["short_term_borrowings", "long_term_borrowings"], {
        "31/12/2023": [("current_liabilities", "Nợ ngắn hạn", "383348951078"), ("long_term_liabilities", "Nợ dài hạn", "123939494657"), ("total_capital", "Tổng nguồn vốn", "5914536094691")],
        "01/01/2023": [("current_liabilities", "Nợ ngắn hạn", "301263026624"), ("long_term_liabilities", "Nợ dài hạn", "140220462435"), ("total_capital", "Tổng nguồn vốn", "4898376539824")]
    }),
    "VNM_23CN_BCTN.pdf": (112, "consolidated", "VND", "missing_required_raw_role", "INCOMPLETE_REQUIRED_ROLE_ABSENT", ["total_capital"], {
        "31/12/2023": [("short_term_borrowings", "Vay ngắn hạn", "8217557172267"), ("long_term_borrowings", "Vay dài hạn", "238476074100"), ("total_liabilities", "Nợ phải trả", "17647627338990")],
        "01/01/2023": [("short_term_borrowings", "Vay ngắn hạn", "4867129839103"), ("long_term_borrowings", "Vay dài hạn", "66028725000"), ("total_liabilities", "Nợ phải trả", "15666145881135")]
    }),
    "YEG_23CN_BCTN.pdf": (73, "consolidated", "VND", "missing_required_raw_role", "INCOMPLETE_REQUIRED_ROLE_ABSENT", ["total_capital"], {
        "31/12/2023": [("short_term_borrowings", "Vay ngắn hạn", "171352690912"), ("long_term_borrowings", "Vay dài hạn", "39014432856"), ("total_liabilities", "Nợ phải trả", "477098903310")],
        "01/01/2023": [("short_term_borrowings", "Vay ngắn hạn", "94317807400"), ("long_term_borrowings", "Vay dài hạn", "1000000000"), ("total_liabilities", "Nợ phải trả", "334197703328")]
    }),
}


def build(queue_path: Path, render_path: Path) -> dict[str, object]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    render = json.loads(render_path.read_text(encoding="utf-8"))
    queue_rows = {(row["source_name"], int(row["page"])): row for row in queue["records"]}
    render_rows = {(row["source_name"], int(row["page"])): row for row in render["records"]}
    discoveries = []
    for input_entry in queue["inputs"]:
        discovery_path = _local_input(str(input_entry["path"]), queue_path.parents[1])
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        discoveries.extend(discovery["records"])
    records = []
    additional_render_hashes = {}
    roles = ("short_term_borrowings", "long_term_borrowings", "total_capital")
    for (source_name, page), source in sorted(queue_rows.items()):
        if source.get("queue_stratum") == "positive_candidate":
            base = {"source_name": source_name, "page": page, "company_key": source["company_key"], "queue_stratum": "positive_candidate", "render_sha256": render_rows[(source_name, page)]["render_sha256"]}
            if source_name in POSITIVE_REJECTED:
                records.append({**base, "decision": "rejected", "reason": POSITIVE_REJECTED[source_name]})
                continue
            scope, unit, current_period, prior_period, current, prior = POSITIVE[source_name]
            records.append({**base, "decision": "approved", "scope": scope, "unit": unit, "periods": {
                current_period: dict(zip(roles, current)), prior_period: dict(zip(roles, prior))
            }})

    for source_name, (page, scope, unit, failure, status, missing, periods) in sorted(NEGATIVE.items()):
        if (source_name, page) in queue_rows:
            source = queue_rows[(source_name, page)]
            render_hash = render_rows[(source_name, page)]["render_sha256"]
        else:
            source = next(row for row in discoveries if row["source_name"] == source_name and int(row["page"]) == page)
            render_file = queue_path.parents[1] / "tmp/pdfs/v12c_joint_calibration_visual_queue_v1" / f"neg__{Path(source_name).stem}__p{page}.png"
            render_hash = _sha256(render_file)
            additional_render_hashes[f"{source_name}:{page}"] = render_hash
        records.append({
            "source_name": source_name, "page": page, "company_key": source["company_key"],
            "queue_stratum": "negative_candidate", "render_sha256": render_hash,
            "decision": "approved_negative", "scope": scope, "unit": unit,
            "proof_failure_type": failure, "required_proof_status": status,
            "missing_required_roles": missing,
            "periods": {period: [{"semantic_role": role, "row_label": label, "value": value} for role, label, value in cells] for period, cells in periods.items()},
        })
    failure_counts = Counter(row["proof_failure_type"] for row in records if row.get("decision") == "approved_negative")
    return {
        "protocol": "v12c-joint-calibration-visual-decisions-v1",
        "classification": "development_only_joint_calibration_visual_qa",
        "protected_evaluation_sources_used": False, "model_invoked": False, "threshold_changed": False,
        "discovery_sha256": _sha256(queue_path), "render_queue_sha256": _sha256(render_path),
        "additional_render_hashes": additional_render_hashes,
        "additional_source_records": [
            row for row in discoveries
            if f"{row['source_name']}:{int(row['page'])}" in additional_render_hashes
        ],
        "records": records,
        "visual_qa_summary": {
            "approved_positive_pages": len(POSITIVE), "approved_positive_controls": 2 * len(POSITIVE),
            "rejected_positive_pages": sorted(POSITIVE_REJECTED),
            "approved_negative_pages": len(NEGATIVE), "approved_negative_controls": sum(len(item[6]) for item in NEGATIVE.values()),
            "negative_company_groups": len(NEGATIVE), "negative_failure_type_counts": dict(sorted(failure_counts.items())),
            "operating_point_rule_changed": False, "quota_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--render-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.queue, args.render_queue, args.output):
        if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
            raise ValueError(f"Protected path is not allowed: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    result = build(args.queue, args.render_queue)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["visual_qa_summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
