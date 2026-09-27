"""Optional native PDF text-layer corroboration for ambiguous OCR table units."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .ontology import normalize_text


def verified_companion_pdf(markdown: Path, markdown_sha256: str):
    provenance = markdown.with_suffix(".provenance.json")
    if not provenance.is_file():
        return None
    record = json.loads(provenance.read_text(encoding="utf8"))
    if record.get("artifact_sha256") != markdown_sha256:
        raise ValueError("companion_pdf_artifact_hash_mismatch")
    relative = Path(record.get("pdf_path", ""))
    expected = record.get("pdf_sha256", "")
    if not relative.parts or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("companion_pdf_provenance_incomplete")
    matches = [root / relative for root in markdown.parents
               if (root / relative).is_file()]
    if len(matches) != 1:
        raise ValueError("companion_pdf_path_ambiguous")
    pdf = matches[0].resolve()
    if sha256(pdf.read_bytes()).hexdigest() != expected:
        raise ValueError("companion_pdf_hash_mismatch")
    return pdf, expected


def _words(pdf: Path, page: int):
    executable = shutil.which("pdftotext")
    if executable is None:
        return ()
    result = subprocess.run(
        [executable, "-f", str(page), "-l", str(page), "-bbox",
         str(pdf), "-"], capture_output=True, timeout=30, check=False)
    if result.returncode or not result.stdout:
        return ()
    root = ET.fromstring(result.stdout)
    return tuple({
        "text": "".join(item.itertext()),
        "bbox": tuple(float(item.attrib[key]) for key in
                      ("xMin", "yMin", "xMax", "yMax")),
    } for item in root.iter() if item.tag.rsplit("}", 1)[-1] == "word")


def _center(box):
    return (box[0] + box[2]) / 2


def _scale_phrase(left, right):
    scale = {"ty": "tỷ đồng", "trieu": "triệu đồng",
             "nghin": "nghìn đồng", "ngan": "nghìn đồng"}
    if normalize_text(right["text"]) != "dong":
        return ""
    return scale.get(normalize_text(left["text"]), "")


def native_column_unit(snapshot, source_name: str, page: int,
                       printed_header: str, raw_value: str):
    """Return a PDF witness only for one aligned year/unit/value column.

    The OCR header is a proposal. The validated native PDF must print an
    unambiguous denomination under the same year and above the exact value.
    """
    match = re.fullmatch(
        r"\s*((?:19|20)\d{2})\s*T\s*(?:đồng|dong|dng)\s*",
        printed_header, re.I)
    companion = snapshot.pdf_sources.get(source_name)
    if not match or not companion or not page:
        return None
    pdf, pdf_hash = companion
    cache = snapshot._pdf_word_cache
    if (source_name, page) not in cache:
        cache[(source_name, page)] = _words(pdf, page)
    words = cache[(source_name, page)]
    values = [word for word in words if word["text"].strip() == raw_value.strip()]
    if len(values) != 1:
        return None
    value = values[0]
    year = match[1]
    headings = []
    for year_word in words:
        if year_word["text"] != year:
            continue
        for index, unit_left in enumerate(words):
            if index + 1 >= len(words):
                continue
            unit_right = words[index + 1]
            unit = _scale_phrase(unit_left, unit_right)
            if not unit:
                continue
            ybox, lbox, rbox, vbox = (year_word["bbox"], unit_left["bbox"],
                                     unit_right["bbox"], value["bbox"])
            unit_center = (lbox[0] + rbox[2]) / 2
            if not (0 <= lbox[1] - ybox[1] <= 28
                    and abs(_center(ybox) - unit_center) <= 24
                    and abs(_center(vbox) - unit_center) <= 24
                    and 0 < vbox[1] - lbox[1] <= 280
                    and abs(lbox[1] - rbox[1]) <= 5
                    and 0 <= rbox[0] - lbox[2] <= 10):
                continue
            headings.append((lbox[1], unit, year_word, unit_left, unit_right))
    if not headings or len({item[1] for item in headings}) != 1:
        return None
    headings.sort(key=lambda item: item[0], reverse=True)
    nearest_y = headings[0][0]
    nearest = [item for item in headings if abs(item[0] - nearest_y) < 3]
    if len({item[1] for item in nearest}) != 1:
        return None
    _, unit, year_word, left, right = nearest[0]
    # A later year heading in the same x lane starts another table.
    if any(word["text"] != year and re.fullmatch(r"(?:19|20)\d{2}", word["text"])
           and abs(_center(word["bbox"]) - _center(value["bbox"])) <= 24
           and nearest_y < word["bbox"][1] < value["bbox"][1]
           for word in words):
        return None
    return {
        "basis": "pdf_native_column_header", "source_name": source_name,
        "pdf_path": str(pdf), "pdf_sha256": pdf_hash, "page": page,
        "year": year, "unit": unit, "unit_text": left["text"] + " " + right["text"],
        "raw_value": raw_value.strip(), "value_bbox": list(value["bbox"]),
        "year_bbox": list(year_word["bbox"]),
        "unit_bbox": [left["bbox"][0], left["bbox"][1],
                      right["bbox"][2], right["bbox"][3]],
    }
