"""Repository-owned function mappings and optional auxiliary API summaries.

The default catalogs describe our actual runtime adapters. External FHE-Coder
summaries are opt-in and never replace the complete local function mapping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


DEFAULT_DOCUMENTATION_ROOT = Path(__file__).resolve().parents[1] / "api"
BACKEND_LIBRARIES = {
    "seal-ckks": "seal", "openfhe-cross": "openfhe", "phantom-ckks": "phantom",
}
DOCUMENTATION_PATHS = {
    "seal": "seal_documentation/summaries_db.json",
    "openfhe": "openfhe_documentation/summaries_db.json",
}
AUTHORITY = (
    "Repository function mappings describe a reviewed runtime adapter and library revision. "
    "Optional external summaries are auxiliary; their upstream revision is unknown. The task's "
    "hash-checked runtime header and compiler contract are authoritative. "
    "target_compatible denotes the library family, not verified API availability."
)


def _read_document(path: Path, max_bytes: int) -> tuple[bytes, list[dict]]:
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("documentation exceeds byte limit")
    document = json.loads(content)
    if not isinstance(document, dict) or not isinstance(document.get("summaries"), list):
        raise ValueError("documentation schema requires a summaries array")
    records = document["summaries"]
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("method_signature"), str):
            raise ValueError("documentation entry requires a method_signature string")
        if not record["method_signature"].strip():
            raise ValueError("documentation method_signature must not be empty")
        for field in ("summary", "doxygen_docstring", "source_file"):
            if field in record and not isinstance(record[field], str):
                raise ValueError(f"documentation {field} must be a string")
        if not (record.get("summary") or record.get("doxygen_docstring")):
            raise ValueError("documentation entry requires a description")
    return content, records


def load_function_mapping(backend: str) -> dict:
    """Load the fixed local IR -> runtime -> library API catalog for a backend.

    Catalog paths and source references are fixed here, never followed from
    descriptive JSON fields. Task preparation checks the reviewed runtime hash.
    """
    if backend not in BACKEND_LIBRARIES:
        raise ValueError(f"unsupported AI documentation backend: {backend}")
    library = BACKEND_LIBRARIES[backend]
    root = DEFAULT_DOCUMENTATION_ROOT.resolve()
    path = root / f"{library}.json"
    content = path.read_bytes()
    document = json.loads(content)
    if document.get("backend") != backend or document.get("library") != library:
        raise ValueError("function mapping backend/library mismatch")
    expected_runtime = f"backend/{library}/{library}_runtime.hpp"
    if (document.get("runtime_header") != expected_runtime
            or document.get("codegen_source") != "compiler/CodeGen/FHE/CodeGen.cpp"):
        raise ValueError("function mapping source references mismatch")
    if not re.fullmatch(r"[a-f0-9]{64}", document.get("runtime_sha256", "")):
        raise ValueError("function mapping requires a runtime SHA256")
    return {**document, "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "authority": "runtime_mapping"}


def _mapping_records(mapping: dict) -> list[dict]:
    """Expose the same local mapping to the bounded symbol retriever."""
    return [{
        "method_signature": "; ".join(record["library_calls"]) or record["runtime"],
        "summary": (f"IR: {', '.join(record['ir_ops']) or 'runtime helper'}. "
                    f"Runtime: {record['runtime']}. {record['description']} "
                    + " ".join(record["constraints"])),
        "source_file": mapping["runtime_header"],
    } for record in mapping["mappings"]]


def load_documentation(
    root: str | Path | None,
    backend: str,
    queries: Iterable[str],
    limit: int = 6,
    *,
    max_document_bytes: int = 16 * 1024 * 1024,
    max_entry_chars: int = 4000,
) -> dict:
    """Retrieve bounded, source-labelled snippets without executing any content.

    The complete local mapping is always included, independently of query/limit.
    By default snippets also use that catalog. An explicit external root reads
    FHE-Coder's target-library summaries as optional auxiliary context; Phantom
    has no external catalog and still retains its local function mapping.
    """
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("documentation limit must be between 1 and 20")
    if type(max_document_bytes) is not int or not 1 <= max_document_bytes <= 32 * 1024 * 1024:
        raise ValueError("invalid documentation byte limit")
    if type(max_entry_chars) is not int or not 128 <= max_entry_chars <= 8192:
        raise ValueError("documentation entry limit must be between 128 and 8192")
    mapping = load_function_mapping(backend)
    target = mapping["library"]
    query_list = [queries] if isinstance(queries, str) else list(queries)
    phrases = [query.lower().strip() for query in query_list if query.strip()]
    tokens = set(re.findall(r"[a-z_][a-z_0-9]*", " ".join(phrases)))
    document_root = Path(root if root is not None else DEFAULT_DOCUMENTATION_ROOT).expanduser().resolve()
    if root is not None and not document_root.is_dir():
        raise ValueError("documentation root must be an existing directory")
    local = document_root == DEFAULT_DOCUMENTATION_ROOT.resolve()
    sources, matches = [], []
    if local:
        # Reuse the single parsed snapshot so source hashes and records agree.
        catalogs = [(target, mapping["path"], mapping["sha256"], mapping["library_revision"],
                     "runtime_mapping", _mapping_records(mapping))]
    elif target in DOCUMENTATION_PATHS:
        path = document_root / DOCUMENTATION_PATHS[target]
        content, records = _read_document(path, max_document_bytes)
        catalogs = [(target, str(path), hashlib.sha256(content).hexdigest(), "unknown", "auxiliary", records)]
    else:
        catalogs = []
    for library, path, checksum, revision, authority, records in catalogs:
        source = {
            "library": library, "target_compatible": library == target,
            "path": path, "sha256": checksum,
            "revision": revision, "authority": authority, "entry_count": len(records),
        }
        sources.append(source)
        for position, record in enumerate(records):
            signature = record["method_signature"]
            description = record.get("summary", "")
            doxygen = record.get("doxygen_docstring", "")
            text = (description + " " + doxygen).lower()
            lowered_signature = signature.lower()
            score = sum(8 for phrase in phrases if phrase in lowered_signature)
            score += sum(2 for phrase in phrases if phrase in text)
            score += sum(3 for token in tokens if token in lowered_signature)
            score += sum(1 for token in tokens if token in text)
            if phrases and not score:
                continue
            # The combined signature/description/source metadata stays bounded.
            snippet_signature = signature[: min(1000, max_entry_chars // 3)]
            snippet_source = record.get("source_file", "")[: min(512, max_entry_chars // 6)]
            budget = max_entry_chars - len(snippet_signature) - len(snippet_source)
            snippet_description = (description + ("\n" + doxygen if doxygen else ""))[:budget]
            entry = {
                **{key: value for key, value in source.items() if key != "entry_count"},
                "signature": snippet_signature, "description": snippet_description,
                "source_file": snippet_source, "source_entry": position,
                "truncated": (len(signature) > len(snippet_signature) or
                              len(record.get("source_file", "")) > len(snippet_source) or
                              len(description) + (1 + len(doxygen) if doxygen else 0) > budget),
            }
            matches.append((score, len(sources), position, entry))
    matches.sort(key=lambda match: (-match[0], match[1], match[2]))
    return {
        "backend": backend, "target_library": target, "authority": AUTHORITY,
        "function_mapping": mapping,
        "sources": sources, "entries": [match[3] for match in matches[:limit]],
    }
