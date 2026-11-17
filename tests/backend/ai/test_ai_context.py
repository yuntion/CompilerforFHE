"""API context respects actual catalog schemas, provenance and library boundaries."""
import hashlib
import json
from pathlib import Path

import pytest

from backend.ai import context as context_module
from backend.ai.context import load_documentation


LOCAL_API_ROOT = Path(__file__).resolve().parents[3] / "backend" / "api"


@pytest.mark.parametrize("backend,library,query,operation,symbol", [
    ("seal-ckks", "seal", "rotate_vector", "rotate", "seal::Evaluator::rotate_vector"),
    ("openfhe-cross", "openfhe", "EvalCKKStoFHEW", "to_logic", "EvalCKKStoFHEW"),
    ("phantom-ckks", "phantom", "rotate", "rotate", "phantom::rotate"),
])
def test_default_context_uses_complete_local_mapping(backend, library, query, operation, symbol):
    result = load_documentation(None, backend, [query], limit=1)
    assert context_module.DEFAULT_DOCUMENTATION_ROOT == LOCAL_API_ROOT
    mapping = result["function_mapping"]
    path = LOCAL_API_ROOT / f"{library}.json"
    source = json.loads(path.read_text())
    assert mapping["backend"] == backend and mapping["library"] == library
    assert mapping["authority"] == "runtime_mapping"
    assert mapping["path"] == str(path)
    assert mapping["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = LOCAL_API_ROOT.parent / library / f"{library}_runtime.hpp"
    assert mapping["runtime_sha256"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert mapping["mappings"] == source["mappings"]
    assert len(mapping["mappings"]) > 1  # limit applies to snippets, not the full mapping.
    binding = next(item for item in mapping["mappings"] if item["operation"] == operation)
    assert any(symbol in call for call in binding["library_calls"])
    assert symbol in json.dumps(result["entries"])
    assert len(result["entries"]) == 1
    assert result["sources"]
    assert all(Path(item["path"]).is_relative_to(LOCAL_API_ROOT) for item in result["sources"])


def catalog(root, library, records):
    path = root / f"{library}_documentation/summaries_db.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summaries": records}))
    return path


def entry(signature, description, **extra):
    return {"method_signature": signature, "summary": description,
            "doxygen_docstring": "@brief " + description, **extra}


def test_retrieval_ranks_symbols_and_preserves_document_provenance(tmp_path):
    path = catalog(tmp_path, "openfhe", [
        entry("EvalMult(a, b)", "Multiply ciphertexts", source_file="pke/include/cryptocontext.h"),
        entry("EvalRotate(ciphertext, index)", "Rotate ciphertext slots"),
    ])
    result = load_documentation(tmp_path, "openfhe-cross", ["EvalRotate"], limit=1)
    assert result["target_library"] == "openfhe"
    assert "runtime header" in result["authority"]
    retrieved, = result["entries"]
    assert retrieved["signature"] == "EvalRotate(ciphertext, index)"
    assert retrieved["target_compatible"] is True
    assert retrieved["revision"] == "unknown"
    assert retrieved["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert retrieved["path"] == str(path)
    assert retrieved["source_entry"] == 1
    # Auxiliary external summaries cannot replace the complete local runtime mapping.
    mapping = result["function_mapping"]
    assert mapping["path"] == str(LOCAL_API_ROOT / "openfhe.json")
    assert mapping["authority"] == "runtime_mapping"
    assert any("EvalCKKStoFHEW" in call for item in mapping["mappings"] for call in item["library_calls"])


def test_other_library_docs_are_explicit_and_never_masquerade_as_backend_apis(tmp_path):
    catalog(tmp_path, "openfhe", [entry("EvalSign(x)", "FHEW sign")])
    catalog(tmp_path, "tfhe", [entry("bootsAND(x, y)", "TFHE gate")])
    catalog(tmp_path, "heongpu", [entry("heongpu::rotate(x)", "CUDA rotate")])
    default = load_documentation(tmp_path, "openfhe-cross", [])
    assert {item["library"] for item in default["entries"]} == {"openfhe"}
    assert all(item["target_compatible"] for item in default["entries"])
    assert {item["library"] for item in default["sources"]} == {"openfhe"}
    phantom = load_documentation(tmp_path, "phantom-ckks", [])
    assert phantom["entries"] == [] and phantom["sources"] == []
    assert phantom["function_mapping"]["library"] == "phantom"
    assert phantom["function_mapping"]["authority"] == "runtime_mapping"


@pytest.mark.parametrize("bad", [[], {"summaries": {}}, {"summaries": [{}]},
                                       {"summaries": [{"method_signature": "x()", "summary": 1}]},
                                       {"summaries": [{"method_signature": "", "summary": "x"}]}])
def test_invalid_documentation_schema_is_rejected_before_retrieval(tmp_path, bad):
    path = catalog(tmp_path, "seal", [])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema|entry|signature|summary"):
        load_documentation(tmp_path, "seal-ckks", [])


def test_document_and_snippet_limits_and_untrusted_source_paths(tmp_path):
    secret = tmp_path / "private-key.txt"
    secret.write_text("must not appear in retrieval")
    path = catalog(tmp_path, "seal", [
        entry("seal::rotate()", "rotate " + "r" * 6000, source_file=str(secret)),
        entry("seal::rotate2()", "rotate another"),
    ])
    result = load_documentation(tmp_path, "seal-ckks", ["rotate"], limit=1, max_entry_chars=128)
    retrieved, = result["entries"]
    assert retrieved["truncated"] is True
    assert sum(len(retrieved[key]) for key in ("signature", "description", "source_file")) <= 128
    assert "must not appear" not in json.dumps(result)
    with pytest.raises(ValueError, match="byte limit"):
        load_documentation(tmp_path, "seal-ckks", [], max_document_bytes=path.stat().st_size - 1)
    assert retrieved["source_file"] == str(secret)[: min(512, 128 // 6)]


def test_tfhe_catalog_schema_without_source_file_is_supported(tmp_path):
    # Optional source metadata uses the same schema; only the target catalog is read.
    catalog(tmp_path, "tfhe", [{
        "method_signature": "void bootsAND(LweSample* result, const LweSample* a, const LweSample* b);",
        "summary": "Homomorphic AND gate",
        "doxygen_docstring": "/** @brief Homomorphic AND gate */",
    }])
    catalog(tmp_path, "openfhe", [{
        "method_signature": "EvalAdd(a, b)", "summary": "Homomorphic addition",
        "doxygen_docstring": "/** @brief Homomorphic addition */",
    }])
    result = load_documentation(tmp_path, "openfhe-cross", ["EvalAdd"])
    retrieved, = result["entries"]
    assert retrieved["source_file"] == ""
    assert retrieved["library"] == "openfhe" and retrieved["target_compatible"] is True
    assert "bootsAND" not in json.dumps(result)


@pytest.mark.parametrize("change", [
    lambda mapping: mapping.pop("runtime_sha256"),
    lambda mapping: mapping.update(backend="phantom-ckks"),
])
def test_local_mapping_rejects_missing_runtime_identity_or_wrong_backend(tmp_path, monkeypatch, change):
    mapping = json.loads((LOCAL_API_ROOT / "seal.json").read_text())
    change(mapping)
    (tmp_path / "seal.json").write_text(json.dumps(mapping))
    monkeypatch.setattr(context_module, "DEFAULT_DOCUMENTATION_ROOT", tmp_path)
    with pytest.raises(ValueError, match="function mapping"):
        context_module.load_function_mapping("seal-ckks")
