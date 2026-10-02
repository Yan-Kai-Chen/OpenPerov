#!/usr/bin/env python3
"""Build the OpenSolar RAG v3 multi-facet dense and lexical index.

This wrapper deliberately reuses the already proven four-GPU Qwen3 embedding
backend from ``innovation_graphrag_v1``.  It only adds v3 corpus validation,
length-sorted facet records, a lightweight FTS5 lane, and a v3-specific
manifest.  The original Stage5 and OpenAlex indexes are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any



MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EXPECTED_CORPUS_SCHEMA = "opensolar_rag_v3_multivector_corpus_manifest_v1_3"
EXPECTED_FACET_SCHEMA = "opensolar_rag_v3_multivector_facet_v1_3"


PUBLIC_MODEL_DIR: Path | None = None

def work_paths(work_root: Path) -> dict[str, Path]:
    return {
        "model": PUBLIC_MODEL_DIR if PUBLIC_MODEL_DIR is not None else work_root / "model",
        "records": work_root / "index" / "records.jsonl",
        "embeddings": work_root / "index" / "embeddings.f16.npy",
        "manifest": work_root / "index" / "manifest.json",
        "shards": work_root / "index" / "shards",
        "status": work_root / "runs" / "dense_index_v1" / "status",
        "logs": work_root / "runs" / "dense_index_v1" / "logs",
        "run_status": work_root / "runs" / "dense_index_v1" / "run_status.json",
    }

def download_model(work_root: Path) -> None:
    """Validate an explicitly supplied local model; never download automatically."""
    model = work_paths(work_root)["model"]
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise FileNotFoundError("Provide the local Qwen3-Embedding-0.6B checkpoint with --model-dir")

def last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    if bool(torch.all(attention_mask[:, -1] == 1)):
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[indices, sequence_lengths]

def encode_shard(
    work_root: Path,
    shard_index: int,
    num_shards: int,
    device: str,
    batch_size: int,
    max_length: int,
) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    paths = work_paths(work_root)
    paths["shards"].mkdir(parents=True, exist_ok=True)
    status_path = paths["status"] / f"shard_{shard_index:02d}.json"
    indices_path = paths["shards"] / f"indices_{shard_index:02d}.npy"
    embeddings_path = paths["shards"] / f"embeddings_{shard_index:02d}.f16.npy"
    if indices_path.is_file() and embeddings_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if prior.get("state") == "COMPLETE":
            print(f"\033[32mShard {shard_index} already complete\033[0m", flush=True)
            return

    records = read_jsonl(paths["records"])
    selected_indices = np.arange(shard_index, len(records), num_shards, dtype=np.int64)
    selected = [records[int(index)] for index in selected_indices]
    if not selected:
        raise RuntimeError(f"Shard {shard_index} is empty")
    atomic_json(
        status_path,
        {
            "state": "LOADING_MODEL",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "device": device,
            "total": len(selected),
            "completed": 0,
            "updated_at": now(),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["model"]), local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        str(paths["model"]),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    output_batches: list[np.ndarray] = []
    started = time.monotonic()
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        encoded = tokenizer(
            [row["document"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            pooled = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = functional.normalize(pooled, p=2, dim=1)
        output_batches.append(pooled.float().cpu().numpy().astype(np.float16))
        completed = min(start + len(batch), len(selected))
        if completed == len(selected) or completed % 256 == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            rate = completed / elapsed
            atomic_json(
                status_path,
                {
                    "state": "ENCODING",
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "device": device,
                    "total": len(selected),
                    "completed": completed,
                    "records_per_sec": round(rate, 3),
                    "eta_sec": round((len(selected) - completed) / rate, 1),
                    "updated_at": now(),
                },
            )
            print(
                f"\033[36mGPU shard {shard_index}: {completed:,}/{len(selected):,} "
                f"({rate:.2f} docs/s)\033[0m",
                flush=True,
            )
    matrix = np.concatenate(output_batches, axis=0)
    with indices_path.with_suffix(indices_path.suffix + ".tmp").open("wb") as handle:
        np.save(handle, selected_indices)
    os.replace(indices_path.with_suffix(indices_path.suffix + ".tmp"), indices_path)
    temporary_embeddings = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    with temporary_embeddings.open("wb") as handle:
        np.save(handle, matrix)
    os.replace(temporary_embeddings, embeddings_path)
    atomic_json(
        status_path,
        {
            "state": "COMPLETE",
            "shard_index": shard_index,
            "num_shards": num_shards,
            "device": device,
            "total": len(selected),
            "completed": len(selected),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "updated_at": now(),
        },
    )

def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backend() -> tuple[ModuleType, Path]:
    # Embedded reference helper functions retain the original embedding algorithm.
    return sys.modules[__name__], Path(__file__).resolve()



def paths(work_root: Path) -> dict[str, Path]:
    backend, _ = load_backend()
    inherited = backend.work_paths(work_root)
    inherited.update(
        {
            "fts": work_root / "index" / "facet_fts.sqlite",
            "pipeline_dir": work_root / "runs" / "index_v1_3",
            "pipeline_status": work_root
            / "runs"
            / "index_v1_3"
            / "pipeline_status.json",
            "prepare_status": work_root
            / "runs"
            / "index_v1_3"
            / "prepare_status.json",
            "logs_v3": work_root / "runs" / "index_v1_3" / "logs",
        }
    )
    return inherited


def read_source_manifest(source_root: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = source_root / "manifest.json"
    records_path = source_root / "facet_records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            f"Source corpus is incomplete: {manifest_path} / {records_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != EXPECTED_CORPUS_SCHEMA:
        raise RuntimeError(
            f"Expected corpus schema {EXPECTED_CORPUS_SCHEMA}, got {manifest.get('schema')}"
        )
    if manifest.get("status") != "READY":
        raise RuntimeError("Source corpus is not READY")
    expected_sha = str(
        ((manifest.get("files") or {}).get("facet_records.jsonl") or {}).get(
            "sha256"
        )
        or ""
    )
    if not expected_sha:
        raise RuntimeError("Source manifest lacks facet_records.jsonl SHA256")
    actual_sha = sha256(records_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Source facet SHA mismatch expected={expected_sha} actual={actual_sha}"
        )
    return manifest, records_path, actual_sha


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def build_fts(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE facet_meta (
                row_id INTEGER PRIMARY KEY,
                facet_id TEXT NOT NULL UNIQUE,
                core_id TEXT NOT NULL,
                facet_type TEXT NOT NULL,
                coverage_tier TEXT NOT NULL,
                title TEXT,
                retrieval_weight_hint REAL NOT NULL
            );
            CREATE INDEX facet_meta_core_idx ON facet_meta(core_id);
            CREATE INDEX facet_meta_type_idx ON facet_meta(facet_type);
            CREATE VIRTUAL TABLE facet_fts USING fts5(
                text,
                content='',
                tokenize='porter unicode61'
            );
            """
        )
        meta_insert = """
            INSERT INTO facet_meta(
                row_id,facet_id,core_id,facet_type,coverage_tier,title,
                retrieval_weight_hint
            ) VALUES(?,?,?,?,?,?,?)
        """
        fts_insert = "INSERT INTO facet_fts(rowid,text) VALUES(?,?)"
        with connection:
            for row_id, row in enumerate(records, start=1):
                connection.execute(
                    meta_insert,
                    (
                        row_id,
                        row["facet_id"],
                        row["core_id"],
                        row["facet_type"],
                        row["coverage_tier"],
                        row.get("title"),
                        float(row.get("retrieval_weight_hint") or 1.0),
                    ),
                )
                connection.execute(fts_insert, (row_id, row["document"]))
        meta_count = int(connection.execute("SELECT count(*) FROM facet_meta").fetchone()[0])
        fts_count = int(connection.execute("SELECT count(*) FROM facet_fts").fetchone()[0])
        if meta_count != len(records) or fts_count != len(records):
            raise RuntimeError(
                f"FTS row mismatch records={len(records)} meta={meta_count} fts={fts_count}"
            )
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    os.replace(temporary, path)


def prepare(source_root: Path, work_root: Path, limit: int | None) -> None:
    output = paths(work_root)
    output["pipeline_dir"].mkdir(parents=True, exist_ok=True)
    source_manifest, source_records, source_sha = read_source_manifest(source_root)
    status_path = output["prepare_status"]
    atomic_json(
        status_path,
        {
            "state": "READING_SOURCE",
            "source_root": str(source_root),
            "source_sha256": source_sha,
            "limit": limit,
            "updated_at": now(),
        },
    )
    records: list[dict[str, Any]] = []
    facet_ids: set[str] = set()
    with source_records.open("r", encoding="utf-8-sig") as handle:
        for source_row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != EXPECTED_FACET_SCHEMA:
                raise RuntimeError(
                    f"Unexpected facet schema at source row {source_row_index + 1}"
                )
            facet_id = str(row.get("facet_id") or "")
            document = str(row.pop("text", "") or "").strip()
            if not facet_id or facet_id in facet_ids or not document:
                raise RuntimeError(
                    f"Invalid facet at source row {source_row_index + 1}: {facet_id!r}"
                )
            facet_ids.add(facet_id)
            row["source_row_index"] = source_row_index
            row["document"] = document
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    expected_records = int(
        ((source_manifest.get("counts") or {}).get("facet_records") or 0)
    )
    if limit is None and len(records) != expected_records:
        raise RuntimeError(
            f"Source row mismatch expected={expected_records} actual={len(records)}"
        )
    records.sort(
        key=lambda row: (
            len(str(row["document"])),
            str(row["core_id"]),
            str(row["facet_type"]),
            str(row["facet_id"]),
        )
    )
    atomic_json(
        status_path,
        {
            "state": "WRITING_RECORDS_AND_FTS",
            "source_sha256": source_sha,
            "records": len(records),
            "updated_at": now(),
        },
    )
    write_records(output["records"], records)
    build_fts(output["fts"], records)
    atomic_json(
        status_path,
        {
            "state": "COMPLETE",
            "source_root": str(source_root),
            "source_sha256": source_sha,
            "records": len(records),
            "records_path": str(output["records"]),
            "records_sha256": sha256(output["records"]),
            "fts_path": str(output["fts"]),
            "fts_sha256": sha256(output["fts"]),
            "length_sorted_for_gpu_padding_efficiency": True,
            "limit": limit,
            "updated_at": now(),
        },
    )
    print(f"\033[32mPrepared {len(records):,} v3 facet records and FTS5 lane\033[0m")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(
    source_root: Path,
    work_root: Path,
    num_shards: int,
    max_length: int,
    backend_path: Path,
) -> dict[str, Any]:
    import numpy as np

    output = paths(work_root)
    source_manifest, _, source_sha = read_source_manifest(source_root)
    records = read_jsonl(output["records"])
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    embedding_dim: int | None = None
    for shard_index in range(num_shards):
        indices_path = output["shards"] / f"indices_{shard_index:02d}.npy"
        embeddings_path = output["shards"] / f"embeddings_{shard_index:02d}.f16.npy"
        indices = np.load(indices_path)
        embeddings = np.load(embeddings_path)
        if len(indices) != embeddings.shape[0]:
            raise RuntimeError(f"Shard {shard_index} row mismatch")
        embedding_dim = embedding_dim or int(embeddings.shape[1])
        if embeddings.shape[1] != embedding_dim:
            raise RuntimeError(f"Shard {shard_index} dimension mismatch")
        pieces.append((indices, embeddings))
    if embedding_dim is None:
        raise RuntimeError("No embedding shards found")
    all_indices = np.concatenate([part[0] for part in pieces])
    if len(all_indices) != len(records):
        raise RuntimeError("Merged shard count does not match records")
    if not np.array_equal(np.sort(all_indices), np.arange(len(records))):
        raise RuntimeError("Shard indices are incomplete or duplicated")
    matrix = np.empty((len(records), embedding_dim), dtype=np.float16)
    for indices, embeddings in pieces:
        matrix[indices] = embeddings
    sample_indices = np.linspace(
        0, len(matrix) - 1, min(1024, len(matrix)), dtype=np.int64
    )
    norms = np.linalg.norm(matrix[sample_indices].astype(np.float32), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 0.01:
        raise RuntimeError("Embedding normalization check failed")
    temporary = output["embeddings"].with_suffix(output["embeddings"].suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, matrix)
    os.replace(temporary, output["embeddings"])
    model_dir = output["model"]
    weight_files = sorted(model_dir.glob("*.safetensors"))
    manifest = {
        "schema": "opensolar_rag_v3_multivector_index_manifest_v1_3",
        "status": "READY",
        "created_at": now(),
        "source_corpus": str(source_root),
        "source_corpus_schema": source_manifest.get("schema"),
        "source_facet_records_sha256": source_sha,
        "model_id": MODEL_ID,
        "model_path": str(model_dir),
        "model_source_policy": "reuse local cache; otherwise hf-mirror.com with proxies cleared",
        "reused_backend": str(backend_path),
        "records": len(records),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype),
        "max_length": max_length,
        "num_gpu_shards": num_shards,
        "document_order": "ascending text characters, then core/facet identifiers",
        "document_embedding_input": "natural-language facet text without query instruction",
        "retrieval_policy": "facet lanes are aggregated to CORE; facet count is not a vote count",
        "files": {
            "records.jsonl": {
                "bytes": output["records"].stat().st_size,
                "sha256": sha256(output["records"]),
            },
            "embeddings.f16.npy": {
                "bytes": output["embeddings"].stat().st_size,
                "sha256": sha256(output["embeddings"]),
            },
            "facet_fts.sqlite": {
                "bytes": output["fts"].stat().st_size,
                "sha256": sha256(output["fts"]),
            },
        },
        "model_files": {
            "config_exists": (model_dir / "config.json").is_file(),
            "weight_files": [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in weight_files
            ],
        },
        "normalization_sample": {
            "count": len(norms),
            "min": round(float(norms.min()), 6),
            "max": round(float(norms.max()), 6),
        },
    }
    atomic_json(output["manifest"], manifest)
    return manifest


def run_all(args: argparse.Namespace, backend: ModuleType, backend_path: Path) -> None:
    output = paths(args.work_root)
    for key in ("shards", "logs_v3", "pipeline_dir"):
        output[key].mkdir(parents=True, exist_ok=True)
    source_manifest, _, source_sha = read_source_manifest(args.source_root)
    if output["manifest"].is_file():
        prior = json.loads(output["manifest"].read_text(encoding="utf-8-sig"))
        if (
            prior.get("status") == "READY"
            and prior.get("source_facet_records_sha256") == source_sha
        ):
            print(f"\033[32mRAG v3 index already READY: {output['manifest']}\033[0m")
            return
    pipeline = {
        "schema": "opensolar_rag_v3_index_pipeline_status_v1_3",
        "state": "STARTING",
        "source_root": str(args.source_root),
        "source_schema": source_manifest.get("schema"),
        "source_sha256": source_sha,
        "work_root": str(args.work_root),
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "started_at": now(),
        "updated_at": now(),
    }
    atomic_json(output["pipeline_status"], pipeline)
    try:
        pipeline.update(state="CHECKING_MODEL", updated_at=now())
        atomic_json(output["pipeline_status"], pipeline)
        backend.download_model(args.work_root)
        pipeline.update(state="PREPARING_CORPUS_AND_FTS", updated_at=now())
        atomic_json(output["pipeline_status"], pipeline)
        prepare(args.source_root, args.work_root, args.limit)

        processes: list[tuple[int, subprocess.Popen[Any], Any, Any]] = []
        for shard_index in range(args.num_shards):
            stdout_path = output["logs_v3"] / f"shard_{shard_index:02d}.stdout.log"
            stderr_path = output["logs_v3"] / f"shard_{shard_index:02d}.stderr.log"
            stdout_handle = stdout_path.open("a", encoding="utf-8", buffering=1)
            stderr_handle = stderr_path.open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-dir",
                str(args.model_dir.resolve()),
                "--mode",
                "encode-shard",
                "--source-root",
                str(args.source_root.resolve()),
                "--work-root",
                str(args.work_root.resolve()),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(args.num_shards),
                "--device",
                f"cuda:{shard_index}",
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
            ]
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=str(args.work_root),
                env=os.environ.copy(),
            )
            processes.append((shard_index, process, stdout_handle, stderr_handle))
        pipeline.update(
            state="ENCODING_4GPU",
            pids={str(index): process.pid for index, process, _, _ in processes},
            updated_at=now(),
        )
        atomic_json(output["pipeline_status"], pipeline)
        failed: list[tuple[int, int]] = []
        try:
            for shard_index, process, _, _ in processes:
                return_code = process.wait()
                if return_code != 0:
                    failed.append((shard_index, return_code))
        finally:
            for _, process, stdout_handle, stderr_handle in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                stdout_handle.close()
                stderr_handle.close()
        if failed:
            raise RuntimeError(f"Encoding shard failures: {failed}")
        pipeline.update(state="MERGING", updated_at=now())
        atomic_json(output["pipeline_status"], pipeline)
        manifest = merge(
            args.source_root,
            args.work_root,
            args.num_shards,
            args.max_length,
            backend_path,
        )
        pipeline.update(
            state="COMPLETE",
            manifest=str(output["manifest"]),
            records=manifest["records"],
            embedding_shape=manifest["embedding_shape"],
            completed_at=now(),
            updated_at=now(),
        )
        atomic_json(output["pipeline_status"], pipeline)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        pipeline.update(
            state="FAILED",
            error=repr(exc),
            traceback=traceback.format_exc(),
            failed_at=now(),
            updated_at=now(),
        )
        atomic_json(output["pipeline_status"], pipeline)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "prepare", "encode-shard", "merge", "self-test"),
        default="all",
    )
    parser.add_argument("--model-dir", type=Path, help="Local Qwen3-Embedding-0.6B weights; required for encoding/merge")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PUBLIC_MODEL_DIR
    PUBLIC_MODEL_DIR = args.model_dir.resolve() if args.model_dir else None
    if args.mode in {"all", "encode-shard", "merge"} and PUBLIC_MODEL_DIR is None:
        raise ValueError("--model-dir is required for encoding/merge; no implicit download")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index is outside --num-shards")
    if args.batch_size < 1 or args.max_length < 128:
        raise ValueError("Invalid batch size or max length")
    backend, backend_path = load_backend()
    if args.mode == "self-test":
        manifest, records_path, source_sha = read_source_manifest(args.source_root)
        print(
            json.dumps(
                {
                    "status": "SELF_TEST_OK",
                    "backend": str(backend_path),
                    "backend_sha256": sha256(backend_path),
                    "source_schema": manifest.get("schema"),
                    "source_records": str(records_path),
                    "source_sha256": source_sha,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.mode == "prepare":
        prepare(args.source_root, args.work_root, args.limit)
        return 0
    if args.mode == "encode-shard":
        backend.encode_shard(
            args.work_root,
            args.shard_index,
            args.num_shards,
            args.device,
            args.batch_size,
            args.max_length,
        )
        return 0
    if args.mode == "merge":
        manifest = merge(
            args.source_root,
            args.work_root,
            args.num_shards,
            args.max_length,
            backend_path,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    run_all(args, backend, backend_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
