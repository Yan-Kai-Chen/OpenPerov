#!/usr/bin/env python3
"""Natural-language hybrid retrieval over the RAG v3 multi-facet CORE index.

Dense scores are normalized independently inside each scientific facet lane.
Each CORE can contribute at most one result per lane, and only its three best
lanes affect the dense aggregate.  This prevents papers with more source packs
from winning merely because they received more retrieval lottery tickets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


QUERY_INSTRUCTIONS = {
    "broad_rag": (
    "Instruct: Given a natural-language research problem in perovskite "
    "photovoltaics or optoelectronics, retrieve multiple scientifically relevant "
    "papers that provide mechanisms, materials evidence, interface knowledge, "
    "processing methods, degradation explanations, comparisons, or transferable "
    "design strategies. Do not try to identify a hidden source paper. Prefer "
    "specific and complementary evidence over generic background.\nQuery: "
    ),
    "source_recovery": (
        "Instruct: Given a detailed scientific question derived from a perovskite "
        "research paper, retrieve the original or closest source paper whose "
        "experiments, materials, device architecture, quantitative observations, "
        "and mechanistic conclusions jointly explain the question. Prioritize "
        "specific evidence matches over general topical similarity.\nQuery: "
    ),
    "source_evidence": (
        "Instruct: Match this perovskite research question to the paper containing "
        "the same distinctive experimental evidence. Use exact materials, layer "
        "stacks, treatments, measurements, numerical values, failure signatures, "
        "and causal mechanism together. Retrieve the most likely source paper, "
        "not merely a paper on the same topic.\nQuery: "
    ),
    "source_identity": (
        "Instruct: Identify the original perovskite paper from which this technical "
        "question could have been constructed. Retrieve papers matching the full "
        "combination of material system, intervention, characterization results, "
        "device behavior, and claimed mechanism. Penalize generic reviews and "
        "single-keyword matches.\nQuery: "
    ),
}

STOPWORDS = {
    "about", "after", "also", "among", "and", "are", "because", "been",
    "before", "between", "both", "build", "can", "cell", "cells", "could",
    "device", "does", "during", "explain", "for", "from", "give", "how",
    "into", "its", "most", "not", "observations", "of", "on", "or",
    "perovskite", "photovoltaic", "please", "research", "should", "solar",
    "system", "than", "that", "the", "their", "then", "these", "this",
    "through", "using", "what", "when", "where", "which", "while", "why",
    "with", "would",
}
KNOWN_PHRASES = (
    "phase segregation",
    "ion migration",
    "halide migration",
    "wide bandgap",
    "narrow bandgap",
    "self assembled monolayer",
    "buried interface",
    "surface passivation",
    "operational stability",
    "tin oxidation",
    "lead tin",
    "mixed halide",
    "nonradiative recombination",
    "charge extraction",
    "open circuit voltage",
    "current voltage hysteresis",
    "thermal degradation",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,}")
FORBIDDEN_QUERY_FIELDS = {
    "core_id",
    "expected_core_ids",
    "reference_expected_answer",
    "gold_answer",
    "expected_answer",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool(torch.all(attention_mask[:, -1] == 1)):
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[indices, sequence_lengths]


def flatten_terms(value: Any) -> list[str]:
    output: list[str] = []
    if value is None:
        return output
    if isinstance(value, str):
        if value.strip():
            output.append(value.strip())
        return output
    if isinstance(value, list):
        for item in value:
            output.extend(flatten_terms(item))
        return output
    if isinstance(value, dict):
        for item in value.values():
            output.extend(flatten_terms(item))
    return output


def row_keywords(row: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "keywords",
        "extra_keywords",
        "retrieval_terms",
        "terms",
        "synonyms",
        "query_plan",
        "raw_content",
    ):
        values.extend(flatten_terms(row.get(key)))
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return " ".join(unique)[:4000]


def lexical_expression(query: str, extra_keywords: str = "") -> str:
    text = f"{query} {extra_keywords}".casefold()
    terms: list[str] = []
    seen: set[str] = set()
    for phrase in KNOWN_PHRASES:
        if phrase in text and phrase not in seen:
            terms.append(f'"{phrase}"')
            seen.add(phrase)
    for token in TOKEN_RE.findall(text):
        token = token.strip(".+-")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        if token.isdigit():
            continue
        escaped = token.replace('"', '""')
        terms.append(f'"{escaped}"')
        seen.add(token)
        if len(terms) >= 64:
            break
    return " OR ".join(terms)


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "natural_language_query"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError("Query row has no question/query/prompt text")


class MultiFacetRetriever:
    def __init__(
        self,
        index_root: Path,
        model_dir: Path,
        device: str,
        max_length: int,
        query_instruction: str,
    ) -> None:
        global torch, functional, AutoModel, AutoTokenizer
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
        manifest_path = index_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("status") != "READY":
            raise RuntimeError("RAG v3 index is not READY")
        self.index_root = index_root
        self.records = read_jsonl(index_root / "records.jsonl")
        matrix = np.load(index_root / "embeddings.f16.npy")
        if matrix.shape[0] != len(self.records):
            raise RuntimeError("Embedding matrix and record count differ")
        if int(manifest.get("records") or 0) != len(self.records):
            raise RuntimeError("Index manifest record count differs")
        self.device = torch.device(device)
        self.gpu_matrix = torch.from_numpy(matrix).to(self.device)
        self.max_length = max_length
        self.query_instruction = query_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        type_indices: dict[str, list[int]] = defaultdict(list)
        self.core_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.records):
            type_indices[str(row["facet_type"])].append(index)
            self.core_to_indices[str(row["core_id"])].append(index)
        self.type_indices = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in sorted(type_indices.items())
        }
        self.fts_uri = f"file:{(index_root / 'facet_fts.sqlite').as_posix()}?mode=ro"

    def encode_queries(self, queries: list[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(queries), batch_size):
            texts = [
                self.query_instruction + query
                for query in queries[start : start + batch_size]
            ]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded)
                pooled = last_token_pool(
                    output.last_hidden_state, encoded["attention_mask"]
                )
                pooled = functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    def dense_scores(self, vectors: np.ndarray, batch_size: int = 64) -> Iterable[np.ndarray]:
        for start in range(0, len(vectors), batch_size):
            query = torch.from_numpy(vectors[start : start + batch_size]).to(
                self.device, dtype=self.gpu_matrix.dtype
            )
            scores = query @ self.gpu_matrix.T
            for row in scores.float().cpu().numpy():
                yield row

    def aggregate_dense(
        self, scores: np.ndarray, lane_top_k: int, dense_core_k: int
    ) -> list[dict[str, Any]]:
        core_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for facet_type, indices in self.type_indices.items():
            lane_scores = scores[indices]
            mean = float(lane_scores.mean())
            std = max(float(lane_scores.std()), 1e-6)
            z_scores = (lane_scores - mean) / std
            keep = min(lane_top_k, len(indices))
            if keep == len(indices):
                local_positions = np.argsort(-z_scores)
            else:
                local_positions = np.argpartition(-z_scores, keep - 1)[:keep]
                local_positions = local_positions[np.argsort(-z_scores[local_positions])]
            for lane_rank, local_position in enumerate(local_positions, start=1):
                record_index = int(indices[int(local_position)])
                record = self.records[record_index]
                raw = float(scores[record_index])
                z_value = float(z_scores[int(local_position)])
                weight = float(record.get("retrieval_weight_hint") or 1.0)
                value = z_value * weight
                core_hits[str(record["core_id"])].append(
                    {
                        "facet_type": facet_type,
                        "record_index": record_index,
                        "lane_rank": lane_rank,
                        "raw_cosine": raw,
                        "lane_z": z_value,
                        "weight": weight,
                        "weighted_lane_z": value,
                    }
                )
        aggregated: list[dict[str, Any]] = []
        for core_id, hits in core_hits.items():
            hits.sort(key=lambda item: (-item["weighted_lane_z"], item["facet_type"]))
            values = [float(item["weighted_lane_z"]) for item in hits]
            primary = values[0]
            secondary = max(values[1], 0.0) if len(values) > 1 else 0.0
            tertiary = max(values[2], 0.0) if len(values) > 2 else 0.0
            aggregate = primary + 0.10 * secondary + 0.025 * tertiary
            aggregated.append(
                {
                    "core_id": core_id,
                    "dense_aggregate_score": aggregate,
                    "dense_primary_score": primary,
                    "dense_support_score": 0.10 * secondary + 0.025 * tertiary,
                    "dense_hits": hits[:3],
                }
            )
        aggregated.sort(
            key=lambda item: (-item["dense_aggregate_score"], item["core_id"])
        )
        output = aggregated[:dense_core_k]
        for rank, item in enumerate(output, start=1):
            item["dense_core_rank"] = rank
        return output

    def lexical_cores(
        self,
        query: str,
        extra_keywords: str,
        facet_limit: int,
        core_limit: int,
    ) -> tuple[list[dict[str, Any]], str]:
        expression = lexical_expression(query, extra_keywords)
        if not expression:
            return [], expression
        connection = sqlite3.connect(self.fts_uri, uri=True)
        try:
            rows = connection.execute(
                """
                SELECT rowid,bm25(facet_fts) AS score
                FROM facet_fts
                WHERE facet_fts MATCH ?
                ORDER BY score,rowid
                LIMIT ?
                """,
                (expression, facet_limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return [], expression
        finally:
            connection.close()
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for facet_rank, (row_id, bm25_score) in enumerate(rows, start=1):
            record_index = int(row_id) - 1
            record = self.records[record_index]
            core_id = str(record["core_id"])
            if core_id in seen:
                continue
            seen.add(core_id)
            output.append(
                {
                    "core_id": core_id,
                    "lexical_core_rank": len(output) + 1,
                    "lexical_facet_rank": facet_rank,
                    "lexical_bm25": float(bm25_score),
                    "lexical_record_index": record_index,
                }
            )
            if len(output) >= core_limit:
                break
        return output, expression

    def fuse(
        self,
        dense: list[dict[str, Any]],
        lexical: list[dict[str, Any]],
        top_k: int,
        rrf_k: int,
        lexical_weight: float,
        preview_chars: int,
        max_facets: int,
    ) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for item in dense:
            core_id = str(item["core_id"])
            entry = candidates.setdefault(core_id, {"core_id": core_id, "rrf": 0.0})
            entry.update(item)
            entry["rrf"] += 1.0 / (rrf_k + int(item["dense_core_rank"]))
        for item in lexical:
            core_id = str(item["core_id"])
            entry = candidates.setdefault(core_id, {"core_id": core_id, "rrf": 0.0})
            entry.update(item)
            entry["rrf"] += lexical_weight / (
                rrf_k + int(item["lexical_core_rank"])
            )
        ordered = sorted(candidates.values(), key=lambda item: (-item["rrf"], item["core_id"]))
        output: list[dict[str, Any]] = []
        for rank, item in enumerate(ordered[:top_k], start=1):
            core_id = str(item["core_id"])
            representative = self.records[self.core_to_indices[core_id][0]]
            facets: list[dict[str, Any]] = []
            used_indices: set[int] = set()
            for hit in item.get("dense_hits") or []:
                if len(facets) >= max_facets:
                    break
                record_index = int(hit["record_index"])
                record = self.records[record_index]
                used_indices.add(record_index)
                facets.append(
                    {
                        "facet_type": record["facet_type"],
                        "source_kind": record.get("source_kind"),
                        "raw_cosine": round(float(hit["raw_cosine"]), 6),
                        "lane_z": round(float(hit["lane_z"]), 6),
                        "lane_rank": int(hit["lane_rank"]),
                        "text_preview": str(record["document"])[:preview_chars],
                    }
                )
            lexical_index = item.get("lexical_record_index")
            if (
                len(facets) < max_facets
                and lexical_index is not None
                and int(lexical_index) not in used_indices
            ):
                record = self.records[int(lexical_index)]
                facets.append(
                    {
                        "facet_type": record["facet_type"],
                        "source_kind": record.get("source_kind"),
                        "lexical_facet_rank": int(item["lexical_facet_rank"]),
                        "lexical_bm25": round(float(item["lexical_bm25"]), 6),
                        "text_preview": str(record["document"])[:preview_chars],
                    }
                )
            output.append(
                {
                    "rank": rank,
                    "core_id": core_id,
                    "title": representative.get("title"),
                    "journal": representative.get("journal"),
                    "year": representative.get("year"),
                    "doi": representative.get("doi"),
                    "openalex_id": representative.get("openalex_id"),
                    "coverage_tier": representative.get("coverage_tier"),
                    "hybrid_rrf_score": round(float(item["rrf"]), 9),
                    "dense_core_rank": item.get("dense_core_rank"),
                    "dense_aggregate_score": (
                        round(float(item["dense_aggregate_score"]), 6)
                        if item.get("dense_aggregate_score") is not None
                        else None
                    ),
                    "lexical_core_rank": item.get("lexical_core_rank"),
                    "best_facets": facets,
                }
            )
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query")
    group.add_argument("--questions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lane-top-k", type=int, default=300)
    parser.add_argument("--dense-core-k", type=int, default=1500)
    parser.add_argument("--lexical-facet-k", type=int, default=2000)
    parser.add_argument("--lexical-core-k", type=int, default=1200)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lexical-weight", type=float, default=0.75)
    parser.add_argument("--preview-chars", type=int, default=700)
    parser.add_argument("--max-facets", type=int, default=4)
    parser.add_argument(
        "--query-mode",
        choices=tuple(QUERY_INSTRUCTIONS),
        default="broad_rag",
        help="Embedding instruction. broad_rag preserves the original behavior; source_* modes recover the likely source paper.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.query is not None:
        inputs = [{"query_id": "interactive_0001", "question": args.query}]
    else:
        inputs = read_jsonl(args.questions)
    for index, row in enumerate(inputs):
        leaked = FORBIDDEN_QUERY_FIELDS.intersection(row)
        if leaked:
            raise RuntimeError(
                f"Private fields found in retrieval input row {index}: {sorted(leaked)}"
            )
    queries = [question_text(row) for row in inputs]
    keywords = [row_keywords(row) for row in inputs]
    started = time.monotonic()
    if args.status:
        atomic_json(
            args.status,
            {
                "schema": "opensolar_rag_v3_retrieval_status_v1_3",
                "state": "LOADING_INDEX_AND_MODEL",
                "queries": len(inputs),
                "completed": 0,
                "private_labels_present": False,
                "updated_at": now(),
            },
        )
    retriever = MultiFacetRetriever(
        args.index_root,
        args.model_dir,
        args.device,
        args.max_length,
        QUERY_INSTRUCTIONS[args.query_mode],
    )
    if args.status:
        atomic_json(
            args.status,
            {
                "schema": "opensolar_rag_v3_retrieval_status_v1_3",
                "state": "ENCODING_QUERIES",
                "queries": len(inputs),
                "completed": 0,
                "private_labels_present": False,
                "updated_at": now(),
            },
        )
    vectors = retriever.encode_queries(queries, args.query_batch_size)
    rows: list[dict[str, Any]] = []
    for index, scores in enumerate(retriever.dense_scores(vectors)):
        dense = retriever.aggregate_dense(
            scores, args.lane_top_k, args.dense_core_k
        )
        lexical, expression = retriever.lexical_cores(
            queries[index],
            keywords[index],
            args.lexical_facet_k,
            args.lexical_core_k,
        )
        results = retriever.fuse(
            dense,
            lexical,
            args.top_k,
            args.rrf_k,
            args.lexical_weight,
            args.preview_chars,
            args.max_facets,
        )
        source = inputs[index]
        output_row = {
            "schema": "opensolar_rag_v3_multivector_retrieval_item_v1_3",
            "query_id": source.get("query_id") or f"query_{index:04d}",
            "benchmark_id": source.get("benchmark_id"),
            "ability_family": source.get("ability_family"),
            "question": queries[index],
            "query_keywords": keywords[index] or None,
            "query_mode": args.query_mode,
            "query_instruction": QUERY_INSTRUCTIONS[args.query_mode],
            "private_labels_present": False,
            "aggregation_policy": (
                "facet-wise z normalization; one hit per CORE/facet lane; "
                "best three lanes only; dense/lexical CORE-level RRF"
            ),
            "lexical_expression": expression,
            "lane_counts": {
                "dense_cores": len(dense),
                "lexical_cores": len(lexical),
            },
            "results": results,
        }
        for passthrough_key in ("parent_query_id", "package_id", "view_type", "view_index"):
            if source.get(passthrough_key) is not None:
                output_row[passthrough_key] = source[passthrough_key]
        rows.append(output_row)
        completed = index + 1
        if args.status and (completed == len(inputs) or completed % 25 == 0):
            elapsed = max(time.monotonic() - started, 0.001)
            atomic_json(
                args.status,
                {
                    "schema": "opensolar_rag_v3_retrieval_status_v1_3",
                    "state": "RETRIEVING",
                    "queries": len(inputs),
                    "completed": completed,
                    "queries_per_sec": round(completed / elapsed, 3),
                    "private_labels_present": False,
                    "updated_at": now(),
                },
            )
    if args.output:
        write_jsonl(args.output, rows)
    else:
        print(json.dumps(rows[0] if len(rows) == 1 else rows, ensure_ascii=False, indent=2))
    if args.status:
        atomic_json(
            args.status,
            {
                "schema": "opensolar_rag_v3_retrieval_status_v1_3",
                "state": "COMPLETE",
                "queries": len(inputs),
                "completed": len(inputs),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "output": str(args.output) if args.output else None,
                "private_labels_present": False,
                "updated_at": now(),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
