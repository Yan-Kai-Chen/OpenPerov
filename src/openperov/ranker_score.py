#!/usr/bin/env python3
"""Score every frozen validation candidate with the trained Qwen3 reranker LoRA.

The script is intentionally inference-only.  It preserves the exact natural-
language prompt and yes/no logit-difference score used during training, writes
one deterministic score per (query_id, candidate_code), and supports torchrun
across four local GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_INSTRUCTION = (
    "Given a natural-language perovskite research question, rank papers by how well "
    "they provide direct evidence or scientifically useful context needed to answer "
    "the question. Prefer condition-matched mechanisms, materials, device structures, "
    "measurements, and interventions over generic topical similarity."
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Candidate:
    query_id: str
    candidate_code: str
    core_id: str
    query: str
    title: str
    evidence: str
    grade: int | None
    family: str

    @property
    def document(self) -> str:
        title = " ".join(self.title.split())
        evidence = " ".join(self.evidence.split())
        return f"Article title: {title}\nScientific evidence: {evidence}"


@dataclass(frozen=True)
class QueryGroup:
    query_id: str
    query: str
    family: str
    candidates: tuple[Candidate, ...]


def read_groups(path: Path, allow_unlabeled: bool = False) -> list[QueryGroup]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    query_text: dict[str, str] = {}
    family: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            candidate_code = str(row["candidate_code"])
            key = (query_id, candidate_code)
            if key in seen:
                raise ValueError(f"Duplicate query/candidate at {path}:{line_number}: {key}")
            seen.add(key)
            raw_grade = row.get("relevance_grade")
            if raw_grade is None:
                if not allow_unlabeled:
                    raise ValueError(f"Missing relevance grade at {path}:{line_number}")
                grade = None
            else:
                grade = int(raw_grade)
                if grade not in {0, 1, 2, 3}:
                    raise ValueError(f"Invalid grade at {path}:{line_number}: {grade}")
            query = str(row["query"]).strip()
            current_family = str(row.get("ability_family") or "unknown")
            if query_id in query_text and query_text[query_id] != query:
                raise ValueError(f"Conflicting query text for {query_id}")
            query_text[query_id] = query
            family[query_id] = current_family
            grouped[query_id].append(
                Candidate(
                    query_id=query_id,
                    candidate_code=candidate_code,
                    core_id=str(row["core_id"]),
                    query=query,
                    title=str(row.get("title") or "").strip(),
                    evidence=str(row.get("scientific_evidence") or "").strip(),
                    grade=grade,
                    family=current_family,
                )
            )
    groups = [
        QueryGroup(
            query_id=query_id,
            query=query_text[query_id],
            family=family[query_id],
            candidates=tuple(grouped[query_id]),
        )
        for query_id in sorted(grouped)
    ]
    for group in groups:
        grades = {candidate.grade for candidate in group.candidates if candidate.grade is not None}
        if not allow_unlabeled and len(grades) < 2:
            raise ValueError(f"Query has fewer than two relevance levels: {group.query_id}")
    return groups


class PromptScorer:
    def __init__(self, tokenizer: Any, max_length: int, instruction: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)
        if self.yes_token_id is None or self.no_token_id is None:
            raise RuntimeError("Could not resolve Qwen3 reranker yes/no token IDs")

    def format(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def score_components(
        self,
        model: torch.nn.Module,
        query: str,
        candidates: list[Candidate],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [self.format(query, candidate.document) for candidate in candidates]
        usable = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=usable,
            return_attention_mask=False,
        )
        for index, token_ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][index] = self.prefix_tokens + token_ids + self.suffix_tokens
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        return (
            logits[:, self.yes_token_id].float(),
            logits[:, self.no_token_id].float(),
        )

    def scores(
        self,
        model: torch.nn.Module,
        query: str,
        candidates: list[Candidate],
        device: torch.device,
    ) -> torch.Tensor:
        yes_logits, no_logits = self.score_components(model, query, candidates, device)
        return yes_logits - no_logits


def ranking_metrics(grades: list[int], k: int = 10) -> dict[str, float]:
    top = (grades[:k] + [0] * k)[:k]
    dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(top))
    ideal = (sorted(grades, reverse=True)[:k] + [0] * k)[:k]
    idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    return {
        "useful_precision10_grade2": sum(grade >= 2 for grade in top) / k,
        "direct_precision10_grade3": sum(grade >= 3 for grade in top) / k,
        "success10_grade2": float(any(grade >= 2 for grade in top)),
        "success10_grade3": float(any(grade >= 3 for grade in top)),
        "mean_grade10": sum(top) / k,
        "ndcg10": dcg / idcg if idcg else 1.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--allow-unlabeled", action="store_true")
    parser.add_argument("--score-filename", default="lora_validation_scores.jsonl")
    parser.add_argument(
        "--distributed-input-shards",
        action="store_true",
        help="Resolve {rank} in validation-jsonl and let each distributed rank read only its query shard.",
    )
    parser.add_argument(
        "--include-base-scores",
        action="store_true",
        help="Also score with the LoRA adapter disabled and emit raw yes/no distributions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.distributed_input_shards and (world_size <= 1 or not args.allow_unlabeled):
        raise ValueError("--distributed-input-shards requires distributed unlabeled inference")
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    started = time.monotonic()
    status_path = args.output_dir / "scoring_status.json"
    score_path = args.output_dir / args.score_filename

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for stale in args.output_dir.glob("lora_scores.rank*.jsonl*"):
            stale.unlink()
        if score_path.exists():
            score_path.unlink()
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_scoring_status_v1",
                "state": "LOADING_DATA",
                "world_size": world_size,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )

    input_path = (
        Path(str(args.validation_jsonl).format(rank=f"{rank:02d}"))
        if args.distributed_input_shards else args.validation_jsonl
    )
    groups = read_groups(input_path, allow_unlabeled=args.allow_unlabeled)
    local_expected_keys = {
        (candidate.query_id, candidate.candidate_code)
        for group in groups for candidate in group.candidates
    }
    if args.distributed_input_shards:
        totals = torch.tensor(
            [len(groups), sum(len(group.candidates) for group in groups)],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        expected_queries = int(totals[0].item())
        expected_candidates = int(totals[1].item())
    else:
        expected_queries = len(groups)
        expected_candidates = sum(len(group.candidates) for group in groups)
    if rank == 0:
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_scoring_status_v1",
                "state": "LOADING_MODEL",
                "world_size": world_size,
                "total_queries": expected_queries,
                "total_candidates": expected_candidates,
                "completed_queries": 0,
                "completed_candidates": 0,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model, str(args.adapter_dir), is_trainable=False, local_files_only=True
    ).to(device)
    model.eval()
    scorer = PromptScorer(tokenizer, args.max_length, args.instruction)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_scoring_status_v1",
                "state": "SCORING",
                "world_size": world_size,
                "total_queries": expected_queries,
                "total_candidates": expected_candidates,
                "completed_queries": 0,
                "completed_candidates": 0,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )

    local_groups = groups if args.distributed_input_shards else groups[rank::world_size]
    if args.distributed_input_shards:
        max_steps_tensor = torch.tensor([len(local_groups)], dtype=torch.int64, device=device)
        dist.all_reduce(max_steps_tensor, op=dist.ReduceOp.MAX)
        max_steps = int(max_steps_tensor.item())
    else:
        max_steps = math.ceil(len(groups) / world_size)
    local_rows: list[dict[str, Any]] = []
    completed_queries = 0
    completed_candidates = 0
    with torch.inference_mode():
        for step in range(max_steps):
            step_queries = 0
            step_candidates = 0
            if step < len(local_groups):
                group = local_groups[step]
                candidates = list(group.candidates)
                component_rows: list[dict[str, float]] = []
                for offset in range(0, len(candidates), args.eval_batch_size):
                    batch = candidates[offset : offset + args.eval_batch_size]
                    lora_yes, lora_no = scorer.score_components(
                        model, group.query, batch, device
                    )
                    if args.include_base_scores:
                        with model.disable_adapter():
                            base_yes, base_no = scorer.score_components(
                                model, group.query, batch, device
                            )
                    else:
                        base_yes = base_no = None
                    for batch_index in range(len(batch)):
                        yes_value = float(lora_yes[batch_index].cpu())
                        no_value = float(lora_no[batch_index].cpu())
                        margin = yes_value - no_value
                        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, margin))))
                        entropy = -(
                            probability * math.log(max(probability, 1e-12))
                            + (1.0 - probability)
                            * math.log(max(1.0 - probability, 1e-12))
                        )
                        components = {
                            "lora_yes_logit": yes_value,
                            "lora_no_logit": no_value,
                            "lora_score": margin,
                            "lora_probability": probability,
                            "lora_binary_entropy": entropy,
                        }
                        if base_yes is not None and base_no is not None:
                            base_yes_value = float(base_yes[batch_index].cpu())
                            base_no_value = float(base_no[batch_index].cpu())
                            base_margin = base_yes_value - base_no_value
                            components.update(
                                {
                                    "base_yes_logit": base_yes_value,
                                    "base_no_logit": base_no_value,
                                    "base_score": base_margin,
                                    "lora_delta_score": margin - base_margin,
                                }
                            )
                        component_rows.append(components)
                for candidate, components in zip(candidates, component_rows):
                    output_row = {
                        "schema": "opensolar_qwen3_reranker_lora_candidate_score_v2",
                        "query_id": candidate.query_id,
                        "ability_family": candidate.family,
                        "candidate_code": candidate.candidate_code,
                        "core_id": candidate.core_id,
                        **components,
                        "human_expert_annotation": False,
                    }
                    if candidate.grade is not None:
                        output_row["relevance_grade"] = candidate.grade
                    local_rows.append(output_row)
                step_queries = 1
                step_candidates = len(candidates)
            progress = torch.tensor(
                [step_queries, step_candidates], dtype=torch.int64, device=device
            )
            if world_size > 1:
                dist.all_reduce(progress, op=dist.ReduceOp.SUM)
            if rank == 0:
                completed_queries += int(progress[0].item())
                completed_candidates += int(progress[1].item())
                elapsed = time.monotonic() - started
                rate = completed_candidates / elapsed if elapsed else 0.0
                eta = (expected_candidates - completed_candidates) / rate if rate else None
                atomic_json(
                    status_path,
                    {
                        "schema": "opensolar_qwen3_reranker_lora_scoring_status_v1",
                        "state": "SCORING",
                        "world_size": world_size,
                        "total_queries": expected_queries,
                        "total_candidates": expected_candidates,
                        "completed_queries": completed_queries,
                        "completed_candidates": completed_candidates,
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds": round(eta, 3) if eta is not None else None,
                        "updated_at": now(),
                        "human_expert_annotation": False,
                    },
                )
                print(
                    "\033[32m"
                    f"scored queries={completed_queries}/{expected_queries} "
                    f"candidates={completed_candidates}/{expected_candidates} "
                    f"eta_min={(eta or 0.0) / 60.0:.1f}"
                    "\033[0m",
                    flush=True,
                )

    shard_path = args.output_dir / f"lora_scores.rank{rank}.jsonl"
    atomic_jsonl(shard_path, local_rows)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for shard_rank in range(world_size):
            path = args.output_dir / f"lora_scores.rank{shard_rank}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (str(row["query_id"]), str(row["candidate_code"]))
                    if key in merged:
                        raise RuntimeError(f"Duplicate merged score: {key}")
                    merged[key] = row
        if args.distributed_input_shards:
            expected_keys: set[tuple[str, str]] = set()
            for shard_rank in range(world_size):
                shard_input = Path(str(args.validation_jsonl).format(rank=f"{shard_rank:02d}"))
                for shard_group in read_groups(shard_input, allow_unlabeled=True):
                    expected_keys.update(
                        (candidate.query_id, candidate.candidate_code)
                        for candidate in shard_group.candidates
                    )
            if set(merged) != expected_keys:
                raise RuntimeError(
                    f"Merged sharded coverage mismatch: expected={len(expected_keys)} actual={len(merged)}"
                )
            ordered_rows = [merged[key] for key in sorted(merged)]
        else:
            expected_keys = {
                (candidate.query_id, candidate.candidate_code)
                for group in groups
                for candidate in group.candidates
            }
            if set(merged) != expected_keys:
                raise RuntimeError(
                    f"Merged score coverage mismatch: expected={len(expected_keys)} actual={len(merged)}"
                )
            ordered_rows = [
                merged[(candidate.query_id, candidate.candidate_code)]
                for group in groups
                for candidate in group.candidates
            ]
        atomic_jsonl(score_path, ordered_rows)
        labels_available = all(
            candidate.grade is not None for group in groups for candidate in group.candidates
        )
        metrics: dict[str, float] | None = None
        if labels_available:
            per_query: list[dict[str, float]] = []
            for group in groups:
                ranked = sorted(
                    group.candidates,
                    key=lambda candidate: -float(
                        merged[(candidate.query_id, candidate.candidate_code)]["lora_score"]
                    ),
                )
                per_query.append(
                    ranking_metrics([int(candidate.grade) for candidate in ranked])
                )
            metrics = aggregate(per_query)
        report = {
            "schema": "opensolar_qwen3_reranker_lora_scoring_report_v1",
            "status": "COMPLETE",
            "created_at": now(),
            "world_size": world_size,
            "queries": expected_queries,
            "candidates": expected_candidates,
            "model_facing_core_ids": False,
            "human_expert_annotation": False,
            "candidate_pool_frozen": True,
            "retrieval_rerun": False,
            "base_scores_included": args.include_base_scores,
            "labels_available": labels_available,
            "metrics": metrics,
            "inputs": {
                "validation_jsonl": (
                    {
                        "path_pattern": str(args.validation_jsonl),
                        "shards": [
                            {
                                "path": str(args.validation_jsonl).format(rank=f"{shard_rank:02d}"),
                                "sha256": sha256(Path(str(args.validation_jsonl).format(rank=f"{shard_rank:02d}"))),
                            }
                            for shard_rank in range(world_size)
                        ],
                    }
                    if args.distributed_input_shards else
                    {"path": str(args.validation_jsonl), "sha256": sha256(args.validation_jsonl)}
                ),
                "base_model": str(args.model_dir),
                "adapter": {
                    "path": str(args.adapter_dir),
                    "sha256": sha256(args.adapter_dir / "adapter_model.safetensors"),
                },
            },
            "output": {"path": str(score_path), "sha256": sha256(score_path)},
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        atomic_json(args.output_dir / "lora_scoring_report.json", report)
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_scoring_status_v1",
                "state": "COMPLETE",
                "world_size": world_size,
                "total_queries": expected_queries,
                "total_candidates": expected_candidates,
                "completed_queries": expected_queries,
                "completed_candidates": expected_candidates,
                "metrics": metrics,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )
        if metrics is None:
            print(
                f"\033[36mCOMPLETE unlabeled queries={expected_queries} "
                f"candidates={expected_candidates}\033[0m",
                flush=True,
            )
        else:
            print(
                "\033[36m"
                f"COMPLETE nDCG@10={metrics['ndcg10']:.6f} "
                f"useful@10={metrics['useful_precision10_grade2']:.6f} "
                f"direct@10={metrics['direct_precision10_grade3']:.6f}"
                "\033[0m",
                flush=True,
            )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
