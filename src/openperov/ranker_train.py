#!/usr/bin/env python3
"""Four-GPU weighted LoRA training for OpenSolar Qwen3 rerankers.

The model keeps Qwen3-Reranker's official next-token yes/no scoring head.  For
each natural-language query we jointly score a graded list of candidate papers
and optimize a LambdaRank-style NDCG@10 loss over 0-3 relevance targets.
Formal, legacy, and calibrated soft labels can carry different confidences.
No CORE/source identifier is included in model-facing text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Candidate:
    query_id: str
    query: str
    title: str
    evidence: str
    grade: float
    confidence: float
    label_source: str
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


def read_groups(path: Path) -> list[QueryGroup]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    query_text: dict[str, str] = {}
    family: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            grade = float(row.get("relevance_target", row["relevance_grade"]))
            confidence = float(row.get("label_confidence", 1.0))
            if not 0.0 <= grade <= 3.0:
                raise ValueError(f"Invalid relevance target at {path}:{line_number}: {grade}")
            if not 0.0 < confidence <= 1.0:
                raise ValueError(f"Invalid label confidence at {path}:{line_number}: {confidence}")
            query = str(row["query"]).strip()
            current_family = str(row.get("ability_family") or "unknown")
            if query_id in query_text and query_text[query_id] != query:
                raise ValueError(f"Conflicting query text for {query_id}")
            query_text[query_id] = query
            family[query_id] = current_family
            grouped[query_id].append(
                Candidate(
                    query_id=query_id,
                    query=query,
                    title=str(row.get("title") or "").strip(),
                    evidence=str(row.get("scientific_evidence") or "").strip(),
                    grade=grade,
                    confidence=confidence,
                    label_source=str(row.get("label_source") or "unspecified"),
                    family=current_family,
                )
            )
    result: list[QueryGroup] = []
    for query_id in sorted(grouped):
        candidates = tuple(grouped[query_id])
        if max(candidate.grade for candidate in candidates) - min(
            candidate.grade for candidate in candidates
        ) <= 1e-6:
            raise ValueError(f"Query has fewer than two relevance levels: {query_id}")
        result.append(
            QueryGroup(
                query_id=query_id,
                query=query_text[query_id],
                family=family[query_id],
                candidates=candidates,
            )
        )
    return result


def dataset_summary(groups: Iterable[QueryGroup]) -> dict[str, Any]:
    groups = list(groups)
    candidates = [candidate for group in groups for candidate in group.candidates]
    grades = Counter(max(0, min(3, round(candidate.grade))) for candidate in candidates)
    families = Counter(group.family for group in groups)
    label_sources = Counter(candidate.label_source for candidate in candidates)
    confidences = [candidate.confidence for candidate in candidates]
    return {
        "queries": len(groups),
        "candidates": len(candidates),
        "rounded_target_distribution": {str(key): grades[key] for key in range(4)},
        "ability_families": dict(sorted(families.items())),
        "label_sources": dict(sorted(label_sources.items())),
        "confidence": {
            "min": min(confidences),
            "mean": sum(confidences) / len(confidences),
            "max": max(confidences),
        },
    }


def target_bin(candidate: Candidate) -> int:
    return max(0, min(3, round(candidate.grade)))


def structured_sample(
    candidates: list[Candidate], count: int, rng: random.Random
) -> tuple[list[Candidate], list[Candidate]]:
    by_grade: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_grade[target_bin(candidate)].append(candidate)
    for values in by_grade.values():
        rng.shuffle(values)
    selected: list[Candidate] = []
    grade_cycle = (3, 0, 2, 1, 3, 0, 2, 1)
    while len(selected) < count and any(by_grade.values()):
        progressed = False
        for grade in grade_cycle:
            if len(selected) >= count:
                break
            if by_grade[grade]:
                selected.append(by_grade[grade].pop())
                progressed = True
        if not progressed:
            break
    remaining = [candidate for values in by_grade.values() for candidate in values]
    rng.shuffle(remaining)
    return selected, remaining


def sample_candidate_list(
    group: QueryGroup,
    docs_per_list: int,
    seed: int,
    min_trusted_per_list: int,
    trusted_confidence_threshold: float,
) -> tuple[Candidate, ...]:
    rng = random.Random(seed)
    trusted = [
        candidate
        for candidate in group.candidates
        if candidate.confidence >= trusted_confidence_threshold
    ]
    auxiliary = [
        candidate
        for candidate in group.candidates
        if candidate.confidence < trusted_confidence_threshold
    ]
    trusted_target = min(docs_per_list, len(trusted), min_trusted_per_list)
    auxiliary_target = min(len(auxiliary), docs_per_list - trusted_target)
    trusted_selected, trusted_remaining = structured_sample(trusted, trusted_target, rng)
    auxiliary_selected, auxiliary_remaining = structured_sample(auxiliary, auxiliary_target, rng)
    selected = trusted_selected + auxiliary_selected
    remaining = trusted_remaining + auxiliary_remaining
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, docs_per_list - len(selected))])
    selected = selected[:docs_per_list]
    rng.shuffle(selected)
    if max(candidate.grade for candidate in selected) - min(
        candidate.grade for candidate in selected
    ) <= 1e-6:
        raise RuntimeError(f"Sampled list lost grade diversity: {group.query_id}")
    return tuple(selected)


def build_epoch_examples(
    groups: list[QueryGroup],
    lists_per_query: int,
    docs_per_list: int,
    seed: int,
    min_trusted_per_list: int,
    trusted_confidence_threshold: float,
) -> list[tuple[QueryGroup, tuple[Candidate, ...]]]:
    examples: list[tuple[QueryGroup, tuple[Candidate, ...]]] = []
    for group_index, group in enumerate(groups):
        for list_index in range(lists_per_query):
            sample_seed = seed + group_index * 1009 + list_index * 9176
            examples.append(
                (
                    group,
                    sample_candidate_list(
                        group,
                        docs_per_list,
                        sample_seed,
                        min_trusted_per_list,
                        trusted_confidence_threshold,
                    ),
                )
            )
    random.Random(seed + 4049).shuffle(examples)
    return examples


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
        if max_length <= len(self.prefix_tokens) + len(self.suffix_tokens) + 64:
            raise ValueError("max_length leaves too little room for query/document text")

    def format(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        )

    def encode(self, query: str, candidates: Iterable[Candidate], device: torch.device) -> dict[str, torch.Tensor]:
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
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    def scores(
        self,
        model: torch.nn.Module,
        query: str,
        candidates: Iterable[Candidate],
        device: torch.device,
    ) -> torch.Tensor:
        inputs = self.encode(query, candidates, device)
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        return (logits[:, self.yes_token_id] - logits[:, self.no_token_id]).float()


def lambda_rank_loss(
    scores: torch.Tensor,
    grades: torch.Tensor,
    confidences: torch.Tensor | None = None,
    k: int = 10,
) -> torch.Tensor:
    """Confidence-weighted LambdaRank pairwise logistic loss at NDCG@k."""
    count = scores.numel()
    if count < 2:
        return scores.sum() * 0.0
    if confidences is None:
        confidences = torch.ones_like(grades, dtype=torch.float32)
    if confidences.numel() != count:
        raise ValueError("Confidence count does not match score count")
    with torch.no_grad():
        gains = torch.pow(2.0, grades.float()) - 1.0
        order = torch.argsort(scores.detach(), descending=True)
        positions = torch.empty_like(order)
        positions[order] = torch.arange(count, device=scores.device)
        discounts = torch.zeros(count, device=scores.device, dtype=torch.float32)
        in_topk = positions < min(k, count)
        discounts[in_topk] = 1.0 / torch.log2(positions[in_topk].float() + 2.0)
        ideal_grades = torch.sort(grades, descending=True).values[:k]
        ideal_gains = torch.pow(2.0, ideal_grades.float()) - 1.0
        ideal_discounts = 1.0 / torch.log2(
            torch.arange(ideal_gains.numel(), device=scores.device).float() + 2.0
        )
        ideal_dcg = torch.sum(ideal_gains * ideal_discounts).clamp_min(1e-8)

    losses: list[torch.Tensor] = []
    normalizers: list[torch.Tensor] = []
    for high in range(count):
        for low in range(count):
            if grades[high] <= grades[low]:
                continue
            with torch.no_grad():
                base_weight = torch.abs(
                    (gains[high] - gains[low]) * (discounts[high] - discounts[low])
                ) / ideal_dcg
                pair_confidence = torch.minimum(confidences[high], confidences[low])
                weight = base_weight * pair_confidence
            if float(base_weight) <= 0.0:
                continue
            losses.append(F.softplus(-(scores[high] - scores[low])) * weight)
            normalizers.append(base_weight)
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).sum() / torch.stack(normalizers).sum().clamp_min(1e-8)


def ndcg_at_k(grades: list[float], k: int = 10) -> float:
    selected = grades[:k]
    dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(selected))
    ideal = sorted(grades, reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate(
    model: torch.nn.Module,
    scorer: PromptScorer,
    groups: list[QueryGroup],
    rank: int,
    world_size: int,
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(13, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for group in groups[rank::world_size]:
            scores: list[float] = []
            candidates = list(group.candidates)
            for offset in range(0, len(candidates), eval_batch_size):
                batch_scores = scorer.scores(
                    model, group.query, candidates[offset : offset + eval_batch_size], device
                )
                scores.extend(float(value) for value in batch_scores.cpu())
            ranked = sorted(
                zip(candidates, scores), key=lambda pair: pair[1], reverse=True
            )
            top = ranked[:10]
            top_grades = [candidate.grade for candidate, _ in top]
            all_ranked_grades = [candidate.grade for candidate, _ in ranked]
            denom = float(len(top))
            totals[0] += 1.0
            totals[1] += sum(grade >= 2 for grade in top_grades) / denom
            totals[2] += sum(grade >= 3 for grade in top_grades) / denom
            totals[3] += float(any(grade >= 2 for grade in top_grades))
            totals[4] += float(any(grade >= 3 for grade in top_grades))
            totals[5] += sum(top_grades) / denom
            totals[6] += ndcg_at_k(all_ranked_grades, 10)
            source_positions = [
                index
                for index, (candidate, _) in enumerate(ranked, start=1)
                if candidate.grade >= 3
            ]
            first_source = min(source_positions) if source_positions else None
            totals[7] += float(first_source is not None and first_source <= 1)
            totals[8] += float(first_source is not None and first_source <= 5)
            totals[9] += float(first_source is not None and first_source <= 10)
            totals[10] += float(first_source is not None and first_source <= 20)
            totals[11] += float(first_source is not None and first_source <= 40)
            totals[12] += 1.0 / first_source if first_source is not None else 0.0
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    queries = max(float(totals[0].item()), 1.0)
    model.train()
    return {
        "queries": int(totals[0].item()),
        "useful_precision10_grade2": float(totals[1].item() / queries),
        "direct_precision10_grade3": float(totals[2].item() / queries),
        "success10_grade2": float(totals[3].item() / queries),
        "success10_grade3": float(totals[4].item() / queries),
        "mean_grade10": float(totals[5].item() / queries),
        "ndcg10": float(totals[6].item() / queries),
        "source_recall1_grade3": float(totals[7].item() / queries),
        "source_recall5_grade3": float(totals[8].item() / queries),
        "source_recall10_grade3": float(totals[9].item() / queries),
        "source_recall20_grade3": float(totals[10].item() / queries),
        "source_recall40_grade3": float(totals[11].item() / queries),
        "source_mrr_grade3": float(totals[12].item() / queries),
    }


def save_adapter(model: torch.nn.Module, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-adapter-dir",
        type=Path,
        default=None,
        help="Optional prior-stage LoRA weights used to initialize a new training stage.",
    )
    parser.add_argument("--training-stage", default="stage1")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lists-per-query", type=int, default=2)
    parser.add_argument("--docs-per-list", type=int, default=8)
    parser.add_argument("--min-trusted-per-list", type=int, default=4)
    parser.add_argument("--trusted-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-validation-queries", type=int, default=0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--selection-metric",
        choices=("ndcg10", "source_recall40_grade3"),
        default="ndcg10",
        help="Primary validation metric used to select adapter_best.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_trusted_per_list <= args.docs_per_list:
        parser.error("--min-trusted-per-list must be between 0 and --docs-per-list")
    if not 0.0 < args.trusted_confidence_threshold <= 1.0:
        parser.error("--trusted-confidence-threshold must be in (0, 1]")
    if args.initial_adapter_dir is not None:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (args.initial_adapter_dir / name).is_file():
                parser.error(f"--initial-adapter-dir is missing {name}")
    return args


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    status_path = args.output_dir / "status.json"
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_status_v1",
                "state": "LOADING_DATA",
                "world_size": world_size,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )
        if metrics_path.exists():
            metrics_path.unlink()

    train_groups = read_groups(args.train_jsonl)
    validation_groups = read_groups(args.validation_jsonl)
    if args.max_train_queries > 0:
        train_groups = train_groups[: args.max_train_queries]
    if args.max_validation_queries > 0:
        validation_groups = validation_groups[: args.max_validation_queries]
    overlap = {group.query_id for group in train_groups}.intersection(
        group.query_id for group in validation_groups
    )
    if overlap:
        raise RuntimeError(f"Train/validation query overlap: {len(overlap)}")
    epoch_examples = len(train_groups) * args.lists_per_query
    if epoch_examples % world_size:
        raise RuntimeError(
            f"Epoch examples ({epoch_examples}) must divide world size ({world_size})"
        )
    examples_per_rank = epoch_examples // world_size
    optimizer_steps_per_epoch = math.ceil(
        examples_per_rank / args.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs

    if rank == 0:
        run_config = {
            "schema": "opensolar_qwen3_reranker_lora_config_v2",
            "created_at": now(),
            "objective": "confidence-weighted graded listwise LambdaRank NDCG@10 over relevance 0-3",
            "training_stage": args.training_stage,
            "optimizer_resumed": False,
            "model": str(args.model_dir),
            "initial_adapter": (
                {
                    "path": str(args.initial_adapter_dir),
                    "adapter_model_sha256": sha256(
                        args.initial_adapter_dir / "adapter_model.safetensors"
                    ),
                }
                if args.initial_adapter_dir is not None
                else None
            ),
            "train": {**dataset_summary(train_groups), "sha256": sha256(args.train_jsonl)},
            "validation": {
                **dataset_summary(validation_groups),
                "sha256": sha256(args.validation_jsonl),
            },
            "hyperparameters": vars(args),
            "world_size": world_size,
            "epoch_examples": epoch_examples,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "model_facing_core_ids": False,
            "human_expert_annotation": False,
        }
        # Paths are not JSON serializable in vars(args).
        run_config["hyperparameters"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        atomic_json(args.output_dir / "run_config.json", run_config)
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_status_v1",
                "state": "LOADING_MODEL",
                "world_size": world_size,
                "train_queries": len(train_groups),
                "validation_queries": len(validation_groups),
                "optimizer_step": 0,
                "total_optimizer_steps": total_optimizer_steps,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )

    tokenizer_source = args.initial_adapter_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), local_files_only=True, padding_side="left"
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
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base_model.enable_input_require_grads()
    if args.initial_adapter_dir is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(args.initial_adapter_dir),
            is_trainable=True,
        )
        active_adapter = model.active_adapter
        loaded_config = model.peft_config[active_adapter]
        expected = (args.lora_r, args.lora_alpha, args.lora_dropout)
        actual = (loaded_config.r, loaded_config.lora_alpha, loaded_config.lora_dropout)
        if actual != expected:
            raise RuntimeError(
                "Initial adapter LoRA configuration does not match requested new-stage "
                f"configuration: actual={actual}, expected={expected}"
            )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if rank == 0:
        print(
            f"\033[36mLoRA trainable parameters: {trainable_parameters:,} / "
            f"{total_parameters:,} ({100.0 * trainable_parameters / total_parameters:.3f}%)\033[0m",
            flush=True,
        )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    scorer = PromptScorer(tokenizer, args.max_length, args.instruction)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup_steps = max(1, round(total_optimizer_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps
    )

    if rank == 0:
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_status_v1",
                "state": "BASELINE_EVALUATION",
                "world_size": world_size,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "optimizer_step": 0,
                "total_optimizer_steps": total_optimizer_steps,
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )
    baseline = evaluate(
        model, scorer, validation_groups, rank, world_size, device, args.eval_batch_size
    )
    history: list[dict[str, Any]] = []
    if rank == 0:
        baseline_stage = "initial_adapter" if args.initial_adapter_dir is not None else "base"
        baseline_row = {"stage": baseline_stage, "created_at": now(), **baseline}
        history.append(baseline_row)
        append_jsonl(metrics_path, baseline_row)
        print(
            f"\033[35m{baseline_stage.upper()} "
            f"nDCG@10={baseline['ndcg10']:.4f} useful@10={baseline['useful_precision10_grade2']:.4f} "
            f"direct@10={baseline['direct_precision10_grade3']:.4f} "
            f"direct-success={baseline['success10_grade3']:.4f}\033[0m",
            flush=True,
        )

    best_selection_value = baseline[args.selection_metric]
    best_ndcg = baseline["ndcg10"]
    best_epoch = 0
    if rank == 0:
        initial_dir = args.output_dir / "adapter_initial"
        best_dir = args.output_dir / "adapter_best"
        for path in (initial_dir, best_dir):
            if path.exists():
                shutil.rmtree(path)
            save_adapter(model, tokenizer, path)
    if world_size > 1:
        dist.barrier()
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        global_examples = build_epoch_examples(
            train_groups,
            args.lists_per_query,
            args.docs_per_list,
            args.seed + epoch * 100_003,
            args.min_trusted_per_list,
            args.trusted_confidence_threshold,
        )
        local_examples = global_examples[rank::world_size]
        local_loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
        local_loss_count = torch.zeros(1, dtype=torch.float64, device=device)
        for example_index, (group, candidates) in enumerate(local_examples, start=1):
            sync_now = (
                example_index % args.gradient_accumulation_steps == 0
                or example_index == len(local_examples)
            )
            sync_context = nullcontext() if sync_now or not isinstance(model, DDP) else model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    scores = scorer.scores(model, group.query, candidates, device)
                    grades = torch.tensor(
                        [candidate.grade for candidate in candidates],
                        dtype=torch.float32,
                        device=device,
                    )
                    confidences = torch.tensor(
                        [candidate.confidence for candidate in candidates],
                        dtype=torch.float32,
                        device=device,
                    )
                    loss = lambda_rank_loss(scores, grades, confidences, k=10)
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
            local_loss_sum += float(loss.detach())
            local_loss_count += 1.0
            if sync_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if rank == 0 and (
                    optimizer_step == 1
                    or optimizer_step % args.log_every == 0
                    or optimizer_step == total_optimizer_steps
                ):
                    elapsed = time.monotonic() - started
                    eta = (
                        elapsed / optimizer_step * (total_optimizer_steps - optimizer_step)
                        if optimizer_step
                        else None
                    )
                    status = {
                        "schema": "opensolar_qwen3_reranker_lora_status_v1",
                        "state": "TRAINING",
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "optimizer_step": optimizer_step,
                        "total_optimizer_steps": total_optimizer_steps,
                        "last_group_loss_rank0": float(loss.detach()),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds": round(eta, 3) if eta is not None else None,
                        "updated_at": now(),
                        "human_expert_annotation": False,
                    }
                    atomic_json(status_path, status)
                    print(
                        "\033[32m"
                        f"epoch={epoch}/{args.epochs} step={optimizer_step}/{total_optimizer_steps} "
                        f"loss={float(loss.detach()):.5f} lr={scheduler.get_last_lr()[0]:.3e} "
                        f"eta_min={(eta or 0.0) / 60.0:.1f}"
                        "\033[0m",
                        flush=True,
                    )

        reduced = torch.stack((local_loss_sum[0], local_loss_count[0]))
        if world_size > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        epoch_train_loss = float(reduced[0].item() / max(reduced[1].item(), 1.0))
        if rank == 0:
            atomic_json(
                status_path,
                {
                    "schema": "opensolar_qwen3_reranker_lora_status_v1",
                    "state": "EVALUATING",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "optimizer_step": optimizer_step,
                    "total_optimizer_steps": total_optimizer_steps,
                    "train_loss": epoch_train_loss,
                    "updated_at": now(),
                    "human_expert_annotation": False,
                },
            )
        metrics = evaluate(
            model, scorer, validation_groups, rank, world_size, device, args.eval_batch_size
        )
        if rank == 0:
            row = {
                "stage": f"epoch_{epoch}",
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "created_at": now(),
                **metrics,
            }
            history.append(row)
            append_jsonl(metrics_path, row)
            print(
                "\033[35m"
                f"EPOCH {epoch} nDCG@10={metrics['ndcg10']:.4f} "
                f"useful@10={metrics['useful_precision10_grade2']:.4f} "
                f"direct@10={metrics['direct_precision10_grade3']:.4f} "
                f"direct-success={metrics['success10_grade3']:.4f}"
                "\033[0m",
                flush=True,
            )
            epoch_dir = args.output_dir / f"adapter_epoch_{epoch}"
            save_adapter(model, tokenizer, epoch_dir)
            selection_value = metrics[args.selection_metric]
            if selection_value > best_selection_value or (
                selection_value == best_selection_value and metrics["ndcg10"] > best_ndcg
            ):
                best_selection_value = selection_value
                best_ndcg = metrics["ndcg10"]
                best_epoch = epoch
                best_dir = args.output_dir / "adapter_best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                save_adapter(model, tokenizer, best_dir)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        report = {
            "schema": "opensolar_qwen3_reranker_lora_training_report_v2",
            "status": "COMPLETE",
            "created_at": now(),
            "objective": "confidence-weighted graded listwise LambdaRank NDCG@10 over relevance 0-3",
            "training_stage": args.training_stage,
            "optimizer_resumed": False,
            "human_expert_annotation": False,
            "model_facing_core_ids": False,
            "base_model": str(args.model_dir),
            "initial_adapter": (
                {
                    "path": str(args.initial_adapter_dir),
                    "adapter_model_sha256": sha256(
                        args.initial_adapter_dir / "adapter_model.safetensors"
                    ),
                }
                if args.initial_adapter_dir is not None
                else None
            ),
            "train": dataset_summary(train_groups),
            "validation": dataset_summary(validation_groups),
            "world_size": world_size,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "best_epoch": best_epoch,
            "best_ndcg10": best_ndcg,
            "selection_metric": args.selection_metric,
            "best_selection_value": best_selection_value,
            "history": history,
            "outputs": {
                "initial_adapter_snapshot": str(args.output_dir / "adapter_initial"),
                "best_adapter": str(args.output_dir / "adapter_best"),
                "metrics": str(metrics_path),
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        atomic_json(args.output_dir / "training_report.json", report)
        atomic_json(
            status_path,
            {
                "schema": "opensolar_qwen3_reranker_lora_status_v1",
                "state": "COMPLETE",
                "epoch": args.epochs,
                "epochs": args.epochs,
                "optimizer_step": optimizer_step,
                "total_optimizer_steps": total_optimizer_steps,
                "best_epoch": best_epoch,
                "best_ndcg10": best_ndcg,
                "selection_metric": args.selection_metric,
                "best_selection_value": best_selection_value,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at": now(),
                "human_expert_annotation": False,
            },
        )
        print(
            f"\033[36mCOMPLETE best_epoch={best_epoch} "
            f"{args.selection_metric}={best_selection_value:.4f} "
            f"best_nDCG@10={best_ndcg:.4f}\033[0m",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
