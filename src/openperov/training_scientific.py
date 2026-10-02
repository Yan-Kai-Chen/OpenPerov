#!/usr/bin/env python3
"""Weighted assistant-only packed SFT for Qwen3.6-27B PVK Stage 3."""

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed


@dataclass
class PackedBlock:
    input_ids: list[int]
    labels: list[int]
    token_weights: list[float]
    valid_length: int
    valid_assistant_tokens: int
    sample_weight_sum: float
    sample_count: int


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def disk_free_gb(path):
    output = subprocess.check_output(["df", "-BG", "--output=avail", str(path)], text=True)
    return int("".join(char for char in output.splitlines()[-1] if char.isdigit()))


def gpu_snapshot():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return output.strip().splitlines()
    except Exception as exc:
        return [f"nvidia-smi failed: {exc!r}"]


def checkpoint_step(path):
    match = re.search(r"checkpoint-step-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def prune_checkpoints(output_dir, recent_limit, milestone_every):
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-step-*") if path.is_dir()], key=checkpoint_step
    )
    protected = set(checkpoints[-recent_limit:]) if recent_limit > 0 else set()
    if milestone_every > 0:
        protected.update(path for path in checkpoints if checkpoint_step(path) % milestone_every == 0)
    removed = []
    for path in checkpoints:
        if path not in protected:
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def lr_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def make_example(tokenizer, row, max_length, min_prefix_tokens, stats):
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        stats["bad_messages"] += 1
        return None
    sample_weight = float(row.get("sample_weight", 1.0))
    if not math.isfinite(sample_weight) or sample_weight <= 0:
        raise RuntimeError(f"invalid sample_weight: {sample_weight}")
    try:
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        stats["template_enable_thinking_false"] += 1
    except TypeError as exc:
        raise RuntimeError("Qwen3 tokenizer must explicitly support enable_thinking=False") from exc

    eos = tokenizer.eos_token
    if not eos:
        raise RuntimeError("tokenizer.eos_token is required")
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(str(messages[-1].get("content") or "") + eos, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        stats["zero_answer_tokens"] += 1
        return None

    original_prefix = len(prefix_ids)
    original_answer = len(answer_ids)
    max_answer = max_length - min_prefix_tokens
    if len(answer_ids) > max_answer:
        answer_ids = answer_ids[:max_answer]
        stats["answer_truncated"] += 1
    prefix_capacity = max_length - len(answer_ids)
    if len(prefix_ids) > prefix_capacity:
        prefix_ids = prefix_ids[-prefix_capacity:]
        stats["prefix_left_truncated"] += 1
    if not prefix_ids or not answer_ids:
        raise RuntimeError("prefix/answer truncation invariant failed")

    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    per_token_weight = sample_weight / len(answer_ids)
    token_weights = [0.0] * len(prefix_ids) + [per_token_weight] * len(answer_ids)
    if len(input_ids) != len(labels) or len(labels) != len(token_weights):
        raise RuntimeError("weighted example lengths do not agree")
    if abs(sum(token_weights) - sample_weight) > 1e-6:
        raise RuntimeError("token weights do not sum to sample_weight")
    stats["examples"] += 1
    stats["original_prefix_tokens"] += original_prefix
    stats["original_answer_tokens"] += original_answer
    stats["kept_assistant_tokens"] += len(answer_ids)
    stats["effective_sample_weight"] += sample_weight
    stats["sample_weight_counts"][str(sample_weight)] += 1
    return input_ids, labels, token_weights, sample_weight


def make_block(ids, labels, weights, sample_weight_sum, sample_count):
    return PackedBlock(
        input_ids=ids,
        labels=labels,
        token_weights=weights,
        valid_length=len(ids),
        valid_assistant_tokens=sum(label != -100 for label in labels),
        sample_weight_sum=sample_weight_sum,
        sample_count=sample_count,
    )


def build_packed_blocks(tokenizer, train_file, max_length, min_prefix_tokens, seed, expected_rows=None):
    examples = []
    stats = {
        "examples": 0,
        "bad_messages": 0,
        "zero_answer_tokens": 0,
        "answer_truncated": 0,
        "prefix_left_truncated": 0,
        "template_enable_thinking_false": 0,
        "original_prefix_tokens": 0,
        "original_answer_tokens": 0,
        "kept_assistant_tokens": 0,
        "effective_sample_weight": 0.0,
        "sample_weight_counts": Counter(),
    }
    with open(train_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                example = make_example(tokenizer, json.loads(line), max_length, min_prefix_tokens, stats)
                if example is not None:
                    examples.append(example)
    if expected_rows is not None and len(examples) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} usable rows, got {len(examples)}")
    random.Random(seed).shuffle(examples)

    blocks = []
    buffer_ids, buffer_labels, buffer_weights = [], [], []
    buffer_sample_weight, buffer_sample_count = 0.0, 0
    for input_ids, labels, token_weights, sample_weight in examples:
        if len(input_ids) > max_length:
            raise RuntimeError("truncation invariant failed")
        if buffer_ids and len(buffer_ids) + len(input_ids) > max_length:
            blocks.append(make_block(
                buffer_ids, buffer_labels, buffer_weights, buffer_sample_weight, buffer_sample_count
            ))
            buffer_ids, buffer_labels, buffer_weights = [], [], []
            buffer_sample_weight, buffer_sample_count = 0.0, 0
        buffer_ids.extend(input_ids)
        buffer_labels.extend(labels)
        buffer_weights.extend(token_weights)
        buffer_sample_weight += sample_weight
        buffer_sample_count += 1
        if len(buffer_ids) == max_length:
            blocks.append(make_block(
                buffer_ids, buffer_labels, buffer_weights, buffer_sample_weight, buffer_sample_count
            ))
            buffer_ids, buffer_labels, buffer_weights = [], [], []
            buffer_sample_weight, buffer_sample_count = 0.0, 0
    if buffer_ids:
        block = make_block(buffer_ids, buffer_labels, buffer_weights, buffer_sample_weight, buffer_sample_count)
        pad_count = max_length - block.valid_length
        block.input_ids.extend([tokenizer.pad_token_id] * pad_count)
        block.labels.extend([-100] * pad_count)
        block.token_weights.extend([0.0] * pad_count)
        blocks.append(block)
    if not blocks or any(block.valid_assistant_tokens <= 0 or block.sample_weight_sum <= 0 for block in blocks):
        raise RuntimeError("packed blocks contain an invalid assistant-label or weight sum")
    packed_weight = sum(block.sample_weight_sum for block in blocks)
    if abs(packed_weight - stats["effective_sample_weight"]) > 1e-5:
        raise RuntimeError("packed sample weights changed during packing")
    stats["sample_weight_counts"] = dict(sorted(stats["sample_weight_counts"].items()))
    stats.update({
        "packed_blocks": len(blocks),
        "packed_token_utilization": sum(block.valid_length for block in blocks) / (len(blocks) * max_length),
        "mean_assistant_tokens_per_block": sum(block.valid_assistant_tokens for block in blocks) / len(blocks),
        "mean_samples_per_block": sum(block.sample_count for block in blocks) / len(blocks),
        "packed_effective_sample_weight": packed_weight,
    })
    return blocks, stats


def tensor_batch(blocks, pad_token_id, device):
    batch_length = max(block.valid_length for block in blocks)
    input_ids = torch.tensor(
        [block.input_ids[:block.valid_length] + [pad_token_id] * (batch_length - block.valid_length) for block in blocks],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        [block.labels[:block.valid_length] + [-100] * (batch_length - block.valid_length) for block in blocks],
        dtype=torch.long,
        device=device,
    )
    token_weights = torch.tensor(
        [block.token_weights[:block.valid_length] + [0.0] * (batch_length - block.valid_length) for block in blocks],
        dtype=torch.float32,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, block in enumerate(blocks):
        attention_mask[row_index, :block.valid_length] = 1
    return input_ids, attention_mask, labels, token_weights


def save_checkpoint(model, tokenizer, output_dir, optimizer_step, micro_step, learning_rate, recent_limit, milestone_every):
    checkpoint_dir = output_dir / f"checkpoint-step-{optimizer_step}"
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    state = {
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "learning_rate": learning_rate,
        "note": "Adapter/tokenizer checkpoint for comparison; this fresh stable-server run does not enable resume.",
    }
    (checkpoint_dir / "TRAINING_STATE_STAGE3_WEIGHTED_PACKED_V1.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return checkpoint_dir, prune_checkpoints(output_dir, recent_limit, milestone_every)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(args.config)

    model_dir = Path(cfg["model_dir"])
    train_file = Path(cfg["train_file"])
    output_dir = Path(cfg["output_dir"])
    report_path = Path(cfg["report_path"])
    metrics_path = Path(cfg["metrics_jsonl"])
    status_path = Path(cfg["status_path"])
    packing_manifest_path = Path(cfg["packing_manifest_path"])
    lora_cfg = load_json(cfg["lora_config"])
    max_length = int(cfg["max_length"])
    min_prefix_tokens = int(cfg.get("min_prefix_tokens", 96))
    physical_batch_size = int(cfg.get("physical_batch_size", 1))
    grad_accum = int(cfg["gradient_accumulation_steps"])
    blocks_per_update = physical_batch_size * grad_accum
    if physical_batch_size <= 0 or grad_accum <= 0:
        raise ValueError("physical_batch_size and gradient_accumulation_steps must be positive")
    learning_rate = float(cfg["learning_rate"])
    warmup_ratio = float(cfg["warmup_ratio"])
    max_grad_norm = float(cfg["max_grad_norm"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    save_every = int(cfg["save_every_optimizer_steps"])
    recent_limit = int(cfg.get("save_recent_limit", 2))
    milestone_every = int(cfg.get("milestone_every_optimizer_steps", 1000))
    expected_rows = (int(cfg["expected_train_rows"]) if cfg.get("expected_train_rows") is not None else None)
    expected_effective_weight = cfg.get("expected_effective_sample_weight")
    min_free_start = int(cfg.get("min_free_gb_start", 80))
    min_free_runtime = int(cfg.get("min_free_gb_runtime", 50))

    disk_guard_path = Path(cfg.get("disk_guard_path", "."))
    free_gb = disk_free_gb(disk_guard_path)
    if free_gb < min_free_start:
        raise RuntimeError(f"shared storage free space too low: {free_gb}G < {min_free_start}G")
    if not model_dir.is_dir() or not train_file.is_file():
        raise FileNotFoundError(f"model/data missing: {model_dir}, {train_file}")
    set_seed(int(cfg.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks, packing = build_packed_blocks(
        tokenizer, train_file, max_length, min_prefix_tokens, int(cfg.get("seed", 42)), expected_rows
    )
    if expected_effective_weight is not None and abs(packing["effective_sample_weight"] - float(expected_effective_weight)) > 1e-5:
        raise RuntimeError(f"effective weight mismatch: {packing['effective_sample_weight']}")
    full_total_steps = math.ceil(len(blocks) / blocks_per_update)
    max_optimizer_steps = int(cfg.get("max_optimizer_steps", 0))
    training_blocks = blocks
    if max_optimizer_steps > 0:
        training_blocks = blocks[:max_optimizer_steps * blocks_per_update]
    total_steps = math.ceil(len(training_blocks) / blocks_per_update)
    packing.update({
        "run_name": cfg["run_name"],
        "train_file": str(train_file),
        "model_dir": str(model_dir),
        "max_length": max_length,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "blocks_per_update": blocks_per_update,
        "optimizer_steps": total_steps,
        "full_optimizer_steps": full_total_steps,
        "max_optimizer_steps": max_optimizer_steps or None,
        "optimizer_tokens_per_step": max_length * blocks_per_update,
        "assistant_only_loss": True,
        "weighted_loss": True,
        "enable_thinking": False,
    })
    packing_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    packing_manifest_path.write_text(json.dumps(packing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "packing_complete", **packing}, ensure_ascii=False), flush=True)
    if args.pack_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError(f"output_dir must be empty for a fresh run: {output_dir}")
    write_json(
        status_path,
        {
            "state": "MODEL_LOADING",
            "run_name": cfg["run_name"],
            "target_optimizer_steps": total_steps,
            "packing": packing,
        },
    )

    max_memory = {
        index: f"{int(cfg.get('max_memory_per_gpu_gib', 24))}GiB"
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = f"{int(cfg.get('max_memory_cpu_gib', 256))}GiB"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=cfg.get("device_map", "balanced"),
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    offload = {str(value) for value in getattr(model, "hf_device_map", {}).values()} & {"cpu", "disk"}
    if offload:
        raise RuntimeError(f"unexpected model offload: {sorted(offload)}")
    model.config.use_cache = False
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**lora_cfg))
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    forbidden = [name for name in trainable_names if ".visual." in name or ".mtp." in name]
    if not trainable or forbidden:
        raise RuntimeError(
            f"invalid adapter match: trainable={len(trainable_names)}, forbidden={forbidden[:5]}"
        )
    match = {
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "linear_attention_tensors": sum("linear_attn" in name for name in trainable_names),
        "full_attention_tensors": sum("self_attn" in name for name in trainable_names),
        "mlp_tensors": sum(".mlp." in name for name in trainable_names),
        "visual_or_mtp_tensors": len(forbidden),
    }
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup_steps, total_steps)
    )
    optimizer.zero_grad(set_to_none=True)
    first_device = model.get_input_embeddings().weight.device
    write_json(
        status_path,
        {
            "state": "RUNNING",
            "run_name": cfg["run_name"],
            "optimizer_step": 0,
            "target_optimizer_steps": total_steps,
            "micro_step": 0,
            "target_micro_steps": len(training_blocks),
            "adapter_match": match,
            "gpu": gpu_snapshot(),
        },
    )
    print(
        json.dumps(
            {"event": "model_loaded", "first_device": str(first_device), "adapter": match, "gpu": gpu_snapshot()},
            ensure_ascii=False,
        ),
        flush=True,
    )

    start = time.time()
    optimizer_step = 0
    raw_losses, weighted_losses = [], []
    status, error = "FAILED", None
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for group_start in range(0, len(training_blocks), blocks_per_update):
                group_end = min(group_start + blocks_per_update, len(training_blocks))
                group = training_blocks[group_start:group_end]
                group_weight_sum = sum(candidate.sample_weight_sum for candidate in group)
                group_valid_tokens = sum(candidate.valid_assistant_tokens for candidate in group)
                group_weighted_numerator = 0.0
                group_raw_numerator = 0.0
                group_raw_tokens = 0
                forward_passes = 0
                for chunk_start in range(0, len(group), physical_batch_size):
                    chunk = group[chunk_start:chunk_start + physical_batch_size]
                    input_ids, attention_mask, labels, token_weights = tensor_batch(
                        chunk, tokenizer.pad_token_id, first_device
                    )
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].to(logits.device).contiguous()
                    shift_weights = token_weights[:, 1:].to(logits.device).contiguous()
                    valid_mask = shift_labels != -100
                    valid_tokens = int(valid_mask.sum().item())
                    if valid_tokens <= 0 or float(shift_weights.sum().item()) <= 0:
                        raise RuntimeError("zero assistant labels or weights after shift")
                    token_losses = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100,
                        reduction="none"
                    ).view_as(shift_labels)
                    weighted_numerator = (token_losses * shift_weights).sum()
                    (weighted_numerator / group_weight_sum).backward()
                    group_weighted_numerator += float(weighted_numerator.detach().cpu())
                    group_raw_numerator += float(token_losses[valid_mask].sum().detach().cpu())
                    group_raw_tokens += valid_tokens
                    forward_passes += 1

                raw_update_loss = group_raw_numerator / group_raw_tokens
                weighted_update_loss = group_weighted_numerator / group_weight_sum
                raw_losses.append(raw_update_loss)
                weighted_losses.append(weighted_update_loss)
                micro_step = group_end
                grad_norm_before = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).detach().cpu())
                grad_norm_after = min(grad_norm_before, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                elapsed = time.time() - start
                free_runtime = disk_free_gb(disk_guard_path)
                record = {
                    "optimizer_step": optimizer_step,
                    "target_optimizer_steps": total_steps,
                    "micro_step": micro_step,
                    "target_micro_steps": len(training_blocks),
                    "raw_loss": raw_update_loss,
                    "mean_raw_loss_recent": sum(raw_losses[-20:]) / min(20, len(raw_losses)),
                    "weighted_loss": weighted_update_loss,
                    "mean_weighted_loss_recent": sum(weighted_losses[-20:]) / min(20, len(weighted_losses)),
                    "physical_batch_size": physical_batch_size,
                    "forward_passes_in_update": forward_passes,
                    "blocks_in_update": len(group),
                    "sum_weight_in_update": group_weight_sum,
                    "samples_in_update": sum(candidate.sample_count for candidate in group),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm_before_clip": grad_norm_before,
                    "grad_norm_after_clip": grad_norm_after,
                    "valid_assistant_tokens": group_raw_tokens,
                    "group_valid_assistant_tokens": group_valid_tokens,
                    "elapsed_sec": round(elapsed, 2),
                    "optimizer_steps_per_sec": round(optimizer_step / elapsed, 6) if elapsed else None,
                    "shared_free_gb": free_runtime,
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                write_json(status_path, {"state": "RUNNING", "run_name": cfg["run_name"], **record})
                if free_runtime < min_free_runtime:
                    raise RuntimeError(f"shared storage below runtime guard: {free_runtime}G")
                if save_every > 0 and optimizer_step % save_every == 0:
                    checkpoint_dir, removed = save_checkpoint(
                        model, tokenizer, output_dir, optimizer_step, micro_step,
                        scheduler.get_last_lr()[0], recent_limit, milestone_every,
                    )
                    print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint_dir), "pruned": removed}, ensure_ascii=False), flush=True)

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        status = "DONE"
    except KeyboardInterrupt:
        status, error = "INTERRUPTED", "KeyboardInterrupt"
        raise
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        report = {
            "status": status,
            "error": error,
            "run_name": cfg["run_name"],
            "route": "qwen36_27b_stage3_targeted_multiview_fiveanswer_weighted_pack1024_lora_r16",
            "model_dir": str(model_dir),
            "output_adapter": str(output_dir),
            "packing": packing,
            "optimizer_steps": optimizer_step,
            "target_optimizer_steps": total_steps,
            "elapsed_sec": round(time.time() - start, 2),
            "mean_raw_loss": sum(raw_losses) / len(raw_losses) if raw_losses else None,
            "mean_weighted_loss": sum(weighted_losses) / len(weighted_losses) if weighted_losses else None,
            "last_raw_loss": raw_losses[-1] if raw_losses else None,
            "last_weighted_loss": weighted_losses[-1] if weighted_losses else None,
            "adapter_match": match,
            "shared_free_gb_final": disk_free_gb(disk_guard_path),
        }
        write_json(report_path, report)
        write_json(status_path, {"state": status, **report})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
