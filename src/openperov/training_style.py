#!/usr/bin/env python3
"""Assistant-loss SFT for the Qwen3.6-27B validated answer-style stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, set_seed


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_free_gb(path: Path) -> int:
    return int(shutil.disk_usage(path).free / (1024**3))


def gpu_snapshot() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return output.strip().splitlines()
    except Exception as exc:  # pragma: no cover
        return [f"nvidia-smi failed: {exc!r}"]


def learning_rate_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def apply_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_rows(
    tokenizer,
    path: Path,
    max_sample_tokens: int,
    status_path: Path,
    run_name: str,
    split: str,
) -> tuple[list[tuple[list[int], list[int]]], dict]:
    encoded: list[tuple[list[int], list[int]]] = []
    task_counts: dict[str, int] = {}
    total_tokens = 0
    assistant_tokens = 0
    maximum = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise RuntimeError(f"{split} row {line_number}: expected exactly three messages")
            if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
                raise RuntimeError(f"{split} row {line_number}: unexpected message roles")
            prefix = apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
            full = apply_template(tokenizer, messages, add_generation_prompt=False)
            if full[: len(prefix)] != prefix:
                raise RuntimeError(f"{split} row {line_number}: assistant prefix mismatch")
            if len(full) > max_sample_tokens:
                raise RuntimeError(
                    f"{split} row {line_number}: {len(full)} tokens exceed {max_sample_tokens}"
                )
            labels = [-100] * len(prefix) + full[len(prefix) :]
            label_count = sum(value != -100 for value in labels)
            if label_count < 4:
                raise RuntimeError(f"{split} row {line_number}: assistant target is empty")
            encoded.append((full, labels))
            total_tokens += len(full)
            assistant_tokens += label_count
            maximum = max(maximum, len(full))
            task = str((row.get("metadata") or {}).get("task_type") or "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
            if len(encoded) % 500 == 0:
                write_json(
                    status_path,
                    {
                        "state": "TOKENIZING_SFT",
                        "run_name": run_name,
                        "split": split,
                        "rows": len(encoded),
                        "source_line": line_number,
                    },
                )
    return encoded, {
        "rows": len(encoded),
        "total_tokens": total_tokens,
        "assistant_target_tokens": assistant_tokens,
        "max_sample_tokens": maximum,
        "task_counts": dict(sorted(task_counts.items())),
        "path": str(path),
        "sha256": sha256_file(path),
    }


def pack_rows(
    rows: list[tuple[list[int], list[int]]],
    block_length: int,
    pad_id: int,
    seed: int,
) -> tuple[list[tuple[list[int], list[int], int]], dict]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    blocks: list[tuple[list[int], list[int], int]] = []
    ids_buffer: list[int] = []
    labels_buffer: list[int] = []
    samples_in_buffer = 0
    samples_per_block: list[int] = []
    for token_ids, labels in shuffled:
        if len(token_ids) > block_length:
            raise RuntimeError(f"sample length {len(token_ids)} exceeds block length {block_length}")
        if ids_buffer and len(ids_buffer) + len(token_ids) > block_length:
            valid = len(ids_buffer)
            ids_buffer.extend([pad_id] * (block_length - valid))
            labels_buffer.extend([-100] * (block_length - valid))
            blocks.append((ids_buffer, labels_buffer, valid))
            samples_per_block.append(samples_in_buffer)
            ids_buffer, labels_buffer, samples_in_buffer = [], [], 0
        ids_buffer.extend(token_ids)
        labels_buffer.extend(labels)
        samples_in_buffer += 1
    if ids_buffer:
        valid = len(ids_buffer)
        ids_buffer.extend([pad_id] * (block_length - valid))
        labels_buffer.extend([-100] * (block_length - valid))
        blocks.append((ids_buffer, labels_buffer, valid))
        samples_per_block.append(samples_in_buffer)

    valid_tokens = sum(block[2] for block in blocks)
    target_tokens = sum(sum(value != -100 for value in block[1]) for block in blocks)
    return blocks, {
        "packed_blocks": len(blocks),
        "block_length": block_length,
        "valid_tokens": valid_tokens,
        "assistant_target_tokens": target_tokens,
        "packing_utilization": valid_tokens / max(1, len(blocks) * block_length),
        "mean_samples_per_block": sum(samples_per_block) / max(1, len(samples_per_block)),
        "shuffle_seed": seed,
    }


def make_batch(block, device: torch.device):
    token_ids, labels, valid_tokens = block
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, :valid_tokens] = 1
    return input_ids, attention_mask, label_tensor


def evaluate(model, blocks, first_device: torch.device, max_blocks: int) -> dict:
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    used = min(max_blocks, len(blocks))
    with torch.no_grad():
        for block in blocks[:used]:
            input_ids, attention_mask, labels = make_batch(block, first_device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            count = int((labels[:, 1:] != -100).sum().item())
            weighted_loss += float(output.loss.detach().cpu()) * count
            target_tokens += count
    model.train()
    mean_loss = weighted_loss / max(1, target_tokens)
    return {
        "blocks": used,
        "assistant_target_tokens": target_tokens,
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


def save_adapter(model, tokenizer, path: Path, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    write_json(path / "training_state.json", state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_json(args.config)

    run_name = str(cfg["run_name"])
    model_dir = Path(cfg["model_dir"])
    initial_adapter_value = cfg.get("initial_adapter_dir")
    initial_adapter_dir = Path(initial_adapter_value) if initial_adapter_value else None
    lora_config_value = cfg.get("lora_config")
    lora_config_path = Path(lora_config_value) if lora_config_value else None
    train_file = Path(cfg["train_file"])
    validation_file = Path(cfg["validation_file"])
    output_adapter_dir = Path(cfg["output_adapter_dir"])
    checkpoint_root = Path(cfg["checkpoint_root"])
    status_path = Path(cfg["status_path"])
    metrics_path = Path(cfg["metrics_jsonl"])
    report_path = Path(cfg["report_path"])
    packing_path = Path(cfg["packing_manifest_path"])
    disk_guard_path = Path(cfg["disk_guard_path"])
    block_length = int(cfg["block_length"])
    max_sample_tokens = int(cfg["max_sample_tokens"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    learning_rate = float(cfg["learning_rate"])
    warmup_ratio = float(cfg["warmup_ratio"])
    max_grad_norm = float(cfg["max_grad_norm"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    save_every = int(cfg["save_every_optimizer_steps"])
    eval_max_blocks = int(cfg["eval_max_blocks"])
    seed = int(cfg["seed"])

    if (initial_adapter_dir is None) == (lora_config_path is None):
        raise SystemExit("configure exactly one of initial_adapter_dir or lora_config")
    required = [model_dir, train_file, validation_file]
    required.append(initial_adapter_dir if initial_adapter_dir is not None else lora_config_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required inputs: " + ", ".join(missing))
    if disk_free_gb(disk_guard_path) < int(cfg["min_free_gb_start"]):
        raise SystemExit("insufficient free disk at launch")

    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(status_path, {"state": "TOKENIZER_LOADING", "run_name": run_name})
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, train_summary = encode_rows(
        tokenizer, train_file, max_sample_tokens, status_path, run_name, "train"
    )
    validation_rows, validation_summary = encode_rows(
        tokenizer, validation_file, max_sample_tokens, status_path, run_name, "validation"
    )
    train_blocks, train_packing = pack_rows(
        train_rows, block_length, tokenizer.pad_token_id, seed
    )
    validation_blocks, validation_packing = pack_rows(
        validation_rows, block_length, tokenizer.pad_token_id, seed + 1
    )
    total_micro_steps = len(train_blocks)
    total_optimizer_steps = math.ceil(total_micro_steps / grad_accum)
    packing = {
        "train": {**train_summary, **train_packing},
        "validation": {**validation_summary, **validation_packing},
        "gradient_accumulation_steps": grad_accum,
        "optimizer_steps": total_optimizer_steps,
    }
    write_json(packing_path, packing)
    print(json.dumps({"event": "packing_complete", **packing}, ensure_ascii=False), flush=True)

    write_json(
        status_path,
        {
            "state": "MODEL_LOADING",
            "run_name": run_name,
            "target_optimizer_steps": total_optimizer_steps,
            "packing": packing,
        },
    )
    max_memory = {
        index: f"{int(cfg['max_memory_per_gpu_gib'])}GiB"
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = f"{int(cfg['max_memory_cpu_gib'])}GiB"
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=cfg.get("device_map", "balanced"),
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    offload = {str(value) for value in getattr(base_model, "hf_device_map", {}).values()} & {"cpu", "disk"}
    if offload:
        raise RuntimeError(f"unexpected model offload: {sorted(offload)}")
    base_model.config.use_cache = False
    if hasattr(base_model, "language_model"):
        base_model.language_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()
    if initial_adapter_dir is not None:
        model = PeftModel.from_pretrained(
            base_model,
            str(initial_adapter_dir),
            is_trainable=True,
            local_files_only=True,
        )
        adapter_origin = {"initial_adapter": str(initial_adapter_dir), "lora_config": None}
    else:
        lora_config = LoraConfig(**load_json(lora_config_path))
        model = get_peft_model(base_model, lora_config)
        adapter_origin = {"initial_adapter": None, "lora_config": str(lora_config_path)}
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
    model.train()
    first_device = model.get_input_embeddings().weight.device
    baseline_validation = evaluate(model, validation_blocks, first_device, eval_max_blocks)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, warmup_steps, total_optimizer_steps),
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    write_json(
        status_path,
        {
            "state": "RUNNING",
            "run_name": run_name,
            "optimizer_step": 0,
            "target_optimizer_steps": total_optimizer_steps,
            "micro_step": 0,
            "target_micro_steps": total_micro_steps,
            "first_device": str(first_device),
            "adapter_match": match,
            "baseline_validation": baseline_validation,
            "gpu": gpu_snapshot(),
        },
    )
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "first_device": str(first_device),
                "adapter": match,
                "baseline_validation": baseline_validation,
                "gpu": gpu_snapshot(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    losses: list[float] = []
    micro_step = 0
    optimizer_step = 0
    started = time.time()
    final_state = "FAILED"
    error = None
    final_validation = None
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for block in train_blocks:
                input_ids, attention_mask, labels = make_batch(block, first_device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss = output.loss
                (loss / grad_accum).backward()
                losses.append(float(loss.detach().cpu()))
                micro_step += 1
                if micro_step % grad_accum and micro_step != total_micro_steps:
                    continue

                remainder = micro_step % grad_accum
                if remainder:
                    correction = grad_accum / remainder
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).detach().cpu()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                elapsed = time.time() - started
                free_gb = disk_free_gb(disk_guard_path)
                label_tokens = int((labels[:, 1:] != -100).sum().item())
                record = {
                    "optimizer_step": optimizer_step,
                    "target_optimizer_steps": total_optimizer_steps,
                    "micro_step": micro_step,
                    "target_micro_steps": total_micro_steps,
                    "loss": losses[-1],
                    "mean_loss_recent": sum(losses[-grad_accum:]) / min(grad_accum, len(losses)),
                    "assistant_target_tokens_last_block": label_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "elapsed_sec": round(elapsed, 2),
                    "steps_per_sec": round(optimizer_step / elapsed, 6),
                    "eta_sec": round(
                        (total_optimizer_steps - optimizer_step) * elapsed / max(1, optimizer_step), 2
                    ),
                    "disk_free_gb": free_gb,
                    "max_memory_allocated_bytes": {
                        f"cuda:{index}": int(torch.cuda.max_memory_allocated(index))
                        for index in range(torch.cuda.device_count())
                    },
                }
                metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                write_json(status_path, {"state": "RUNNING", "run_name": run_name, **record})
                print(json.dumps(record, ensure_ascii=False), flush=True)

                if free_gb < int(cfg["min_free_gb_runtime"]):
                    raise RuntimeError(f"runtime disk guard triggered: {free_gb} GiB")
                if save_every and optimizer_step % save_every == 0:
                    checkpoint = checkpoint_root / f"checkpoint-step-{optimizer_step}"
                    save_adapter(
                        model,
                        tokenizer,
                        checkpoint,
                        {
                            "optimizer_step": optimizer_step,
                            "micro_step": micro_step,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "resume_supported": False,
                        },
                    )
                    print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint)}), flush=True)

        final_validation = evaluate(model, validation_blocks, first_device, eval_max_blocks)
        save_adapter(
            model,
            tokenizer,
            output_adapter_dir,
            {
                "status": "DONE",
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                **adapter_origin,
                "base_model": str(model_dir),
                "baseline_validation": baseline_validation,
                "final_validation": final_validation,
            },
        )
        final_state = "DONE"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        elapsed = time.time() - started
        report = {
            "status": final_state,
            "error": error,
            "run_name": run_name,
            "base_model": str(model_dir),
            **adapter_origin,
            "output_adapter": str(output_adapter_dir),
            "packing": packing,
            "adapter_match": match,
            "optimizer_steps": optimizer_step,
            "target_optimizer_steps": total_optimizer_steps,
            "micro_steps": micro_step,
            "target_micro_steps": total_micro_steps,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "last_loss": losses[-1] if losses else None,
            "baseline_validation": baseline_validation,
            "final_validation": final_validation,
            "elapsed_sec": round(elapsed, 2),
            "disk_free_gb": disk_free_gb(disk_guard_path),
            "gpu": gpu_snapshot(),
        }
        write_json(report_path, report)
        write_json(status_path, {"state": final_state, **report})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
