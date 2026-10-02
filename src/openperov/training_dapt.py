#!/usr/bin/env python3
"""Packed text-only DAPT LoRA for Qwen3.6-27B on four 24GB GPUs."""

import argparse
import json
import math
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, set_seed


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def disk_free_gb(path):
    out = subprocess.check_output(["df", "-BG", "--output=avail", str(path)], text=True)
    return int("".join(ch for ch in out.splitlines()[-1] if ch.isdigit()))


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return out.strip().splitlines()
    except Exception as exc:
        return [f"nvidia-smi failed: {exc!r}"]


def checkpoint_step(path):
    match = re.search(r"checkpoint-step-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def prune_checkpoints(output_dir, recent_limit, milestone_every):
    checkpoints = sorted(
        [path for path in output_dir.glob("checkpoint-step-*") if path.is_dir()],
        key=checkpoint_step,
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


def prepare_packed_sequences(tokenizer, train_file, max_length, status_path, run_name):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is required")

    packed = []
    buffer = []
    document_count = 0
    text_token_count = 0
    started = time.time()

    with open(train_file, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            document_count += 1
            text_token_count += len(token_ids)
            buffer.extend(token_ids)
            buffer.append(eos_token_id)

            while len(buffer) >= max_length:
                packed.append((buffer[:max_length], max_length))
                buffer = buffer[max_length:]

            if document_count % 1000 == 0:
                write_json(
                    status_path,
                    {
                        "state": "PACKING",
                        "run_name": run_name,
                        "documents_scanned": document_count,
                        "packed_sequences": len(packed),
                        "elapsed_sec": round(time.time() - started, 2),
                    },
                )

    last_block_tokens = len(buffer)
    if buffer:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("tokenizer.pad_token_id is required")
        packed.append((buffer + [pad_token_id] * (max_length - len(buffer)), len(buffer)))

    total_train_tokens = text_token_count + document_count
    return packed, {
        "source_documents": document_count,
        "text_tokens": text_token_count,
        "document_separator_tokens": document_count,
        "total_train_tokens": total_train_tokens,
        "packed_sequences": len(packed),
        "last_block_valid_tokens": last_block_tokens,
        "max_length": max_length,
        "packing_utilization": total_train_tokens / max(1, len(packed) * max_length),
    }


def packed_batch(sequence, valid_tokens, first_device):
    input_ids = torch.tensor([sequence], dtype=torch.long, device=first_device)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, :valid_tokens] = 1
    labels = input_ids.clone()
    labels[:, valid_tokens:] = -100
    return input_ids, attention_mask, labels


def save_checkpoint(model, tokenizer, output_dir, optimizer_step, micro_step, learning_rate, recent_limit, milestone_every):
    checkpoint_dir = output_dir / f"checkpoint-step-{optimizer_step}"
    model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint_dir))
    write_json(
        checkpoint_dir / "training_state.json",
        {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "learning_rate": learning_rate,
            "resume_supported": False,
        },
    )
    return checkpoint_dir, prune_checkpoints(output_dir, recent_limit, milestone_every)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)

    model_dir = Path(cfg["model_dir"])
    train_file = Path(cfg["train_file"])
    output_dir = Path(cfg["output_dir"])
    report_path = Path(cfg["report_path"])
    metrics_path = Path(cfg["metrics_jsonl"])
    status_path = Path(cfg["status_path"])
    packing_manifest_path = Path(cfg["packing_manifest_path"])
    lora_match_path = Path(cfg["lora_match_path"])
    lora_cfg = load_json(cfg["lora_config"])
    max_length = int(cfg["max_length"])
    grad_accum = int(cfg["gradient_accumulation_steps"])
    learning_rate = float(cfg["learning_rate"])
    warmup_ratio = float(cfg["warmup_ratio"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    max_grad_norm = float(cfg["max_grad_norm"])
    save_every = int(cfg["save_every_optimizer_steps"])
    save_recent_limit = int(cfg["save_recent_limit"])
    milestone_every = int(cfg["milestone_every_optimizer_steps"])
    disk_guard_path = Path(cfg["disk_guard_path"])
    min_free_start = int(cfg["min_free_gb_start"])
    min_free_runtime = int(cfg["min_free_gb_runtime"])
    run_name = cfg["run_name"]

    if disk_free_gb(disk_guard_path) < min_free_start:
        raise SystemExit("Insufficient free disk space for training")
    if not model_dir.exists() or not train_file.exists():
        raise SystemExit("Model or training corpus is missing")

    set_seed(int(cfg["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"state": "TOKENIZER_LOADING", "run_name": run_name})
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_sequences, packing = prepare_packed_sequences(
        tokenizer, train_file, max_length, status_path, run_name
    )
    total_micro_steps = len(packed_sequences)
    total_steps = math.ceil(total_micro_steps / grad_accum)
    packing.update(
        {
            "run_name": run_name,
            "gradient_accumulation_steps": grad_accum,
            "optimizer_steps": total_steps,
            "optimizer_tokens_per_step": max_length * grad_accum,
            "train_file": str(train_file),
        }
    )
    write_json(packing_manifest_path, packing)
    print(json.dumps({"event": "packing_complete", **packing}, ensure_ascii=False), flush=True)

    write_json(
        status_path,
        {
            "state": "MODEL_LOADING",
            "run_name": run_name,
            "source_documents": packing["source_documents"],
            "packed_sequences": total_micro_steps,
            "target_optimizer_steps": total_steps,
        },
    )
    max_memory = {
        index: f"{int(cfg['max_memory_per_gpu_gib'])}GiB"
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = f"{int(cfg['max_memory_cpu_gib'])}GiB"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=cfg.get("device_map", "balanced"),
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    offload_targets = {str(value) for value in getattr(model, "hf_device_map", {}).values()} & {"cpu", "disk"}
    if offload_targets:
        raise RuntimeError(f"Unexpected model offload targets: {sorted(offload_targets)}")

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(**lora_cfg))
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise RuntimeError("LoRA matched no trainable parameters")
    forbidden = [name for name in trainable_names if ".visual." in name or ".mtp." in name]
    if forbidden:
        raise RuntimeError(f"LoRA entered frozen visual/MTP branches: {forbidden[:10]}")
    match_summary = {
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "linear_attention_tensors": sum("linear_attn" in name for name in trainable_names),
        "full_attention_tensors": sum("self_attn" in name for name in trainable_names),
        "mlp_tensors": sum(".mlp." in name for name in trainable_names),
        "visual_or_mtp_tensors": len(forbidden),
        "sample_names": trainable_names[:30],
    }
    write_json(lora_match_path, match_summary)
    model.train()
    model.print_trainable_parameters()

    first_device = model.get_input_embeddings().weight.device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup_steps, total_steps)
    )
    optimizer.zero_grad(set_to_none=True)
    write_json(
        status_path,
        {
            "state": "RUNNING",
            "run_name": run_name,
            "optimizer_step": 0,
            "target_optimizer_steps": total_steps,
            "micro_step": 0,
            "target_micro_steps": total_micro_steps,
            "first_device": str(first_device),
            "lora_match": match_summary,
        },
    )
    print(
        json.dumps(
            {"event": "model_loaded", "first_device": str(first_device), "lora": match_summary, "gpu": gpu_snapshot()},
            ensure_ascii=False,
        ),
        flush=True,
    )

    micro_step = 0
    optimizer_step = 0
    losses = []
    valid_token_counts = []
    started = time.time()
    status = "FAILED"
    error = None

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for sequence, valid_tokens in packed_sequences:
                input_ids, attention_mask, labels = packed_batch(sequence, valid_tokens, first_device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                loss = outputs.loss
                (loss / grad_accum).backward()
                losses.append(float(loss.detach().cpu()))
                valid_token_counts.append(int((labels[:, 1:] != -100).sum().item()))
                micro_step += 1

                if micro_step % grad_accum:
                    continue

                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).detach().cpu())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                elapsed = time.time() - started
                free_gb = disk_free_gb(disk_guard_path)
                record = {
                    "optimizer_step": optimizer_step,
                    "target_optimizer_steps": total_steps,
                    "micro_step": micro_step,
                    "target_micro_steps": total_micro_steps,
                    "loss": losses[-1],
                    "mean_loss_recent": sum(losses[-grad_accum:]) / grad_accum,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "valid_tokens": valid_token_counts[-1],
                    "mean_valid_tokens_recent": sum(valid_token_counts[-grad_accum:]) / grad_accum,
                    "elapsed_sec": round(elapsed, 2),
                    "optimizer_steps_per_sec": round(optimizer_step / elapsed, 6) if elapsed else None,
                    "disk_free_gb": free_gb,
                    "max_memory_allocated_bytes": {
                        f"cuda:{index}": int(torch.cuda.max_memory_allocated(index))
                        for index in range(torch.cuda.device_count())
                    },
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                metrics_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_handle.flush()
                write_json(status_path, {"state": "RUNNING", "run_name": run_name, **record})

                if free_gb < min_free_runtime:
                    raise RuntimeError(f"Free disk below runtime guard: {free_gb}G")
                if save_every > 0 and optimizer_step % save_every == 0:
                    checkpoint_dir, removed = save_checkpoint(
                        model,
                        tokenizer,
                        output_dir,
                        optimizer_step,
                        micro_step,
                        scheduler.get_last_lr()[0],
                        save_recent_limit,
                        milestone_every,
                    )
                    print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint_dir), "pruned": removed}), flush=True)

            remainder = micro_step % grad_accum
            if remainder:
                correction = grad_accum / remainder
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
        status = "DONE"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        error = "KeyboardInterrupt"
        raise
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        elapsed = time.time() - started
        summary = {
            "status": status,
            "error": error,
            "run_name": run_name,
            "route": "qwen36_27b_text_only_packed1024_bf16_lora_r16_device_map_balanced",
            "packing": packing,
            "optimizer_steps": optimizer_step,
            "target_optimizer_steps": total_steps,
            "micro_steps": micro_step,
            "target_micro_steps": total_micro_steps,
            "gradient_accumulation_steps": grad_accum,
            "elapsed_sec": round(elapsed, 2),
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "last_loss": losses[-1] if losses else None,
            "metrics_jsonl": str(metrics_path),
            "disk_free_gb_final": disk_free_gb(disk_guard_path),
            "lora_match": match_summary,
        }
        write_json(report_path, summary)
        write_json(status_path, {"state": status, **summary})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            cli = argparse.ArgumentParser(add_help=False)
            cli.add_argument("--config", required=True)
            known, _ = cli.parse_known_args()
            failed_cfg = load_json(known.config)
            failed_status = Path(failed_cfg["status_path"])
            current = {}
            if failed_status.exists():
                try:
                    current = load_json(failed_status)
                except (OSError, json.JSONDecodeError):
                    current = {}
            current.update(state="FAILED", error=repr(exc), failed_at=time.time())
            write_json(failed_status, current)
        except Exception:
            pass
        traceback.print_exc()
        raise
