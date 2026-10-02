#!/usr/bin/env python3
"""Merge the completed Qwen3.6-27B DAPT adapter into a standalone BF16 model."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    base_model = Path(args.base_model)
    adapter = Path(args.adapter)
    output = Path(args.output)
    report = Path(args.report)
    status = Path(args.status)
    adapter_weight = adapter / "adapter_model.safetensors"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"merge output is not empty: {output}")
    if not adapter_weight.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors missing in {adapter}")

    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    write_json(status, {"state": "MERGE_LOADING", "base_model": str(base_model), "adapter": str(adapter)})
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model), local_files_only=True, trust_remote_code=True
        )
        processor = None
        processor_error = None
        try:
            processor = AutoProcessor.from_pretrained(
                str(base_model), local_files_only=True, trust_remote_code=True
            )
        except Exception as exc:
            processor_error = repr(exc)

        max_memory = {index: "20GiB" for index in range(torch.cuda.device_count())}
        max_memory["cpu"] = "256GiB"
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(base_model),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        device_map = getattr(model, "hf_device_map", None)
        if any(str(device) in {"cpu", "disk"} for device in (device_map or {}).values()):
            raise RuntimeError(f"Unexpected CPU/disk offload during merge: {device_map}")
        write_json(status, {"state": "MERGING", "device_map": device_map})

        peft_model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
        merged = peft_model.merge_and_unload(safe_merge=True)
        write_json(status, {"state": "MERGE_SAVING", "output": str(output)})
        merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="5GB")
        if processor is not None:
            processor.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))

        index_path = output / "model.safetensors.index.json"
        shard_paths = sorted(output.glob("model-*.safetensors"))
        report_data = {
            "status": "DONE",
            "base_model": str(base_model),
            "adapter": str(adapter),
            "adapter_sha256": sha256(adapter_weight),
            "output": str(output),
            "elapsed_sec": round(time.time() - start, 2),
            "torch_dtype": "bfloat16",
            "device_map": device_map,
            "weight_shards": len(shard_paths),
            "weight_bytes": sum(path.stat().st_size for path in shard_paths),
            "model_index_sha256": sha256(index_path) if index_path.is_file() else None,
            "processor_saved": processor is not None,
            "processor_error": processor_error,
        }
        write_json(report, report_data)
        write_json(status, {"state": "MERGE_DONE", **report_data})
        print(json.dumps(report_data, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_json(status, {"state": "MERGE_FAILED", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
