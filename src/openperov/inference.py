"""Explicit local-model or OpenAI-compatible endpoint inference.

Importing this module never loads weights or contacts a service.
"""
from __future__ import annotations
import json
import os
import urllib.request
from pathlib import Path


class Generator:
    def __init__(self, model, endpoint=None, api_key_env=None, device_map="auto",
                 max_input_tokens=24576, repetition_penalty=1.05):
        self.model_id, self.endpoint = str(model), endpoint
        self.api_key_env, self.device_map = api_key_env, device_map
        self.max_input_tokens = max_input_tokens
        self.repetition_penalty = repetition_penalty
        self.model = self.tokenizer = None
        self.last_finish_reason = None

    def generate(self, messages, max_tokens=2048):
        if self.endpoint:
            headers = {"Content-Type": "application/json"}
            if self.api_key_env:
                key = os.environ.get(self.api_key_env)
                if not key:
                    raise ValueError("Requested API key environment variable is unset")
                headers["Authorization"] = "Bearer " + key
            payload = {"model": self.model_id, "messages": messages,
                       "temperature": 0, "max_tokens": max_tokens,
                       "enable_thinking": False,
                       "repetition_penalty": self.repetition_penalty}
            request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(), headers)
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)
            choice = result["choices"][0]
            self.last_finish_reason = choice.get("finish_reason")
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Endpoint returned no final answer")
            return answer.strip()
        if self.model is None:
            if not Path(self.model_id).is_dir():
                raise FileNotFoundError("Local model path is required; download weights separately")
            import torch
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.model_id, local_files_only=True, dtype=torch.bfloat16,
                device_map=self.device_map, low_cpu_mem_usage=True)
            self.model.eval()
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = encoded["input_ids"].shape[1]
        if count > self.max_input_tokens:
            raise ValueError(f"Input has {count} tokens; limit {self.max_input_tokens}; no silent truncation")
        device = self.model.get_input_embeddings().weight.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=False,
                max_new_tokens=max_tokens, repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        generated = output[0, count:]
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.last_finish_reason = "length" if len(generated) >= max_tokens and int(generated[-1]) not in eos else "stop"
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        if not text.strip():
            raise RuntimeError("Model produced no final answer")
        return text.strip()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
