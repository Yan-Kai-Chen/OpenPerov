"""Generate final answers locally or through an explicitly supplied endpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openperov.inference import Generator, read_jsonl, write_jsonl
from openperov.routing import question_record, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-label", default="OpenPerov Flash")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--endpoint", help="Full chat-completions URL; network is used only when specified")
    parser.add_argument("--api-key-env")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    cfg = json.loads(args.config.read_text()) if args.config else {}
    if cfg.get("temperature", 0) != 0 or cfg.get("enable_thinking", False):
        parser.error("This Flash runner uses deterministic final-answer generation")
    rows = read_jsonl(args.questions)
    records = [question_record(r) for r in rows]
    if len({r[0] for r in records}) != len(records):
        parser.error("Question IDs must be unique")
    generator = Generator(args.model, args.endpoint, args.api_key_env, args.device_map,
        cfg.get("max_input_tokens", 24576), cfg.get("repetition_penalty", 1.05))
    completed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row, (qid, question, family) in zip(rows, records):
            if "system_prompt" in row and "user_prompt" in row:
                prompt = [{"role": "system", "content": row["system_prompt"]},
                          {"role": "user", "content": row["user_prompt"]}]
                route = "frozen_messages"
            else:
                prompt = messages(question, family)
                route = "task_family" if family else "general"
            answer = generator.generate(prompt, cfg.get("max_output_tokens", 2048))
            result = {"benchmark_id": qid, "model": args.system_label, "checkpoint": args.model,
                      "answer": answer, "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                      "prompt_route": route, "finish_reason": generator.last_finish_reason,
                      "nonempty_length_limited_retained": generator.last_finish_reason == "length"}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
    print(json.dumps({"output": str(args.output), "answers": completed}))


if __name__ == "__main__":
    main()
