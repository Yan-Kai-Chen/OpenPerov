"""Run portable OpenPerov Pro on a user-supplied evidence corpus.

The paper's private retrieval corpus and multi-route candidate pools are not
included. Both neural ranking stages are optional but must be supplied together.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openperov.inference import Generator, read_jsonl, write_jsonl
from openperov.retrieval import EvidenceIndex, NeuralRanker, compile_evidence
from openperov.revision import revise
from openperov.routing import question_record, messages, condition_id, baseline_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="Frozen Flash predictions; missing answers are generated")
    parser.add_argument("--baseline-system", default="OpenPerov Flash", help="Exact model/system label to select from a public prediction ledger")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path, help="Optional local audit containing user evidence; do not publish without review")
    parser.add_argument("--save-index", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key-env")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--ranker-device", default="cuda:0")
    parser.add_argument("--selector-model")
    parser.add_argument("--selector-adapter")
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-adapter")
    args = parser.parse_args()
    for path in [args.output, args.audit, args.save_index]:
        if path and path.exists():
            parser.error(f"Output already exists: {path}")
    neural = [args.selector_model, args.selector_adapter, args.reranker_model, args.reranker_adapter]
    if any(neural) and not all(neural):
        parser.error("Supply all four selector/reranker paths, or none for lexical own-corpus mode")
    cfg = json.loads(args.config.read_text())
    if cfg.get("retrieval") != "own_corpus_bm25":
        parser.error("This portable entrypoint requires own_corpus_bm25 retrieval")
    rows = read_jsonl(args.questions)
    records = [question_record(r) for r in rows]
    conditions = [condition_id(row) for row in rows]
    if len({r[0] for r in records}) != len(records):
        parser.error("Question IDs must be unique")
    if cfg.get("require_condition") and any(value not in {"E", "S"} for value in conditions):
        parser.error("S20 profile requires condition experimental_context/limited_context or condition_id E/S")
    top_k = int(cfg.get("top_k", 40))
    if not 1 <= top_k <= 40:
        parser.error("top_k must be between 1 and 40")
    index = EvidenceIndex(read_jsonl(args.evidence))
    if args.save_index:
        args.save_index.parent.mkdir(parents=True, exist_ok=True)
        index.save(args.save_index)
    pools = [index.search(q, int(cfg.get("candidate_pool_size", 240)), row.get("exclude_document_ids", []))
             for row, (_, q, _) in zip(rows, records)]
    if all(neural):
        selector = NeuralRanker(args.selector_model, args.selector_adapter, 1536, args.ranker_device,
            "Given a detailed perovskite research question derived from a paper, rank the paper that contains the same distinctive materials, experiments, quantitative observations, and mechanism above merely related literature.")
        try:
            pools = [selector.rank(q, docs)[:top_k] for (_, q, _), docs in zip(records, pools)]
        finally:
            selector.close()
        reranker = NeuralRanker(args.reranker_model, args.reranker_adapter, 1024, args.ranker_device)
        try:
            pools = [reranker.rank(q, docs) for (_, q, _), docs in zip(records, pools)]
        finally:
            reranker.close()
    else:
        pools = [docs[:top_k] for docs in pools]
    baseline = baseline_answers(read_jsonl(args.baseline), args.baseline_system) if args.baseline else {}
    generator = Generator(args.model, args.endpoint, args.api_key_env, args.device_map,
                          cfg.get("max_input_tokens", 24576))
    output, audit = [], []
    for row, (qid, question, family), docs, condition in zip(rows, records, pools, conditions):
        if "system_prompt" in row and "user_prompt" in row:
            prompt = [{"role": "system", "content": row["system_prompt"]}, {"role": "user", "content": row["user_prompt"]}]
        else:
            prompt = messages(question, family)
        draft = baseline.get(qid) or row.get("baseline_answer") or generator.generate(prompt, 2048)
        chunks, provenance = compile_evidence(docs, question)
        result = revise(generator, question, draft, chunks, cfg["acceptance_policy"],
                        cfg.get("revision_budget"), condition)
        answer = result["answer"]
        output.append({"benchmark_id": qid, "model": args.model + "+Pro", "answer": answer,
                       "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                       "profile": cfg.get("profile"), "retrieval_mode": "bm25_with_neural_top40" if all(neural) else "bm25_only",
                       "accepted_patch_count": len(result["accepted_patches"]), "status": result["status"]})
        audit.append({"benchmark_id": qid, "baseline_sha256": hashlib.sha256(draft.encode()).hexdigest(),
                      "provenance": provenance, **result})
    write_jsonl(args.output, output)
    if args.audit:
        write_jsonl(args.audit, audit)
    print(json.dumps({"answers": len(output), "output": str(args.output), "profile": cfg.get("profile")}))


if __name__ == "__main__":
    main()
