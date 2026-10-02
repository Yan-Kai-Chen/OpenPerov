"""Portable own-corpus retrieval, distinct from the paper's private index.

BM25 candidate retrieval is provided for portability. Trained Qwen selectors
can select Top40 and rerank that same membership using the released adapters.
This adapter does not reproduce the paper's multivector/citation-graph pool.
"""
from __future__ import annotations
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:[.+-][a-z0-9]+)*", text.lower())


class EvidenceIndex:
    def __init__(self, documents):
        self.documents = []
        seen = set()
        for row in documents:
            doc = {key: str(row.get(key, "")) for key in ("document_id", "title", "text", "source_uri")}
            if not doc["document_id"] or not doc["text"].strip() or doc["document_id"] in seen:
                raise ValueError("Documents require unique document_id and nonempty text")
            seen.add(doc["document_id"])
            self.documents.append(doc)
        if not self.documents:
            raise ValueError("The evidence collection is empty")
        self.term_counts = [Counter(tokens(d["title"] + " " + d["text"])) for d in self.documents]
        self.lengths = [sum(c.values()) for c in self.term_counts]
        self.average = sum(self.lengths) / len(self.lengths) or 1
        self.df = Counter(term for counts in self.term_counts for term in counts)

    def search(self, query, limit=240, exclude_ids=()):
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = set(tokens(query))
        excluded = set(exclude_ids)
        scored = []
        for doc, counts, length in zip(self.documents, self.term_counts, self.lengths):
            if doc["document_id"] in excluded:
                continue
            score = 0.0
            for term in query_terms:
                tf = counts[term]
                if tf:
                    idf = math.log(1 + (len(self.documents) - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / self.average))
            if score > 0:
                scored.append({**doc, "retrieval_score": score})
        return sorted(scored, key=lambda d: (-d["retrieval_score"], d["document_id"]))[:limit]

    def save(self, path):
        Path(path).write_text(json.dumps({"schema": "openperov_own_corpus_bm25_v1", "documents": self.documents}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "openperov_own_corpus_bm25_v1":
            raise ValueError("Unsupported index schema")
        return cls(payload["documents"])


class NeuralRanker:
    """Same Qwen yes-minus-no scoring prompt as reference ranker_score."""
    def __init__(self, model_path, adapter_path, max_length=1024, device="cuda:0", instruction=None):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from .ranker_score import PromptScorer, DEFAULT_INSTRUCTION
        self.torch, self.device = torch, torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
        self.model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True,
            is_trainable=False).to(self.device).eval()
        self.scorer = PromptScorer(tokenizer, max_length, instruction or DEFAULT_INSTRUCTION)

    def rank(self, question, documents, batch_size=2):
        from types import SimpleNamespace
        scored = []
        with self.torch.inference_mode():
            for offset in range(0, len(documents), batch_size):
                docs = documents[offset:offset + batch_size]
                candidates = [SimpleNamespace(document="Article title: " + " ".join(d["title"].split()) + "\nScientific evidence: " + " ".join(d["text"].split())) for d in docs]
                scores = self.scorer.scores(self.model, question, candidates, self.device).tolist()
                scored.extend({**d, "ranker_score": float(s)} for d, s in zip(docs, scores))
        return sorted(scored, key=lambda d: (-d["ranker_score"], d["document_id"]))

    def close(self):
        self.model.to("cpu")
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def compile_evidence(documents, question):
    """Select complete supplied sentences under the reference rank budgets.

    This portable adapter has no private science-topic summaries. It preserves
    complete sentences, exact wording and a separate provenance manifest.
    """
    query_terms = set(tokens(question))
    blocks, provenance, seen = [], [], set()
    for rank, doc in enumerate(documents[:40], 1):
        budget = 1900 if rank <= 8 else 1300 if rank <= 20 else 800
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", doc["text"]) if s.strip()]
        ordered = sorted(enumerate(sentences), key=lambda p: (-len(query_terms & set(tokens(p[1]))), p[0]))
        chosen, used = [], 0
        for position, sentence in ordered:
            key = " ".join(sentence.lower().split())
            if key not in seen and used + len(sentence) + 1 <= budget:
                chosen.append((position, sentence)); used += len(sentence) + 1; seen.add(key)
        if chosen:
            text = " ".join(s for _, s in sorted(chosen))
            eid = f"E{len(blocks) + 1:03d}"
            blocks.append({"id": eid, "text": text})
            provenance.append({"evidence_id": eid, "document_id": doc["document_id"],
                               "source_uri": doc.get("source_uri", ""), "rank": rank})
    return blocks, provenance
