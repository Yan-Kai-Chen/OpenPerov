from pathlib import Path
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openperov.retrieval import EvidenceIndex, compile_evidence
from openperov.routing import question_record, condition_id, baseline_answers


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"document_id": "D1", "title": "Toy heat", "text": "Heat changes the fictional coating. A matched toy control checks heat history."},
                     {"document_id": "D2", "title": "Toy bias", "text": "Bias causes reversible fictional signal changes. Reverse bias tests recovery."}]

    def test_query_relevance_and_exclusion(self):
        index = EvidenceIndex(self.docs)
        self.assertEqual(index.search("reversible bias")[0]["document_id"], "D2")
        self.assertEqual(index.search("reversible bias", exclude_ids=["D2"]), [])
        self.assertEqual(index.search("unrelatedword"), [])

    def test_index_roundtrip_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.json"
            EvidenceIndex(self.docs).save(path)
            self.assertEqual(EvidenceIndex.load(path).search("heat")[0]["document_id"], "D1")
        with self.assertRaises(ValueError):
            EvidenceIndex([self.docs[0], self.docs[0]])

    def test_complete_sentences_and_provenance(self):
        chunks, provenance = compile_evidence(self.docs, "heat")
        self.assertTrue(chunks)
        self.assertEqual(provenance[0]["document_id"], "D1")
        self.assertTrue(all(c["text"].endswith(".") for c in chunks))
        self.assertEqual([c["id"] for c in chunks], [p["evidence_id"] for p in provenance])
        long = [{"document_id": "LONG", "title": "", "text": "x" * 2000 + "."}]
        self.assertEqual(compile_evidence(long, "x")[0], [])

    def test_public_question_schemas(self):
        self.assertEqual(question_record({"id": "P1", "question": "Toy?", "task_family": "stability_failure"}), ("P1", "Toy?", "stability_failure"))
        self.assertEqual(question_record({"id": "E1", "system_prompt": "Frozen", "user_prompt": "Frozen toy?"})[1], "Frozen toy?")

    def test_published_s20_conditions_do_not_mutate_prompts(self):
        row = {"id": "E1", "condition": "experimental_context", "system_prompt": "Original system", "user_prompt": "Original user"}
        original = dict(row)
        self.assertEqual(condition_id(row), "E")
        self.assertEqual(row, original)
        self.assertEqual(condition_id({"condition": "limited_context"}), "S")
        self.assertEqual(condition_id({"condition_id": "S"}), "S")

    def test_public_multisystem_baseline_is_filtered_first(self):
        rows = [{"benchmark_id": "Q1", "model": "OpenPerov Flash", "answer": "Frozen answer"},
                {"benchmark_id": "Q1", "model": "OpenPerov Pro", "answer": "Revised answer"}]
        self.assertEqual(baseline_answers(rows), {"Q1": "Frozen answer"})
        self.assertEqual(baseline_answers([{"id": "E1", "system": "OpenPerov Flash", "answer": "Expert task answer"}]), {"E1": "Expert task answer"})
        with self.assertRaises(ValueError):
            baseline_answers(rows, "Unknown system")

    def test_nonempty_length_limited_answer_is_retained(self):
        from openperov.inference import Generator
        response = {"choices": [{"finish_reason": "length", "message": {"content": "A nonempty fictional answer."}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            generator = Generator("synthetic", endpoint="https://example.invalid/v1/chat/completions")
            answer = generator.generate([{"role": "user", "content": "Toy question"}], 10)
        self.assertEqual(answer, "A nonempty fictional answer.")
        self.assertEqual(generator.last_finish_reason, "length")

    def test_reference_index_builds_real_fts_on_synthetic_facets(self):
        from openperov import reference_index
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, work = root / "source", root / "work"
            source.mkdir()
            rows = [{"schema": reference_index.EXPECTED_FACET_SCHEMA,
                     "facet_id": "SYNTH_FACET_1", "core_id": "SYNTH_PAPER_1",
                     "facet_type": "openalex_abstract", "coverage_tier": "synthetic",
                     "title": "Fictional heat control", "text": "A fictional coating changes under heat. A matched control tests temperature history."}]
            text = "".join(json.dumps(row) + "\n" for row in rows)
            (source / "facet_records.jsonl").write_bytes(text.encode("utf-8"))
            manifest = {"schema": reference_index.EXPECTED_CORPUS_SCHEMA, "status": "READY",
                        "counts": {"facet_records": len(rows)}, "files": {"facet_records.jsonl": {"sha256": hashlib.sha256(text.encode()).hexdigest()}}}
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            reference_index.prepare(source, work, None)
            connection = sqlite3.connect(work / "index/facet_fts.sqlite")
            try:
                count = connection.execute("SELECT count(*) FROM facet_fts WHERE facet_fts MATCH 'heat'").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)
            self.assertFalse((work / "index/embeddings.f16.npy").exists())

    def test_reference_top40_assembler_runs_without_private_assets(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            question = {"query_id": "SYNTH_Q1", "benchmark_id": "SYNTH_Q1", "ability_family": "mechanism_diagnostic", "question": "How can matched fictional controls distinguish heat damage from instrument drift?"}
            ranking = {"query_id": "SYNTH_Q1", "ranking": [{"rank": i, "core_id": f"SYNTH_PAPER_{i}"} for i in range(1, 41)]}
            facets = [{"facet_id": f"SYNTH_FACET_{i}", "core_id": f"SYNTH_PAPER_{i}", "facet_type": "openalex_abstract", "title": f"Fictional toy control study {i}", "text": "This invented example tests a toy coating under heat. A matched fictional reference sample distinguishes instrument drift from changes in the coating. No real experiment or material result is represented."} for i in range(1, 41)]
            for name, rows in [("questions", [question]), ("rankings", [ranking]), ("facets", facets), ("topics", [])]:
                (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            command = [sys.executable, "-B", str(project / "scripts/reference_retrieval.py"), "evidence",
                       "--questions", str(root / "questions.jsonl"), "--rankings", str(root / "rankings.jsonl"),
                       "--science-topic-packs", str(root / "topics.jsonl"), "--facet-corpus", str(root / "facets.jsonl"),
                       "--expected-queries", "1", "--output-dir", str(root / "output")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            packet = json.loads((root / "output/evidence_packets.jsonl").read_text().splitlines()[0])
            self.assertEqual(len(packet["cards"]), 40)


if __name__ == "__main__":
    unittest.main()
