import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openperov.revision import deduplicate_sentences, draft_anchors, materialize_all, revise, validate_candidates


class FakeGenerator:
    def __init__(self, values):
        self.values = iter(values)
        self.prompts = []
    def generate(self, messages, max_tokens):
        self.prompts.append(messages)
        return json.dumps(next(self.values))


def patch(cid="C1", anchor="S001", obligation="O1", text="A matched toy control distinguishes the proposed mechanism."):
    return {"candidate_id": cid, "obligation_id": obligation,
            "effect_class": "ADDS_REQUIRED_CONTROL", "operation": "APPEND_AFTER_ANCHOR",
            "target_anchor_id": anchor, "revised_text": text, "evidence_chunk_ids": ["E001"],
            "scientific_delta": "Adds a discriminating control", "counterfactual_consequence": "Alternative mechanism remains unresolved", "confidence": "high"}


class RevisionTests(unittest.TestCase):
    def test_exact_dedup_preserves_distinct_numbers(self):
        a = "The fictional sample retained 90 percent of its signal."
        b = "The fictional sample retained 91 percent of its signal."
        output, removed = deduplicate_sentences(a + " " + a + " " + b)
        self.assertEqual(output, a + " " + b)
        self.assertEqual(len(removed), 1)

    def test_all_valid_same_anchor_preserves_every_patch(self):
        base = "The toy baseline is unchanged. The second toy statement remains."
        p1, p2 = patch(), patch("C2", text="A second independent control tests a different explanation.")
        output, actions = materialize_all(base, [p1, p2])
        self.assertIn(p1["revised_text"], output)
        self.assertIn(p2["revised_text"], output)
        self.assertIn("The second toy statement remains.", output)
        self.assertEqual(len(actions), 1)

    def test_unknown_evidence_and_anchor_rejected(self):
        p = patch(anchor="S999")
        p["evidence_chunk_ids"] = ["E999"]
        valid, rejected = validate_candidates({"decision": "CANDIDATES", "candidates": [p]},
            {a["id"]: a for a in draft_anchors("The toy baseline remains valid.")}, {"E001"}, {"O1"})
        self.assertEqual(valid, [])
        self.assertIn("unknown_target_anchor", rejected[0]["reasons"])
        self.assertIn("invalid_evidence_ids", rejected[0]["reasons"])

    def test_policy_and_budget_are_effective(self):
        base = "The first fictional claim is valid. The second fictional claim is valid."
        obligations = {"obligations": [{"id": "O1", "requirement": "test first mechanism", "draft_anchor_ids": ["S001"]},
                                       {"id": "O2", "requirement": "test second mechanism", "draft_anchor_ids": ["S002"]}]}
        p1, p2 = patch(), patch("C2", "S002", "O2", "An independent measurement tests the second toy mechanism.")
        proposals = {"decision": "CANDIDATES", "candidates": [p1, p2]}
        chunks = [{"id": "E001", "text": "Fictional matched controls test two independent mechanisms."}]
        structural = FakeGenerator([obligations, proposals])
        out = revise(structural, "How can both toy mechanisms be tested?", base, chunks)
        self.assertEqual(len(out["accepted_patches"]), 2)
        critic = FakeGenerator([obligations, proposals, {"decision": "PATCH", "selected_candidate_ids": ["C2", "C1"]}])
        out = revise(critic, "How can both toy mechanisms be tested?", base, chunks, "critic", 1, "S")
        self.assertEqual([p["candidate_id"] for p in out["accepted_patches"]], ["C2"])
        self.assertIn("Condition S", critic.prompts[-1][1]["content"])

    def test_no_evidence_never_calls_model(self):
        out = revise(FakeGenerator([]), "A toy question?", "A toy answer.", [])
        self.assertEqual(out["answer"], "A toy answer.")

    def test_reference_s20_rejects_external_fact_as_target(self):
        from openperov.revision_s20 import deterministic_gate, default_reject
        base = "The fictional interface may change under bias."
        query = {"question": "How would the toy interface respond?", "base_answer": base, "condition_id": "S"}
        review = default_reject("C1", "")
        review.update(decision="ACCEPT", necessity="FILLS_MISSING_OBLIGATION",
            target_system_match="MATCH", prompt_support="ABSENT", constraint_check="PASS",
            constraint_preserved=True, evidence_role="TARGET_OBSERVATION",
            claim_strength="OBSERVED", novel_information="HIGH", obligation_gain="Tests a mechanism", reason_codes=[])
        result = deterministic_gate(query, {}, patch(), review,
            {a["id"]: a for a in draft_anchors(base)}, {"E001"})
        self.assertFalse(result["eligible_after_gates"])
        self.assertIn("G3_EXTERNAL_FACT_AS_TARGET_OBSERVATION", result["gate_reasons"])

    def test_reference_s20_detects_target_paper_evidence(self):
        from openperov.revision_s20 import deterministic_gate, default_reject
        base = "The fictional interface may change under bias."
        query = {"question": "How would the toy interface respond?", "base_answer": base, "condition_id": "E"}
        result = deterministic_gate(query, {"target_article_exact_hit_count": 1}, patch(),
            default_reject("C1", ""), {a["id"]: a for a in draft_anchors(base)}, {"E001"})
        self.assertFalse(result["eligible_after_gates"])
        self.assertIn("G0_TARGET_ARTICLE_LEAKAGE", result["gate_reasons"])


if __name__ == "__main__":
    unittest.main()
