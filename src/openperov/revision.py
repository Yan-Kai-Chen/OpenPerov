"""Reference local-patch prompts and deterministic materialization.

Transport and own-corpus retrieval are portable adapters. The original paper's
private candidate pools and evidence records are not distributed.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any
OBLIGATION_SYSTEM = """You are a senior perovskite scientist mapping the explicit obligations of a scientific question onto a frozen expert answer. Do not use external knowledge, do not score the answer, and do not rewrite it. Return strict JSON only."""

PROPOSAL_SYSTEM = """You are a senior perovskite scientist performing a high-recall evidence comparison. The frozen Stage5 answer remains the base answer. Surface up to six candidate local evidence deltas that could materially correct or complete an explicit question obligation. This is a proposal stage, so include plausible consequential candidates but never add decorative facts. Do not regenerate the answer. Return strict JSON only."""

CRITIC_SYSTEM = """You are the final conservative scientific patch critic. From prevalidated local candidates, retain zero to three patches only when the cited evidence changes a required causal link, scientific boundary, decisive control, or go/no-go consequence. Reject merely related facts, narrower examples that displace a valid general statement, repetition, style edits, and unsupported specificity. The frozen answer is authoritative. Return strict JSON only."""

SCHEMA_REPAIR_SYSTEM = """You are a deterministic schema repairer. Convert the supplied scientific analysis into the requested JSON object without changing its scientific conclusions. Preserve candidate IDs, anchor IDs, obligation IDs, evidence IDs, and scientific text. Return strict JSON only, with no markdown or commentary."""

MAX_SELECTED_PATCHES = 6

ALLOWED_EFFECTS = {
    "CORRECTS_CAUSAL_ERROR",
    "FILLS_REQUIRED_CAUSAL_LINK",
    "LIMITS_OR_BOUNDS",
    "ADDS_REQUIRED_CONTROL",
    "ADDS_DECISION_THRESHOLD",
}

ALLOWED_OPERATIONS = {"REPLACE_ANCHOR", "APPEND_AFTER_ANCHOR"}

STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "these", "those", "into", "for", "are", "was", "were",
    "what", "when", "where", "which", "while", "their", "then", "than", "have", "has", "had", "also", "one",
    "two", "three", "using", "used", "use", "must", "should", "would", "could", "answer", "question", "required",
    "provide", "explain", "include", "between", "under", "over", "only", "each", "most", "more", "less", "not",
}

def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model output is not an object")
    return payload

def schema_repair_prompt(stage: str, raw: str) -> str:
    schemas = {
        "obligation": '{"obligations":[{"id":"O1","requirement":"...","kind":"mechanism|diagnostic|control|boundary|comparison|decision|quantitative","draft_anchor_ids":["S001"],"coverage":"complete|partial|implicit|missing|possibly_wrong","missing_or_weak_dimensions":["..."]}]}',
        "proposal": '{"decision":"CANDIDATES|NONE","candidates":[{"candidate_id":"C1","obligation_id":"O1","effect_class":"CORRECTS_CAUSAL_ERROR|FILLS_REQUIRED_CAUSAL_LINK|LIMITS_OR_BOUNDS|ADDS_REQUIRED_CONTROL|ADDS_DECISION_THRESHOLD","operation":"REPLACE_ANCHOR|APPEND_AFTER_ANCHOR","target_anchor_id":"S001","revised_text":"...","evidence_chunk_ids":["E001"],"scientific_delta":"...","counterfactual_consequence":"...","confidence":"high|medium"}]}',
        "critic": '{"decision":"PATCH|NONE","selected_candidate_ids":["C1"],"rejection_summary":"brief reason"}',
    }
    limits = {
        "obligation": "Keep two to six obligations.",
        "proposal": "Keep no more than six candidates.",
        "critic": f"Keep no more than {MAX_SELECTED_PATCHES} selected candidate IDs.",
    }
    return f"""Convert the analysis below to exactly this JSON schema.

Schema:
{schemas[stage]}

Constraint:
{limits[stage]}

Scientific analysis to preserve:
{raw}"""

def evidence_chunks(evidence: str) -> list[dict[str, str]]:
    raw_parts = [part.strip() for part in re.split(r"\n(?=##\s|[-*]\s|\d+[.)]\s)", evidence) if part.strip()]
    parts: list[str] = []
    for part in raw_parts:
        if len(part) <= 2200:
            parts.append(part)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", part)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > 1800:
                parts.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            parts.append(current)
    return [{"id": f"E{index:03d}", "text": text} for index, text in enumerate(parts, start=1)]

def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z][A-Za-z0-9+.-]{2,}", text.lower()) if token not in STOPWORDS}

def focus_chunks(question: str, obligation: dict[str, Any], chunks: list[dict[str, str]], limit: int = 6) -> list[str]:
    query = tokens(question + " " + str(obligation.get("requirement") or "") + " " + " ".join(obligation.get("missing_or_weak_dimensions") or []))
    document_frequency = Counter(token for chunk in chunks for token in tokens(chunk["text"]))
    scored = []
    for chunk in chunks:
        present = tokens(chunk["text"])
        score = sum(math.log((len(chunks) + 1) / (document_frequency[token] + 1)) + 1.0 for token in query & present)
        scored.append((score, chunk["id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for score, chunk_id in scored[:limit] if score > 0]

def anchor_block(anchors: list[dict[str, Any]]) -> str:
    return "\n".join(f'[{item["id"]}] {item["text"]}' for item in anchors)

def evidence_block(chunks: list[dict[str, str]], only: set[str] | None = None) -> str:
    return "\n\n".join(f'[{item["id"]}] {item["text"]}' for item in chunks if only is None or item["id"] in only)

def obligation_prompt(question: str, draft: str, anchors: list[dict[str, Any]]) -> str:
    return f"""# Scientific question\n\n{question}\n\n# Frozen Stage5 answer\n\n{draft}\n\n# Deterministic sentence anchors\n\n{anchor_block(anchors)}\n\n# Task\n\nMap the question into two to six non-overlapping explicit obligations. For each obligation, name the required conclusion or action, its causal link, and any boundary/control/decision consequence that the question requires. Assess the frozen answer using anchor IDs only.\n\nReturn exactly:\n{{\"obligations\":[{{\"id\":\"O1\",\"requirement\":\"...\",\"kind\":\"mechanism|diagnostic|control|boundary|comparison|decision|quantitative\",\"draft_anchor_ids\":[\"S001\"],\"coverage\":\"complete|partial|implicit|missing|possibly_wrong\",\"missing_or_weak_dimensions\":[\"specific missing causal link, boundary, control, or decision consequence; empty when complete\"]}}]}}\n\nDo not quote or rewrite the answer. Do not invent hidden obligations. Anchor IDs are backend navigation only."""

def validate_obligations(payload: dict[str, Any], anchor_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = payload.get("obligations")
    if not isinstance(raw, list):
        return [], ["obligations_not_list"]
    valid, reasons, seen = [], [], set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            reasons.append(f"obligation_{index}_not_object")
            continue
        oid = str(item.get("id") or "").strip()
        requirement = str(item.get("requirement") or "").strip()
        ids = [str(value).strip() for value in item.get("draft_anchor_ids") or [] if str(value).strip() in anchor_ids]
        dimensions = [str(value).strip() for value in item.get("missing_or_weak_dimensions") or [] if str(value).strip()]
        if not oid or oid in seen or not requirement:
            reasons.append(f"obligation_{index}_invalid_identity")
            continue
        seen.add(oid)
        valid.append({"id": oid, "requirement": requirement, "kind": str(item.get("kind") or "").strip(), "draft_anchor_ids": ids, "coverage": str(item.get("coverage") or "").strip(), "missing_or_weak_dimensions": dimensions})
    return (valid, reasons) if valid else ([], reasons + ["no_valid_obligations"])

def proposal_prompt(question: str, draft: str, anchors: list[dict[str, Any]], obligations: list[dict[str, Any]], chunks: list[dict[str, str]], focus: dict[str, list[str]]) -> str:
    focus_ids = {value for values in focus.values() for value in values}
    return f"""# Scientific question\n\n{question}\n\n# Frozen Stage5 answer\n\n{draft}\n\n# Deterministic answer anchors\n\n{anchor_block(anchors)}\n\n# Obligation map\n\n{json.dumps(obligations, ensure_ascii=False, indent=2)}\n\n# Soft-focus evidence shortlist\nThe shortlist is only a navigation aid. The full evidence appendix remains authoritative and must also be considered.\n\n{evidence_block(chunks, focus_ids)}\n\n# Full evidence appendix\n\n{evidence_block(chunks)}\n\n# High-recall candidate task\n\nPropose zero to six independent local evidence deltas. Prefer a coherent replacement when a causal obligation is incomplete; use append only for a genuinely missing control, boundary, threshold, or decision rule.\n\nReturn exactly:\n{{\"decision\":\"CANDIDATES|NONE\",\"candidates\":[{{\"candidate_id\":\"C1\",\"obligation_id\":\"O1\",\"effect_class\":\"CORRECTS_CAUSAL_ERROR|FILLS_REQUIRED_CAUSAL_LINK|LIMITS_OR_BOUNDS|ADDS_REQUIRED_CONTROL|ADDS_DECISION_THRESHOLD\",\"operation\":\"REPLACE_ANCHOR|APPEND_AFTER_ANCHOR\",\"target_anchor_id\":\"S001\",\"revised_text\":\"complete replacement sentence(s), or only new appended sentence(s)\",\"evidence_chunk_ids\":[\"E001\"],\"scientific_delta\":\"what material conclusion changes\",\"counterfactual_consequence\":\"what required weakness remains without it\",\"confidence\":\"high|medium\"}}]}}\n\nRules: no style edits, no paper narration, no citations or backend IDs in revised_text, no benchmark reference or score. Every candidate must affect an explicit obligation and be supported by one to five evidence chunks."""

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))

def validate_candidates(payload: dict[str, Any], anchors: dict[str, dict[str, Any]], evidence_ids: set[str], obligation_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if str(payload.get("decision") or "").upper() == "NONE":
        return [], []
    raw = payload.get("candidates")
    if str(payload.get("decision") or "").upper() != "CANDIDATES" or not isinstance(raw, list):
        return [], [{"index": 0, "reasons": ["invalid_candidate_decision"]}]
    if len(raw) > 6:
        return [], [{"index": 0, "reasons": ["more_than_six_candidates"]}]
    valid, rejected, seen = [], [], set()
    for index, item in enumerate(raw, start=1):
        reasons = []
        if not isinstance(item, dict):
            rejected.append({"index": index, "reasons": ["candidate_not_object"]})
            continue
        candidate = {key: str(item.get(key) or "").strip() for key in ("candidate_id", "obligation_id", "effect_class", "operation", "target_anchor_id", "revised_text", "scientific_delta", "counterfactual_consequence", "confidence")}
        raw_ids = [str(value).strip() for value in item.get("evidence_chunk_ids") or []]
        ids = list(dict.fromkeys(value for value in raw_ids if value in evidence_ids))[:5]
        candidate["evidence_chunk_ids"] = ids
        cid = candidate["candidate_id"]
        if not cid or cid in seen:
            reasons.append("invalid_candidate_id")
        if candidate["obligation_id"] not in obligation_ids:
            reasons.append("unknown_obligation_id")
        if candidate["effect_class"] not in ALLOWED_EFFECTS:
            reasons.append("invalid_effect_class")
        if candidate["operation"] not in ALLOWED_OPERATIONS:
            reasons.append("invalid_operation")
        if candidate["target_anchor_id"] not in anchors:
            reasons.append("unknown_target_anchor")
        if not candidate["revised_text"]:
            reasons.append("empty_revision")
        if not ids:
            reasons.append("invalid_evidence_ids")
        if not candidate["scientific_delta"] or not candidate["counterfactual_consequence"]:
            reasons.append("missing_consequence_fields")
        if candidate["confidence"].lower() not in {"high", "medium"}:
            reasons.append("invalid_confidence")
        anchor = anchors.get(candidate["target_anchor_id"])
        added = word_count(candidate["revised_text"]) if candidate["operation"] == "APPEND_AFTER_ANCHOR" else max(0, word_count(candidate["revised_text"]) - word_count(str(anchor["text"]) if anchor else ""))
        candidate["added_words"] = added
        candidate["exceeds_soft_160_word_target"] = added > 160
        if reasons:
            rejected.append({"index": index, "reasons": reasons, "candidate": candidate})
        else:
            valid.append(candidate)
            seen.add(cid)
    return valid, rejected

def critic_prompt(question: str, draft: str, anchors: list[dict[str, Any]], obligations: list[dict[str, Any]], candidates: list[dict[str, Any]], chunks: list[dict[str, str]], focus: dict[str, list[str]]) -> str:
    used_ids = {eid for candidate in candidates for eid in candidate["evidence_chunk_ids"]}
    used_ids.update(value for values in focus.values() for value in values)
    return f"""# Scientific question\n\n{question}\n\n# Frozen Stage5 answer\n\n{draft}\n\n# Answer anchors\n\n{anchor_block(anchors)}\n\n# Explicit obligations\n\n{json.dumps(obligations, ensure_ascii=False, indent=2)}\n\n# Prevalidated candidate patches\n\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n# Candidate-linked and soft-focus evidence\n\n{evidence_block(chunks, used_ids)}\n\n# Full evidence appendix\n\n{evidence_block(chunks)}\n\n# Final critic task\n\nSelect zero to three candidate IDs. Retain a candidate only if it materially improves the answer to this exact question and its consequence is supported by the evidence. Avoid multiple patches to the same obligation unless they repair distinct causal or decision defects.\n\nReturn exactly:\n{{\"decision\":\"PATCH|NONE\",\"selected_candidate_ids\":[\"C1\"],\"rejection_summary\":\"brief reason for rejecting the rest\"}}\n\nDo not rewrite candidates, do not score, and do not use hidden references."""

def select_candidates(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    if str(payload.get("decision") or "").upper() == "NONE":
        return [], []
    ids = payload.get("selected_candidate_ids")
    if str(payload.get("decision") or "").upper() != "PATCH" or not isinstance(ids, list):
        return [], ["invalid_critic_decision"]
    ids = [str(value).strip() for value in ids]
    if len(ids) > MAX_SELECTED_PATCHES or len(set(ids)) != len(ids):
        return [], ["critic_selection_count_or_duplicates"]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if any(value not in by_id for value in ids):
        return [], ["critic_selected_unknown_candidate"]
    chosen = [dict(by_id[value]) for value in ids]
    used_anchors, used_obligations, total_added = set(), set(), 0
    final, reasons = [], []
    for candidate in chosen:
        if candidate["target_anchor_id"] in used_anchors:
            reasons.append(f'{candidate["candidate_id"]}_duplicate_anchor')
            continue
        if candidate["obligation_id"] in used_obligations:
            reasons.append(f'{candidate["candidate_id"]}_duplicate_obligation')
            continue
        if total_added + int(candidate["added_words"]) > 320:
            reasons.append(f'{candidate["candidate_id"]}_total_added_words')
            continue
        final.append(candidate)
        used_anchors.add(candidate["target_anchor_id"])
        used_obligations.add(candidate["obligation_id"])
        total_added += int(candidate["added_words"])
    return final, reasons

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])" )

ANCHOR_BOUNDARY = re.compile(
    r"[.!?](?=\s+(?:[A-Z0-9(\[]|However\b|Thus\b|First\b|Second\b|Finally\b))"
)

def normalize_sentence(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = re.sub(r"[.!?]+$", "", value).strip()
    return value

def lexical_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    # Hyphens and mojibake separators must not make equivalent scientific
    # phrases look different (for example, additive-precursor-solvent versus
    # additive<bad-separator>precursor<bad-separator>solvent).
    return set(re.findall(r"\d+(?:\.\d+)?|[a-z][a-z0-9+]*", normalized))

def draft_anchors(draft: str) -> list[dict[str, Any]]:
    spans: list[tuple[int, int]] = []
    paragraphs = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", draft, flags=re.S))
    for paragraph in paragraphs:
        pstart, pend = paragraph.span()
        starts = [pstart]
        starts.extend(match.end() for match in ANCHOR_BOUNDARY.finditer(draft, pstart, pend))
        starts = sorted(set(starts))
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else pend
            while start < end and draft[start].isspace():
                start += 1
            while end > start and draft[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end))
    return [
        {"id": f"S{index:03d}", "start": start, "end": end, "text": draft[start:end]}
        for index, (start, end) in enumerate(spans, start=1)
    ]

def deduplicate_sentences(text: str) -> tuple[str, list[str]]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    seen: set[str] = set()
    cleaned_paragraphs: list[str] = []
    removed: list[str] = []
    for paragraph in paragraphs:
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(paragraph) if part.strip()]
        kept: list[str] = []
        for sentence in sentences:
            key = normalize_sentence(sentence)
            if key and len(key) >= 24 and key in seen:
                removed.append(sentence)
                continue
            if key:
                seen.add(key)
            kept.append(sentence)
        if kept:
            cleaned_paragraphs.append(" ".join(kept))
    return "\n\n".join(cleaned_paragraphs).strip(), removed

def rematerialize(
    base: str, patches: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    anchors = {row["id"]: row for row in draft_anchors(base)}
    positioned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for patch in patches:
        anchor_id = str(patch.get("target_anchor_id") or "")
        if anchor_id not in anchors:
            raise RuntimeError(f"Unknown anchor {anchor_id}")
        positioned.append((anchors[anchor_id], patch))

    answer = base
    actions: list[dict[str, Any]] = []
    for anchor, patch in sorted(positioned, key=lambda item: int(item[0]["start"]), reverse=True):
        start, end = int(anchor["start"]), int(anchor["end"])
        original = str(anchor["text"])
        revised = str(patch.get("revised_text") or "").strip()
        operation = str(patch.get("operation") or "")
        effect = str(patch.get("effect_class") or "")
        if not revised:
            raise RuntimeError("Empty revised_text in accepted patch")
        if operation == "APPEND_AFTER_ANCHOR":
            replacement = original + " " + revised
            action = "append_then_deduplicate"
        elif operation == "REPLACE_ANCHOR":
            original_key = normalize_sentence(original)
            revised_key = normalize_sentence(revised)
            already_preserved = bool(original_key and original_key in revised_key)
            if effect == "CORRECTS_CAUSAL_ERROR" or already_preserved:
                replacement = revised
                action = (
                    "destructive_replace_allowed_causal_correction"
                    if effect == "CORRECTS_CAUSAL_ERROR"
                    else "replace_already_preserves_anchor"
                )
            else:
                replacement = original + " " + revised
                action = "noncorrective_replace_preserved_anchor"
        else:
            raise RuntimeError(f"Unsupported operation: {operation}")
        answer = answer[:start] + replacement + answer[end:]
        actions.append(
            {
                "candidate_id": patch.get("candidate_id"),
                "operation": operation,
                "effect_class": effect,
                "target_anchor_id": anchor["id"],
                "action": action,
                "original_anchor": original,
                "revised_text": revised,
            }
        )
    actions.reverse()
    return answer, actions

def materialize_all(
    base: str, candidates: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    anchors = {row["id"]: row for row in draft_anchors(base)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        anchor_id = str(candidate.get("target_anchor_id") or "")
        if anchor_id not in anchors:
            raise RuntimeError(f"Unknown target anchor: {anchor_id}")
        operation = str(candidate.get("operation") or "")
        if operation not in {"APPEND_AFTER_ANCHOR", "REPLACE_ANCHOR"}:
            raise RuntimeError(f"Unsupported operation: {operation}")
        revised = str(candidate.get("revised_text") or "").strip()
        if not revised:
            raise RuntimeError(f"Empty revised_text: {candidate.get('candidate_id')}")
        groups[anchor_id].append(candidate)

    answer = base
    actions: list[dict[str, Any]] = []
    ordered_groups = sorted(
        groups.items(), key=lambda item: int(anchors[item[0]]["start"]), reverse=True
    )
    for anchor_id, patches in ordered_groups:
        anchor = anchors[anchor_id]
        original = str(anchor["text"])
        replacements = [
            patch for patch in patches if patch["operation"] == "REPLACE_ANCHOR"
        ]
        appends = [
            patch for patch in patches if patch["operation"] == "APPEND_AFTER_ANCHOR"
        ]
        if replacements:
            # All replacement candidates must remain represented. The first is
            # the replacement body; additional replacements and appends become
            # local continuation text instead of silently overwriting one another.
            ordered = replacements + appends
            replacement = " ".join(str(patch["revised_text"]).strip() for patch in ordered)
            group_action = "replace_then_append_all_same_anchor_candidates"
        else:
            ordered = appends
            replacement = original + " " + " ".join(
                str(patch["revised_text"]).strip() for patch in ordered
            )
            group_action = "append_all_same_anchor_candidates"

        start, end = int(anchor["start"]), int(anchor["end"])
        answer = answer[:start] + replacement + answer[end:]
        actions.append(
            {
                "target_anchor_id": anchor_id,
                "candidate_ids": [str(patch.get("candidate_id") or "") for patch in ordered],
                "operations": [str(patch.get("operation") or "") for patch in ordered],
                "candidate_count": len(ordered),
                "action": group_action,
            }
        )
    actions.reverse()
    return answer.strip(), actions

def apply_patches(draft: str, anchors: dict[str, dict[str, Any]], patches: list[dict[str, Any]]) -> str:
    answer = draft
    positioned = [(anchors[patch["target_anchor_id"]], patch) for patch in patches]
    for anchor, patch in sorted(positioned, key=lambda item: int(item[0]["start"]), reverse=True):
        start, end = int(anchor["start"]), int(anchor["end"])
        if patch["operation"] == "APPEND_AFTER_ANCHOR":
            answer = answer[:end] + " " + patch["revised_text"] + answer[end:]
        else:
            answer = answer[:start] + patch["revised_text"] + answer[end:]
    return answer

def revise(generator, question, draft, chunks, policy="structural_validity", revision_budget=None, condition=None):
    """Run reference proposal prompts over supplied evidence, with explicit policy.

    condition E/S adds the published information boundary to the critic prompt.
    This portable boundary instruction does not reproduce S20's original G0-G7
    verifier, calibrated gate decisions or frozen evidence packets.
    """
    if policy not in {"structural_validity", "critic"}:
        raise ValueError("policy must be structural_validity or critic")
    if revision_budget is not None and (not isinstance(revision_budget, int) or revision_budget < 0):
        raise ValueError("revision_budget must be a nonnegative integer or null")
    if condition not in {None, "E", "S"}:
        raise ValueError("condition must be E or S")
    anchors = draft_anchors(draft)
    if not anchors or not chunks:
        return {"answer": draft, "status": "no_anchors_or_evidence", "accepted_patches": [], "proposals": []}
    def call(stage, system, prompt, limit):
        raw = generator.generate([{"role": "system", "content": system}, {"role": "user", "content": prompt}], max_tokens=limit)
        try:
            return parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            repaired = generator.generate([{"role": "system", "content": SCHEMA_REPAIR_SYSTEM},
                        {"role": "user", "content": schema_repair_prompt(stage, raw)}], max_tokens=limit)
            return parse_json_object(repaired)
    obligations, obligation_rejections = validate_obligations(
        call("obligation", OBLIGATION_SYSTEM, obligation_prompt(question, draft, anchors), 1400),
        {a["id"] for a in anchors})
    if not obligations:
        return {"answer": draft, "status": "no_valid_obligations", "accepted_patches": [], "proposals": [], "validation_rejections": obligation_rejections}
    focus = {o["id"]: focus_chunks(question, o, chunks) for o in obligations}
    proposed = call("proposal", PROPOSAL_SYSTEM,
                   proposal_prompt(question, draft, anchors, obligations, chunks, focus), 4200)
    candidates, rejected = validate_candidates(proposed, {a["id"]: a for a in anchors},
                             {c["id"] for c in chunks}, {o["id"] for o in obligations})
    critic_rejections = []
    if policy == "critic" and candidates:
        prompt = critic_prompt(question, draft, anchors, obligations, candidates, chunks, focus)
        if condition:
            boundary = ("Condition E: facts in the question are target-case observations. Retrieved evidence may only add a causal link, boundary or discriminating experiment, and should not repeat supplied background."
                        if condition == "E" else
                        "Condition S: target-case information is sparse. External devices, values, processes and observations must not be presented as already observed in the target case. Use external evidence for mechanisms, hypotheses and discriminating tests.")
            prompt = boundary + "\n\n" + prompt
        chosen, critic_rejections = select_candidates(call("critic", CRITIC_SYSTEM, prompt, 1400), candidates)
    else:
        chosen = candidates
    if revision_budget is not None:
        chosen = chosen[:revision_budget]
    if policy == "structural_validity":
        amended, actions = materialize_all(draft, chosen)
    else:
        amended = apply_patches(draft, {a["id"]: a for a in anchors}, chosen)
        actions = [{"candidate_id": p["candidate_id"], "operation": p["operation"], "target_anchor_id": p["target_anchor_id"]} for p in chosen]
    final, removed = deduplicate_sentences(amended)
    return {"answer": final, "status": "complete", "policy": policy,
            "revision_budget": revision_budget, "obligations": obligations,
            "proposals": candidates, "accepted_patches": chosen,
            "validation_rejections": rejected, "critic_rejections": critic_rejections,
            "assembly_actions": actions, "exact_duplicates_removed": len(removed)}

from pathlib import Path

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
