"""Scientific Mastery v4.3. Frozen-component aggregation is fully reproducible.
Exact new-answer components require separately supplied original source supports.
Component algorithms copied from the frozen scorer without threshold changes.
"""
from __future__ import annotations
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import scoring_strict, scoring_features
from .statistics import huber

V4_WEIGHTS = {
    "semantic_adequacy": 0.20,
    "rubric_coverage": 0.20,
    "supported_claim_precision": 0.40,
    "decision_control": 0.20,
}

SEMANTIC_MASTERY_THRESHOLD = 0.80

KEY_OVERLAP_FULL = 0.50

CLAIM_F1_FULL = 0.36

CLAIM_PRECISION_FULL = 0.72

DECISION_OVERLAP_FULL = 0.50

MISSING_KEY_THRESHOLD = 0.18

def item_id(row: dict[str, Any]) -> str:
    return str(row.get("benchmark_id") or row.get("source_benchmark_id") or "")

def answer_text(row: dict[str, Any]) -> str:
    return str(
        row.get("candidate_answer")
        or row.get("prediction_text")
        or row.get("answer")
        or ""
    )

def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def round_half(value: float) -> float:
    return max(0.0, min(4.0, round(value * 2) / 2))

def unified_rescue_v1(
    strict: float,
    row: dict[str, Any],
    reference: str,
    keys: list[str],
    support: str,
    features_module: Any,
) -> tuple[float, str, dict[str, Any]]:
    answer = answer_text(row)
    feats = features_module.answer_features(answer, reference, keys, support)
    flags = {
        str(value)
        for value in (row.get("heuristic_flags") or row.get("strict_flags") or [])
    }
    diagnostics = row.get("diagnostics") or {}
    strict_key = as_float(diagnostics.get("key_point_credit_fraction"))
    base_meta = {
        "answer_token_count": int(feats["answer_token_count"]),
        "semantic_key_fraction": float(feats["semantic_key_fraction"]),
        "semantic_ref_cosine": float(feats["semantic_ref_cosine"]),
        "semantic_ref_coverage": float(feats["semantic_ref_coverage"]),
        "semantic_specific_coverage": float(feats["semantic_specific_coverage"]),
        "semantic_numeric_coverage": float(feats["semantic_numeric_coverage"]),
        "reasoning_marker_count": int(feats["reasoning_marker_count"]),
        "strict_key_fraction": strict_key,
    }
    if strict >= 4.0:
        return strict, "already_4", base_meta
    if not answer.strip() or "empty_answer" in flags:
        return strict, "no_rescue_empty", base_meta
    if int(feats["answer_token_count"]) < 22 or "very_short_answer_cap_1" in flags:
        return strict, "no_rescue_short", base_meta

    sem_key = float(feats["semantic_key_fraction"])
    sem_ref_cos = float(feats["semantic_ref_cosine"])
    sem_ref_cov = float(feats["semantic_ref_coverage"])
    sem_ref = max(sem_ref_cos, sem_ref_cov)
    sem_specific = float(feats["semantic_specific_coverage"])
    sem_numeric = float(feats["semantic_numeric_coverage"])
    answer_tokens = int(feats["answer_token_count"])
    evidence_ok = sem_specific >= 0.25 or sem_numeric >= 0.45 or sem_ref >= 0.42
    ref_ok = sem_ref >= 0.38 or (sem_ref_cos >= 0.32 and sem_ref_cov >= 0.28)
    keyish = sem_key >= 0.50 or strict_key >= 0.50
    broad_complete = answer_tokens >= 45 and ref_ok and evidence_ok and keyish
    high_complete = (
        answer_tokens >= 55
        and sem_key >= 0.65
        and sem_ref >= 0.35
        and evidence_ok
    )
    target = strict
    reason = "no_rescue"
    if strict <= 2.5 and broad_complete:
        target = max(target, 3.0)
        reason = "v1_science_complete_to_3"
    if strict <= 3.0 and high_complete:
        target = max(target, 3.5)
        reason = "v1_high_coverage_to_3_5"
    strong_low_score = high_complete or (
        broad_complete and sem_ref >= 0.45 and sem_specific >= 0.35
    )
    rescued = round_half(min(target, strict + 0.5))
    if rescued <= strict:
        rescued = strict
        reason = "no_rescue"
    return rescued, reason, {
        **base_meta,
        "evidence_ok": evidence_ok,
        "ref_ok": ref_ok,
        "keyish": keyish,
        "broad_complete": broad_complete,
        "high_complete": high_complete,
        "strong_low_score": strong_low_score,
        "max_delta": 0.5,
    }

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

def scientific_claims(text: str, features_module: Any) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"(?<=[.!?;])\s+|\n+", text or ""):
        claim_tokens = features_module.tokens(chunk)
        if len(claim_tokens) < 4:
            continue
        signature = " ".join(sorted(set(claim_tokens)))
        if signature in seen:
            continue
        seen.add(signature)
        claims.append(chunk)
    return claims

def token_f1(left: str, right: str, features_module: Any) -> float:
    left_tokens = features_module.token_set(left)
    right_tokens = features_module.token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    precision = len(left_tokens & right_tokens) / len(left_tokens)
    recall = len(left_tokens & right_tokens) / len(right_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def score_components(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    strict_module: Any,
    features_module: Any,
    model_label: str,
    input_index: int,
) -> dict[str, Any]:
    benchmark_id = item_id(reference)
    answer = answer_text(prediction).strip()
    expected = str(reference.get("expected_answer") or "")
    keys = [str(value) for value in reference.get("scoring_key_points", [])]
    accepted = [
        str(value) for value in reference.get("accepted_equivalents", [])
    ]
    source_anchor = str(reference.get("source_anchor_for_judge_only") or "")
    support = "\n".join(
        part
        for part in (
            source_anchor,
            "Accepted equivalents: " + " | ".join(accepted),
        )
        if part.strip()
    )
    if len(keys) != 4:
        raise RuntimeError(f"expected four key points for {benchmark_id}")

    strict_row = strict_module.score_row(
        {
            "benchmark_id": benchmark_id,
            "core_id": reference.get("core_id"),
            "model_under_evaluation": model_label,
            "candidate_answer": answer,
            "reference_expected_answer": expected,
            "source_support": support,
            "scoring_key_points": keys,
        }
    )
    strict_score = float(strict_row["score"])
    semantic_score, rescue_reason, semantic_meta = unified_rescue_v1(
        strict_score,
        {**strict_row, "candidate_answer": answer},
        expected,
        keys,
        support,
        features_module,
    )

    gold_claims = (
        scientific_claims(expected, features_module)
        + keys
        + accepted
        + scientific_claims(source_anchor, features_module)
    )
    support_text = "\n".join(gold_claims)
    features = features_module.answer_features(
        answer,
        expected,
        keys,
        support_text,
    )
    key_overlaps = [
        float(row["semantic_overlap"])
        for row in features["semantic_key_details"]
    ]
    rubric_coverage = statistics.mean(
        clamp01(value / KEY_OVERLAP_FULL) for value in key_overlaps
    )
    answer_claims = scientific_claims(answer, features_module)
    per_claim_support = [
        clamp01(
            max(
                (
                    token_f1(claim, gold, features_module)
                    for gold in gold_claims
                ),
                default=0.0,
            )
            / CLAIM_F1_FULL
        )
        for claim in answer_claims
    ]
    raw_claim_precision = (
        statistics.mean(per_claim_support) if per_claim_support else 0.0
    )
    supported_claim_precision = clamp01(
        raw_claim_precision / CLAIM_PRECISION_FULL
    )
    semantic_adequacy = 25.0 * float(semantic_score)
    decision_quality = clamp01(key_overlaps[-1] / DECISION_OVERLAP_FULL)
    missing_key_count = sum(
        value < MISSING_KEY_THRESHOLD for value in key_overlaps
    )

    raw_v4 = 100 * (
        V4_WEIGHTS["semantic_adequacy"] * (semantic_adequacy / 100)
        + V4_WEIGHTS["rubric_coverage"] * rubric_coverage
        + V4_WEIGHTS["supported_claim_precision"] * supported_claim_precision
        + V4_WEIGHTS["decision_control"] * decision_quality
    )
    hard_gate = min(
        1.0,
        (semantic_adequacy / 100) / SEMANTIC_MASTERY_THRESHOLD,
    )
    soft_gate = math.sqrt(max(0.0, min(1.0, hard_gate)))
    cap = 100.0
    if missing_key_count == 1:
        cap = 85.0
    elif missing_key_count == 2:
        cap = 70.0
    elif missing_key_count >= 3:
        cap = 50.0
    final_v4_1 = min(raw_v4 * soft_gate, cap)

    return {
        "input_index": input_index,
        "benchmark_id": benchmark_id,
        "core_id": reference.get("core_id"),
        "ability_family": reference.get("ability_family"),
        "component_subset": reference.get("component_subset"),
        "model": model_label,
        "strict_score_0_4": strict_score,
        "unified_semantic_rescue_v1_score_0_4": float(semantic_score),
        "rescue_reason": rescue_reason,
        **semantic_meta,
        "unified_semantic_adequacy_0_100": round(semantic_adequacy, 6),
        "rubric_coverage_0_1": round(rubric_coverage, 6),
        "raw_claim_support_mean_0_1": round(raw_claim_precision, 6),
        "supported_claim_precision_0_1": round(
            supported_claim_precision,
            6,
        ),
        "decision_control_quality_0_1": round(decision_quality, 6),
        "missing_key_count": missing_key_count,
        "scientific_claim_count": len(answer_claims),
        "scientific_mastery_raw_before_gate_0_100": round(raw_v4, 6),
        "semantic_mastery_gate_0_1": round(hard_gate, 6),
        "scientific_mastery_v4_1_soft_gate_0_1": round(soft_gate, 8),
        "scientific_mastery_v4_1_missing_key_cap_0_100": cap,
        "scientific_mastery_v4_1_softgate_score_0_100": round(
            final_v4_1,
            6,
        ),
        "answer_word_count_diagnostic_only": len(answer.split()),
        "answer_length_used_in_score": False,
        "finish_reason": prediction.get("finish_reason"),
        "candidate_answer": answer,
        "reference_expected_answer": expected,
        "scoring_key_points": keys,
        "accepted_equivalents": accepted,
    }

V43_FIELD = 'scientific_mastery_v4_3_gatecap20_score_0_100'
COMPONENT_FIELDS = ('unified_semantic_adequacy_0_100','rubric_coverage_0_1','decision_control_quality_0_1','supported_claim_precision_0_1','semantic_mastery_gate_0_1','missing_key_count')

def scientific_mastery_v43(row):
    """Accepted gate-cap-20 formula, using recorded rounded component inputs."""
    values={key:float(row[key]) for key in COMPONENT_FIELDS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Score components must be finite')
    if not (0 <= values['unified_semantic_adequacy_0_100'] <= 100):
        raise ValueError('Semantic adequacy must be on 0--100')
    for key in COMPONENT_FIELDS[1:-1]:
        if not 0 <= values[key] <= 1: raise ValueError(f'{key} must be on 0--1')
    missing=int(values['missing_key_count'])
    if missing!=values['missing_key_count'] or not 0<=missing<=4:
        raise ValueError('Missing criterion count must be an integer from zero to four')
    additive=(values['unified_semantic_adequacy_0_100']+100*values['rubric_coverage_0_1']+100*values['decision_control_quality_0_1'])/3
    gate=math.sqrt(max(0,min(1,values['semantic_mastery_gate_0_1'])))
    precision=math.sqrt(max(0,min(1,values['supported_claim_precision_0_1']/0.75)))
    cap=85.0 if missing==1 else 70.0 if missing==2 else 50.0 if missing>=3 else 100.0
    return round(max(min(additive*gate*precision,cap),additive-20.0),6)

def score_answer_exact(prediction, reference, source_support):
    """Recompute components with user-supplied original judge support.

    An explicit support record is mandatory. Passing the public reference alone
    cannot reproduce the historical text-to-component calculation. An empty
    support string is allowed only when explicitly present in that record.
    The support is used in memory and is never returned in the score record.
    """
    bid=str(reference.get('benchmark_id') or reference.get('id') or '')
    if not bid: raise ValueError('Reference ID is required')
    if source_support is None or 'source_anchor_for_judge_only' not in source_support:
        raise ValueError('Exact scoring requires an explicit original source-support record')
    sid=str(source_support.get('benchmark_id') or source_support.get('id') or '')
    if sid != bid: raise ValueError('Source-support and reference IDs differ')
    pid=str(prediction.get('benchmark_id') or prediction.get('id') or '')
    if pid != bid: raise ValueError('Prediction and reference IDs differ')
    pred={**prediction,'candidate_answer':str(prediction.get('answer') or prediction.get('prediction_text') or prediction.get('candidate_answer') or '')}
    ref={**reference,'benchmark_id':bid,'ability_family':reference.get('task_family',reference.get('ability_family')),'source_anchor_for_judge_only':source_support['source_anchor_for_judge_only']}
    result=score_components(pred,ref,scoring_strict,scoring_features,str(prediction.get('model','user_model')),1)
    keep={key:result[key] for key in COMPONENT_FIELDS}
    keep.update(benchmark_id=bid,model=str(prediction.get('model','user_model')))
    keep[V43_FIELD]=scientific_mastery_v43(keep)
    return keep

def aggregate_component_rows(rows):
    """Recompute item scores and Huber/mean summaries from frozen components."""
    groups=defaultdict(list);seen=set();max_difference=0.0
    for row in rows:
        identity=(str(row['model']),str(row['benchmark_id']))
        if identity in seen: raise ValueError(f'Duplicate model/item pair: {identity}')
        seen.add(identity)
        score=scientific_mastery_v43(row)
        if V43_FIELD in row:
            difference=abs(score-float(row[V43_FIELD]));max_difference=max(max_difference,difference)
            if difference>2e-6: raise ValueError(f'Frozen item score mismatch: {identity}')
        groups[identity[0]].append(score)
    if not groups: raise ValueError('No component rows')
    return {'method':'Scientific Mastery v4.3 gatecap20; Huber c=1.345','max_item_difference':max_difference,'systems':{name:{'items':len(values),'arithmetic':statistics.mean(values),'huber':huber(values)['location']} for name,values in sorted(groups.items())}}

def anchored_pro_score(frozen_flash_score, paired_rows):
    """Use matched within-batch Pro minus Flash increments, per S20 condition."""
    if not paired_rows: raise ValueError('No matched S20 rows')
    conditions={r['condition'] for r in paired_rows}
    if len(conditions)!=1: raise ValueError('S20 conditions must be scored separately')
    ids=[r['id'] for r in paired_rows]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate S20 question IDs')
    increment=statistics.mean(float(r['pro_score'])-float(r['flash_score']) for r in paired_rows)
    return {'condition':next(iter(conditions)),'items':len(paired_rows),'frozen_flash_score':float(frozen_flash_score),'paired_increment':increment,'anchored_pro_score':float(frozen_flash_score)+increment}

def score_external_mcq(questions, predictions):
    """Exact match, no judge model. Requires a user-supplied upstream question file."""
    if len(questions)!=49: raise ValueError('Expected the 49-question Table S4 set, not the 197-question set')
    if len(predictions)!=49: raise ValueError('Expected 49 predictions')
    choices={int(row['source_index']):str(row['predicted_option']).strip().upper() for row in predictions}
    if set(choices)!=set(range(1,50)): raise ValueError('source_index must uniquely cover 1--49')
    correct=sum(choices[i]==str(q['correct_option']).strip().upper() for i,q in enumerate(questions,1))
    return {'items':49,'correct':correct,'accuracy':correct/49,'method':'exact_match_to_released_option'}

def human_family_agreement(component_rows, rating_rows):
    """Agreement across 16 system-by-family means, not 12,800 independent raters."""
    human=defaultdict(list);automatic=defaultdict(list)
    for row in rating_rows:
        human[(row['model'],row['task_family'])].append(float(row['human_criterion_score_0_100']))
    for row in component_rows:
        key=(row['model'],row['task_family'])
        if key in human: automatic[key].append(float(row[V43_FIELD]))
    if set(human)!=set(automatic):raise ValueError('Human/automatic family groups do not match')
    keys=sorted(human);x=[statistics.mean(human[k]) for k in keys];y=[statistics.mean(automatic[k]) for k in keys]
    def ranks(values):
        order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.0]*len(values);start=0
        while start<len(order):
            end=start+1
            while end<len(order) and values[order[end]]==values[order[start]]:end+=1
            for index in order[start:end]:out[index]=(start+1+end)/2
            start=end
        return out
    return {'groups':len(keys),'pearson_r':statistics.correlation(x,y),'spearman_rho':statistics.correlation(ranks(x),ranks(y)),'unit':'system_by_task_family_mean','means':[{'model':k[0],'task_family':k[1],'human_mean':x[i],'automatic_mean':y[i]} for i,k in enumerate(keys)]}
