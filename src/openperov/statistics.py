"""Robust aggregation copied from the accepted v4.3 implementation."""
from __future__ import annotations
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import random

MAD_NORMALIZER = 0.6744897501960817

HUBER_C = 1.345

def huber(values: list[float]) -> dict[str, float | int]:
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    scale = mad / MAD_NORMALIZER
    if scale <= 1e-12:
        return {"location": statistics.mean(values), "robust_scale": 0.0, "downweighted": 0}
    for _ in range(1000):
        weights = []
        for value in values:
            standardized = abs((value - center) / scale)
            weights.append(1.0 if standardized <= HUBER_C else HUBER_C / standardized)
        updated = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)
        if abs(updated - center) < 1e-12:
            center = updated
            break
        center = updated
    downweighted = sum(abs((value - center) / scale) > HUBER_C for value in values)
    return {"location": center, "robust_scale": scale, "downweighted": downweighted}

def paired_bootstrap(left, right, iterations=10000, seed=20260903, statistic='mean'):
    """Paired item bootstrap. Keep question pairing and record this RNG profile.

    This portable helper uses Python random.Random and nearest-rank interval
    endpoints. Historical saved intervals remain separately labeled; different
    RNG/percentile implementations need not reproduce their final digits.
    """
    if len(left)!=len(right) or not left: raise ValueError('Nonempty paired arrays of equal length required')
    if iterations<40: raise ValueError('At least 40 replicates required')
    estimator=statistics.mean if statistic=='mean' else lambda x: huber(x)['location']
    if statistic not in ('mean','huber'): raise ValueError('Unknown statistic')
    rng=random.Random(seed);n=len(left);samples=[]
    for _ in range(iterations):
        idx=[rng.randrange(n) for _ in range(n)]
        samples.append(estimator([left[i] for i in idx])-estimator([right[i] for i in idx]))
    samples.sort()
    return {'difference':estimator(left)-estimator(right),'ci95':[samples[max(0,math.ceil(.025*iterations)-1)],samples[max(0,math.ceil(.975*iterations)-1)]],'iterations':iterations,'seed':seed,'unit':'paired_question','rng':'python_random','interval':'nearest_rank'}

def study_paired_bootstrap(rows, iterations=10000, seed=20260913):
    """Resample study means, keeping all matched tasks from a study together."""
    conditions={r['condition'] for r in rows}
    if len(conditions)!=1: raise ValueError('Do not pool S20 conditions')
    studies=defaultdict(list)
    for row in rows: studies[row['study_id']].append(float(row['pro_score'])-float(row['flash_score']))
    means=[statistics.mean(v) for _,v in sorted(studies.items())]
    result=paired_bootstrap(means,[0.0]*len(means),iterations,seed)
    result.update(unit='study',studies=len(studies),condition=next(iter(conditions)))
    return result
