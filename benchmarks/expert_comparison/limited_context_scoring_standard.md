# S20 Sparse-Information Perovskite Expert Extension — Frozen Scoring Standard v1

## Construct

This extension evaluates whether a model can use internalized perovskite-domain knowledge to reason scientifically when the target paper, supplementary information, retrieval context, full experimental history and published conclusion are unavailable. The model receives only a deliberately incomplete but scientifically interpretable problem state.

This is an extension stress test and does not replace the main S20 evidence-conditioned challenge.

## Content-only scoring rule

Score scientific substance only. Do not reward or penalize answer length, headings, formatting, prose style, number of paragraphs, resemblance to a reference answer, or whether the answer presents one or several scientifically justified possibilities. A response may receive full credit through a mechanism, terminology or experimental strategy different from the reference material if it is scientifically defensible.

Do not penalize calibrated uncertainty. Under sparse information, explicitly distinguishing what is likely, possible and unresolved is evidence of expertise. Do not require the model to guess a hidden paper-specific fact.

Operational preferences such as choosing exactly one intervention, using a continue/revise/stop template, or prescribing a unique device layer are not scoring obligations. Multiple hypotheses or complementary measurements are acceptable when scientifically motivated. They should reduce the score only if they are internally contradictory, physically implausible, or prevent meaningful causal discrimination.

## Six expert dimensions

Each answer is rated from 0 to 4 on every dimension. The normalized score is the weighted sum below.

| Dimension | Weight | What is evaluated |
|---|---:|---|
| Perovskite mechanistic validity and domain specificity | 30% | Correct use of perovskite chemistry, photophysics, interfaces, phase behaviour, ion migration, degradation or processing knowledge relevant to the problem. |
| Sparse-evidence integration | 20% | Whether the limited observations are connected causally without inventing unavailable results. |
| Competing-mechanism discrimination | 20% | Whether plausible alternatives, artefacts and confounders are recognized and meaningfully distinguished. |
| Measurement-to-mechanism mapping | 15% | Whether proposed measurements, controls or predicted signatures can actually inform the scientific mechanism. |
| Uncertainty and claim calibration | 10% | Whether conclusions remain bounded by the information available and avoid false certainty. |
| Falsifiability and scientific usefulness | 5% | Whether the answer yields testable predictions or useful next scientific reasoning. |

Rating anchors:

- 4: expert-level, scientifically correct, domain-specific and well calibrated;
- 3: strong, with only limited omissions or ambiguity;
- 2: partly useful but incomplete, generic or weakly discriminating;
- 1: major scientific weakness, unsupported extrapolation or substantial confusion;
- 0: absent, fundamentally incorrect or incompatible with the stated evidence.

## Later-paper comparison

The post-freeze paper is an independent scientific comparator, not a single canonical answer key. Record core-finding reconstruction separately as:

- aligned;
- partially aligned;
- scientifically plausible alternative;
- unsupported or contradicted.

A plausible alternative must not be scored as wrong merely because the later paper pursued a different experimental route.

## Aggregation

- Primary endpoint: mean content-only expert score, first averaged within paper and then across papers.
- Statistical unit: target paper.
- Uncertainty: paper-cluster bootstrap.
- Secondary endpoints: six dimension scores, paper-level win/tie/loss and core-finding reconstruction category.

## Blinding

Candidate identity, model identity and prior scores must be hidden from the scorer. Both candidates must be scored under the same scientific standard.
