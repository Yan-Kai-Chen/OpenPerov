# Evaluation

`python scripts/evaluate.py reproduce` recalculates the 11,200 PSM item scores and all 14 model aggregates from public component ledgers. It also reconstructs the two expert-comparison anchored Pro means and human/automatic agreement. The automatic protocol is Scientific Mastery v4.3 gate-cap-20 with Huber c=1.345, not a V5 scorer inferred from historical archive names.

The reproduced Flash/Pro Huber values are 87.5659/89.0539. Human/automatic agreement over 16 system-by-family means is Pearson r=0.707059 and Spearman rho=0.729412.

## Scoring new answers

```sh
python scripts/evaluate.py components --predictions outputs/flash.jsonl --references benchmarks/psm_bench/references.jsonl --source-support private/source_support.jsonl --output outputs/components.jsonl
```

The original deterministic component algorithm uses original source-support text in addition to public reference answers. This support must be supplied explicitly; it is not distributed. Missing support is an error. Public frozen component ledgers need no such private dependency for aggregation.

## Expert comparison

The two settings use separate scientific rubrics and archived blinded ratings with GPT-5.6 Sol xhigh assistance. Experts determined and checked the Expert + AI reference content; DeepSeek V4 Pro assisted wording/organization. The published Pro mean equals the frozen Flash mean plus the matched within-batch Pro-minus-Flash mean. The released baseline and paired files reproduce 88.2082 and 96.7500. The command does not generate new scientific judgments.

Preserve question pairing on PSM and study-level clustering on the later-study tasks. Statistical helpers document RNG/percentile conventions; archived intervals are not claimed to be bit-identical under a new RNG.

## External 49-question evaluation

```sh
python scripts/evaluate.py mcq --questions /path/to/Benchmark.json --predictions outputs/r1_predictions.jsonl
```

Obtain Perovskite-R1 Table S4 separately and retain its upstream license. The adapter checks the recorded question-file hash and exact option agreement. Its source/member metadata are in `benchmarks/external/perovskite_r1_adapter.json`. The nearby 197-question open-ended file is a different resource. Author-released predictions and OpenPerov local predictions retain distinct provenance.
