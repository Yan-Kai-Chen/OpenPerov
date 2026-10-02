# OpenPerov

### Expert language models for perovskite photovoltaics

OpenPerov connects expert-calibrated literature learning, scientific reasoning and materials decisions in perovskite photovoltaics. Flash is the adapted Qwen3.6-27B answer model. Pro adds evidence retrieval, learned ranking and local answer revision.

**Code and evaluation release v0.1.0rc1.** Research resources for *Evidence-grounded language models guide perovskite interface design*, by Yankai Chen, Zhi Wan and Tao Jing.

This repository provides OpenPerov training and inference code, PSM-Bench, expert-comparison tasks and reproducible evaluation results. Model weights are prepared for a separate release; download links will be added when available.

## Resources

| Resource | Contents |
|---|---|
| PSM-Bench | 800 frozen questions, original references and criteria |
| Expert comparison | 40 experimental-context and 40 limited-context tasks and their frozen prompts |
| Model evaluation | 11,200 final answers and component records across 14 systems |
| Human assessment | 3,200 anonymous answer records and 12,800 criterion judgments |
| Expert-comparison results | 240 answers, rubric ratings, baseline and paired scores |
| Code | Inference, training/merging, retrieval/revision, learned ranking, scoring and statistics |

The literature and training corpora, article extractions, private source-support passages, indexes and evidence packets are excluded. Third-party Perovskite-R1 questions are obtained separately through the documented input adapter.

## Quick verification

From this repository root, using Python 3.10 or later:

```sh
python -m pip install --no-deps .
python scripts/evaluate.py reproduce
python -m unittest discover -s tests -v
python scripts/validate_release.py
```

These offline commands use no model weights or network services. The evaluation recovers Flash **87.5659**, Pro **89.0539**, and the two expert-comparison Pro means **88.2082 / 96.7500**. It reproduces frozen scores, not new model generation or scientific judging.

## Flash inference

```sh
python -m pip install '.[inference]'
python scripts/run_inference.py --model /path/to/OpenPerov-Flash-27B --questions benchmarks/psm_bench/questions.jsonl --config configs/flash.json --output outputs/flash.jsonl
```

The same entrypoint accepts either expert-comparison JSONL and preserves its system/user prompts. An OpenAI-compatible endpoint can be supplied explicitly with `--endpoint`; no service is configured by default. Nonempty answers reaching the output limit are retained according to the evaluation protocol.

## Pro with user-supplied evidence

```sh
python scripts/run_pro.py --model /path/to/OpenPerov-Flash-27B --questions benchmarks/psm_bench/questions.jsonl --baseline results/psm_bench/predictions.jsonl --evidence /path/to/your/evidence.jsonl --config configs/pro_psm.json --output outputs/pro.jsonl
```

The released multi-system baseline is filtered to OpenPerov Flash. The portable candidate pool uses BM25. Supply the selector model/adapter and reranker model/adapter paths together to use both learned ranking stages. Lexical-only and learned-ranking runs are labeled separately. The synthetic example in `examples/` illustrates the evidence schema.

The paper's original private retrieval collection is not included. Reference retrieval tools and the original S20 G0--G7 verification/assembly implementation are provided separately from the portable runtime. The two-edit portable S20 profile does not claim to regenerate the original private-evidence run.

## Training and weights

Flash uses **Broad-DAPT -> scientific instruction -> answer-style tuning**, without PF6. `scripts/train.py` supports these stages and the selector/reranker stages; the matching JSON configurations accept user-owned data. `scripts/merge.py` supports sequential adapter merging. The separate weight materials provide the exact three-stage reconstruction order. Stage5 alone must not be applied to the original base as if it were the complete model.

The weight release will include the final merged Flash model and the learned Pro retrieval adapters. Weight downloads will be hosted separately from this code repository. The literature and training corpora remain private; the released training tools accept user-supplied data.

## Documentation and licensing

See [evaluation](docs/EVALUATION.md), [dataset card](docs/DATA_CARD.md), [runtime](docs/ENVIRONMENT.md) and [public boundary](docs/PUBLIC_BOUNDARY.md). Code is Apache-2.0; original evaluation data and project-generated evaluation outputs are CC BY 4.0. Third-party models/data retain their notices. Use [CITATION.cff](CITATION.cff) for attribution. `release_manifest.json` records the released file contents and SHA-256 hashes.
