# OpenPerov evaluation data

PSM-Bench has 800 questions from 446 source studies: mechanism diagnosis368, stability-constrained design300, stability-failure analysis73, cross-constraint synthesis59. The recorded construction contains600 core questions and200 selected using model-informed discrimination. Selection history and article-level training exclusion are distinct dataset properties.

The later-study evaluation comprises10 studies x4 tasks under each of two conditions. Experimental context provides observations for mechanism diagnosis, a decisive experiment, an evidence update or a scientific decision. Limited context provides sparse information for mechanism diagnosis, competing mechanisms, an evidence update or a falsifiable prediction. The studies appeared after the recorded14April2026 literature cutoff.

There are880 task records in total. The3200 human-assessed answers refer to the same800PSM questions; they are not new questions. The49 external Perovskite-R1 MCQs remain separate.

## Data organization

Question files contain exact frozen text and stable IDs. Expert-comparison IDs combine condition, study and question because Q1--Q4 repeat across studies. Frozen prompts and their hashes are preserved; historical naming inside a prompt remains unchanged. References and criteria are stored separately from inference inputs.

`results/psm_bench/` contains final answers and numerical components. `results/expert_comparison/` contains answers, archived rubric ratings, baseline scores and paired increments. `results/human_assessment/` contains anonymous answer-level judgments and agreement summaries. No reviewer IDs or private blind keys are included.

Domain experts determined and checked scientific content in the Expert + AI reference condition, with AI assistance for wording. This is separate from the human assessment of model answers. PSM closed-book systems received no retrieved passages, whereas Pro used the curated evidence collection. Later-study target articles and SI were withheld under their separate protocol.

Original questions, references, annotations and project-generated results are CC BY4.0. Cite OpenPerov, retain the notice and identify modifications. The private training/retrieval corpus is not part of this dataset. Exact score reproduction from public components is distinct from recomputing source-dependent text components on new answers; see EVALUATION.md.
