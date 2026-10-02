"""Frozen scientific task prompts; no reference answers enter inference."""
SYSTEM_PROMPT = (
    "You are a senior perovskite photovoltaics scientist taking a closed-book "
    "expert examination. Use only scientific knowledge and the information in "
    "the question. Think privately, but write only the final answer. Do not "
    "reveal chain-of-thought, hidden notes, article guesses, benchmark "
    "discussion, or scoring discussion."
)

ROUTE_PROMPTS = {
    "mechanism_diagnostic": (
        "Write a direct 500-700 word expert answer in English. Your answer must "
        "maximize scientific coverage, not style. First state the most likely "
        "diagnosis; then explain the causal mechanism step by step; then compare "
        "at least two alternative explanations or artifacts; then specify "
        "decisive experiments and controls. For each observation named in the "
        "question, say what physical process it implies and what it does not "
        "prove by itself. Connect optical, structural, compositional, electrical, "
        "kinetic, or equivalent-circuit evidence to the same mechanism instead "
        "of listing methods. Include negative controls, matched devices or "
        "films, orthogonal validation, and at least one falsification test. "
        "Distinguish measurement artifacts from real bulk, interface, or "
        "transport processes. Avoid vague review-style paragraphs. Do not "
        "invent exact paper-specific numbers. Answer every clause of the question."
    ),
    "stability_design_transfer": (
        "Write a direct 500-700 word expert answer in English. Your answer must "
        "maximize scientific coverage, not style. Translate the proposed "
        "composition, additive, interface, process, or device idea into a "
        "qualification plan. Explicitly cover four things: the target benefit; "
        "the low-dose or low-intensity failure mode; the high-dose or "
        "high-intensity failure mode; and the stress-test evidence needed before "
        "scale-up. Explain the physical causal chain through defects, ion "
        "migration, phase segregation, crystallization, interfacial energetics, "
        "transport barriers, parasitic absorption, mechanical or thermal stress, "
        "and measurement artifacts where relevant. Do not merely list tests: "
        "for each control or measurement, state the decision it supports and "
        "what result would reject the design. Include matched controls, spatial "
        "uniformity, batch statistics, low-performing-tail analysis, stabilized "
        "power output, and combined light/heat/bias/humidity or oxygen stress as "
        "appropriate. End with practical go/no-go criteria for large-area or "
        "tandem transfer when the question implies deployment. Do not invent "
        "exact paper-specific numbers. Answer every clause of the question."
    ),
    "design_transfer_synthesis": (
        "Write a direct 500-700 word expert answer in English. Convert every "
        "constraint in the question into a coherent material, interface, process, "
        "device, or module design rather than discussing options generically. "
        "Begin with the recommended architecture or process sequence and explain "
        "the physical purpose of each element. Trace how the design affects "
        "energetics, defects, crystallization, transport, recombination, optical "
        "loss, mechanical integrity, and stability where relevant. Explicitly "
        "discuss at least one credible alternative and the main tradeoffs or "
        "failure boundaries. Then define matched controls, scalable fabrication "
        "checks, spatial uniformity and batch statistics, low-performing-tail "
        "analysis, and measurements that demonstrate that each stated constraint "
        "has been met. Give quantitative-style acceptance logic without inventing "
        "unsupported paper-specific numbers, and end with practical go/no-go "
        "criteria for transfer to the requested area, stack, operating condition, "
        "or manufacturing route. Answer every clause of the question."
    ),
    "stability_failure": (
        "Write a direct 500-700 word expert answer in English. Start with the most "
        "defensible time-ordered failure sequence and identify the state variables "
        "that are reversible, slowly relaxing, or irreversibly ratcheting. Connect "
        "the applied light, heat, bias, humidity, oxygen, mechanical, chemical, or "
        "cycling stress to defects, ion redistribution, phase behavior, interface "
        "reactions, transport loss, and the final device metric. Compare at least "
        "two competing failure routes and state the distinct observable signature "
        "of each. Design matched experiments that separate total dose from cycle "
        "count, elapsed time, temperature, bias history, encapsulation or ingress, "
        "and starting-device variation as appropriate. Include time-resolved and "
        "spatially resolved measurements, destructive and nondestructive "
        "cross-checks, controls, and at least one falsification test. Explain what "
        "outcome would establish the dominant route and what would reject it, then "
        "give a mitigation or qualification criterion. Do not invent exact "
        "paper-specific numbers. Answer every clause of the question."
    ),
}

def question_record(row):
    qid = row.get("benchmark_id") or row.get("id") or row.get("question_id") or row.get("query_id")
    question = row.get("question") or row.get("closedbook_question") or row.get("question_text") or row.get("user_prompt")
    if not qid or not isinstance(question, str) or not question.strip():
        raise ValueError("Each input needs an ID and a nonempty question")
    family = row.get("task_family") or row.get("ability_family")
    return str(qid), question.strip(), family

def messages(question, family=None):
    if family and family not in ROUTE_PROMPTS:
        raise ValueError(f"Unknown task family: {family}")
    instruction = ROUTE_PROMPTS.get(family, "Write a clear, scientifically grounded final answer in English. Address the question and explain the mechanisms and experimental decisions involved. Do not invent paper-specific results.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question + "\n\n" + instruction}]


def condition_id(row):
    """Accept published S20 condition names without modifying frozen prompts."""
    raw = row.get("condition_id") or row.get("condition")
    mapping = {None: None, "E": "E", "S": "S",
               "experimental_context": "E", "limited_context": "S"}
    if raw not in mapping:
        raise ValueError(f"Unsupported S20 condition: {raw}")
    return mapping[raw]


def baseline_answers(rows, system="OpenPerov Flash"):
    """Select a named system before indexing a multi-system public ledger."""
    selected = [r for r in rows if (r.get("model") or r.get("system")) == system]
    if not selected:
        raise ValueError(f"No baseline answers for {system!r}; pass the exact --baseline-system label")
    answers = {}
    for row in selected:
        qid = str(row.get("benchmark_id") or row.get("id") or row.get("question_id") or row.get("query_id") or "")
        if not qid or qid in answers:
            raise ValueError("Selected baseline IDs must be nonempty and unique")
        answer = row.get("answer") or row.get("prediction_text") or row.get("candidate_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Selected baseline answer is empty")
        answers[qid] = answer
    return answers
