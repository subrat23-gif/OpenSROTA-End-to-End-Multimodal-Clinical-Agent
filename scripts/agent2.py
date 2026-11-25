"""
Local clinical agent:
    transcript -> Qwen3 -> validated ClinicalState -> deterministic SOAP

Uses direct Transformers inference on the local GPU.
No Ollama and no vLLM required.

Example:
    python agent.py transcript.json \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --mode full \
        --out state.json
"""

from __future__ import annotations

import os
import argparse
import difflib
import json
import re
import time
from typing import List

from schema2 import (
    FIELDS,
    Attribute,
    ClinicalState,
    Extraction,
    StoredFact,
    Turn,
    Assertion,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """
You extract clinical facts from synthetic clinical utterances.

IMPORTANT DATASET CONTEXT:
- The input is primarily synthetic clinical text, not necessarily a genuine doctor-patient conversation.
- The speaker label may be "Patient" because the audio pipeline uses a single-speaker mode.
- DO NOT assume that every clinical statement is patient-reported merely because the speaker is labelled Patient.
- Determine the clinical meaning and source from the actual wording of the utterance.
- Labels such as "Premise", "Prompting", and "Hypothesis" are part of the input and are semantically meaningful.
- These labels describe the structure or role of the statement. They are NOT clinical source labels.

CORE RULES:
- Extract ONLY clinical information explicitly supported by the input.
- Never invent a diagnosis, cause, test, treatment, or clinical relationship.
- Preserve the wording and meaning of the source.
- Do not silently convert hypothetical statements into present or absent facts.
- Do not merge two distinct clinical facts merely because they are related.
- Do not infer information that is only medically plausible but not explicitly stated.
- If the text is ambiguous, use model_uncertain rather than guessing.

QUESTIONS:
- A question in the source is NOT itself a clinical fact, investigation, diagnosis, or plan.
- Do not convert educational/test questions into clinical facts.
- For example, "Occlusion of which vessel is the most likely cause?" must NOT become an investigation or diagnosis.
- Extract only facts explicitly stated in the question's surrounding clinical description.
- Do not treat "most likely" questions as evidence that the proposed condition is true.
- Do not extract answer choices or implied answers unless the source explicitly states the answer.

CLINICAL TERM MENTIONS AND COMPARISONS:
- Do not extract a clinical term as an investigation merely because the term is mentioned.
- A test, laboratory marker, imaging study, or procedure should be extracted as an investigation only when the text explicitly states that the test/procedure was performed, obtained, ordered, planned, or that its result is being reported.
- Do NOT extract tests or biomarkers mentioned only as part of a comparison, discussion, educational statement, or question.
- Examples:
  - "Comparison of elastase, amylase, lipase, and trypsin in the diagnosis of acute pancreatitis."
    -> Do NOT extract elastase, amylase, lipase, or trypsin as patient investigations.
  - "The patient's lipase was 450 U/L."
    -> extract lipase as an investigation/result.
  - "Lipase was ordered."
    -> extract lipase as an investigation with source = explicit_plan.
  - "The patient underwent a CT scan."
    -> extract CT scan as an investigation.
  - "Which test is more useful for diagnosing pancreatitis?"
    -> do NOT extract the mentioned tests as patient investigations.
- Mentioning a disease or diagnostic condition in an educational/question context does not mean the patient has that condition.

ASSERTION:
- present = explicitly stated as true/current.
- absent = explicitly denied or explicitly stated not to exist.
- uncertain = possible, suspected, might, may, unclear, or otherwise uncertain.
- hypothetical = explicitly introduced as a hypothesis, hypothetical statement, or hypothetical scenario.

IMPORTANT LABEL HANDLING:
- "Premise" does NOT mean hypothetical. Extract the stated clinical fact normally.
- "Prompting" indicates a stated action, relation, request, or clinical event. Extract only what is explicitly stated without inventing additional details.
- "Hypothesis" means the associated statement should normally have assertion = hypothetical.
- The words "Premise", "Prompting", and "Hypothesis" themselves should NOT be extracted as clinical facts.
- Do not use "premise", "prompting", or "hypothesis" as values for the source field.

SOURCE:
The source field describes WHO or WHAT clinically supports the extracted fact.

The source field MUST be exactly one of:
- patient_reported
- doctor_observed
- doctor_assessed
- explicit_plan
- model_uncertain

Never output any of the following as a source value:
- premise
- prompting
- hypothesis

Interpret the dataset labels separately:

- Premise → stated clinical information; determine the source from the actual wording.
- Prompting → a stated action/relation/request; determine the appropriate clinical source from the wording.
- Hypothesis → the statement is hypothetical; set assertion = "hypothetical" and determine the source from the wording.



SOURCE DEFINITIONS:
- patient_reported = the patient explicitly reports, describes, experiences, denies, or provides the information.
- doctor_observed = a clinician explicitly describes an examination finding, measured observation, vital sign, physical finding, or directly observed clinical finding.
- doctor_assessed = a clinician explicitly states an assessment, diagnosis, impression, interpretation, or clinical judgment.
- explicit_plan = a clinician explicitly states an order, advice, treatment plan, follow-up, referral, procedure plan, or other intended clinical action.
- model_uncertain = use when the wording does not support a more specific source.

INVESTIGATIONS AND TEST RESULTS:
- Distinguish between a planned investigation and a completed investigation/result.
- explicit_plan is ONLY for an explicitly stated future or intended clinical action.
- Do NOT use explicit_plan for a test, procedure, biopsy, imaging study, or laboratory result that has already occurred or whose result is already stated.
- Examples:
  - "Order a biopsy." -> investigation, source = explicit_plan.
  - "The patient will undergo a biopsy." -> investigation, source = explicit_plan.
  - "A biopsy specimen shows sheets of lymphocytes..." -> investigation/result, NOT explicit_plan.
  - "Serology for Epstein-Barr virus is positive." -> investigation/result, NOT explicit_plan.
- A laboratory result, pathology result, imaging result, or completed procedure is not automatically doctor_observed.
- doctor_observed should primarily be used for directly observed physical examination findings or measurements.
- If the schema does not provide a dedicated source for laboratory/pathology results and the wording does not establish who interpreted or reported the result, use model_uncertain rather than inventing a source.

FIELD PLACEMENT:
- examination = physical examination findings and directly observed clinical findings.
- investigations = tests, laboratory studies, imaging, biopsies, diagnostic procedures, and their explicitly stated results.
- assessment = clinician-stated diagnoses, impressions, interpretations, or clinical assessments.
- advice = explicitly stated clinical advice or instructions.
- follow_up = explicitly stated follow-up arrangements or timing.
- referrals = explicitly stated referrals.
- Do not place an investigation result in assessment unless the text explicitly presents it as a clinician's assessment or interpretation.
- Do not place a completed investigation or test result in advice or explicit_plan.

IMPORTANT SOURCE RULE:
- DO NOT infer doctor/patient attribution from the speaker label alone.
- A speaker labelled "Patient" does NOT automatically mean patient_reported.
- Use the semantic wording of the utterance to determine the source.
- For example, "fetal heart rate noted to be depressed" is an observation and may be doctor_observed even if the speaker label is "Patient".
- "Prompting delivery by cesarean section" may indicate an explicit clinical plan if the wording clearly describes a planned clinical action.
- If the wording does not provide enough information to determine the source, use model_uncertain.

EXAMPLES:
- "Premise, patient has diabetes."
  -> diabetes, assertion = present

- "Premise, fetal heart rate noted to be depressed."
  -> depressed fetal heart rate, assertion = present, source = doctor_observed

- "Prompting delivery by cesarean section."
  -> delivery by cesarean section, extract only the explicitly stated action/relation; do not invent additional details

- "Hypothesis, delivery was via C-section."
  -> delivery via C-section, assertion = hypothetical

- "Patient denies asthma."
  -> asthma, assertion = absent, source = patient_reported

- "Patient might have pneumonia."
  -> pneumonia, assertion = uncertain, source = patient_reported

- "The examination revealed right-sided weakness."
  -> right-sided weakness, assertion = present, source = doctor_observed

- "The clinician suspects pneumonia."
  -> pneumonia, assertion = uncertain, source = doctor_assessed

- "The patient should return in two weeks."
  -> follow-up in two weeks, source = explicit_plan

EXPERIENCER:
- patient = the patient/person described by the statement.
- family = a family member is explicitly described as having the condition.
- other = another explicitly identified person/entity.
- Never assign family merely because a condition is mentioned in a family-history context.
- Preserve explicit family history as family experiencer.
- Do not assume the patient has a condition merely because a family member has it.

EVIDENCE:
- Every fact needs at least one evidence item.
- turn_id must refer to an actual input turn.
- quote must be copied VERBATIM from that turn.
- quote must be sufficient to support the extracted fact.
- Do not use a fragment that omits the key clinical concept.
- Evidence should contain enough surrounding wording to establish the meaning, assertion, and source when possible.
- Do not use evidence from a different turn unless that turn is actually needed.
- Never invent evidence.
- Never modify, paraphrase, correct, or clean up the evidence quote.

NEGATION:
- "no history of X", "denies X", "without X" -> X is absent.
- Do not treat a negated fact as present.
- Preserve the distinction between an explicitly absent condition and a condition that is simply not mentioned.

HYPOTHETICAL AND UNCERTAIN LANGUAGE:
- "hypothesis", "hypothetical", "if", "assuming", or explicitly hypothetical scenarios -> assertion = hypothetical.
- "possible", "suspected", "might", "may", "could", "likely", "unclear" or equivalent uncertainty -> assertion = uncertain unless the wording explicitly establishes a hypothetical scenario.
- Do not convert uncertainty into present.
- Do not convert a hypothesis into present.
- Do not treat a question containing "most likely" as evidence that the proposed diagnosis is true.

CLINICAL DETAILS:
- Preserve medication dosage, frequency, route, duration, and other explicit attributes.
- Preserve family history as family experiencer.
- Preserve vitals and examination findings when explicitly stated.
- Preserve explicit temporal information when present.
- Preserve explicit relationships between clinical entities.
- Normal findings may be extracted when clinically meaningful and explicitly stated.
- Do not invent missing units, dosages, frequencies, diagnoses, causes, or relationships.
- Do not "correct" ASR wording unless the intended clinical meaning is explicitly supported by the surrounding text.

DUPLICATION:
- Do not intentionally output the same clinical fact multiple times.
- If the same fact appears repeatedly in the input, return one fact with combined evidence when appropriate.
- Keep genuinely distinct facts separate even if they have similar names.
- Do not merge facts that differ in assertion, experiencer, or clinical meaning.

EMPTY EXTRACTION:
- If there is no explicit clinical information, return empty lists for the relevant fields.
- If explicit clinical information exists, do NOT return an entirely empty ClinicalState.
- Do not manufacture facts merely to avoid an empty field.

RESEARCH / EDUCATIONAL / EXPERIMENTAL TASKS:
- Do not treat a research experiment, cognitive task, behavioral task, educational
  exercise, benchmark task, or study procedure as a patient clinical investigation
  merely because it uses clinical terminology.
- Examples include dot-probe tasks, comparison studies, experimental protocols,
  educational questions, and research measurements.
- Extract an investigation only when the text explicitly describes a clinical test,
  examination, imaging study, laboratory test, pathology study, or clinical procedure
  performed on or planned for the patient/person.

OUTPUT:
- Return JSON only.
- Do not include Markdown.
- Do not include explanations outside the JSON.
- Use empty lists for fields with nothing to extract.
- Follow the provided JSON schema exactly.
- Every enum value must exactly match the allowed schema value.
- The "source" field must always be one of:
  patient_reported,
  doctor_observed,
  doctor_assessed,
  explicit_plan,
  model_uncertain
- Never output "premise", "prompting", or "hypothesis" as a source value.
"""

SYSTEM_FT = """Extract clinical facts from the transcript into JSON.
Fields: complaints, symptoms, history, medications, allergies, vitals, examination, assessment, investigations, advice, follow_up, referrals.
Each fact: name, attributes [{key,value}], assertion (present|absent|uncertain|hypothetical), experiencer (patient|family|other),
source (patient_reported|doctor_observed|doctor_assessed|explicit_plan|model_uncertain), status, evidence [{turn_id, quote}].
Rules: extract only explicitly stated information; quotes are verbatim from the cited turn (<= 25 words);
never put negation in the name (use assertion=absent); never turn exam-question text or answer options into facts;
use model_uncertain when the source is unclear; return JSON only, empty lists for empty fields."""

# ============================================================
# DIRECT TRANSFORMERS / QWEN BACKEND
# ============================================================

class TruncatedOutput(ValueError):
    """Generation hit max_new_tokens before emitting EOS (greedy decoding => retrying identically is useless)."""
    def __init__(self, text: str):
        super().__init__("generation truncated at max_new_tokens")
        self.text = text


class TransformersLLM:

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 4096,
        adapter: str | None = None,
        schema_in_prompt: bool = True,
        system: str | None = None,
        verbose: bool = False,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.schema_in_prompt = schema_in_prompt
        self.system = system
        self.verbose = verbose
        self.salvage_count = 0

        print(f"\nLoading model: {model_name}")
        print("Using direct Transformers inference on GPU...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            print(f"Loaded LoRA adapter: {adapter}")

        self.model.eval()
        print("Model loaded successfully.")

        # A silent CPU/disk offload makes generation 100-1000x slower.
        dmap = getattr(self.model, "hf_device_map", None) or {}
        if any(str(v) in ("cpu", "disk") for v in dmap.values()):
            print("!!! WARNING: some layers are offloaded to CPU/disk -> extremely slow. "
                  "Free GPU memory (check nvidia-smi for other processes).")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def json(self, system: str, user: str, schema: dict, max_new_tokens: int | None = None) -> dict:

        import torch

        budget = max_new_tokens or self.max_new_tokens
        content = user
        if self.schema_in_prompt:
            content += (
                "\n\nReturn JSON matching this schema exactly:\n"
                + json.dumps(schema, ensure_ascii=False)
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        # Qwen3 supports disabling thinking for structured extraction.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        t0 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                use_cache=True,
            )

        generated = outputs[0, input_length:]
        n_new = int(generated.shape[-1])
        dt = time.time() - t0

        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        truncated = n_new >= budget and int(generated[-1]) not in eos

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        print(f"  [gen] prompt_tok={input_length} new_tok={n_new} {n_new / max(dt, 1e-6):.1f} tok/s"
              + (" TRUNCATED" if truncated else ""))
        if self.verbose:
            print("\n========== RAW OUTPUT ==========")
            print(text)
            print("================================\n")

        if truncated:
            raise TruncatedOutput(text)

        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict:

        text = text.strip()

        # Remove markdown code fences.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Model did not return valid JSON.\nRaw output:\n{text[:500]}")


def repair_truncated_json(text: str) -> dict | None:
    """Salvage every COMPLETE fact from a truncated Extraction JSON.

    Scans the text tracking strings/brackets, remembers the last position where a whole fact
    object (or a whole field array) had just been closed, cuts there and closes the open brackets.
    Returns None if nothing usable.
    """
    start = text.find("{")
    if start == -1:
        return None
    stack, in_str, esc, best = [], False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append(ch)
        elif ch in "}]":
            if not stack: break
            stack.pop()
            if len(stack) <= 2:           # just closed a fact object (depth 2) or a field array (depth 1)
                best = (i, list(stack))
    if best is None:
        return None
    pos, st = best
    cand = text[start:pos + 1] + "".join("}" if c == "{" else "]" for c in reversed(st))
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


# ============================================================
# TURN FORMATTING
# ============================================================

def fmt_turns(turns: List[Turn]) -> str:
    return "\n".join(
        f"[T{t.id}] {t.speaker}: {t.text}"
        for t in turns
    )


# ============================================================
# MODEL CALL + VALIDATION
# ============================================================


def sanitize_llm_output(data: dict) -> dict:
    """Fixes common LLM formatting hallucinations before Pydantic validation."""
    fields = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
              "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]
    
    if not isinstance(data, dict):
        return {}
        
    for field in fields:
        val = data.get(field, [])
        
        # Fix 1: Handle if the model outputted a single string instead of a list
        if isinstance(val, str):
            data[field] = [{"name": val}] if val.strip() else []
        elif isinstance(val, dict):
            data[field] = [val]
        elif isinstance(val, list):
            clean_list = []
            for fact in val:
                if isinstance(fact, str):
                    clean_list.append({"name": fact})
                elif isinstance(fact, dict):
                    # Fix 2: "value" -> "name"
                    if "value" in fact and "name" not in fact:
                        fact["name"] = fact.pop("value")
                    
                    # Fix 3: "T1" -> 1
                    evs = fact.get("evidence", [])
                    if isinstance(evs, dict): evs = [evs]
                    for ev in evs:
                        if isinstance(ev, dict):
                            tid = ev.get("turn_id")
                            if isinstance(tid, str):
                                digits = re.sub(r"\D", "", tid)
                                if digits: ev["turn_id"] = int(digits)
                    fact["evidence"] = evs
                    clean_list.append(fact)
            data[field] = clean_list
            
    return data

def _call(
    llm,
    user: str,
    retries: int = 2,
    max_budget: int = 8192,
) -> Extraction:

    schema = Extraction.model_json_schema()
    system = getattr(llm, "system", None) or SYSTEM
    budget = getattr(llm, "max_new_tokens", 4096)
    last, last_text = None, None

    for attempt in range(retries + 1):

        try:
            try:
                result = llm.json(system, user, schema, max_new_tokens=budget)
            except TypeError:                      # llm without per-call budget (mocks etc.)
                result = llm.json(system, user, schema)
            return Extraction.model_validate(sanitize_llm_output(result))

        except TruncatedOutput as e:
            # Greedy decoding: an identical retry gives an identical truncation. Give it more room instead.
            last, last_text = e, e.text
            budget = min(budget * 2, max_budget)
            print(f"Extraction attempt {attempt + 1}/{retries + 1} truncated -> retry with max_new_tokens={budget}")

        except Exception as e:
            last = e
            print(f"Extraction attempt {attempt + 1}/{retries + 1} failed: {str(e)[:200]}")
            if attempt < retries:
                user = (
                    user
                    + "\n\nIMPORTANT: Your previous response was invalid. "
                      "Return ONLY valid JSON matching the schema. "
                      "Do not return explanations or reasoning."
                )

    # Last resort: keep every complete fact from the final truncated output, and count it.
    if last_text:
        data = repair_truncated_json(last_text)
        if data is not None:
            try:
                ext = Extraction.model_validate(data)
                llm.salvage_count = getattr(llm, "salvage_count", 0) + 1
                print("  !! salvaged complete facts from truncated output (file is INCOMPLETE)")
                return ext
            except Exception:
                pass

    raise RuntimeError(f"extraction failed after retries: {str(last)[:300]}")


# ============================================================
# STATE MERGING
# ============================================================

def _key(name: str) -> str:
    return re.sub(
        r"[^a-z0-9 ]",
        "",
        name.lower(),
    ).strip()


def _fact_key(f):
    # Assertion is part of identity: "chest pain"/present and "chest pain"/absent are DIFFERENT facts.
    # (Old code collapsed them and silently kept only one -> hidden assertion flips.)
    return (_key(f.name), f.experiencer, f.assertion)


def _absorb(hit: StoredFact, new: StoredFact) -> None:
    attrs = {a.key: a.value for a in hit.attributes}
    attrs.update({a.key: a.value for a in new.attributes})
    hit.attributes = [Attribute(key=k, value=v) for k, v in attrs.items()]
    if new.status:
        hit.status = new.status
    seen = {(e.turn_id, e.quote) for e in hit.evidence}
    hit.evidence += [e for e in new.evidence if (e.turn_id, e.quote) not in seen]


def merge(state: ClinicalState, ext: Extraction) -> None:
    for field in FIELDS:
        bucket = getattr(state, field)
        for f in getattr(ext, field):
            new = StoredFact(**f.model_dump())
            hit = next((x for x in bucket if _fact_key(x) == _fact_key(new)), None)
            if hit is None:
                bucket.append(new)
            else:
                _absorb(hit, new)


def deduplicate_state(state: ClinicalState) -> None:
    """Final deterministic dedup on (field, normalised name, experiencer, assertion)."""
    for field in FIELDS:
        unique = {}
        for fact in getattr(state, field):
            k = _fact_key(fact)
            if k not in unique:
                unique[k] = fact
            else:
                _absorb(unique[k], fact)
        setattr(state, field, list(unique.values()))


def find_conflicts(state: ClinicalState) -> list:
    """Same name+experiencer asserted differently inside one field -> needs human/second-pass review."""
    out = []
    for field in FIELDS:
        groups = {}
        for f in getattr(state, field):
            groups.setdefault((_key(f.name), f.experiencer), set()).add(f.assertion.value)
        out += [(field, k[0], sorted(v)) for k, v in groups.items() if len(v) > 1]
    return out

# ============================================================
# FULL EXTRACTION
# ============================================================

def extract_full(llm, turns: List[Turn]) -> ClinicalState:

    st = ClinicalState()

    ext = _call(
        llm,
        "Conversation:\n" + fmt_turns(turns)
    )

    merge(
        st,
        ext
    )

    deduplicate_state(st)

    return st

# ============================================================
# INCREMENTAL EXTRACTION
# ============================================================

def _compact(
    state: ClinicalState,
) -> str:

    out = {
        f: [
            {
                "name": x.name,
                "assertion": x.assertion.value,
                "attrs": {
                    a.key: a.value
                    for a in x.attributes
                },
            }
            for x in getattr(
                state,
                f,
            )
        ]
        for f in FIELDS
        if getattr(state, f)
    }

    return json.dumps(
        out,
        ensure_ascii=False,
    )


def extract_incremental(
    llm,
    turns: List[Turn],
    step: int = 2,
    context: int = 3
) -> ClinicalState:

    st = ClinicalState()

    for i in range(0, len(turns), step):

        ctx = turns[
            max(0, i - context):i
        ]

        new = turns[
            i:i + step
        ]

        user = (
            f"Current state "
            f"(already extracted, do not repeat "
            f"unless updated):\n"
            f"{_compact(st)}\n\n"

            f"Earlier context "
            f"(for resolving short replies only):\n"
            f"{fmt_turns(ctx) or '(none)'}\n\n"

            f"New turns to extract from:\n"
            f"{fmt_turns(new)}"
        )

        merge(
            st,
            _call(llm, user)
        )

    deduplicate_state(st)

    return st

# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def _norm(s: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^\w\s]",
            "",
            s.lower(),
        ),
    ).strip()


def quote_in_turn(
    quote: str,
    text: str,
    thr: float = 0.85,
) -> bool:

    q = _norm(quote)
    t = _norm(text)

    if not q:
        return False

    if q in t:
        return True

    qw = q.split()
    tw = t.split()

    n = len(qw)
    best = 0.0

    for i in range(
        max(
            1,
            len(tw) - n + 1,
        )
    ):

        best = max(
            best,
            difflib.SequenceMatcher(
                None,
                " ".join(
                    tw[i:i + n]
                ),
                q,
            ).ratio(),
        )

    return best >= thr


_ROLE = {
    "patient_reported": {"patient"},
    "doctor_observed": {"doctor"},
    "doctor_assessed": {"doctor"},
    "explicit_plan": {"doctor"},
}


def validate(state: ClinicalState, turns: List[Turn]) -> dict:
    by = {t.id: t for t in turns}
    total = unsupported = weak = 0

    for field in FIELDS:
        for f in getattr(state, field):
            good = [
                e for e in f.evidence
                if e.turn_id in by and quote_in_turn(e.quote, by[e.turn_id].text)
            ]

            # Evidence grounding is the hard validation gate.
            # Do NOT infer source from speaker labels.
            f.supported = bool(good)

            # Diagnostic warning only; do not require lexical overlap between
            # fact name and quote because valid paraphrases are common.
            f.weak_link = bool(good) and any(
                len(_norm(e.quote).split()) <= 1 for e in good
            )

            if f.weak_link:
                weak += 1

            # Grounding diagnostics beyond "the quote exists in the transcript":
            qtxt = _norm(" ".join(e.quote for e in good))
            toks = [t for t in _norm(f.name).split() if len(t) > 3]
            f.name_ungrounded = bool(good) and bool(toks) and not any(t[:5] in qtxt for t in toks)
            nums = re.findall(r"\d+(?:\.\d+)?", " ".join(a.value for a in f.attributes))
            qnums = set(re.findall(r"\d+(?:\.\d+)?", " ".join(e.quote for e in good)))
            f.value_ungrounded = bool(good) and any(n not in qnums for n in nums)

            if good:
                f.t_start = min(by[e.turn_id].start for e in good)
                f.t_end = max(by[e.turn_id].end for e in good)

            total += 1
            if not f.supported:
                unsupported += 1

    return {
        "n_facts": total,
        "n_unsupported": unsupported,
        "unsupported_rate": unsupported / total if total else 0.0,
        "n_name_ungrounded": sum(f.name_ungrounded is True for fld in FIELDS for f in getattr(state, fld)),
        "n_value_ungrounded": sum(f.value_ungrounded is True for fld in FIELDS for f in getattr(state, fld)),
        "n_weak_link": weak,
        "weak_link_rate": weak / total if total else 0.0,
    }

# ============================================================
# DETERMINISTIC SOAP
# ============================================================

def fact_line(
    f: StoredFact,
) -> str:

    a = ", ".join(
        f"{x.key}: {x.value}"
        for x in f.attributes
    )

    s = (
        f.name
        + (f" ({a})" if a else "")
        + (
            f" [{f.status}]"
            if f.status
            else ""
        )
    )

    if f.experiencer.value == "family":
        s += " — family history"

    return {
        Assertion.absent:
            "Denies " + s,

        Assertion.uncertain:
            "Possible " + s,

        Assertion.hypothetical:
            "Hypothetical: " + s,
    }.get(
        f.assertion,
        s,
    )


def render_soap(
    state: ClinicalState,
    only_supported: bool = True,
) -> str:

    def sec(fields):

        rows = [
            f"- {fact_line(f)}"
            for fld in fields
            for f in getattr(
                state,
                fld,
            )
            if (
                f.supported
                or not only_supported
            )
        ]

        return (
            "\n".join(rows)
            or "- (none stated)"
        )

    return (
        "SUBJECTIVE\n"
        + sec([
            "complaints",
            "symptoms",
            "history",
            "medications",
            "allergies",
        ])
        + "\n\n"

        "OBJECTIVE\n"
        + sec([
            "vitals",
            "examination",
        ])
        + "\n\n"

        "ASSESSMENT (clinician-stated only)\n"
        + sec([
            "assessment",
        ])
        + "\n\n"

        "PLAN\n"
        + sec([
            "investigations",
            "advice",
            "follow_up",
            "referrals",
        ])
        + "\n"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "transcript",
        nargs="?",
        help="Single transcript JSON file",
    )

    ap.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing transcript JSON files",
    )

    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for extracted JSON files",
    )

    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    ap.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
    )

    ap.add_argument(
        "--out",
        default="prediction.json",
    )

    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,   # was 2048: long notes were truncated mid-JSON (see issues.md, file 2526)
    )

    ap.add_argument("--verbose", action="store_true", help="print raw model output for every call")
    ap.add_argument("--ft", action="store_true",
                    help="fine-tuned student: short SYSTEM_FT prompt, no schema suffix (must match train_qlora.py)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (use with --model <base>)")

    a = ap.parse_args()

    # ========================================================
    # BATCH MODE
    # ========================================================

    if a.input_dir is not None:

        if a.output_dir is None:
            ap.error(
                "--output-dir is required with --input-dir"
            )

        os.makedirs(
            a.output_dir,
            exist_ok=True,
        )

        files = sorted(
            f
            for f in os.listdir(a.input_dir)
            if f.endswith(".json") and not f.startswith("_")
        )

        print("=" * 70)
        print("QWEN BATCH EXTRACTION")
        print("=" * 70)
        print(f"Input directory : {a.input_dir}")
        print(f"Output directory: {a.output_dir}")
        print(f"Files           : {len(files)}")
        print(f"Model           : {a.model}")
        print(f"Mode            : {a.mode}")
        print("=" * 70)

        # ----------------------------------------------------
        # LOAD MODEL ONCE
        # ----------------------------------------------------

        print("\nLoading Qwen model...")

        t_load = time.time()

        llm = TransformersLLM(
            model_name=a.model,
            max_new_tokens=a.max_new_tokens,
            adapter=a.adapter,
            schema_in_prompt=not a.ft,
            system=SYSTEM_FT if a.ft else None,
            verbose=a.verbose,
        )

        print(
            f"Model loaded in "
            f"{time.time() - t_load:.1f}s"
        )

        successful = 0
        failed = 0

        total_start = time.time()

        # ----------------------------------------------------
        # PROCESS FILES
        # ----------------------------------------------------

        for i, filename in enumerate(files, start=1):

            input_path = os.path.join(
                a.input_dir,
                filename,
            )

            output_path = os.path.join(
                a.output_dir,
                filename,
            )

            print(
                f"\n[{i}/{len(files)}] "
                f"{filename}"
            )

            # ------------------------------------------------
            # RESUME SUPPORT
            # ------------------------------------------------

            if os.path.exists(output_path):

                print(
                    "  -> exists, skipping"
                )

                successful += 1
                continue

            t0 = time.time()
            llm.salvage_count = 0

            try:

                # --------------------------------------------
                # READ TRANSCRIPT
                # --------------------------------------------

                with open(
                    input_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    turns = [
                        Turn(**t)
                        for t in json.load(f)
                    ]

                # --------------------------------------------
                # EXTRACT
                # --------------------------------------------

                if a.mode == "full":

                    st = extract_full(
                        llm,
                        turns,
                    )

                else:

                    st = extract_incremental(
                        llm,
                        turns,
                    )

                # --------------------------------------------
                # VALIDATE
                # --------------------------------------------

                rep = validate(
                    st,
                    turns,
                )

                # --------------------------------------------
                # SAVE
                # --------------------------------------------

                with open(
                    output_path,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        st.model_dump(
                            mode="json"
                        ),
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                dt = time.time() - t0

                n_facts = sum(
                    len(getattr(st, field))
                    for field in FIELDS
                )

                conflicts = find_conflicts(st)
                print(
                    f"  -> facts={n_facts} "
                    f"time={dt:.2f}s "
                    f"unsupported={rep['n_unsupported']} "
                    f"name_ungrounded={rep['n_name_ungrounded']} "
                    f"value_ungrounded={rep['n_value_ungrounded']} "
                    f"conflicts={len(conflicts)}"
                )

                if llm.salvage_count or conflicts:
                    with open(os.path.join(a.output_dir, "_review.jsonl"), "a", encoding="utf-8") as lf:
                        lf.write(json.dumps({
                            "file": filename,
                            "salvaged_incomplete": bool(llm.salvage_count),
                            "conflicts": conflicts,
                        }) + "\n")

                successful += 1

            except Exception as e:

                failed += 1

                print(
                    f"  ERROR: "
                    f"{type(e).__name__}: {str(e)[:300]}"
                )

                # Failed files vanish from the dataset (and bias it toward short notes): record them.
                with open(os.path.join(a.output_dir, "_failed.txt"), "a", encoding="utf-8") as lf:
                    lf.write(filename + "\n")

                # Continue instead of killing
                # the complete 10k run.
                continue

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total_time = (
            time.time() - total_start
        )

        print()
        print("=" * 70)
        print("QWEN BATCH COMPLETE")
        print("=" * 70)

        print(
            f"Successful : {successful}"
        )

        print(
            f"Failed     : {failed}"
        )

        print(
            f"Total time : {total_time:.1f}s"
        )

        if successful:

            print(
                f"Avg/file   : "
                f"{total_time / successful:.2f}s"
            )

    # ========================================================
    # SINGLE FILE MODE
    # ========================================================

    else:

        if a.transcript is None:
            ap.error(
                "Provide a transcript file "
                "or use --input-dir"
            )

        with open(
            a.transcript,
            "r",
            encoding="utf-8",
        ) as f:

            turns = [
                Turn(**t)
                for t in json.load(f)
            ]

        t0 = time.time()

        llm = TransformersLLM(
            model_name=a.model,
            max_new_tokens=a.max_new_tokens,
            adapter=a.adapter,
            schema_in_prompt=not a.ft,
            system=SYSTEM_FT if a.ft else None,
            verbose=a.verbose,
        )

        if a.mode == "full":

            st = extract_full(
                llm,
                turns,
            )

        else:

            st = extract_incremental(
                llm,
                turns,
            )

        dt = time.time() - t0

        rep = validate(
            st,
            turns,
        )

        with open(
            a.out,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                st.model_dump(
                    mode="json"
                ),
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            render_soap(st)
        )

        print(
            f"mode={a.mode} "
            f"seconds={dt:.1f} "
            f"{rep}"
        )