"""
MedLex-CoT: End-to-End Clinical Speech-to-SOAP & Order Extraction Blueprint
===========================================================================
Reference implementation addressing the catastrophic failure modes of naive
Audio -> Direct JSON fine-tuning on MTS-Dialog:
  1. Energy-aware windowed long-audio loading (fixes 30s Whisper truncation &
     raw-path processor bugs).
  2. Med-SpeechCLIP + CTC-Guided Modality Adapter (phonetic-semantic alignment
     against frozen UMLS/RxNorm/LOINC SapBERT embeddings with LASA hard negatives).
  3. Prefix-Trie Formulary Biasing / Constrained Decoding LogitsProcessor for
     exact medication & lab spelling during Stage-1 Speech-CoT transcription.
  4. Multi-dimensional Clinical GRPO Reward Engine for Stage-3 RL post-training.
  5. Medically Relevant Safety Evaluation Harness replacing dangerous fuzzy
     difflib >= 0.75 matching.

Run standalone self-test (no GPU required):
    python3 medlex_cot_clinical_pipeline.py
"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# 1. LONG-AUDIO CHUNKING (Fixes Qwen2-Audio 30s truncation & str-path bug)
# =============================================================================

def chunk_audio_with_overlap(
    waveform: Sequence[float],
    sample_rate: int = 16000,
    window_sec: float = 28.0,
    overlap_sec: float = 4.0,
) -> List[Tuple[float, float, List[float]]]:
    """
    Splits mono 16 kHz audio into overlapping windows under Whisper's 30s cap,
    snapping split boundaries to lowest-RMS quiet frames within the overlap.
    Returns list of (start_sec, end_sec, chunk_samples).
    """
    win = int(window_sec * sample_rate)
    ov = int(overlap_sec * sample_rate)
    total = len(waveform)
    if total <= win:
        return [(0.0, total / sample_rate, list(waveform))]

    chunks: List[Tuple[float, float, List[float]]] = []
    start = 0
    frame_step = int(0.05 * sample_rate)  # 50 ms RMS search step

    while start < total:
        raw_end = min(start + win, total)
        if raw_end < total:
            search_lo = max(start + win - ov, start + frame_step)
            best_end = raw_end
            best_rms = float("inf")
            for pos in range(search_lo, raw_end - frame_step, frame_step):
                seg = waveform[pos : pos + frame_step]
                rms = math.sqrt(sum(x * x for x in seg) / max(1, len(seg)))
                if rms < best_rms:
                    best_rms = rms
                    best_end = pos + frame_step // 2
            end = best_end
        else:
            end = raw_end

        chunks.append((start / sample_rate, end / sample_rate, list(waveform[start:end])))
        if end >= total:
            break
        start = max(end - ov // 2, start + frame_step)

    return chunks


# =============================================================================
# 2. CTC-GUIDED ADAPTER + MED-SPEECHCLIP / DINOSR PHONETIC-SEMANTIC LOSS
# =============================================================================

if TORCH_AVAILABLE:

    class CTCGuidedPhoneticAdapter(nn.Module):
        """
        Compresses speech encoder frames using CTC blank-probability gating
        (inspired by CoT-ASR, 2026) and projects both:
          - LLM-aligned speech tokens (for downstream Speech-CoT generation)
          - Entity-aligned phonetic embeddings (for Med-SpeechCLIP InfoNCE
            contrastive training against frozen SapBERT/UMLS embeddings).
        """

        def __init__(
            self,
            audio_dim: int = 1280,
            llm_dim: int = 2048,
            umls_dim: int = 768,
            vocab_size: int = 1024,
            blank_threshold: float = 0.85,
        ) -> None:
            super().__init__()
            self.blank_threshold = blank_threshold
            self.ctc_head = nn.Linear(audio_dim, vocab_size)  # index 0 = CTC blank
            self.llm_proj = nn.Sequential(
                nn.Linear(audio_dim, llm_dim),
                nn.GELU(),
                nn.LayerNorm(llm_dim),
                nn.Linear(llm_dim, llm_dim),
            )
            self.umls_proj = nn.Sequential(
                nn.Linear(audio_dim, umls_dim),
                nn.GELU(),
                nn.Linear(umls_dim, umls_dim),
            )
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def forward(
            self,
            audio_hidden: torch.Tensor,
            entity_spans: Optional[List[Tuple[int, int, int]]] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            audio_hidden: [B, T, audio_dim]
            entity_spans: optional list of (batch_idx, start_frame, end_frame)
                          for spoken drug/lab mentions in the batch.
            """
            ctc_logits = self.ctc_head(audio_hidden)  # [B, T, V]
            blank_probs = F.softmax(ctc_logits, dim=-1)[..., 0]  # [B, T]
            # Soft retention weight: suppresses silence/filler frames while staying differentiable
            keep_weight = 1.0 - torch.sigmoid((blank_probs - self.blank_threshold) * 10.0)
            gated_hidden = audio_hidden * keep_weight.unsqueeze(-1)
            llm_tokens = self.llm_proj(gated_hidden)

            out = {
                "ctc_logits": ctc_logits,
                "llm_tokens": llm_tokens,
                "keep_weight": keep_weight,
            }

            if entity_spans:
                span_embs = []
                for b_idx, s_f, e_f in entity_spans:
                    pooled = audio_hidden[b_idx, s_f : max(s_f + 1, e_f)].mean(dim=0)
                    span_embs.append(F.normalize(self.umls_proj(pooled), dim=-1))
                out["span_umls_embs"] = torch.stack(span_embs, dim=0)  # [N_ent, umls_dim]

            return out

        def med_speechclip_lasa_loss(
            self,
            span_umls_embs: torch.Tensor,
            pos_text_embs: torch.Tensor,
            hard_neg_text_embs: torch.Tensor,
        ) -> torch.Tensor:
            """
            Phonetic-Semantic Contrastive Loss with Look-Alike/Sound-Alike (LASA)
            hard negatives (e.g. metformin vs metoprolol, ceftriaxone vs ceftazidime).
            span_umls_embs:    [N, D] normalized speech span embeddings
            pos_text_embs:     [N, D] frozen SapBERT embeddings of true RxNorm/LOINC term
            hard_neg_text_embs:[N, K, D] frozen SapBERT embeddings of K LASA distractors
            """
            scale = self.logit_scale.exp().clamp(max=100.0)
            pos_text_embs = F.normalize(pos_text_embs, dim=-1)
            hard_neg_text_embs = F.normalize(hard_neg_text_embs, dim=-1)

            pos_sim = (span_umls_embs * pos_text_embs).sum(dim=-1, keepdim=True)  # [N, 1]
            neg_sim = torch.bmm(
                hard_neg_text_embs, span_umls_embs.unsqueeze(-1)
            ).squeeze(-1)  # [N, K]
            logits = torch.cat([pos_sim, neg_sim], dim=-1) * scale  # [N, 1 + K]
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, targets)


# =============================================================================
# 3. ZERO-SHOT PREFIX-TRIE FORMULARY BIASING (RxNorm + LOINC Spelling Guard)
# =============================================================================

class FormularyPrefixTrie:
    """
    Token-ID prefix trie for exact medication (RxNorm) and lab (LOINC) spellings.
    Used inside a decoding LogitsProcessor whenever the model emits <RX> or <LAB>.
    """

    def __init__(self) -> None:
        self.root: Dict[int, Any] = {}
        self.END_TOKEN = -1

    def insert(self, token_ids: Sequence[int]) -> None:
        node = self.root
        for tid in token_ids:
            node = node.setdefault(tid, {})
        node[self.END_TOKEN] = True

    def allowed_next_tokens(self, prefix: Sequence[int]) -> Set[int]:
        node = self.root
        for tid in prefix:
            if tid not in node:
                return set()
            node = node[tid]
        return {k for k in node.keys() if k != self.END_TOKEN}

    def is_complete(self, prefix: Sequence[int]) -> bool:
        node = self.root
        for tid in prefix:
            if tid not in node:
                return False
            node = node[tid]
        return self.END_TOKEN in node


def apply_formulary_trie_bias(
    logits: List[float],
    active_entity_prefix: Sequence[int],
    trie: FormularyPrefixTrie,
    end_tag_token_id: int,
    bias_strength: float = 4.5,
    strict_constrain: bool = False,
) -> List[float]:
    """
    Applies soft contextual biasing (or hard constrained decoding when
    strict_constrain=True) inside <RX>...</RX> or <LAB>...</LAB> spans.
    """
    allowed = trie.allowed_next_tokens(active_entity_prefix)
    if trie.is_complete(active_entity_prefix):
        allowed.add(end_tag_token_id)

    if not allowed:
        return logits

    adjusted = list(logits)
    if strict_constrain:
        for i in range(len(adjusted)):
            if i not in allowed:
                adjusted[i] = -1e9
    else:
        for tid in allowed:
            if 0 <= tid < len(adjusted):
                adjusted[tid] += bias_strength
    return adjusted


# =============================================================================
# 4. STAGE-3 VERIFIABLE CLINICAL GRPO REWARD ENGINE
# =============================================================================

# Curated subset of high-risk Look-Alike/Sound-Alike (LASA) clinical pairs
# (ISMP-style safety list; expand with full RxNorm phonetic index in production)
LASA_CRITICAL_PAIRS: Set[frozenset] = {
    frozenset({"metformin", "metoprolol"}),
    frozenset({"ceftriaxone", "ceftazidime"}),
    frozenset({"hydralazine", "hydroxyzine"}),
    frozenset({"hypoglycemia", "hyperglycemia"}),
    frozenset({"hypertension", "hypotension"}),
    frozenset({"prednisone", "prednisolone"}),
    frozenset({"clonidine", "clonazepam"}),
    frozenset({"celecoxib", "celexa"}),
    frozenset({"glipizide", "glyburide"}),
    frozenset({"lamictal", "lamisil"}),
}

CANONICAL_SYNONYMS: Dict[str, str] = {
    "a1c": "hemoglobin a1c",
    "hba1c": "hemoglobin a1c",
    "cbc": "complete blood count",
    "cmp": "comprehensive metabolic panel",
    "bmp": "basic metabolic panel",
    "tsh": "thyroid stimulating hormone",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "lipitor": "atorvastatin",
    "glucophage": "metformin",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "norvasc": "amlodipine",
}


def normalize_clinical_term(term: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s\-\.]", " ", str(term).lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return CANONICAL_SYNONYMS.get(cleaned, cleaned)


def all_aliases_for(term: str) -> Set[str]:
    norm = normalize_clinical_term(term)
    aliases = {term.lower().strip(), norm}
    for alias, canon in CANONICAL_SYNONYMS.items():
        if canon == norm:
            aliases.add(alias)
    return aliases


def extract_dosage_signature(attributes: Sequence[str]) -> Set[str]:
    """Extracts normalized dosage/frequency tokens like '500mg', 'twice daily'."""
    sigs: Set[str] = set()
    dose_pat = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|meq)\b")
    freq_pat = re.compile(r"\b(qd|bid|tid|qid|prn|qhs|daily|twice daily|once daily|weekly)\b")
    for attr in attributes:
        low = str(attr).lower()
        for num, unit in dose_pat.findall(low):
            sigs.add(f"{num}{unit}")
        for freq in freq_pat.findall(low):
            sigs.add(freq)
    return sigs


@dataclass
class ClinicalSafetyReport:
    rx_exact_f1: float
    rx_spelling_error_rate: float
    dosage_exact_match: float
    lab_order_precision: float
    lab_order_recall: float
    lab_order_f1: float
    assertion_accuracy: float
    lasa_critical_errors: List[Tuple[str, str]] = field(default_factory=list)
    ungrounded_hallucinations: List[str] = field(default_factory=list)
    negation_flips: List[str] = field(default_factory=list)
    json_valid: bool = True

    def grpo_scalar_reward(self) -> float:
        """
        Verifiable reward for Stage-3 GRPO optimization.
        Heavily penalizes LASA confusions, negation flips, and hallucinations.
        """
        if not self.json_valid:
            return -2.0
        reward = (
            1.5 * self.rx_exact_f1
            + 1.0 * self.dosage_exact_match
            + 1.5 * self.lab_order_f1
            + 1.0 * self.assertion_accuracy
            - 2.0 * len(self.lasa_critical_errors)
            - 1.5 * len(self.negation_flips)
            - 1.0 * len(self.ungrounded_hallucinations)
        )
        return round(reward, 4)


def evaluate_clinical_prediction(
    pred: Dict[str, Any],
    gold: Dict[str, Any],
    stage1_transcript: Optional[str] = None,
) -> ClinicalSafetyReport:
    """
    Medically relevant evaluation replacing fuzzy difflib >= 0.75.
    Checks:
      1. Exact canonical medication name match (NO fuzzy credit for LASA pairs!)
      2. Medication Spelling Error Rate (MSER)
      3. Dosage & frequency exactness
      4. Lab ordering precision/recall (filtering status == 'ordered' vs historical)
      5. Assertion polarity (present vs absent/negated)
      6. Alias-aware grounding check against Stage-1 verbatim transcript
    """
    # --- 1. Medications & LASA safety ---
    gold_meds = {
        normalize_clinical_term(m["name"]): m for m in gold.get("medications", [])
    }
    pred_meds = {
        normalize_clinical_term(m["name"]): m for m in pred.get("medications", [])
    }

    rx_tp = len(gold_meds.keys() & pred_meds.keys())
    rx_p = rx_tp / len(pred_meds) if pred_meds else (1.0 if not gold_meds else 0.0)
    rx_r = rx_tp / len(gold_meds) if gold_meds else (1.0 if not pred_meds else 0.0)
    rx_f1 = (2 * rx_p * rx_r / (rx_p + rx_r)) if (rx_p + rx_r) else 0.0

    lasa_errors: List[Tuple[str, str]] = []
    spelling_errors = 0
    unmatched_pred = pred_meds.keys() - gold_meds.keys()
    unmatched_gold = gold_meds.keys() - pred_meds.keys()
    for pm in unmatched_pred:
        for gm in unmatched_gold:
            if frozenset({pm, gm}) in LASA_CRITICAL_PAIRS:
                lasa_errors.append((pm, gm))
            elif difflib.SequenceMatcher(None, pm, gm).ratio() >= 0.70:
                spelling_errors += 1

    mser = (spelling_errors + len(lasa_errors)) / max(1, len(gold_meds))

    # Dosage exactness on matched medications
    dose_hits = 0
    dose_total = 0
    for k in gold_meds.keys() & pred_meds.keys():
        g_dose = extract_dosage_signature(gold_meds[k].get("attributes", []))
        p_dose = extract_dosage_signature(pred_meds[k].get("attributes", []))
        if g_dose:
            dose_total += 1
            if g_dose == p_dose:
                dose_hits += 1
    dosage_exact = dose_hits / dose_total if dose_total else 1.0

    # --- 2. Lab Orders (strictly status == 'ordered') ---
    def get_ordered_labs(record: Dict[str, Any]) -> Set[str]:
        labs = set()
        for item in record.get("investigations", []):
            status = str(item.get("status", "ordered")).lower()
            if status in {"ordered", "planned", "requested", "to_order"}:
                labs.add(normalize_clinical_term(item["name"]))
        return labs

    gold_labs = get_ordered_labs(gold)
    pred_labs = get_ordered_labs(pred)
    lab_tp = len(gold_labs & pred_labs)
    lab_p = lab_tp / len(pred_labs) if pred_labs else (1.0 if not gold_labs else 0.0)
    lab_r = lab_tp / len(gold_labs) if gold_labs else (1.0 if not pred_labs else 0.0)
    lab_f1 = (2 * lab_p * lab_r / (lab_p + lab_r)) if (lab_p + lab_r) else 0.0

    # --- 3. Assertion / Negation Flip Check across all clinical fields ---
    negation_flips: List[str] = []
    assertion_total = 0
    assertion_correct = 0
    for field_name in ["symptoms", "complaints", "history", "assessment", "allergies"]:
        g_map = {
            normalize_clinical_term(x["name"]): x.get("assertion", "present")
            for x in gold.get(field_name, [])
        }
        p_map = {
            normalize_clinical_term(x["name"]): x.get("assertion", "present")
            for x in pred.get(field_name, [])
        }
        for p_term in p_map.keys() - g_map.keys():
            for g_term in g_map.keys() - p_map.keys():
                if frozenset({p_term, g_term}) in LASA_CRITICAL_PAIRS:
                    lasa_errors.append((p_term, g_term))

        for k in g_map.keys() & p_map.keys():
            assertion_total += 1
            if g_map[k] == p_map[k]:
                assertion_correct += 1
            else:
                negation_flips.append(f"{k}: gold={g_map[k]} vs pred={p_map[k]}")

    assertion_acc = assertion_correct / assertion_total if assertion_total else 1.0

    # --- 4. Alias-Aware Transcript Grounding (Hallucination Detection) ---
    hallucinations: List[str] = []
    if stage1_transcript is not None:
        t_low = re.sub(r"[^a-z0-9\s\-\.]", " ", stage1_transcript.lower())
        for field_name in ["medications", "investigations"]:
            for item in pred.get(field_name, []):
                aliases = all_aliases_for(item["name"])
                if not any(re.search(rf"\b{re.escape(a)}\b", t_low) for a in aliases):
                    hallucinations.append(f"{field_name}:{item['name']}")

    return ClinicalSafetyReport(
        rx_exact_f1=round(rx_f1, 4),
        rx_spelling_error_rate=round(mser, 4),
        dosage_exact_match=round(dosage_exact, 4),
        lab_order_precision=round(lab_p, 4),
        lab_order_recall=round(lab_r, 4),
        lab_order_f1=round(lab_f1, 4),
        assertion_accuracy=round(assertion_acc, 4),
        lasa_critical_errors=lasa_errors,
        ungrounded_hallucinations=hallucinations,
        negation_flips=negation_flips,
    )


# =============================================================================
# 5. STANDALONE SELF-TEST DEMONSTRATING OLD vs NEW METRICS + TRIE BIASING
# =============================================================================

def _old_broken_difflib_match(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.75


if __name__ == "__main__":
    print("=== 1. Demonstrating why old difflib >= 0.75 is clinically dangerous ===")
    dangerous_pairs = [
        ("hypoglycemia", "hyperglycemia"),
        ("prednisone", "prednisolone"),
        ("hypertension", "hypotension"),
        ("ceftriaxone", "ceftazidime"),
    ]
    for g, p in dangerous_pairs:
        ratio = difflib.SequenceMatcher(None, g, p).ratio()
        print(
            f"  Gold='{g}' vs Pred='{p}' -> difflib={ratio:.2f} | "
            f"OldEval={'TRUE POSITIVE (DANGEROUS!)' if _old_broken_difflib_match(g, p) else 'Mismatch'}"
        )

    print("\n=== 2. ClinicalSafetyEvaluator + GRPO Reward on Adversarial Prediction ===")
    gold_sample = {
        "symptoms": [{"name": "chest pain", "assertion": "absent"}],
        "assessment": [{"name": "hypoglycemia", "assertion": "present"}],
        "medications": [{"name": "metformin", "attributes": ["500 mg", "twice daily"]}],
        "investigations": [
            {"name": "HbA1c", "status": "ordered"},
            {"name": "CBC", "status": "historical"},
        ],
    }
    bad_pred = {
        "symptoms": [{"name": "chest pain", "assertion": "present"}],  # Negation flip!
        "assessment": [{"name": "hyperglycemia", "assertion": "present"}],  # LASA!
        "medications": [{"name": "metoprolol", "attributes": ["50 mg", "daily"]}],  # LASA!
        "investigations": [
            {"name": "hemoglobin a1c", "status": "ordered"},  # Canonical match!
            {"name": "lipid panel", "status": "ordered"},  # Hallucinated!
        ],
    }
    transcript = "Doctor: No chest pain. Your blood sugar dropped, so hypoglycemia. Start metformin 500 mg twice daily and let's order an A1c today."
    report = evaluate_clinical_prediction(bad_pred, gold_sample, stage1_transcript=transcript)
    print(json.dumps(report.__dict__, indent=2))
    print(f"GRPO Scalar Reward: {report.grpo_scalar_reward()}")

    print("\n=== 3. Formulary Prefix-Trie Biasing Verification ===")
    trie = FormularyPrefixTrie()
    trie.insert([10, 20, 30])
    trie.insert([10, 21, 31])
    raw_logits = [0.0] * 40
    biased = apply_formulary_trie_bias(raw_logits, active_entity_prefix=[10], trie=trie, end_tag_token_id=39)
    print(f"  Allowed next token IDs after [10]: {sorted(trie.allowed_next_tokens([10]))}")
    print(f"  Biased logits at 20, 21, 22: {biased[20]:.1f}, {biased[21]:.1f}, {biased[22]:.1f}")

    print("\n=== 4. Windowed Long-Audio Chunker (90s synthetic waveform) ===")
    fake_wav = [0.01] * (90 * 16000)
    chunks = chunk_audio_with_overlap(fake_wav, sample_rate=16000)
    print(f"  90s audio split into {len(chunks)} overlapping windows: "
          f"{[(round(s, 1), round(e, 1)) for s, e, _ in chunks]}")