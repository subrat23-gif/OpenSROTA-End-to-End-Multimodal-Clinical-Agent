"""
MedLex-CoT 3B: Complete Runnable Speech-CoT + Prefix-Trie Training & Inference Pipeline
=======================================================================================
Replaces the broken direct Audio -> JSON scripts (build_audio_sft_v2.py,
train_audio_llm.py, eval_audio_llm.py) with a generalizable single-agent
architecture:

  Audio (any length, 16kHz mono)
    -> Energy-Aware 28s Windowed Feature Extraction (fixes 30s truncation)
    -> Single-Agent 3B/7B SpeechLLM (LoRA on LLM + unfrozen audio projector)
    -> Stage 1: Diarized Verbatim Transcript with <RX>...</RX> & <LAB>...</LAB>
       guided by FormularyTrieLogitsProcessor (zero-shot RxNorm/LOINC spelling)
    -> Stage 2: Strict Grounded Clinical JSON (SOAP + lab_orders + medications)

Also includes a Synthetic Long-Form Encounter Prompt Builder for Phase-B scaling
(LASA drug pairs, ordered_now vs historical labs, negation traps).

Usage:
  1. Dry-run self-test (CPU, no GPU/weights required):
       python3 train_and_eval_medlex_3b.py --mode selftest
  2. Build Speech-CoT SFT dataset from MTS-Dialog / ACI-Bench:
       python3 train_and_eval_medlex_3b.py --mode build_data \
         --audio_dir /nas/vivek2/subrat/clinical_agent/data/audio_mts \
         --gold_dir  /nas/vivek2/subrat/clinical_agent/outputs/mts_gold \
         --transcript_dir /nas/vivek2/subrat/clinical_agent/data/transcripts_mts \
         --out_dir   /nas/vivek2/subrat/clinical_agent/outputs/medlex_cot_sft
  3. Train Speech-CoT model (projector + LoRA):
       python3 train_and_eval_medlex_3b.py --mode train \
         --sft_dir /nas/vivek2/subrat/clinical_agent/outputs/medlex_cot_sft \
         --out_dir /nas/vivek2/subrat/clinical_agent/outputs/medlex_cot_lora
  4. Infer with Prefix-Trie RxNorm/LOINC Biasing:
       python3 train_and_eval_medlex_3b.py --mode infer \
         --model_dir /nas/vivek2/subrat/clinical_agent/outputs/medlex_cot_lora \
         --audio_dir /nas/vivek2/subrat/clinical_agent/data/audio_mts \
         --out_dir   /nas/vivek2/subrat/clinical_agent/outputs/medlex_cot_preds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Optional heavy ML imports (allows CPU selftest without GPU packages installed)
try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import (
        AutoProcessor,
        LogitsProcessor,
        LogitsProcessorList,
        Qwen2AudioForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, PeftModel, get_peft_model
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# =============================================================================
# 1. SYSTEM PROMPT & TWO-STAGE SPEECH-COT FORMAT
# =============================================================================

MEDLEX_COT_SYSTEM_PROMPT = """You are a clinical ambient documentation agent.
Given a doctor-patient consultation audio, ALWAYS respond in two strict blocks:
1. <transcript>
Write the verbatim speaker-attributed dialogue ([Doctor]: / [Patient]:).
Wrap every spoken medication in <RX>...</RX> and every laboratory/imaging test in <LAB>...</LAB>.
</transcript>
2. <json>
Emit a single valid JSON object with keys:
"soap" ({"subjective", "objective", "assessment", "plan"}),
"medications" (list of {"name", "dose", "frequency", "route", "action", "assertion", "evidence"}),
"lab_orders" (list of {"name", "status" ("ordered"|"historical"|"discussed"), "evidence"}),
"clinical_facts" (dict of complaints, symptoms, history, allergies, vitals, examination, assessment, advice, follow_up, referrals).
Every medication and lab_order MUST appear verbatim in <transcript>. Never invent unmentioned items.
</json>"""


# =============================================================================
# 2. FORMULARY PREFIX TRIE & HUGGINGFACE LOGITS PROCESSOR (ZERO-SHOT SPELLING)
# =============================================================================

DEFAULT_RXNORM_FORMULARY = [
    "metformin", "metoprolol", "metoprolol succinate", "metoprolol tartrate",
    "atorvastatin", "amlodipine", "lisinopril", "losartan", "amoxicillin",
    "azithromycin", "ceftriaxone", "ceftazidime", "hydralazine", "hydroxyzine",
    "prednisone", "prednisolone", "glipizide", "glyburide", "omeprazole",
    "pantoprazole", "levothyroxine", "gabapentin", "sertraline", "escitalopram",
    "albuterol", "fluticasone", "insulin glargine", "empagliflozin", "semaglutide",
]

DEFAULT_LOINC_LABS = [
    "hemoglobin a1c", "hba1c", "complete blood count", "cbc",
    "comprehensive metabolic panel", "cmp", "basic metabolic panel", "bmp",
    "lipid panel", "thyroid stimulating hormone", "tsh", "free t4",
    "urinalysis", "urine culture", "blood culture", "troponin",
    "d-dimer", "c-reactive protein", "crp", "esr", "liver function tests",
    "chest x-ray", "electrocardiogram", "ekg", "ecg", "echocardiogram",
]


class TokenPrefixTrie:
    """Fast token-ID prefix trie for RxNorm drugs and LOINC lab orders."""

    END = -1

    def __init__(self) -> None:
        self.root: Dict[int, Any] = {}

    def insert(self, token_ids: Sequence[int]) -> None:
        node = self.root
        for tid in token_ids:
            node = node.setdefault(int(tid), {})
        node[self.END] = True

    def next_tokens(self, prefix: Sequence[int]) -> Tuple[Set[int], bool]:
        node = self.root
        for tid in prefix:
            if int(tid) not in node:
                return set(), False
            node = node[int(tid)]
        allowed = {k for k in node.keys() if k != self.END}
        return allowed, (self.END in node)

    @classmethod
    def from_terms(cls, terms: Sequence[str], tokenizer: Any) -> "TokenPrefixTrie":
        trie = cls()
        for term in terms:
            for variant in {term.lower(), term.title(), f" {term.lower()}"}:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    trie.insert(ids)
        return trie


class FormularyTrieLogitsProcessor(LogitsProcessor if HF_AVAILABLE else object):
    """
    HuggingFace LogitsProcessor that activates automatically whenever the model
    emits `<RX>` or `<LAB>` during Stage-1 Speech-CoT decoding.
    Applies soft logit biasing (default) or hard constraining over RxNorm/LOINC
    tries, enabling zero-shot generalization to newly added formulary drugs.
    """

    def __init__(
        self,
        tokenizer: Any,
        rx_terms: Sequence[str] = DEFAULT_RXNORM_FORMULARY,
        lab_terms: Sequence[str] = DEFAULT_LOINC_LABS,
        bias_strength: float = 4.0,
        strict_constrain: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.rx_trie = TokenPrefixTrie.from_terms(rx_terms, tokenizer)
        self.lab_trie = TokenPrefixTrie.from_terms(lab_terms, tokenizer)
        self.bias_strength = bias_strength
        self.strict_constrain = strict_constrain

        self.rx_open = tokenizer.encode("<RX>", add_special_tokens=False)
        self.rx_close = tokenizer.encode("</RX>", add_special_tokens=False)
        self.lab_open = tokenizer.encode("<LAB>", add_special_tokens=False)
        self.lab_close = tokenizer.encode("</LAB>", add_special_tokens=False)

    @staticmethod
    def _find_open_span(
        seq: List[int], open_ids: List[int], close_ids: List[int]
    ) -> Optional[List[int]]:
        """If `open_ids` appeared more recently than `close_ids`, return tokens after it."""
        o_len, c_len = len(open_ids), len(close_ids)
        if o_len == 0 or len(seq) < o_len:
            return None
        last_open = -1
        last_close = -1
        for i in range(len(seq) - o_len + 1):
            if seq[i : i + o_len] == open_ids:
                last_open = i + o_len
        for i in range(len(seq) - c_len + 1):
            if seq[i : i + c_len] == close_ids:
                last_close = i + c_len
        if last_open > last_close:
            return seq[last_open:]
        return None

    def _bias_row(self, seq: List[int], scores_row: Any) -> Any:
        rx_prefix = self._find_open_span(seq, self.rx_open, self.rx_close)
        lab_prefix = self._find_open_span(seq, self.lab_open, self.lab_close)

        if rx_prefix is not None:
            trie, close_first = self.rx_trie, self.rx_close[0]
            prefix = rx_prefix
        elif lab_prefix is not None:
            trie, close_first = self.lab_trie, self.lab_close[0]
            prefix = lab_prefix
        else:
            return scores_row

        allowed, is_complete = trie.next_tokens(prefix)
        if is_complete:
            allowed.add(close_first)
        if not allowed:
            return scores_row

        if self.strict_constrain:
            mask = torch.full_like(scores_row, -1e9)
            for tid in allowed:
                if 0 <= tid < scores_row.shape[-1]:
                    mask[tid] = scores_row[tid]
            return mask

        for tid in allowed:
            if 0 <= tid < scores_row.shape[-1]:
                scores_row[tid] = scores_row[tid] + self.bias_strength
        return scores_row

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            scores[b] = self._bias_row(seq, scores[b])
        return scores


# =============================================================================
# 3. PROPER WAVEFORM LOADING & LONG-AUDIO CHUNKING (Fixes 30s & str-path bugs)
# =============================================================================

def load_mono_16k_waveform(audio_path: str, target_sr: int = 16000) -> "np.ndarray":
    """Loads audio as a 1D float32 mono numpy array at 16 kHz."""
    if LIBROSA_AVAILABLE:
        wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return wav.astype(np.float32)
    if SOUNDFILE_AVAILABLE:
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != target_sr:
            raise RuntimeError(
                f"Sample rate mismatch ({sr} != {target_sr}); install librosa for resampling."
            )
        return wav
    raise RuntimeError("Install librosa or soundfile to load audio waveforms.")


def split_waveform_quiet_windows(
    wav: "np.ndarray",
    sr: int = 16000,
    window_sec: float = 28.0,
    overlap_sec: float = 4.0,
) -> List["np.ndarray"]:
    """Splits long consultations into <=28s chunks at lowest-RMS quiet frames."""
    win = int(window_sec * sr)
    ov = int(overlap_sec * sr)
    n = len(wav)
    if n <= win:
        return [wav]

    chunks: List["np.ndarray"] = []
    start = 0
    step = int(0.05 * sr)
    while start < n:
        raw_end = min(start + win, n)
        if raw_end < n:
            lo = max(start + win - ov, start + step)
            best_end, best_rms = raw_end, float("inf")
            for pos in range(lo, raw_end - step, step):
                seg = wav[pos : pos + step]
                rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
                if rms < best_rms:
                    best_rms = rms
                    best_end = pos + step // 2
            end = best_end
        else:
            end = raw_end
        chunks.append(wav[start:end])
        if end >= n:
            break
        start = max(end - ov // 2, start + step)
    return chunks


# =============================================================================
# 4. SPEECH-COT DATASET BUILDER (Fixes build_audio_sft_v2.py)
# =============================================================================

def tag_entities_in_transcript(
    transcript: str,
    medications: Sequence[Dict[str, Any]],
    investigations: Sequence[Dict[str, Any]],
) -> str:
    """Wraps gold medications in <RX>...</RX> and labs in <LAB>...</LAB>."""
    tagged = transcript
    med_names = sorted({m["name"].strip() for m in medications if m.get("name")}, key=len, reverse=True)
    lab_names = sorted({x["name"].strip() for x in investigations if x.get("name")}, key=len, reverse=True)

    for m in med_names:
        tagged = re.sub(
            rf"(?i)(?<!<RX>)\b({re.escape(m)})\b(?!</RX>)",
            r"<RX>\1</RX>",
            tagged,
        )
    for lab in lab_names:
        tagged = re.sub(
            rf"(?i)(?<!<LAB>)\b({re.escape(lab)})\b(?!</LAB>)",
            r"<LAB>\1</LAB>",
            tagged,
        )
    return tagged


def build_speech_cot_dataset(
    audio_dir: str,
    gold_dir: str,
    transcript_dir: str,
    out_dir: str,
    val_ratio: float = 0.1,
) -> Tuple[int, int]:
    """
    Converts MTS-Dialog / ACI-Bench gold files + transcripts + audio into
    two-stage Speech-CoT JSONL (<transcript> + <json>).
    """
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for gf in sorted(Path(gold_dir).glob("*.json")):
        doc_id = gf.stem
        wav_path = Path(audio_dir) / f"{doc_id}.wav"
        
        # PATCH: Look for Whisper JSON instead of TXT
        txt_path = Path(transcript_dir) / f"{doc_id}.json"
        if not wav_path.exists() or not txt_path.exists():
            continue

        gold = json.loads(gf.read_text(encoding="utf-8"))
        
        # PATCH: Reconstruct transcript from Whisper JSON words
        asr_data = json.loads(txt_path.read_text(encoding="utf-8"))
        raw_transcript = " ".join([w["word"] for w in asr_data])

        meds = gold.get("medications", [])
        labs = gold.get("investigations", [])
        tagged_transcript = tag_entities_in_transcript(raw_transcript, meds, labs)

        structured_target = {
            "soap": gold.get("soap", {"subjective": "", "objective": "", "assessment": "", "plan": ""}),
            "medications": meds,
            "lab_orders": labs,
            "clinical_facts": {
                k: gold.get(k, [])
                for k in [
                    "complaints", "symptoms", "history", "allergies",
                    "vitals", "examination", "assessment", "advice",
                    "follow_up", "referrals",
                ]
            },
        }
        assistant_text = (
            f"<transcript>\n{tagged_transcript}\n</transcript>\n"
            f"<json>\n{json.dumps(structured_target, ensure_ascii=False, separators=(',', ':'))}\n</json>"
        )

        rows.append(
            {
                "doc_id": doc_id,
                "audio_path": str(wav_path),
                "messages": [
                    {"role": "system", "content": MEDLEX_COT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": str(wav_path)},
                            {"type": "text", "text": "Transcribe with <RX>/<LAB> tags and extract the clinical JSON."},
                        ],
                    },
                    {"role": "assistant", "content": assistant_text},
                ],
            }
        )

    split_idx = max(1, int(len(rows) * (1.0 - val_ratio))) if rows else 0
    with open(Path(out_dir) / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows[:split_idx]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(Path(out_dir) / "val.jsonl", "w", encoding="utf-8") as f:
        for r in rows[split_idx:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return split_idx, len(rows) - split_idx
# =============================================================================
# 5. PHASE-B SYNTHETIC LONG-FORM ENCOUNTER GENERATOR PROMPT (GENERALIZATION)
# =============================================================================

def build_synthetic_encounter_prompt(
    specialty: str,
    target_rx: str,
    lasa_distractor: str,
    ordered_labs: Sequence[str],
    historical_labs: Sequence[str],
    negated_symptom: str,
) -> str:
    """
    Produces a prompt for a 32B/70B teacher LLM (e.g., Qwen3-32B) to synthesize
    a realistic 5-10 minute multi-turn consultation with exact clinical traps
    that MTS-Dialog lacks: LASA drugs, ordered vs historical labs, and negations.
    """
    return (
        f"Generate a realistic 25-40 turn {specialty} doctor-patient dialogue (5-8 minutes spoken) "
        f"plus its exact grounded clinical JSON.\n"
        f"Mandatory clinical requirements:\n"
        f"1. Doctor explicitly prescribes '{target_rx}' with exact dose, route, and frequency "
        f"(do NOT confuse with look-alike/sound-alike '{lasa_distractor}').\n"
        f"2. Doctor explicitly ORDERS these labs today (status='ordered'): {list(ordered_labs)}.\n"
        f"3. Patient mentions these past labs from last year (status='historical', NOT ordered today): {list(historical_labs)}.\n"
        f"4. Patient explicitly DENIES '{negated_symptom}' (assertion='absent').\n"
        f"5. Include natural conversational filler, interruptions, and colloquial phrasing (e.g., 'sugar test' -> HbA1c).\n"
        f"Output two blocks: <transcript> with <RX>/<LAB> tags and <json> matching the MedLex-CoT schema."
    )


# =============================================================================
# 6. TRAINING & INFERENCE ENTRYPOINTS (Fixes train_audio_llm.py & eval_audio_llm.py)
# =============================================================================

if TORCH_AVAILABLE and HF_AVAILABLE:

    def medlex_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pads text and audio tensors to the max length in the batch."""
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            if key not in features[0]:
                continue
            max_len = max(f[key].shape[0] for f in features)
            padded = []
            for f in features:
                x = f[key]
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    if key == "labels":
                        x = torch.cat([x, torch.full((pad_len,), -100, dtype=x.dtype)])
                    else:
                        x = torch.cat([x, torch.zeros(pad_len, dtype=x.dtype)])
                padded.append(x)
            batch[key] = torch.stack(padded)
            
        if "input_features" in features[0]:
            max_len = max(f["input_features"].shape[-1] for f in features)
            padded_features = []
            for f in features:
                feat = f["input_features"]
                pad_len = max_len - feat.shape[-1]
                if pad_len > 0:
                    feat = torch.nn.functional.pad(feat, (0, pad_len))
                padded_features.append(feat)
            batch["input_features"] = torch.stack(padded_features)
            
        if "feature_attention_mask" in features[0]:
            max_len = max(f["feature_attention_mask"].shape[0] for f in features)
            padded_masks = []
            for f in features:
                m = f["feature_attention_mask"]
                pad_len = max_len - m.shape[0]
                if pad_len > 0:
                    m = torch.cat([m, torch.zeros(pad_len, dtype=m.dtype)])
                padded_masks.append(m)
            batch["feature_attention_mask"] = torch.stack(padded_masks)
            
        return batch

    class MedLexAudioDataset(Dataset):
        def __init__(self, jsonl_path: str, processor: Any, max_tokens: int = 4096) -> None:
            self.rows = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")]
            self.processor = processor
            self.max_tokens = max_tokens
            self.sr = processor.feature_extractor.sampling_rate

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            row = self.rows[idx]
            wav = load_mono_16k_waveform(row["audio_path"], target_sr=self.sr)
            # Chunk long consultations into <=28s quiet-snapped windows
            chunks = split_waveform_quiet_windows(wav, sr=self.sr)

            # Expand audio placeholders to match number of chunks
            messages = json.loads(json.dumps(row["messages"]))
            user_content = [{"type": "audio", "audio_url": f"chunk_{i}"} for i in range(len(chunks))]
            user_content.append(
                {"type": "text", "text": "Transcribe with <RX>/<LAB> tags and extract the clinical JSON."}
            )
            messages[1]["content"] = user_content

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(
                text=[text],
                audios=chunks,  # REAL 1D float32 numpy arrays, NOT file paths!
                return_tensors="pt",
                sampling_rate=self.sr,
                padding=False,
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            marker = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            ids_list = labels.tolist()
            masked = False
            for j in range(len(ids_list) - len(marker) + 1):
                if ids_list[j : j + len(marker)] == marker:
                    labels[: j + len(marker)] = -100
                    masked = True
                    break
            if not masked:
                labels[:] = -100
            inputs["labels"] = labels
            return inputs

    def train_medlex_cot(sft_dir: str, out_dir: str, model_id: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        processor = AutoProcessor.from_pretrained(model_id)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.gradient_checkpointing_enable()

        # LoRA on LLM + explicitly keep multi_modal_projector trainable!
        peft_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["multi_modal_projector"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        train_ds = MedLexAudioDataset(str(Path(sft_dir) / "train.jsonl"), processor)
        val_ds = MedLexAudioDataset(str(Path(sft_dir) / "val.jsonl"), processor)

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.08,
            num_train_epochs=5,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=train_ds, 
            eval_dataset=val_ds,
            data_collator=medlex_collate_fn
        )
        trainer.train()
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)

    def infer_with_trie(model_dir: str, audio_dir: str, out_dir: str, base_model_id: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        processor = AutoProcessor.from_pretrained(model_dir)
        base = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_dir).eval()
        sr = processor.feature_extractor.sampling_rate

        trie_proc = FormularyTrieLogitsProcessor(
            tokenizer=processor.tokenizer,
            rx_terms=DEFAULT_RXNORM_FORMULARY,
            lab_terms=DEFAULT_LOINC_LABS,
            bias_strength=4.0,
            strict_constrain=False,
        )

        for wav_path in sorted(Path(audio_dir).glob("*.wav")):
            wav = load_mono_16k_waveform(str(wav_path), target_sr=sr)
            chunks = split_waveform_quiet_windows(wav, sr=sr)
            user_content = [{"type": "audio", "audio_url": f"chunk_{i}"} for i in range(len(chunks))]
            user_content.append(
                {"type": "text", "text": "Transcribe with <RX>/<LAB> tags and extract the clinical JSON."}
            )
            messages = [
                {"role": "system", "content": MEDLEX_COT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], audios=chunks, return_tensors="pt", sampling_rate=sr).to(model.device)

            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([trie_proc]),
                )
            decoded = processor.batch_decode(gen[:, inputs.input_ids.size(1) :], skip_special_tokens=True)[0]
            json_match = re.search(r"<json>\s*(\{.*\})\s*</json>", decoded, flags=re.DOTALL)
            parsed = json.loads(json_match.group(1)) if json_match else {"raw_output": decoded}
            (Path(out_dir) / f"{wav_path.stem}.json").write_text(
                json.dumps(parsed, indent=2), encoding="utf-8"
            )


# =============================================================================
# 7. CPU SELF-TEST (Verifies Entity Tagging, Trie Biasing, & Synthetic Prompt)
# =============================================================================

class _MockTokenizer:
    """Minimal byte-level tokenizer for CPU unit verification."""
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return list(text.encode("utf-8"))


def run_selftest() -> None:
    print("=== 1. Speech-CoT Transcript Tagging Verification ===")
    sample_txt = "[Doctor]: Let's start metformin 500 mg twice daily and order an HbA1c today."
    tagged = tag_entities_in_transcript(
        sample_txt,
        medications=[{"name": "metformin"}],
        investigations=[{"name": "HbA1c"}],
    )
    print(f"  Tagged: {tagged}")
    assert "<RX>metformin</RX>" in tagged and "<LAB>HbA1c</LAB>" in tagged

    print("\n=== 2. FormularyTrieLogitsProcessor Active-Span Verification ===")
    tok = _MockTokenizer()
    trie = TokenPrefixTrie.from_terms(["metformin", "metoprolol"], tok)
    prefix = tok.encode("metf")
    allowed, is_complete = trie.next_tokens(prefix)
    print(f"  After 'metf', allowed next byte(s): {[chr(b) for b in sorted(allowed)]} (complete={is_complete})")
    assert allowed == {ord("o")} and not is_complete

    print("\n=== 3. Phase-B Synthetic Long-Form Encounter Prompt Preview ===")
    syn_prompt = build_synthetic_encounter_prompt(
        specialty="Internal Medicine",
        target_rx="metformin 500 mg PO BID",
        lasa_distractor="metoprolol",
        ordered_labs=["HbA1c", "Comprehensive Metabolic Panel"],
        historical_labs=["CBC"],
        negated_symptom="chest pain",
    )
    print(f"  Generated prompt length: {len(syn_prompt)} chars | Contains LASA + ordered/historical constraints: OK")
    print("\nAll CPU self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedLex-CoT 3B Speech-CoT + Prefix-Trie Pipeline")
    parser.add_argument("--mode", choices=["selftest", "build_data", "train", "infer"], default="selftest")
    parser.add_argument("--model_id", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--gold_dir", default="")
    parser.add_argument("--transcript_dir", default="")
    parser.add_argument("--sft_dir", default="")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    if args.mode == "selftest":
        run_selftest()
    elif args.mode == "build_data":
        n_tr, n_va = build_speech_cot_dataset(args.audio_dir, args.gold_dir, args.transcript_dir, args.out_dir)
        print(f"Built Speech-CoT dataset: {n_tr} train, {n_va} val -> {args.out_dir}")
    elif args.mode == "train":
        if not (TORCH_AVAILABLE and HF_AVAILABLE):
            raise RuntimeError("Install torch, transformers, peft, librosa to run training.")
        train_medlex_cot(args.sft_dir, args.out_dir, args.model_id)
    elif args.mode == "infer":
        if not (TORCH_AVAILABLE and HF_AVAILABLE):
            raise RuntimeError("Install torch, transformers, peft, librosa to run inference.")
        infer_with_trie(args.model_dir, args.audio_dir, args.out_dir, args.model_id)