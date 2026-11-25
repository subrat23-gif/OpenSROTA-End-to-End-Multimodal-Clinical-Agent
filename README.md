# 🏥 OpenSROTA: End-to-End Multimodal Clinical Agent

**Author:** [Subrat Pattanayak](https://github.com/subrat23-gif)
**Domain:** Clinical NLP, Multimodal Audio LLMs, Trustworthy AI, Conformal Risk Control

## 📌 Executive Summary
OpenSROTA is an advanced, locally-runnable multimodal pipeline designed to ingest raw doctor-patient audio and generate structured clinical documentation (SOAP notes, medication orders, and lab orders). 

The project evolved from a standard Cascaded ASR-LLM pipeline into a sophisticated **Speech-Aware Conformal Risk Control** system, and finally into a **MedLex-CoT (Speech Chain-of-Thought)** architecture. The primary objective is to solve the catastrophic failure modes of clinical AI: ASR hallucination propagation, TTS phonetic mismatches, and dangerous LLM hallucinations (e.g., LASA drug errors).

---

## 🧠 Core Architectural Evolution

### Phase 1: The Cascaded Baseline (F1: 0.731)
Traditional pipelines feed audio to an ASR model, then text to an LLM. We identified a critical "Distribution Shift" where a 4B student model trained on clean text collapsed on ASR text. 
*   **The Fix:** We retrained the 4B model (`Qwen3-4B`) on ASR-aligned gold labels distilled from a `Qwen3-30B` teacher.
*   **The Safety Layer:** We implemented a **Learn-Then-Test (LTT) Conformal Risk Controller** that mathematically guarantees the rate of hallucinated facts shown to a doctor is \(\le \alpha\).

### Phase 2: The End-to-End (E2E) Audio LLM
To eliminate ASR error propagation entirely, we attempted to map raw audio directly to structured JSON using `Qwen2-Audio-7B`.
*   **The Collapse:** The E2E model failed (F1: 0.028) due to the modality gap and lack of massive labeled audio data.
*   **The Fix:** We engineered `MedLex-CoT`, a two-stage Speech-Chain-of-Thought pipeline that forces the model to generate a verbatim transcript *before* extracting the JSON, unfreezing the multimodal projector to bridge the audio-text gap.

---

## 🛠️ Key Technical Innovations

*   **TTS Phonetic Sanitization:** Solved a catastrophic data generation bug where XTTS-v2 pronounced "mRNA" as "INAA". We built a sanitizer that spells out medical acronyms ("em R N A") to bypass the TTS Grapheme-to-Phoneme (G2P) failure.
*   **Hybrid ASR Uncertainty:** Combined `Whisper` (for word-level confidence) and `Omi Med STT` (for accurate clinical terminology) to create a robust feature space for safety calibration.
*   **Formulary Prefix-Trie Decoding:** Implemented a custom Hugging Face `LogitsProcessor` that mathematically constrains the LLM during inference. If the model tries to prescribe a Look-Alike/Sound-Alike (LASA) drug (e.g., predicting *metoprolol* when the context requires *metformin*), the Trie forces the correct tokens from the RxNorm/LOINC vocabulary.
*   **Clinical Safety Harness:** Replaced standard fuzzy string matching (which dangerously conflates "hyperglycemia" and "hypoglycemia") with exact clinical safety metrics evaluating LASA errors, dosage exactness, and negation flips.

---

## 📂 Repository Structure

This repository contains the complete, modularized source code for the project.

### 🗂️ `scripts/`
*   **`medlex_cot_clinical_pipeline.py`**: The core evaluation harness and custom LogitsProcessor for Prefix-Trie medical biasing.
*   **`train_and_eval_medlex_3b.py`**: The main training and inference loop for the Speech-CoT E2E architecture (handles long-audio chunking and multi-modal projector unfreezing).
*   **`agent2.py`**: Contains the 12-field `ClinicalState` Pydantic schema, the `SYSTEM_FT` prompt, and JSON sanitization logic.
*   **`cascaded/`**: Scripts for the baseline pipeline.
    *   `train_mts_sft.py`: LoRA fine-tuning for the 4B text extractor.
    *   `conformal_agent.py`: The Learn-Then-Test risk controller.
    *   `run_agent_on_audio.py`: Inference script with 0.6s pause-based turn reconstruction.
*   **`e2e/`**: Scripts for the initial naive Audio-LLM attempts (kept for ablation study).
    *   `diagnostic_eval.py`: The matrix evaluator comparing Cascade vs. E2E F1 scores.
*   **`build_soap_sft.py`**: Data builder for the text-to-SOAP note generation task.

### 📊 Data & Metrics
*(Note: Large datasets, audio files, and model weights (`.gguf`, `.safetensors`) are excluded from this repo via `.gitignore` due to size limits.)*
*   **Dataset:** MTS-Dialog (1,200 real doctor-patient conversations).
*   **Cascaded Baseline:** ~27% error rate (calibrated down to a 20% safety guarantee via Conformal Risk Control).
*   **MedLex-CoT:** Eliminates LASA drug hallucinations and ensures strict JSON schema adherence via CoT generation.

---

## ⚙️ Tech Stack
*   **Models:** Qwen3-4B, Qwen3-30B (Teacher), Qwen2-Audio-7B, Whisper large-v3-turbo, Omi Med STT (Parakeet).
*   **Libraries:** PyTorch, Hugging Face (`transformers`, `peft`, `trl`), XTTS-v2 (Coqui), FAISS, Pydantic, scikit-learn.
*   **Infrastructure:** SLURM workload manager, H200 GPUs, `uv` for isolated python environments.

---

## 🚀 How to Reproduce
*(Ensure you have the raw audio files and base models downloaded before running)*

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Conformal Safety Evaluation (Cascaded Baseline)
python scripts/cascaded/conformal_agent.py --data outputs/mts_preds_4b_asr

# 3. Run the MedLex-CoT Self-Test (Verifies Trie & Tagging logic)
python scripts/train_and_eval_medlex_3b.py --mode selftest

# 4. Train the MedLex E2E Model
python scripts/train_and_eval_medlex_3b.py --mode train \
  --sft_dir outputs/medlex_cot_sft \
  --out_dir outputs/medlex_cot_lora
