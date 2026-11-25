#!/usr/bin/env python3
"""
Extract per-word confidence from Whisper with Medical Vocabulary Biasing (initial_prompt).
Skips corrupt/silent audio gracefully.
"""
import argparse, json, os
from pathlib import Path
from faster_whisper import WhisperModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True, help="Manifest to extract medical terms from")
    ap.add_argument("--asr", default="large-v3-turbo")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    
    # 1. Build Medical Prompt from Manifest
    medical_terms = set()
    if os.path.exists(a.manifest):
        for line in open(a.manifest, encoding="utf-8"):
            if not line.strip(): continue
            try:
                row = json.loads(line)
                text = row.get("target_text", "")
                # Extract acronyms and words > 5 chars (likely medical terms)
                for w in str(text).split():
                    w_clean = "".join(c for c in w if c.isalnum())
                    if (w_clean.isupper() and len(w_clean) > 2) or len(w_clean) > 7:
                        medical_terms.add(w_clean)
            except Exception:
                continue
    
    # Limit to 100 terms to avoid blowing out Whisper's context window
    prompt_str = "Medical terms: " + ", ".join(sorted(list(medical_terms))[:100]) + "."
    print(f"Built initial_prompt with {len(medical_terms)} terms.")
    print(f"Prompt (first 150 chars): {prompt_str[:150]}...")

    print(f"\nLoading Whisper: {a.asr}")
    model = WhisperModel(a.asr, device="auto", compute_type="auto")

    wavs = sorted(Path(a.audio_dir).glob("*.wav"))
    failed_files = []
    
    for i, wav in enumerate(wavs, 1):
        out_path = Path(a.out_dir) / f"{wav.stem}.json"
        
        # RESUME LOGIC
        if out_path.exists():
            continue
        
        try:
            segments, info = model.transcribe(
                str(wav), 
                word_timestamps=True, 
                vad_filter=True,
                initial_prompt=prompt_str # <--- THE FIX
            )
            words = []
            for seg in segments:
                for w in (seg.words or []):
                    words.append({
                        "word": w.word.strip(),
                        "prob": w.probability,
                        "start": w.start,
                        "end": w.end
                    })
            
            with open(out_path, "w") as f:
                json.dump(words, f, indent=2)
                
        except Exception as e:
            print(f"[{i}/{len(wavs)}] FAILED on {wav.name}: {type(e).__name__}")
            failed_files.append(wav.name)
            with open(out_path, "w") as f:
                json.dump([], f)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(wavs)}")

    print("\nDone.")
    if failed_files:
        print(f"Failed {len(failed_files)} files.")
        with open(f"{a.out_dir}/_failed_conf.txt", "w") as f:
            for fn in failed_files:
                f.write(fn + "\n")

if __name__ == "__main__":
    main()