import json
import os
from pathlib import Path
from schema import Turn
from agent2 import extract_full, TransformersLLM, SYSTEM_FT

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio_mts")
ASR_DIR = os.path.join(BASE_DIR, "data/transcripts/mts_whisper")
OUT_DIR = os.path.join(BASE_DIR, "outputs/mts_preds_4b_asr")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    llm = TransformersLLM(
        model_name="Qwen/Qwen3-4B-Instruct-2507", 
        adapter=os.path.join(BASE_DIR, "outputs/mts_lora_4b_asr"),
        schema_in_prompt=False,
        system=SYSTEM_FT,
        max_new_tokens=8192
    )
    
    for wav_file in Path(AUDIO_DIR).glob("*.wav"):
        doc_id = wav_file.stem
        asr_file = Path(ASR_DIR) / f"{doc_id}.json"
        out_file = Path(OUT_DIR) / f"{doc_id}.json"
        
        if not asr_file.exists() or out_file.exists(): 
            continue
            
        print(f"Extracting from audio for {doc_id}...")
        asr_data = json.load(open(asr_file))
        
        # THE FIX: Reconstruct turns using pauses (0.6 seconds)
        turns = []
        current_text = []
        speaker = "Doctor" # Assume Doctor starts
        last_end = 0.0
        
        for w in asr_data:
            pause = w.get("start", 0.0) - last_end
            if pause > 0.6 and current_text:
                turns.append(Turn(id=len(turns), speaker=speaker, text=" ".join(current_text)))
                speaker = "Patient" if speaker == "Doctor" else "Doctor"
                current_text = []
            
            current_text.append(w["word"])
            last_end = w.get("end", 0.0)
            
        if current_text:
            turns.append(Turn(id=len(turns), speaker=speaker, text=" ".join(current_text)))
            
        if not turns:
            continue
            
        try:
            st = extract_full(llm, turns)
            with open(out_file, "w") as f:
                json.dump(st.model_dump(), f, indent=2)
            n_facts = sum(len(getattr(st, f)) for f in ["complaints", "symptoms", "history", "medications", "allergies", "vitals", "examination", "assessment", "investigations", "advice", "follow_up", "referrals"])
            print(f"  -> Extracted {n_facts} facts")
        except Exception as e:
            print(f"  -> FAILED: {e}")

if __name__ == "__main__":
    main()