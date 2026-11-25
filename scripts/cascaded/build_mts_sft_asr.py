import json
import os
from pathlib import Path
from agent2 import SYSTEM_FT

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
ASR_DIR = os.path.join(BASE_DIR, "data/transcripts/mts_whisper")
GOLD_ASR_DIR = os.path.join(BASE_DIR, "outputs/mts_gold_asr")
OUT_DIR = os.path.join(BASE_DIR, "outputs/mts_sft_asr")
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
          "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]

def reconstruct_turns(asr_data):
    turns = []
    current_text = []
    speaker = "Doctor"
    last_end = 0.0
    for w in asr_data:
        pause = w.get("start", 0.0) - last_end
        if pause > 0.6 and current_text:
            turns.append({"id": len(turns), "speaker": speaker, "text": " ".join(current_text)})
            speaker = "Patient" if speaker == "Doctor" else "Doctor"
            current_text = []
        current_text.append(w["word"])
        last_end = w.get("end", 0.0)
    if current_text:
        turns.append({"id": len(turns), "speaker": speaker, "text": " ".join(current_text)})
    return turns

def main():
    rows = []
    for gold_file in Path(GOLD_ASR_DIR).glob("*.json"):
        doc_id = gold_file.stem
        asr_file = Path(ASR_DIR) / f"{doc_id}.json"
        if not asr_file.exists(): continue
        
        asr_data = json.load(open(asr_file))
        turns = reconstruct_turns(asr_data)
        if not turns: continue
        
        conv_text = "\n".join([f"[T{t['id']}] {t['speaker']}: {t['text']}" for t in turns])
        gold_state = json.load(open(gold_file))
        target = json.dumps({f: gold_state.get(f, []) for f in FIELDS}, ensure_ascii=False, separators=(",", ":"))
        
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_FT},
                       {"role": "user", "content": "Conversation:\n" + conv_text}],
            "completion": [{"role": "assistant", "content": target}]
        })
    
    split_idx = int(len(rows) * 0.9)
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for r in rows[:split_idx]: f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT_DIR, "val.jsonl"), "w") as f:
        for r in rows[split_idx:]: f.write(json.dumps(r) + "\n")
    print(f"Wrote {split_idx} train, {len(rows)-split_idx} val")

if __name__ == "__main__":
    main()