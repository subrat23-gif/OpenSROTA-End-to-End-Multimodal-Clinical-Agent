import json
import os
from pathlib import Path
from agent2 import SYSTEM_FT

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio_mts")
GOLD_DIR = os.path.join(BASE_DIR, "outputs/mts_gold") # <--- CLEAN TEXT GOLD
OUT_DIR = os.path.join(BASE_DIR, "outputs/audio_sft_v2")
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["complaints", "symptoms", "history", "medications", "allergies", "vitals",
          "examination", "assessment", "investigations", "advice", "follow_up", "referrals"]

def strip_stored_fact(fact):
    """Keep only the core Fact fields, drop supported/weak_link/etc."""
    return {
        "name": fact.get("name", ""),
        "attributes": fact.get("attributes", []),
        "assertion": fact.get("assertion", "present"),
        "experiencer": fact.get("experiencer", "patient"),
        "source": fact.get("source", "patient_reported"),
        "status": fact.get("status"),
        "evidence": fact.get("evidence", [])
    }

def main():
    rows = []
    for gold_file in Path(GOLD_DIR).glob("*.json"):
        doc_id = gold_file.stem
        audio_file = Path(AUDIO_DIR) / f"{doc_id}.wav"
        
        if not audio_file.exists():
            continue
            
        gold_data = json.load(open(gold_file))
        target = json.dumps({f: [strip_stored_fact(x) for x in gold_data.get(f, [])] for f in FIELDS}, ensure_ascii=False, separators=(",", ":"))
        
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_FT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the clinical facts from this conversation."},
                        {"type": "audio", "audio": str(audio_file)}
                    ]
                },
                {
                    "role": "assistant",
                    "content": target
                }
            ]
        })
        
    split_idx = int(len(rows) * 0.9)
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w") as f:
        for r in rows[:split_idx]:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT_DIR, "val.jsonl"), "w") as f:
        for r in rows[split_idx:]:
            f.write(json.dumps(r) + "\n")
            
    print(f"Wrote {split_idx} train and {len(rows)-split_idx} val examples to {OUT_DIR}")

if __name__ == "__main__":
    main()