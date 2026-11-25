import json
import os
from pathlib import Path

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
# Use the raw MTS-Dialog which has the actual SOAP sections
MTS_FILE = os.path.join(BASE_DIR, "data/mts_dialog.jsonl")
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio_mts")
OUT_DIR = os.path.join(BASE_DIR, "outputs/soap_sft")
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an expert clinical scribe. 
Listen to the doctor-patient conversation and generate a standard SOAP note.
Include the following sections:
SUBJECTIVE: Patient's history, complaints, and symptoms.
OBJECTIVE: Vitals, physical exam, and test results.
ASSESSMENT: Clinician's diagnoses and clinical impressions.
PLAN: Medications, lab orders, procedures, and follow-up instructions.
Use clear, professional medical terminology. Do not invent facts."""

def main():
    rows = []
    for line in open(MTS_FILE):
        row = json.loads(line)
        doc_id = row["id"]
        audio_file = Path(AUDIO_DIR) / f"{doc_id}.wav"
        
        if not audio_file.exists():
            continue
            
        # MTS-Dialog provides the gold note. We format it as a unified SOAP text.
        # If your JSONL structure differs, adjust the keys accordingly.
        gold_note = row.get("gold_note", "")
        
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate the SOAP note for this conversation."},
                        {"type": "audio", "audio": str(audio_file)}
                    ]
                },
                {
                    "role": "assistant",
                    "content": gold_note
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