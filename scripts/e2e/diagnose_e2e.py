import os, json, torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel
from pathlib import Path
import sys
sys.path.append("/nas/vivek2/subrat/clinical_agent/scripts")
from agent2 import SYSTEM_FT

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
MODEL_DIR = os.path.join(BASE_DIR, "outputs/qwen2_audio_lora_v2")
GOLD_DIR = os.path.join(BASE_DIR, "outputs/mts_gold")
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio_mts")

def main():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    base = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    
    files = list(Path(GOLD_DIR).glob("*.json"))[:50]
    total_match = 0
    total_gold = 0
    
    for gf in files:
        doc_id = gf.stem
        wav = Path(AUDIO_DIR) / f"{doc_id}.wav"
        if not wav.exists(): continue
        
        gold = json.load(open(gf))
        gold_facts = [f["name"] for field in gold.values() for f in field]
        
        # 1. Prompt the model with the gold JSON as a "hint" (Teacher Forcing)
        messages = [
            {"role": "system", "content": SYSTEM_FT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Extract facts from this audio. The expected facts are: {gold_facts}"},
                {"type": "audio", "audio": str(wav)}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], audios=[str(wav)], return_tensors="pt", sampling_rate=16000).to(model.device)
        
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
        out = processor.batch_decode(gen[:, inputs.input_ids.size(1):], skip_special_tokens=True)[0]
        
        try:
            out = out.strip().split("```json")[-1].split("```")[0] if "```json" in out else out
            parsed = json.loads(out)
            pred_facts = [f["name"] for field in parsed.values() for f in field]
            
            # Count matches
            for p in pred_facts:
                if any(p.lower().strip() == g.lower().strip() for g in gold_facts):
                    total_match += 1
            total_gold += len(gold_facts)
            print(f"{doc_id}: Gold={len(gold_facts)}, Pred={len(pred_facts)}, Match={total_match}")
        except Exception as e:
            print(f"{doc_id}: FAILED to parse JSON - {e}")
    
    print(f"\nTeacher-Forcing Precision: {total_match / max(1, total_gold):.3f}")

if __name__ == "__main__":
    main()