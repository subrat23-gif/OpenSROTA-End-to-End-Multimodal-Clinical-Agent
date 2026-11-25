import os
import json
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel
from pathlib import Path

# Import SYSTEM_FT to match training
import sys
sys.path.append("/nas/vivek2/subrat/clinical_agent/scripts")
from agent2 import SYSTEM_FT

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
MODEL_DIR = os.path.join(BASE_DIR, "outputs/qwen2_audio_lora_v2")
AUDIO_DIR = os.path.join(BASE_DIR, "data/audio_mts")
OUT_PREDS_DIR = os.path.join(BASE_DIR, "outputs/mts_preds_e2e_v2")
os.makedirs(OUT_PREDS_DIR, exist_ok=True)

def main():
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    print("Loading End-to-End Audio Model (v2)...")
    
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()

    wav_files = list(Path(AUDIO_DIR).glob("*.wav"))
    print(f"Found {len(wav_files)} audio files for E2E inference.")
    
    for i, wav_file in enumerate(wav_files):
        doc_id = wav_file.stem
        out_file = Path(OUT_PREDS_DIR) / f"{doc_id}.json"
        
        if out_file.exists(): continue
        
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(wav_files)}...")
        
        messages = [
            {"role": "system", "content": SYSTEM_FT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the clinical facts from this conversation."},
                    {"type": "audio", "audio": str(wav_file)}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], audios=[str(wav_file)], return_tensors="pt", sampling_rate=16000).to(model.device)
        
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
                
            parsed_json = json.loads(response)
            with open(out_file, "w") as f:
                json.dump(parsed_json, f, indent=2)
        except Exception as e:
            with open(out_file, "w") as f:
                json.dump({}, f)

    print("E2E Inference Complete.")

if __name__ == "__main__":
    main()