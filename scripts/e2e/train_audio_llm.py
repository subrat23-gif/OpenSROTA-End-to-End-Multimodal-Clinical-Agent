import os
import torch
import json
from torch.utils.data import Dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

BASE_DIR = "/nas/vivek2/subrat/clinical_agent"
SFT_DIR = os.path.join(BASE_DIR, "outputs/audio_sft_v2")
OUT_DIR = os.path.join(BASE_DIR, "outputs/qwen2_audio_lora_v2")
os.makedirs(OUT_DIR, exist_ok=True)

class AudioSFTDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
    
    def _find_audio(self, messages):
        """Robustly find the audio path regardless of message position."""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "audio":
                        return c["audio"]
        return None
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        
        # 1. Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 2. Find audio (robust)
        audio_path = self._find_audio(messages)
        if audio_path is None:
            raise ValueError(f"No audio found for sample {idx}")
        
        # 3. Process text + audio
        inputs = self.processor(
            text=[text], 
            audios=[audio_path], 
            return_tensors="pt", 
            padding="max_length",
            max_length=4096,
            truncation=True,
            sampling_rate=16000
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 4. Mask everything before assistant response
        labels = inputs["input_ids"].clone()
        assistant_token_ids = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        
        input_ids_list = labels.tolist()
        for j in range(len(input_ids_list) - len(assistant_token_ids) + 1):
            if input_ids_list[j:j+len(assistant_token_ids)] == assistant_token_ids:
                labels[:j+len(assistant_token_ids)] = -100
                break
                
        # Mask padding
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs

def collate_fn(features):
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
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

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    print(f"Loading processor and model: {model_id}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = AudioSFTDataset(f"{SFT_DIR}/train.jsonl", processor)
    val_ds = AudioSFTDataset(f"{SFT_DIR}/val.jsonl", processor)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )

    print("Starting End-to-End Audio LLM Training (v2)...")
    trainer.train()
    
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(f"Saved model to {OUT_DIR}")

if __name__ == "__main__":
    main()