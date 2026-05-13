"""
Fine-tune Gemma 4 2B on a Mac Mini (24 GB) using QLoRA via the
Hugging Face ecosystem (transformers + peft + trl).

Goal: teach the model a custom command vocabulary so it performs
better when deployed back to the Raspberry Pi.

Requirements:
    pip install transformers peft trl datasets bitsandbytes accelerate torch

Usage:
    python finetune.py --dataset my_commands.jsonl --output ./gemma2b-finetuned
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL = "google/gemma-4-2b-it"   # instruction-tuned base


def load_jsonl(path: str) -> Dataset:
    """
    Expect each line to be: {"prompt": "...", "response": "..."}
    We format it as a chat turn for instruction fine-tuning.
    """
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "text": (
                    f"<start_of_turn>user\n{row['prompt']}<end_of_turn>\n"
                    f"<start_of_turn>model\n{row['response']}<end_of_turn>"
                )
            })
    log.info("Loaded %d examples from %s", len(records), path)
    return Dataset.from_list(records)


def build_model_and_tokenizer(quantize: bool = True):
    """
    Load Gemma 4 2B. On a 24 GB Mac we can run in fp16 without
    quantization, but 4-bit NF4 is faster and uses only ~2 GB.
    """
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",          # uses MPS on Apple Silicon if available
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def apply_lora(model):
    """
    Add LoRA adapters. We only train a small fraction of weights,
    which is why this fits on a Mac Mini.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                   # rank: higher = more capacity, more VRAM
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],   # attention layers only
    )
    return get_peft_model(model, lora_config)


def train(dataset_path: str, output_dir: str) -> None:
    dataset = load_jsonl(dataset_path)
    model, tokenizer = build_model_and_tokenizer()
    model = apply_lora(model)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 8
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
    )

    log.info("Starting fine-tune…")
    trainer.train()

    # Save the LoRA adapter (tiny, ~50 MB)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved adapter to %s", output_dir)
    log.info("Convert to GGUF for Ollama: see post.md §5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to .jsonl training file")
    parser.add_argument("--output", default="./gemma2b-finetuned")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    train(args.dataset, args.output)
