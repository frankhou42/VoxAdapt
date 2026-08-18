from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA-tune a small seq2seq voice post-editor")
    parser.add_argument("--dataset", default="data/train.jsonl")
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--output", default="artifacts/flan-t5-voxadapt-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: uv sync --extra ml") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q", "v"],
        ),
    )
    dataset = load_dataset("json", data_files=args.dataset, split="train").train_test_split(
        test_size=0.1, seed=7
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        inputs = [f"Polish spoken text: {text}" for text in batch["input"]]
        encoded = tokenizer(inputs, max_length=192, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=192, truncation=True)
        encoded["labels"] = labels["input_ids"]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    output = Path(args.output)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-4,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            report_to=[],
            predict_with_generate=True,
            seed=7,
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
