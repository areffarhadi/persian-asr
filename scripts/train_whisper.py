#!/usr/bin/env python3
"""Fine-tune a Whisper checkpoint on a Persian CSV manifest.

The manifest is the same CSV used for inference: columns ``path`` and ``text``.
Audio is read as 16 kHz mono. Whisper consumes 30-second windows, so segment
longer recordings before training.

Example::

    python scripts/train_whisper.py \\
        --model openai/whisper-large-v3 \\
        --train_csv manifests/train.csv \\
        --eval_csv manifests/dev.csv \\
        --output_dir checkpoints/Taraneh-WLarge \\
        --max_steps 40000
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import IterableDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taraneh.models import MODELS, resolve_model  # noqa: E402


class StreamingAudioDataset(IterableDataset):
    def __init__(self, csv_path: str, processor, chunk_size: int = 32):
        self.csv_path = csv_path
        self.processor = processor
        self.chunk_size = chunk_size

    def __iter__(self):
        buf = []
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                try:
                    audio, _ = sf.read(row["path"])
                    audio = np.asarray(audio)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    feats = self.processor.feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features.squeeze(0)
                    labels = self.processor.tokenizer(
                        str(row["text"]), return_tensors="pt"
                    ).input_ids.squeeze(0)
                    buf.append({"input_features": feats, "labels": labels})
                    if len(buf) >= self.chunk_size:
                        yield from buf
                        buf = []
                        gc.collect()
                except Exception as exc:  # noqa: BLE001
                    print(f"Error processing {row.get('path')}: {exc}")
        yield from buf


@dataclass
class StreamingDataCollator:
    processor: Any

    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]
        batch = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        labels = batch["input_ids"].masked_fill(batch.attention_mask.ne(1), -100)
        return {"input_features": input_features, "labels": labels}


def main():
    p = argparse.ArgumentParser(description="Fine-tune Whisper for Persian ASR")
    p.add_argument("--model", default="openai/whisper-large-v3", help="Base model or a Taraneh name")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--eval_csv", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=5000)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--logging_steps", type=int, default=100)
    args = p.parse_args()

    spec = resolve_model(args.model)
    init = spec["base"] or spec["checkpoint"]
    if args.model in MODELS:
        init = MODELS[args.model]["base"]

    import evaluate
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(init, language="persian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(init)
    model.generation_config.language = "persian"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="persian", task="transcribe")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyp = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": 100 * wer_metric.compute(predictions=hyp, references=ref),
            "cer": 100 * cer_metric.compute(predictions=hyp, references=ref),
        }

    train_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps" if args.eval_csv else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        load_best_model_at_end=bool(args.eval_csv),
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )
    try:
        training_args = Seq2SeqTrainingArguments(**train_args_kwargs)
    except TypeError:
        train_args_kwargs["evaluation_strategy"] = train_args_kwargs.pop("eval_strategy")
        training_args = Seq2SeqTrainingArguments(**train_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=StreamingAudioDataset(args.train_csv, processor),
        eval_dataset=StreamingAudioDataset(args.eval_csv, processor) if args.eval_csv else None,
        data_collator=StreamingDataCollator(processor),
        compute_metrics=compute_metrics if args.eval_csv else None,
    )
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=processor.feature_extractor)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=processor.feature_extractor)
    trainer.train()
    save_dir = Path(args.output_dir) / "final"
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved {save_dir}")


if __name__ == "__main__":
    main()
