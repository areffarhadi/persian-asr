import os
import json
import pandas as pd
import numpy as np
import torch
import gc
import evaluate
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)
import matplotlib.pyplot as plt

# Constants
CHECKPOINT_DIR = "./whisper_farsi_train3.py"
METRICS_FILE = os.path.join(CHECKPOINT_DIR, "training_metrics.json")
TRAINING_STATE_FILE = os.path.join(CHECKPOINT_DIR, "training_state.json")

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class StreamingAudioDataset(TorchIterableDataset):
    def __init__(self, csv_path, chunk_size=32, processor=None):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.processor = processor

    def __iter__(self):
        chunk_buffer = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                try:
                    audio_data, sample_rate = sf.read(row['path'])
                    
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    input_features = self.processor.feature_extractor(
                        audio_data,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.squeeze(0)
                    
                    labels = self.processor.tokenizer(
                        row['text'],
                        return_tensors="pt"
                    ).input_ids.squeeze(0)
                    
                    chunk_buffer.append({
                        "input_features": input_features,
                        "labels": labels
                    })
                    
                    if len(chunk_buffer) >= self.chunk_size:
                        for item in chunk_buffer:
                            yield item
                        chunk_buffer = []
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"Error processing file {row['path']}: {str(e)}")
                    continue
        
        for item in chunk_buffer:
            yield item

@dataclass
class StreamingDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not features:
            return None
            
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), 
            -100
        )
        
        return {
            "input_features": input_features,
            "labels": labels
        }

def get_training_state():
    """Load training state from file or create new state"""
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "current_step": 0,
        "best_wer": float('inf'),
        "best_step": 0
    }

def save_training_state(state):
    """Save training state to file"""
    with open(TRAINING_STATE_FILE, 'w') as f:
        json.dump(state, f)

def get_training_metrics():
    """Load training metrics from file or create new metrics"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {"train_metrics": [], "eval_metrics": []}

def save_training_metrics(metrics):
    """Save training metrics to file"""
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

def main():
    # Load training state
    training_state = get_training_state()
    training_metrics = get_training_metrics()
    current_step = training_state["current_step"]
    
    # Initialize metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Load model components
    if current_step == 0:
        print("Loading initial model components...")
        model_name = "openai/whisper-medium"
        tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Persian", task="transcribe")
        processor = WhisperProcessor.from_pretrained(model_name, language="Persian", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        print(f"Loading model from checkpoint at step {current_step}...")
        last_checkpoint = os.path.join(CHECKPOINT_DIR, f"step_{current_step}")
        tokenizer = WhisperTokenizer.from_pretrained(last_checkpoint)
        processor = WhisperProcessor.from_pretrained(last_checkpoint)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(last_checkpoint)
        model = WhisperForConditionalGeneration.from_pretrained(
            last_checkpoint,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    
    # Create streaming datasets
    train_dataset = StreamingAudioDataset(
        "./filtered_train_file10.csv",
        chunk_size=32,
        processor=processor
    )

    eval_dataset = StreamingAudioDataset(
        "./dev_filtered5_norm.csv",
        chunk_size=32,
        processor=processor
    )
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=12000,  # Adjust this for daily training length
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=4000,
        logging_steps=100,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=True if current_step > 0 else None,
    )
    
    # Set forced decoder IDs
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Persian", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=StreamingDataCollator(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # Train model
    print(f"Starting training from step {current_step}...")
    train_result = trainer.train(resume_from_checkpoint=True if current_step > 0 else None)
    
    # Update training state and metrics
    final_step = current_step + training_args.max_steps
    training_state["current_step"] = final_step
    
    # Evaluate on test set
    print("Evaluating on test set...")
    new_manifest_path = "./test_filtered5_norm.csv"
    new_manifest_df = pd.read_csv(new_manifest_path)
    
    def prepare_test_dataset(examples):
        audio_data, _ = sf.read(examples["path"])
        input_features = feature_extractor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]
        
        labels = tokenizer(examples["text"], return_tensors="pt").input_ids[0]
        return {
            "input_features": input_features,
            "labels": labels
        }
    
    # Process test dataset
    test_dataset = StreamingAudioDataset(
        new_manifest_path,
        chunk_size=32,
        processor=processor
    )
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_texts = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    
    # Calculate metrics
    ground_truth_texts = new_manifest_df["text"].tolist()
    test_wer = 100 * wer_metric.compute(predictions=pred_texts, references=ground_truth_texts)
    test_cer = 100 * cer_metric.compute(predictions=pred_texts, references=ground_truth_texts)
    
    # Save results
    new_manifest_df["Generated_Text"] = pred_texts
    new_manifest_df.to_csv(f"{CHECKPOINT_DIR}/test_predictions_step_{final_step}.csv", index=False)
    
    # Update best metrics if necessary
    if test_wer < training_state["best_wer"]:
        training_state["best_wer"] = test_wer
        training_state["best_step"] = final_step
    
    # Save state and metrics
    save_training_state(training_state)
    training_metrics["train_metrics"].append({
        "step": final_step,
        "metrics": train_result.metrics
    })
    training_metrics["eval_metrics"].append({
        "step": final_step,
        "metrics": {"test_wer": test_wer, "test_cer": test_cer}
    })
    save_training_metrics(training_metrics)
    
    # Save model components
    save_dir = os.path.join(CHECKPOINT_DIR, f"step_{final_step}")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    feature_extractor.save_pretrained(save_dir)
    
    print(f"Training completed at step {final_step}")
    print(f"Current best WER: {training_state['best_wer']} (Step {training_state['best_step']})")
    print(f"Latest test WER: {test_wer}, CER: {test_cer}")

if __name__ == "__main__":
    main()
