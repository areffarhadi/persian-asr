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
from datetime import datetime
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Constants
CHECKPOINT_DIR = "./whisper_farsi_checkpoints_all_3andmusic"
METRICS_FILE = os.path.join(CHECKPOINT_DIR, "training_metrics.json")
TRAINING_STATE_FILE = os.path.join(CHECKPOINT_DIR, "training_state.json")
STEPS_PER_DAY = 18000
BACKUP_DIR = os.path.join(CHECKPOINT_DIR, "backups")

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Previously defined classes remain the same (StreamingAudioDataset, StreamingDataCollator, TrainingManager)
# ... (copy the previous implementation of these classes)
class StreamingAudioDataset(TorchIterableDataset):
    """Dataset for streaming audio data to avoid loading everything into memory"""
    
    def __init__(self, csv_path: str, chunk_size: int = 32, processor: Any = None):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.processor = processor
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def __iter__(self):
        chunk_buffer = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                try:
                    # Load and process audio file
                    audio_path = row['path']
                    if not os.path.exists(audio_path):
                        logging.warning(f"Audio file not found: {audio_path}")
                        continue
                        
                    audio_data, sample_rate = sf.read(audio_path)
                    
                    # Convert audio to float32 if needed
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    # Extract features
                    input_features = self.processor.feature_extractor(
                        audio_data,
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_features.squeeze(0)
                    
                    # Process text
                    labels = self.processor.tokenizer(
                        row['text'],
                        return_tensors="pt"
                    ).input_ids.squeeze(0)
                    
                    chunk_buffer.append({
                        "input_features": input_features,
                        "labels": labels,
                        "path": audio_path  # Store path for debugging
                    })
                    
                    # Yield when buffer is full
                    if len(chunk_buffer) >= self.chunk_size:
                        for item in chunk_buffer:
                            yield item
                        chunk_buffer = []
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    logging.error(f"Error processing file {row['path']}: {str(e)}")
                    continue
        
        # Yield remaining items in buffer
        for item in chunk_buffer:
            yield item

@dataclass
class StreamingDataCollator:
    """Collates streaming data into batches"""
    
    processor: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not features:
            return None
            
        # Stack input features
        input_features = torch.stack([f["input_features"] for f in features])
        
        # Process labels
        labels = [f["labels"] for f in features]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            return_tensors="pt"
        )
        
        # Mask padding in labels
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), 
            -100
        )
        
        return {
            "input_features": input_features,
            "labels": labels
        }

class TrainingManager:
    """Manages training state and checkpoints"""
    
    def __init__(self):
        self.state = self._load_state()
        self.metrics = self._load_metrics()
        
    def _load_state(self) -> Dict:
        """Load or initialize training state"""
        if os.path.exists(TRAINING_STATE_FILE):
            try:
                with open(TRAINING_STATE_FILE, 'r') as f:
                    state = json.load(f)
                logging.info(f"Loaded training state from step {state['current_step']}")
                return state
            except Exception as e:
                logging.error(f"Error loading training state: {e}")
                return self._create_initial_state()
        return self._create_initial_state()
    
    def _create_initial_state(self) -> Dict:
        """Create initial training state"""
        return {
            "current_step": 0,
            "best_wer": float('inf'),
            "best_step": 0,
            "daily_runs": 0,
            "last_checkpoint": None,
            "start_date": datetime.now().isoformat()
        }
    
    def _load_metrics(self) -> Dict:
        """Load or initialize training metrics"""
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metrics: {e}")
                return {"train_metrics": [], "eval_metrics": []}
        return {"train_metrics": [], "eval_metrics": []}
    
    def save_state(self):
        """Save training state with backup"""
        # Create backup of current state file
        if os.path.exists(TRAINING_STATE_FILE):
            backup_path = os.path.join(
                BACKUP_DIR, 
                f"state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.replace(TRAINING_STATE_FILE, backup_path)
        
        # Save new state
        with open(TRAINING_STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        logging.info("Saved training state")
    
    def save_metrics(self):
        """Save training metrics"""
        with open(METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logging.info("Saved training metrics")
    
    def update_metrics(self, train_metrics: Dict, eval_metrics: Dict):
        """Update training and evaluation metrics"""
        self.metrics["train_metrics"].append({
            "step": self.state["current_step"],
            "metrics": train_metrics
        })
        self.metrics["eval_metrics"].append({
            "step": self.state["current_step"],
            "metrics": eval_metrics
        })
        self.save_metrics()
        
def main(starting_step: int = None):
    # Initialize training manager
    manager = TrainingManager()

    # Set the current step based on user input or the saved state
    if starting_step is not None:
        manager.state["current_step"] = starting_step
    current_step = manager.state["current_step"]

    # Calculate steps for this run
    total_desired_steps = (manager.state["daily_runs"] + 1) * STEPS_PER_DAY
    steps_this_run = total_desired_steps - current_step

    logging.info(f"Starting daily run {manager.state['daily_runs'] + 1}")
    logging.info(f"Manually specified starting step: {starting_step}")
    logging.info(f"Current step: {current_step}")
    logging.info(f"Steps to run today: {steps_this_run}")

    # Initialize metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    try:
        # Load model components from the specified step
        last_step_dir = os.path.join(CHECKPOINT_DIR, f"step_{current_step}")
        logging.info(f"Loading model from directory: {last_step_dir}")

        # Attempt to load from checkpoint
        try:
            tokenizer = WhisperTokenizer.from_pretrained(
                last_step_dir, 
                language="Persian", 
                task="transcribe"
            )
            processor = WhisperProcessor.from_pretrained(
                last_step_dir, 
                language="Persian", 
                task="transcribe"
            )
            feature_extractor = WhisperFeatureExtractor.from_pretrained(last_step_dir)
            model = WhisperForConditionalGeneration.from_pretrained(
                last_step_dir,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            logging.info(f"Successfully loaded model from step {current_step}")
        except Exception as e:
            logging.error(f"Failed to load model from {last_step_dir}: {e}")
            raise RuntimeError(f"Could not load checkpoint from {last_step_dir}")

        # Create datasets
        train_dataset = StreamingAudioDataset(
            "./new_cleaned_manifests_with_music/train_combined.csv",
            chunk_size=32,
            processor=processor
        )

        eval_dataset = StreamingAudioDataset(
            "./new_cleaned_manifests_with_music/dev_combined.csv",
            chunk_size=32,
            processor=processor
        )

        def compute_metrics(pred):
            """Compute WER and CER metrics"""
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
            learning_rate=8e-6,
            warmup_steps=500,
            max_steps=steps_this_run,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=min(1000, steps_this_run // 5),
            save_steps=max(4000, steps_this_run // 2),
            logging_steps=100,
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            resume_from_checkpoint=False,  # Always load explicitly
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
        logging.info(f"Starting training from step {current_step}...")
        train_result = trainer.train(resume_from_checkpoint=False)

        # Update training state
        final_step = current_step + steps_this_run
        manager.state["current_step"] = final_step
        manager.state["daily_runs"] += 1

        # Save training state
        manager.save_state()

        # Save model components
        save_dir = os.path.join(CHECKPOINT_DIR, f"step_{final_step}")
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        feature_extractor.save_pretrained(save_dir)

        logging.info(f"Training completed at step {final_step}")
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    # Example usage: 
    # To start training from step 16000, call main(16000)
    # To start training from step 32000, call main(32000)

    
    import sys
    
    if len(sys.argv) > 1:
        try:
            last_step = int(sys.argv[1])
            main(last_step)
        except ValueError:
            print("Please provide a valid integer step number.")
            sys.exit(1)
    else:
        print("Please provide a step number as a command-line argument.")
        print("Example: python script.py 16000")
        sys.exit(1)
     
