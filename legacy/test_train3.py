import os
import pandas as pd
import torch
import evaluate
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import IterableDataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor
)

class StreamingTestDataset(IterableDataset):
    def __init__(self, csv_path, chunk_size=32, processor=None):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.processor = processor

    def __iter__(self):
        chunk_buffer = []
        
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            for _, row in chunk.iterrows():
                try:
                    audio_data, _ = sf.read(row['path'])
                    
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
                    
                    yield {
                        "input_features": input_features,
                        "labels": labels
                    }
                    
                except Exception as e:
                    print(f"Error processing file {row['path']}: {str(e)}")
                    continue

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

def transcribe_audio(audio_path, model, processor, forced_decoder_ids):
    """Transcribe a single audio file"""
    audio_data, _ = sf.read(audio_path)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    input_features = processor.feature_extractor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    
    if torch.cuda.is_available():
        input_features = input_features.to("cuda")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            language="fa",
            task="transcribe",
            max_length=225
        )
    
    transcription = processor.tokenizer.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription

def main():
    # Path to your fine-tuned model
    MODEL_PATH = "./whisper_farsi_train3.py/step_24000"  # Adjust this to your model path
    NEW_MANIFEST_PATH = "./fleurs/data_fa_ir_test6.csv"  # Your new test dataset
    OUTPUT_DIR = "./fleurs/inference_results2"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and processor with Persian language settings
    print("Loading model and processor...")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, language="fa", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, language="fa", task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Set up Persian language generation
    model.config.language = "fa"
    model.config.task = "transcribe"
    
    # Get forced decoder IDs for Persian
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="fa", task="transcribe")
    
    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Read test manifest
    print("Processing test dataset...")
    test_df = pd.read_csv(NEW_MANIFEST_PATH)
    results = []
    
    # Process each file
    for idx, row in test_df.iterrows():
        print(f"Processing file {idx + 1}/{len(test_df)}")
        try:
            transcription = transcribe_audio(row['path'], model, processor, forced_decoder_ids)
            results.append({
                'path': row['path'],
                'original_text': row['text'],
                'generated_text': transcription
            })
        except Exception as e:
            print(f"Error processing {row['path']}: {str(e)}")
            results.append({
                'path': row['path'],
                'original_text': row['text'],
                'generated_text': 'ERROR',
                'error': str(e)
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    valid_results = results_df[results_df['generated_text'] != 'ERROR']
    wer = 100 * wer_metric.compute(
        predictions=valid_results['generated_text'].tolist(),
        references=valid_results['original_text'].tolist()
    )
    cer = 100 * cer_metric.compute(
        predictions=valid_results['generated_text'].tolist(),
        references=valid_results['original_text'].tolist()
    )
    
    # Save results
    results_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    
    # Save metrics
    with open(f"{OUTPUT_DIR}/metrics.txt", 'w') as f:
        f.write(f"Word Error Rate (WER): {wer:.2f}%\n")
        f.write(f"Character Error Rate (CER): {cer:.2f}%\n")
        f.write(f"Total files: {len(results_df)}\n")
        f.write(f"Successfully processed: {len(valid_results)}\n")
        f.write(f"Failed: {len(results_df) - len(valid_results)}\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")

if __name__ == "__main__":
    main()
