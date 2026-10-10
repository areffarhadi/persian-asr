# Taraneh-ASR: Persian speech recognition

Taraneh is a family of five Persian ASR models, fine-tuned on about 1,560 hours of Persian speech. Three are Whisper models and two are Qwen3-ASR models.

| Name | Architecture | Base checkpoint |
|---|---|---|
| Taraneh-WBase | Whisper-Base | `openai/whisper-base` |
| Taraneh-WMedium | Whisper-Medium | `openai/whisper-medium` |
| Taraneh-WLarge | Whisper-Large-v3 | `openai/whisper-large-v3` |
| Taraneh-Q0.6B | Qwen3-ASR-0.6B | `Qwen/Qwen3-ASR-0.6B` |
| Taraneh-Q1.7B | Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` |

## Paper

Aref Farhadipour. **Taraneh-ASR: Scaling Persian Speech Recognition and Forced Alignment Across Singing and Conversational Domains.** University of Zurich.

The PDF is in this repository: [ArefFarhadi-TaranehASR.pdf](ArefFarhadi-TaranehASR.pdf). Each model card on Hugging Face links to the same paper.

Model pages:

- [areffarhadi/Taraneh-WBase](https://huggingface.co/areffarhadi/Taraneh-WBase)
- [areffarhadi/Taraneh-WMedium](https://huggingface.co/areffarhadi/Taraneh-WMedium)
- [areffarhadi/Taraneh-WLarge](https://huggingface.co/areffarhadi/Taraneh-WLarge)
- [areffarhadi/Taraneh-Q0.6B](https://huggingface.co/areffarhadi/Taraneh-Q0.6B)
- [areffarhadi/Taraneh-Q1.7B](https://huggingface.co/areffarhadi/Taraneh-Q1.7B)

Pass either that Hugging Face id or a local checkpoint directory to `--model`.

## License

The models are free for commercial use.

| Models | License | Why |
|---|---|---|
| Taraneh-WBase, Taraneh-WMedium, Taraneh-WLarge | [MIT](https://opensource.org/license/mit) | Same license as OpenAI Whisper |
| Taraneh-Q0.6B, Taraneh-Q1.7B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Same license as Qwen3-ASR |

MIT and Apache-2.0 both allow commercial products, including hosted APIs. Apache-2.0 also includes a patent grant. Keep the original license notice with the weights.

This repository covers inference, manifest preparation, text normalization for scoring, and fine-tuning.

## Results

Word error rate / character error rate (%), after the unified text normalization in `persian_text_unify.py`. Lower is better. Bold marks the best fine-tuned model in each column. All five models were trained on the full ~1,560 h mix. PSRB and VisualEars were unseen at training time.

| Model | FLEURS | Common Voice | YouTube | Music | Media | PSRB | VisualEars |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taraneh-WBase | 7.88 / 2.93 | 10.76 / 4.45 | 19.02 / 10.30 | 37.69 / 17.67 | 24.43 / 13.46 | 26.88 / 12.26 | 11.58 / 3.97 |
| Taraneh-WMedium | 4.16 / 1.52 | 3.47 / 1.67 | 14.29 / 8.26 | **22.79 / 10.92** | **14.97 / 8.99** | 19.14 / 8.73 | 6.23 / 1.89 |
| Taraneh-WLarge | **3.86 / 1.33** | **3.34 / 1.53** | **13.42 / 7.99** | 22.98 / 11.00 | 16.06 / 9.54 | **18.00 / 8.17** | **5.31 / 1.56** |
| Taraneh-Q0.6B | 7.11 / 2.54 | 9.65 / 3.86 | 18.87 / 10.12 | 35.97 / 15.83 | 22.32 / 11.73 | 24.61 / 10.67 | 9.44 / 3.12 |
| Taraneh-Q1.7B | 6.25 / 2.15 | 7.85 / 3.13 | 18.21 / 9.91 | 34.50 / 15.05 | 21.91 / 11.46 | 22.79 / 9.98 | 8.05 / 2.64 |

## Install

```bash
pip install -r requirements.txt
```

Qwen inference and fine-tuning also need:

```bash
pip install -r requirements-qwen.txt
```

## Audio

Use 16 kHz mono audio. WAV is the format the training code expects. Whisper reads a 30-second window, so split longer recordings before Whisper inference or training. Qwen3-ASR accepts longer files.

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 utt.wav
```

## Manifest

One CSV is shared by inference and Whisper training. Two columns, header required:

```csv
path,text
/data/utt0001.wav,این یک جمله آزمایشی است
/data/utt0002.wav,سلام
```

- `path`: audio file. Absolute paths are safest. Relative paths are resolved from `--base_path` when you rebuild the manifest.
- `text`: reference transcript, plain Persian. Leave the column empty only if you are transcribing with no reference.

Build a CSV from audio files that each have a sibling `.txt` transcript, and the Qwen JSONL at the same time:

```bash
python scripts/make_manifest.py \
    --audio_dir /data/wavs \
    --output manifests/test.csv \
    --jsonl manifests/test.jsonl
```

Or rewrite an existing CSV onto absolute paths:

```bash
python scripts/make_manifest.py \
    --csv raw.csv \
    --base_path /data/persian-asr \
    --output manifests/test.csv \
    --jsonl manifests/test.jsonl
```

Qwen fine-tuning does not read the CSV directly. Each JSONL line is:

```json
{"audio": "/data/utt0001.wav", "text": "language fa<asr_text>این یک جمله آزمایشی است"}
```

The prefix `language fa<asr_text>` is what the released Taraneh-Q models were trained with. `make_manifest.py` adds it. Do not put that prefix in the CSV `text` column.

Example files: `examples/manifest.csv`, `examples/manifest.jsonl`.

## Inference

Inference writes `inference_results.csv` with columns `path`, `original_text`, `generated_text`.

Whisper and Qwen use different fast paths.

### Whisper

`scripts/infer_whisper.py` batches log-mel features and decodes with beam size 1, Persian forced from the start, and no timestamp tokens. When [CTranslate2](https://github.com/OpenNMT/CTranslate2) is installed it converts the checkpoint once into `~/.cache/taraneh/ct2/` and runs that engine. Otherwise it uses Transformers in float16 and skips the temperature-fallback retries.

```bash
pip install ctranslate2
python scripts/infer_whisper.py \
    --model Taraneh-WLarge \
    --manifest manifests/test.csv \
    --output_dir results/Taraneh-WLarge \
    --batch_size 16
```

`--compute_type int8_float16` is faster still. `--compatible` turns the CTranslate2 path off and restores the slower Transformers decoder. Raise `--batch_size` until the GPU is full. A local checkpoint is `--model /path/to/checkpoint`.

### Qwen

`scripts/infer_qwen.py` runs Taraneh-Q0.6B and Taraneh-Q1.7B with vLLM. Streaming exists only on that backend. Install it with Python 3.10 or newer; the Qwen ASR package pins vLLM 0.14:

```bash
pip install "qwen-asr[vllm]"
```

Leave `--language` empty for the automatic language identification used in the table. `--language Persian` forces Persian.

Offline. The manifest is decoded in batches:

```bash
python scripts/infer_qwen.py \
    --model Taraneh-Q1.7B \
    --manifest manifests/test.csv \
    --output_dir results/Taraneh-Q1.7B \
    --mode offline \
    --batch_size 32
```

`--batch_size` is how many waveforms are loaded at once. vLLM decodes that group in one call. Lower it if the GPU runs out of memory. `--backend transformers` is the non-vLLM offline path (SDPA or FlashAttention). It cannot stream.

Streaming. Audio is 16 kHz, and the model updates the transcript after every chunk. The final line is written to `inference_results.csv`, same columns as offline. The Qwen API default chunk is 2 seconds. The streaming demo uses 1 second.

```bash
python scripts/infer_qwen.py \
    --model Taraneh-Q1.7B \
    --manifest manifests/test.csv \
    --output_dir results/Taraneh-Q1.7B-stream \
    --mode streaming \
    --chunk_size_sec 2
```

`scripts/infer.py` picks the family from the model name and calls the same engines. One file:

```bash
python scripts/infer.py --model Taraneh-WMedium --audio utt.wav --output_dir results/one
```

`--model` accepts `Taraneh-WBase`, `Taraneh-WMedium`, `Taraneh-WLarge`, `Taraneh-Q0.6B`, `Taraneh-Q1.7B`, a local checkpoint, or any Hugging Face repo id.

## Text normalization

Reported WER and CER use `persian_text_unify.py` on both the reference and the hypothesis. The pipeline is:

1. Digits to Persian words
2. Punctuation, ZWNJ, and diacritics removed
3. Arabic letters mapped to Persian
4. `می` / `نمی` joined, and a basic `ها` split
5. A small compound split/join list
6. Colloquial forms mapped toward formal forms
7. Clitics and suffixes detached (`ترین`, `ها`, `اش`, ...)
8. `آ` mapped to `ا`, and a standalone ezafe `ی` dropped
9. A few verb plurals unified, and `هایی` mapped to `های`
10. If two strings differ only by spaces, the hypothesis spacing is aligned to the reference

Score a result file:

```bash
python persian_text_unify.py \
    --csv results/Taraneh-WLarge/inference_results.csv \
    --output_dir results/Taraneh-WLarge
```

The input CSV must contain `original_text` and `generated_text` (or `reference` / `prediction`). The script writes `metrics_unified_final.txt`. The line `Unified + spacing` is the number to report.

Normalize a string from Python:

```python
from persian_text_unify import normalize_unified, normalize_pair

print(normalize_unified("این یک تست ۱۲ است."))
gt, pred = normalize_pair(reference, hypothesis)
```

## Fine-tuning

### Whisper

```bash
python scripts/train_whisper.py \
    --model openai/whisper-large-v3 \
    --train_csv manifests/train.csv \
    --eval_csv manifests/dev.csv \
    --output_dir checkpoints/Taraneh-WLarge \
    --max_steps 40000 \
    --learning_rate 1e-5 \
    --train_batch_size 16 \
    --grad_acc 2
```

`--model` may also be `openai/whisper-base` or `openai/whisper-medium`. The trainer streams the CSV, so it needs `--max_steps` rather than an epoch count. Checkpoints land in `--output_dir`, and the final weights in `--output_dir/final`. Run that folder with `scripts/infer.py --model`.

The released Whisper models used this recipe: learning rate `1e-5`, batch size 16, gradient accumulation 2, Persian decoder prompt, and saves every 5,000 steps.

### Qwen3-ASR

`finetuning/qwen3_asr_sft.py` is the upstream Qwen3-ASR fine-tuning script (Apache-2.0, Copyright 2026 The Alibaba Qwen team). It reads JSONL, not CSV.

```bash
python finetuning/qwen3_asr_sft.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --train_file manifests/train.jsonl \
    --eval_file manifests/dev.jsonl \
    --output_dir checkpoints/Taraneh-Q1.7B \
    --batch_size 32 \
    --grad_acc 4 \
    --lr 2e-5 \
    --epochs 1 \
    --save_steps 200
```

For the 0.6B model, set `--model_path Qwen/Qwen3-ASR-0.6B`. Multi-GPU:

```bash
torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --train_file manifests/train.jsonl \
    --output_dir checkpoints/Taraneh-Q1.7B
```

Resume with `--resume 1` or `--resume_from checkpoints/Taraneh-Q1.7B/checkpoint-200`. Inference on a saved checkpoint:

```bash
python scripts/infer.py \
    --model checkpoints/Taraneh-Q1.7B/checkpoint-200 \
    --manifest manifests/test.csv \
    --output_dir results/Taraneh-Q1.7B
```

## Layout

```
scripts/infer_whisper.py     fast Whisper inference (CTranslate2)
scripts/infer_qwen.py        fast Qwen inference (vLLM)
scripts/infer.py            either family, same engines
scripts/make_manifest.py    CSV manifest and Qwen JSONL
scripts/train_whisper.py    Whisper fine-tuning
finetuning/qwen3_asr_sft.py Qwen3-ASR fine-tuning
persian_text_unify.py       normalization and WER/CER
examples/                   tiny manifest samples
legacy/                     earlier scripts from this repo
```
