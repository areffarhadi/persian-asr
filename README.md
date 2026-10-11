# Taraneh-ASR: Persian speech recognition

<p align="center">
<b>درود بر پارسی‌زبانان،<br>و بر آن خاکِ گرامی.</b>
</p>

<p align="center">
📑 <a href="ArefFarhadi-TaranehASR.pdf">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;🤗 <a href="https://huggingface.co/areffarhadi">Hugging Face</a>
</p>

Taraneh is a family of five Persian ASR models, fine-tuned on about 1,560 hours of Persian speech. Three are Whisper models and two are Qwen3-ASR models.

| Name | Base checkpoint |
|---|---|
| [Taraneh-WBase](https://huggingface.co/areffarhadi/Taraneh-WBase) | `openai/whisper-base` |
| [Taraneh-WMedium](https://huggingface.co/areffarhadi/Taraneh-WMedium) | `openai/whisper-medium` |
| [Taraneh-WLarge](https://huggingface.co/areffarhadi/Taraneh-WLarge) | `openai/whisper-large-v3` |
| [Taraneh-Q0.6B](https://huggingface.co/areffarhadi/Taraneh-Q0.6B) | `Qwen/Qwen3-ASR-0.6B` |
| [Taraneh-Q1.7B](https://huggingface.co/areffarhadi/Taraneh-Q1.7B) | `Qwen/Qwen3-ASR-1.7B` |

## Paper

Aref Farhadipour. **Taraneh-ASR: Scaling Persian Speech Recognition and Forced Alignment Across Singing and Conversational Domains.** University of Zurich.

The PDF is in this repository: [ArefFarhadi-TaranehASR.pdf](ArefFarhadi-TaranehASR.pdf). The models and evaluation sets are on the [Hugging Face account](https://huggingface.co/areffarhadi). Each model card links to the same paper.

Pass a model name from the table, such as `Taraneh-WLarge`, or a local checkpoint directory to `--model`. The script resolves the name to `areffarhadi/Taraneh-WLarge` on Hugging Face.

## License

The models are free for commercial use.

| Models | License | Why |
|---|---|---|
| Taraneh-WBase, Taraneh-WMedium, Taraneh-WLarge | [MIT](https://opensource.org/license/mit) | Same license as OpenAI Whisper |
| Taraneh-Q0.6B, Taraneh-Q1.7B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Same license as Qwen3-ASR |

MIT and Apache-2.0 both allow commercial products, including hosted APIs. Apache-2.0 also includes a patent grant. Keep the original license notice with the weights.

This repository covers inference, forced alignment, manifest preparation, text normalization for scoring, and fine-tuning.

## Results

Word error rate / character error rate (%), after the unified text normalization in `persian_text_unify.py`. Lower is better. Bold marks the best fine-tuned model in each column. All five models were trained on the full ~1,560 h mix. PSRB and VisualEars were unseen at training time.

| <small>Model</small> | <small>FLEURS</small> | <small>Common&nbsp;Voice</small> | <small>YouTube</small> | <small>[Music](https://huggingface.co/datasets/areffarhadi/persian-music-eval)</small> | <small>[Media](https://huggingface.co/datasets/areffarhadi/persian-media-eval)</small> | <small>[PSRB](https://huggingface.co/datasets/PartAI/PSRB)</small> | <small>[VisualEars](https://huggingface.co/datasets/Reza2kn/visualears-golden-6669)</small> |
|---|---:|---:|---:|---:|---:|---:|---:|
| <small>[Taraneh&#8209;WBase](https://huggingface.co/areffarhadi/Taraneh-WBase)</small> | <small>7.88&nbsp;/&nbsp;2.93</small> | <small>10.76&nbsp;/&nbsp;4.45</small> | <small>19.02&nbsp;/&nbsp;10.30</small> | <small>37.69&nbsp;/&nbsp;17.67</small> | <small>24.43&nbsp;/&nbsp;13.46</small> | <small>26.88&nbsp;/&nbsp;12.26</small> | <small>11.58&nbsp;/&nbsp;3.97</small> |
| <small>[Taraneh&#8209;WMedium](https://huggingface.co/areffarhadi/Taraneh-WMedium)</small> | <small>4.16&nbsp;/&nbsp;1.52</small> | <small>3.47&nbsp;/&nbsp;1.67</small> | <small>14.29&nbsp;/&nbsp;8.26</small> | <small>**22.79&nbsp;/&nbsp;10.92**</small> | <small>**14.97&nbsp;/&nbsp;8.99**</small> | <small>19.14&nbsp;/&nbsp;8.73</small> | <small>6.23&nbsp;/&nbsp;1.89</small> |
| <small>[Taraneh&#8209;WLarge](https://huggingface.co/areffarhadi/Taraneh-WLarge)</small> | <small>**3.86&nbsp;/&nbsp;1.33**</small> | <small>**3.34&nbsp;/&nbsp;1.53**</small> | <small>**13.42&nbsp;/&nbsp;7.99**</small> | <small>22.98&nbsp;/&nbsp;11.00</small> | <small>16.06&nbsp;/&nbsp;9.54</small> | <small>**18.00&nbsp;/&nbsp;8.17**</small> | <small>**5.31&nbsp;/&nbsp;1.56**</small> |
| <small>[Taraneh&#8209;Q0.6B](https://huggingface.co/areffarhadi/Taraneh-Q0.6B)</small> | <small>7.11&nbsp;/&nbsp;2.54</small> | <small>9.65&nbsp;/&nbsp;3.86</small> | <small>18.87&nbsp;/&nbsp;10.12</small> | <small>35.97&nbsp;/&nbsp;15.83</small> | <small>22.32&nbsp;/&nbsp;11.73</small> | <small>24.61&nbsp;/&nbsp;10.67</small> | <small>9.44&nbsp;/&nbsp;3.12</small> |
| <small>[Taraneh&#8209;Q1.7B](https://huggingface.co/areffarhadi/Taraneh-Q1.7B)</small> | <small>6.25&nbsp;/&nbsp;2.15</small> | <small>7.85&nbsp;/&nbsp;3.13</small> | <small>18.21&nbsp;/&nbsp;9.91</small> | <small>34.50&nbsp;/&nbsp;15.05</small> | <small>21.91&nbsp;/&nbsp;11.46</small> | <small>22.79&nbsp;/&nbsp;9.98</small> | <small>8.05&nbsp;/&nbsp;2.64</small> |

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

### ASR, or ASR plus forced alignment

`--task asr` is the default and writes only `inference_results.csv`. `--task asr+fa` also writes `alignments.jsonl`: one JSON object per clip, with the ASR transcript and a `words` list of `{word, start, end}` in seconds.

Forced alignment is the paper readout, not a second trained model. It is available for Taraneh-WMedium, Taraneh-WLarge, Taraneh-Q0.6B, and Taraneh-Q1.7B. Taraneh-WBase is ASR only. Whisper alignment reads decoder cross-attention with eager attention, so it loads the Hugging Face checkpoint even when ASR itself used CTranslate2. Qwen alignment reads thinker self-attention the same way, even when ASR used vLLM. Whisper still sees at most 30 seconds.

The recipe that matches the FarsDat table is: the heads below, a median filter of width 3, column L2-normalization, dynamic time warping on the negative attention matrix, and a −35 dB energy trim of the first-word start and last-word end. Whisper uses the attention probabilities as Hugging Face returns them, with no second softmax. Qwen uses three heads and does not refine boundaries inside an audio token. Dynamic time warping is the standard dynamic program in `alignment/whisper_forced_align.py`. No DTW weights were trained.

Heads are 0-based `[layer, head]`. These are the heads selected on a 300-clip gold pool and applied to held-out speech, which is the set behind the reported test-half numbers. The paper names the same layers: L16 and L19 for Taraneh-WMedium, L23 and L26 for Taraneh-WLarge, and L20–L25 for Taraneh-Q0.6B. Taraneh-Q1.7B uses the same selection procedure; its three heads also sit in L20–L25.

| Model | Heads used by `--task asr+fa` | FarsDat MAE |
|---|---|---:|
| Taraneh-WMedium | L19/H8, L16/H12 | 37.8 ms |
| Taraneh-WLarge | L23/H3, L26/H7 | 38.1 ms |
| Taraneh-Q0.6B | L22/H7, L24/H12, L20/H15 | 47.2 ms |
| Taraneh-Q1.7B | L22/H7, L20/H14, L25/H5 | 48.5 ms |

The full 6,039-clip FarsDat number is cross-fit, so clips in the dev half were scored with a second list chosen on a disjoint 300-clip pool. That list is identical for Taraneh-WLarge. For the other three models it is L19/H8 and L16/H3 (Taraneh-WMedium), L22/H7, L24/H12, and L25/H10 (Taraneh-Q0.6B), and L22/H7, L23/H14, and L22/H12 (Taraneh-Q1.7B). New audio uses the table above. Both lists are in `alignment/paper_heads.json`.

```bash
python scripts/infer.py \
    --model Taraneh-WMedium \
    --audio utt.wav \
    --output_dir results/fa \
    --task asr+fa
```

The scripts that ranked every head against gold word times are `alignment/whisper_head_select.py` and `alignment/qwen_head_select.py`. See `alignment/README.md`.

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
