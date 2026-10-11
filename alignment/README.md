# Forced alignment

Inference for the paper recipe lives in `scripts/infer.py` with `--task asr+fa`. The heads it uses are `paper_heads.json`.

This folder is the exploration that chose those heads, plus the aligners it calls.

| Script | What it does |
|---|---|
| `whisper_forced_align.py` | Whisper decoder cross-attention, median filter, column normalization, dynamic time warping |
| `qwen_attn_align.py` | Qwen thinker self-attention from text queries onto audio keys, then the same warping |
| `whisper_head_select.py` | Rank every Whisper cross-attention head on a gold word-time split, then score the top-k means |
| `qwen_head_select.py` | The same ranking for Qwen self-attention heads |

Dynamic time warping is the `dtw` function in `whisper_forced_align.py`. It is the usual dynamic program on the attention cost matrix. No DTW weights were trained.

Head selection needs a gold JSONL (one record per clip, with `path` and `words: [{start, end}]`) and a CSV of `path,text` for the selection split. Run it without `--resoftmax` for Whisper. The released readout is the top 2 Whisper heads and the top 3 Qwen heads from that ranking, with silence trim on.

```bash
python alignment/whisper_head_select.py \
    --model_path /path/to/Taraneh-WMedium \
    --manifest gold.jsonl \
    --csv dev300.csv \
    --out_dir head_select/Taraneh-WMedium

python alignment/qwen_head_select.py \
    --model_path /path/to/Taraneh-Q0.6B \
    --manifest gold.jsonl \
    --csv dev300.csv \
    --out_dir head_select/Taraneh-Q0.6B
```

`heads_top2.json` (Whisper) or `heads_top3.json` (Qwen) is the list to pass as `--heads`. The paper numbers also use a second list, chosen on a disjoint 300-clip pool, when scoring the other half of FarsDat. That second list is `crossfit_on_dev_half` in `paper_heads.json`. New audio uses the `heads` list.
