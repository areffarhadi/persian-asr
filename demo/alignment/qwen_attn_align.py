#!/usr/bin/env python
"""Forced alignment from the *internal* self-attention of a fine-tuned Qwen3-ASR checkpoint.

Idea (extension of "Whisper has an internal word aligner" to a decoder-only audio LLM):
  * teacher-force  <chat prefix> <audio tokens> ... language Persian<asr_text> w1 w2 ... wn <|im_end|>
  * for every layer/head take attention from text-query positions to audio-key positions
    -> [L*H, n_text_rows, n_audio_tokens]  (row r = query whose *next* token is text token r)
  * unsupervised head selection (column/row-norm score, as in the paper) or a fixed head list,
    median filter, column-normalise, DTW on -matrix, word boundaries at token jumps
  * audio-token grid is ~80 ms; optional sub-token refinement by attention centroid.
Outputs alignments.jsonl compatible with evaluate_gold.py.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from alignment.whisper_forced_align import dtw, filter_attention, median_filter, remove_punctuation  # noqa: E402
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor  # noqa: E402

AUDIO_TOKEN_ID = 151676


class QwenAttnAligner:
    def __init__(self, model_path, device="cuda:0", dtype=torch.bfloat16, language="Persian"):
        self.device = device
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
        self.processor = Qwen3ASRProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        self.tok = self.processor.tokenizer
        self.language = language
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        self.prefix = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        self.prefix += f"language {language}<asr_text>"
        self.asr_text_id = self.tok.convert_tokens_to_ids("<asr_text>")
        self.im_end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.audio_token_id = getattr(self.model.config.thinker_config, "audio_token_id", AUDIO_TOKEN_ID)
        # capture per-layer attention weights via hooks (eager attention returns (out, weights))
        self._attn_buf = {}
        self.attn_layers = [layer.self_attn for layer in self.model.thinker.model.layers]
        for li, mod in enumerate(self.attn_layers):
            mod.register_forward_hook(self._make_hook(li))

    def _make_hook(self, li):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self._attn_buf[li] = out[1].detach()
        return hook

    def word_tokens(self, words):
        toks = []
        for i, w in enumerate(words):
            ids = self.tok.encode((" " if i > 0 else "") + w, add_special_tokens=False)
            toks.append(ids)
        return toks

    @torch.inference_mode()
    def get_attention(self, wav, words):
        """Return (attn [L,H,n_rows,n_audio], n_audio, row layout)."""
        wtoks = self.word_tokens(words)
        text_ids = [t for ws in wtoks for t in ws]
        inputs = self.processor(text=[self.prefix], audio=[wav], return_tensors="pt", padding=True)
        ids = inputs["input_ids"][0].tolist()
        full = ids + text_ids + [self.im_end_id]
        input_ids = torch.tensor([full], device=self.device)
        attn_mask = torch.ones_like(input_ids)
        feats = inputs["input_features"].to(self.device, self.model.dtype)
        fmask = inputs["feature_attention_mask"].to(self.device)
        self._attn_buf.clear()
        out = self.model.thinker(
            input_ids=input_ids,
            attention_mask=attn_mask,
            input_features=feats,
            feature_attention_mask=fmask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attns = out.attentions
        if attns is None:
            attns = [self._attn_buf[i] for i in range(len(self.attn_layers)) if i in self._attn_buf]
        if not attns:
            raise RuntimeError("no attention weights captured; is attn_implementation='eager'?")
        audio_pos = (input_ids[0] == self.audio_token_id).nonzero().squeeze(-1)
        asr_pos = len(ids) - 1  # position of <asr_text>: its query predicts the first text token
        n_text = len(text_ids)
        # n_text+1 rows: row r predicts text token r; the final row (last text token) predicts <|im_end|>
        rows = torch.arange(asr_pos, asr_pos + n_text + 1, device=self.device)
        # attentions: list(L) of [1,H,T,T]
        A = torch.stack([a[0][:, rows][:, :, audio_pos] for a in attns], 0)  # [L,H,n_rows,n_audio]
        return A.float(), wtoks, out

    def align(self, wav, text, **cfg):
        words = remove_punctuation(text).split()
        if not words:
            return [], None
        A, wtoks, _ = self.get_attention(wav, words)
        return self.align_from_attention(A, wtoks, words, wav, **cfg)

    def align_from_attention(self, A, wtoks, words, wav, topk=10, medfilt=3, aggregation="topk", heads=None,
                             w_colnorm=1.0, w_rownorm=1.0, refine=False, sil_trim=False, energy_thr_db=-35.0):
        L, H, n_rows, n_audio = A.shape
        duration = len(wav) / 16000.0
        sec_per_tok = duration / n_audio if n_audio > 0 else 0.08
        if medfilt > 1:
            A = median_filter(A, medfilt)
        if heads is not None:
            sel = torch.stack([A[l, h] for l, h in heads], 0)
            scores = None
        elif aggregation == "topk":
            selected, scores = filter_attention(A, topk, w_colnorm, w_rownorm, 0.0)
            sel = torch.cat(selected, 0)
        else:
            sel = A.reshape(L * H, n_rows, n_audio)
            scores = None
        col_norm = sel.norm(dim=-2, keepdim=True).clamp_min(1e-8)
        matrix = torch.mean(sel / col_norm, 0).cpu().numpy()  # [n_rows, n_audio]
        text_idx, time_idx = dtw(-matrix)
        jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
        jump_frames = time_idx[jumps]  # frame where the path first enters row r (len n_rows)
        # word k starts when the path enters row (first token index of word k); ends at the
        # entry into the row of the first token of word k+1 (or the final row for the last word)
        starts_tok = np.cumsum([0] + [len(t) for t in wtoks[:-1]])
        ends_tok = np.append(starts_tok[1:], n_rows - 1)
        max_idx = len(jump_frames) - 1
        st = jump_frames[np.clip(starts_tok, 0, max_idx)].astype(float)
        en = jump_frames[np.clip(ends_tok, 0, max_idx)].astype(float)
        if refine:
            # sub-token refinement: move each boundary to the attention-mass crossover between
            # the two adjacent rows within +-1 frame
            for k in range(1, len(words)):
                r_prev, r_cur = ends_tok[k - 1] - 1, starts_tok[k]
                f = int(en[k - 1])
                lo, hi = max(0, f - 1), min(n_audio - 1, f + 1)
                if hi > lo and 0 <= r_prev < n_rows and r_cur < n_rows:
                    a = matrix[r_prev, lo:hi + 1]
                    b = matrix[r_cur, lo:hi + 1]
                    w = np.clip(b - a, 0, None)
                    if w.sum() > 0:
                        pos = lo + (w * np.arange(len(w))).sum() / w.sum()
                        st[k] = en[k - 1] = pos
        st_s = st * sec_per_tok
        en_s = en * sec_per_tok
        en_s[-1] = min(en_s[-1], duration)
        if sil_trim:
            st_s, en_s = trim_to_speech(wav, st_s, en_s, energy_thr_db)
        out = [{"word": w, "start": round(float(s), 3), "end": round(float(e), 3)} for w, s, e in zip(words, st_s, en_s)]
        return out, scores


def frame_energy_db(wav, sr=16000, win=0.02):
    n = int(win * sr)
    m = len(wav) // n
    if m == 0:
        return np.array([0.0])
    x = wav[: m * n].reshape(m, n)
    e = (x ** 2).mean(1) + 1e-10
    return 10 * np.log10(e / (e.max() if e.max() > 0 else 1.0))


def trim_to_speech(wav, st, en, thr_db=-35.0, win=0.02):
    e = frame_energy_db(wav, win=win)
    speech = np.where(e > thr_db)[0]
    if len(speech) == 0:
        return st, en
    first = speech[0] * win
    last = (speech[-1] + 1) * win
    st = st.copy(); en = en.copy()
    if st[0] < first < en[0]:
        st[0] = first
    if en[-1] > last > st[-1]:
        en[-1] = last
    return st, en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Taraneh-Q0.6B or Taraneh-Q1.7B checkpoint")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--medfilt", type=int, default=3)
    ap.add_argument("--aggr", default="topk", choices=["topk", "all"])
    ap.add_argument("--heads", default=None, help="json file with list of [layer,head] to use instead of topk")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--sil_trim", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save_scores", action="store_true", help="store selected head list per utt")
    ap.add_argument("--variants", default=None,
                    help="JSON file: {name: {topk, medfilt, aggregation, heads(list|file), refine, sil_trim}} ; "
                         "one forward pass, one alignments.jsonl per variant under output_dir/<name>/")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    heads = json.load(open(args.heads)) if args.heads else None
    base_cfg = dict(topk=args.topk, medfilt=args.medfilt, aggregation=args.aggr, heads=heads,
                    refine=args.refine, sil_trim=args.sil_trim)
    if args.variants:
        variants = json.load(open(args.variants))
        for name, cfg in variants.items():
            if isinstance(cfg.get("heads"), str):
                cfg["heads"] = json.load(open(cfg["heads"]))
            full = dict(base_cfg); full.update(cfg); variants[name] = full
    else:
        variants = {"": base_cfg}
    al = QwenAttnAligner(args.model_path, device=args.device)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    head_counter = {}
    files = {}
    for name in variants:
        d = os.path.join(args.output_dir, name) if name else args.output_dir
        os.makedirs(d, exist_ok=True)
        files[name] = open(os.path.join(d, "alignments.jsonl"), "w", encoding="utf-8")
    for i, r in enumerate(rows):
        wav, sr = sf.read(r["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(-1)
        words_txt = remove_punctuation(r["text"]).split()
        try:
            A, wtoks, _ = al.get_attention(wav, words_txt)
            err = None
        except Exception as e:  # noqa: BLE001
            A, err = None, str(e)
        for name, cfg in variants.items():
            if err is not None:
                rec = {"path": r["path"], "text": r["text"], "error": err}
            else:
                try:
                    words, scores = al.align_from_attention(A, wtoks, words_txt, wav, **cfg)
                    rec = {"path": r["path"], "text": r["text"], "words": words}
                    if scores is not None and args.save_scores:
                        for _, (l, h) in scores:
                            head_counter[f"{l},{h}"] = head_counter.get(f"{l},{h}", 0) + 1
                except Exception as e:  # noqa: BLE001
                    rec = {"path": r["path"], "text": r["text"], "error": str(e)}
            files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    if head_counter:
        json.dump(head_counter, open(os.path.join(args.output_dir, "head_counts.json"), "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
