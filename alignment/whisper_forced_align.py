#!/usr/bin/env python3
"""Forced word alignment with fine-tuned HF Whisper (attention + DTW).

Method: Yeh et al., Whisper Has an Internal Word Aligner (ASRU 2025)
https://github.com/30stomercury/whisper-char-alignment
"""
from __future__ import annotations

import argparse
import csv
import json
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAMPLE_RATE = 16000
HOP_LENGTH = 160
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
TOKENS_PER_SECOND = SAMPLE_RATE // AUDIO_SAMPLES_PER_TOKEN
MAX_AUDIO_SAMPLES = SAMPLE_RATE * 30

_PUNCT = string.punctuation.replace("'", "") + "«»٬٫،؛؟…"


@dataclass
class WordSpan:
    word: str
    start: float
    end: float
    chars: Optional[List[Dict[str, Any]]] = None  # per-character spans [{c,start,end}]


def remove_punctuation(text: str) -> str:
    table = str.maketrans("", "", _PUNCT)
    text = text.translate(table)
    return " ".join(w.strip(_PUNCT) for w in text.split() if w.strip(_PUNCT))


def encode_units(text: str, tokenizer, aligned_unit_type: str = "char") -> List[int]:
    if aligned_unit_type == "subword":
        return tokenizer.encode(text, add_special_tokens=False)
    tokens: List[int] = []
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    words = text.split()
    for i, wrd in enumerate(words):
        for c in wrd:
            tokens.extend(tokenizer.encode(c, add_special_tokens=False))
        if i < len(words) - 1:
            tokens.extend(space_ids)
    return tokens


def split_tokens_on_spaces(
    tokens: Sequence[int], tokenizer, aligned_unit_type: str = "char"
) -> Tuple[List[str], List[List[int]]]:
    if aligned_unit_type == "subword":
        words: List[str] = []
        word_tokens: List[List[int]] = []
        for tid in tokens:
            if tid >= tokenizer.eos_token_id:
                words.append(tokenizer.decode([tid], skip_special_tokens=False))
                word_tokens.append([tid])
                continue
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            if piece.startswith(" ") or not words:
                words.append(piece.lstrip(" "))
                word_tokens.append([tid])
            else:
                words[-1] += piece
                word_tokens[-1].append(tid)
        return words, word_tokens

    subwords, subword_tokens_list = [], []
    for tid in tokens:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        subwords.append(piece)
        subword_tokens_list.append([tid])

    words, word_tokens = [], []
    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword == " "
        if special or with_space or not words:
            words.append(subword)
            word_tokens.append(list(subword_tokens))
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
    return words, word_tokens


def median_filter(x: torch.Tensor, filter_width: int) -> torch.Tensor:
    """Median filter along last dim. Use constant pad (this CUDA build rejects reflect/replicate)."""
    if filter_width <= 1 or x.shape[-1] <= 1:
        return x
    if filter_width % 2 == 0:
        filter_width += 1
    pad = filter_width // 2
    if x.shape[-1] <= pad:
        return x
    x_pad = torch.nn.functional.pad(x, (pad, pad), mode="constant", value=0.0)
    return x_pad.unfold(-1, filter_width, 1).median(dim=-1).values


def dtw(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(cost[i - 1, j - 1])
            acc[i, j] = c + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            argmin = int(np.argmin(choices))
            if argmin == 0:
                i, j = i - 1, j - 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()
    text_idx = np.array([p[0] for p in path if p[0] >= 0 and p[1] >= 0], dtype=np.int32)
    time_idx = np.array([p[1] for p in path if p[0] >= 0 and p[1] >= 0], dtype=np.int32)
    return text_idx, time_idx


def frame_energy_db(wav, sr=16000, win=0.02):
    n = int(win * sr)
    m = len(wav) // n
    if m == 0:
        return np.array([0.0])
    x = np.asarray(wav[: m * n], dtype=np.float32).reshape(m, n)
    e = (x ** 2).mean(1) + 1e-10
    return 10 * np.log10(e / max(e.max(), 1e-10))


def trim_to_speech(wav, st, en, thr_db=-35.0, win=0.02):
    """Clamp the first word start / last word end to the first/last frame above an energy
    threshold (relative to the loudest frame). Attention-DTW forces the first word to t=0."""
    e = frame_energy_db(wav, win=win)
    speech = np.where(e > thr_db)[0]
    st = np.array(st, dtype=float).copy()
    en = np.array(en, dtype=float).copy()
    if len(speech) == 0:
        return st, en
    first = speech[0] * win
    last = (speech[-1] + 1) * win
    if st[0] < first < en[0]:
        st[0] = first
    if en[-1] > last > st[-1]:
        en[-1] = last
    return st, en


def coverage_penalty(attn: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    coverage = torch.sum(attn, dim=0)
    penalty = torch.max(coverage, coverage.clone().fill_(threshold)).sum(-1)
    return penalty - coverage.size(-1) * threshold


def filter_attention(attns, topk=20, w_colnorm=1.0, w_rownorm=1.0, w_coverage=0.0):
    n_layers, n_heads = attns.size(0), attns.size(1)
    score_matrix = torch.zeros(n_layers, n_heads, device=attns.device)
    if w_colnorm > 0:
        score_matrix += w_colnorm * attns.norm(dim=-2).sum(-1)
    if w_rownorm > 0:
        score_matrix += w_rownorm * attns.norm(dim=-1).sum(-1)
    scores = []
    for l in range(n_layers):
        for h in range(n_heads):
            score = score_matrix[l, h]
            if w_coverage > 0:
                score = score - w_coverage * coverage_penalty(attns[l, h])
            scores.append((score.item(), (l, h)))
    scores_sorted = sorted(scores)[-topk:]
    selected = [attns[l, h].unsqueeze(0) for _, (l, h) in scores_sorted]
    return selected, scores_sorted


def force_align_from_weights(
    ws, text_tokens, tokenizer, aligned_unit_type="char", aggregation="topk",
    topk=10, sot_sequence_len=3, w_colnorm=1.0, w_rownorm=1.0, w_coverage=0.0, heads=None,
):
    scores = None
    if heads is not None:
        sel = torch.stack([ws[l, h] for l, h in heads], 0)
        col_norm = sel.norm(dim=-2, keepdim=True).clamp_min(1e-8)
        matrix = torch.mean(sel / col_norm, 0)
    elif aggregation == "mean":
        ws = ws / ws.norm(dim=-2, keepdim=True)
        ws = ws[ws.size(0) // 2 :]
        matrix = ws.mean(axis=(0, 1))
    elif aggregation == "topk":
        selected, scores = filter_attention(ws, topk, w_colnorm, w_rownorm, w_coverage)
        matrix = torch.cat(selected, 0)
        col_norm = matrix.norm(dim=-2, keepdim=True).clamp_min(1e-8)
        matrix = torch.mean(matrix / col_norm, 0)
    else:
        raise ValueError(aggregation)
    matrix = matrix[sot_sequence_len:-1].detach().float().cpu().numpy()
    text_indices, time_indices = dtw(-matrix)
    words, word_tokens = split_tokens_on_spaces(
        list(text_tokens) + [tokenizer.eos_token_id], tokenizer, aligned_unit_type
    )
    if len(word_tokens) <= 1:
        return [], np.array([]), np.array([]), scores
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / float(TOKENS_PER_SECOND)
    max_idx = len(jump_times) - 1
    if max_idx < 0:
        return [], np.array([]), np.array([]), scores
    wb = np.clip(word_boundaries, 0, max_idx)
    # per-character (token) times: token k spans jump_times[k] .. jump_times[k+1]
    char_spans = []
    for wi, toks in enumerate(word_tokens[:-1]):
        base = int(word_boundaries[wi])
        cs, acc, k0 = [], [], 0
        for k, tid in enumerate(toks):
            acc.append(tid)
            piece = tokenizer.decode(acc, skip_special_tokens=False)
            if "\ufffd" in piece and k < len(toks) - 1:
                continue  # multi-token UTF-8 character: keep accumulating
            a = int(min(base + k0, max_idx)); b = int(min(base + k + 1, max_idx))
            cs.append((piece, float(jump_times[a]), float(jump_times[b])))
            acc, k0 = [], k + 1
        char_spans.append(cs)
    scores = {"scores": scores, "char_spans": char_spans}
    return words, jump_times[wb[:-1]], jump_times[wb[1:]], scores


def load_audio(path: str, target_sr: int = SAMPLE_RATE):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio, float(len(audio) / target_sr)


def build_decoder_ids(tokenizer, text_tokens):
    ids = list(tokenizer.prefix_tokens) + list(text_tokens) + [tokenizer.eos_token_id]
    return torch.tensor(ids, dtype=torch.long)


class WhisperForcedAligner:
    def __init__(self, model_path, language="Persian", device=None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained(
            model_path, language=language, task="transcribe"
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="eager",
            dtype=dtype if str(self.device).startswith("cuda") else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.sot_sequence_len = max(len(self.tokenizer.prefix_tokens) - 1, 1)

    @torch.no_grad()
    def get_cross_attentions(self, audio, decoder_ids, max_frames, medfilt_width=3, qk_scale=1.0,
                             resoftmax=True):
        """HF returns post-softmax cross-attention probabilities. The original paper code hooks
        pre-softmax QK logits and applies softmax; re-applying softmax to probabilities flattens the
        matrices (resoftmax=True keeps the legacy behaviour, False uses the probabilities directly)."""
        feats = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        feats = feats.to(self.device, dtype=self.model.dtype)
        decoder_ids = decoder_ids.unsqueeze(0).to(self.device)
        out = self.model(
            input_features=feats, decoder_input_ids=decoder_ids,
            output_attentions=True, return_dict=True, use_cache=False,
        )
        weights = torch.stack([a[0] for a in out.cross_attentions], dim=0)
        weights = weights[..., :max_frames]
        weights = median_filter(weights, medfilt_width)
        if resoftmax:
            weights = (weights * qk_scale).softmax(dim=-1)
        return weights.float()

    def align(self, audio, text, *, aligned_unit_type="char", aggregation="topk",
              topk=10, medfilt_width=3, w_colnorm=1.0, w_rownorm=1.0, w_coverage=0.0,
              clean_text=True, resoftmax=True, heads=None, sil_trim=False, energy_thr_db=-35.0):
        prep = self.prepare(audio, text, aligned_unit_type=aligned_unit_type, medfilt_width=medfilt_width,
                            clean_text=clean_text, resoftmax=resoftmax)
        if prep is None:
            return []
        ws, text_tokens = prep
        return self.align_from_ws(ws, text_tokens, audio, aligned_unit_type=aligned_unit_type,
                                  aggregation=aggregation, topk=topk, w_colnorm=w_colnorm, w_rownorm=w_rownorm,
                                  w_coverage=w_coverage, heads=heads, sil_trim=sil_trim, energy_thr_db=energy_thr_db)

    def prepare(self, audio, text, *, aligned_unit_type="char", medfilt_width=3, clean_text=True, resoftmax=True):
        """Forward pass only: returns (cross-attention weights [L,H,T_dec,T_frames], text_tokens)."""
        if clean_text:
            text = remove_punctuation(text)
        if not text.strip():
            return None
        max_frames = max(1, min(int(len(audio) // AUDIO_SAMPLES_PER_TOKEN), 1500))
        text_tokens = encode_units(text, self.tokenizer, aligned_unit_type)
        if not text_tokens:
            return None
        decoder_ids = build_decoder_ids(self.tokenizer, text_tokens)
        if decoder_ids.numel() > 448:
            raise ValueError(f"Decoder sequence too long ({decoder_ids.numel()} > 448)")
        ws = self.get_cross_attentions(audio, decoder_ids, max_frames, medfilt_width=medfilt_width,
                                       resoftmax=resoftmax)
        return ws, text_tokens

    def align_from_ws(self, ws, text_tokens, audio, *, aligned_unit_type="char", aggregation="topk", topk=10,
                      w_colnorm=1.0, w_rownorm=1.0, w_coverage=0.0, heads=None, sil_trim=False,
                      energy_thr_db=-35.0):
        duration = len(audio) / float(SAMPLE_RATE)
        words, starts, ends, extra = force_align_from_weights(
            ws, text_tokens, self.tokenizer, aligned_unit_type=aligned_unit_type,
            aggregation=aggregation, topk=topk, sot_sequence_len=self.sot_sequence_len,
            w_colnorm=w_colnorm, w_rownorm=w_rownorm, w_coverage=w_coverage, heads=heads,
        )
        spans = []
        char_spans = (extra or {}).get("char_spans") if isinstance(extra, dict) else None
        n = min(len(words), len(starts), len(ends))
        for i in range(n):
            w = words[i]
            if (not w or not w.strip() or w.startswith("<|")):
                continue
            s = float(max(0.0, min(starts[i], duration)))
            e = float(max(s, min(ends[i], duration)))
            chars = None
            if char_spans is not None and i < len(char_spans):
                chars = [{"c": c, "start": float(max(0.0, min(a, duration))), "end": float(max(0.0, min(b, duration)))}
                         for c, a, b in char_spans[i] if c.strip(" \u200c\t\n")]
            spans.append(WordSpan(word=w.strip(), start=s, end=e, chars=chars))
        if sil_trim and spans:
            st, en = trim_to_speech(audio, np.array([s.start for s in spans]), np.array([s.end for s in spans]),
                                    energy_thr_db)
            new = []
            for sp, a, b in zip(spans, st, en):
                chars = sp.chars
                if chars:
                    chars = [dict(c) for c in chars]
                    chars[0]["start"] = float(a); chars[-1]["end"] = float(b)
                    for c in chars:  # keep chars inside the (possibly shrunk) word span
                        c["start"] = min(max(c["start"], float(a)), float(b)); c["end"] = min(max(c["end"], c["start"]), float(b))
                new.append(WordSpan(word=sp.word, start=float(a), end=float(b), chars=chars))
            spans = new
        return spans

    def align_file(self, audio_path, text, **kwargs):
        audio, _ = load_audio(audio_path)
        if len(audio) > MAX_AUDIO_SAMPLES:
            audio = audio[:MAX_AUDIO_SAMPLES]
        return self.align(audio, text, **kwargs)


def _align_kwargs(args):
    return dict(
        aligned_unit_type=args.aligned_unit_type, aggregation=args.aggr, topk=args.topk,
        medfilt_width=args.medfilt_width, w_colnorm=args.w_colnorm, w_rownorm=args.w_rownorm,
        w_coverage=args.w_coverage, clean_text=not args.no_clean,
        resoftmax=not args.no_resoftmax, sil_trim=args.sil_trim,
        heads=[tuple(h) for h in json.load(open(args.heads))] if args.heads else None,
    )


def run_single(args, aligner):
    spans = aligner.align_file(args.audio, args.text, **_align_kwargs(args))
    rows = [asdict(s) for s in spans]
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            with out.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["word", "start", "end"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        else:
            out.write_text(json.dumps({"audio": args.audio, "text": args.text, "words": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(rows)} words -> {out}")
    else:
        for s in spans:
            print(f"{s.start:.2f}\t{s.end:.2f}\t{s.word}")


def run_csv_variants(args, aligner):
    """One forward pass per clip; several post-processing variants -> output_dir/<name>/alignments.jsonl.
    Variant keys: aggr, topk, heads (list or json path), sil_trim, w_colnorm, w_rownorm, w_coverage.
    Shared (forward-pass) settings come from the CLI: --aligned_unit_type --medfilt_width --no_resoftmax."""
    variants = json.load(open(args.variants))
    for name, cfg in variants.items():
        if isinstance(cfg.get("heads"), str):
            cfg["heads"] = [tuple(h) for h in json.load(open(cfg["heads"]))]
        elif cfg.get("heads"):
            cfg["heads"] = [tuple(h) for h in cfg["heads"]]
    out_dir = Path(args.output_dir)
    files = {}
    for name in variants:
        d = out_dir / name; d.mkdir(parents=True, exist_ok=True)
        files[name] = (d / "alignments.jsonl").open("w", encoding="utf-8")
    n = 0
    with open(args.csv, encoding="utf-8") as f_in:
        for i, row in enumerate(csv.DictReader(f_in)):
            if args.limit is not None and i >= args.limit:
                break
            path, text = row["path"], row["text"]
            try:
                audio, _ = load_audio(path)
                if len(audio) > MAX_AUDIO_SAMPLES:
                    audio = audio[:MAX_AUDIO_SAMPLES]
                prep = aligner.prepare(audio, text, aligned_unit_type=args.aligned_unit_type,
                                       medfilt_width=args.medfilt_width, clean_text=not args.no_clean,
                                       resoftmax=not args.no_resoftmax)
                err = None
            except Exception as e:
                prep, err = None, str(e)
            for name, cfg in variants.items():
                if err is not None or prep is None:
                    rec = {"path": path, "text": text, "error": err or "empty"}
                else:
                    try:
                        spans = aligner.align_from_ws(
                            prep[0], prep[1], audio, aligned_unit_type=args.aligned_unit_type,
                            aggregation=cfg.get("aggr", args.aggr), topk=cfg.get("topk", args.topk),
                            w_colnorm=cfg.get("w_colnorm", args.w_colnorm), w_rownorm=cfg.get("w_rownorm", args.w_rownorm),
                            w_coverage=cfg.get("w_coverage", args.w_coverage), heads=cfg.get("heads"),
                            sil_trim=cfg.get("sil_trim", False))
                        rec = {"path": path, "text": text, "words": [asdict(s) for s in spans]}
                    except Exception as e:
                        rec = {"path": path, "text": text, "error": str(e)}
                files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n} clips", flush=True)
    for f in files.values():
        f.close()
    print(f"Done variants -> {out_dir}")


def run_csv(args, aligner):
    if args.variants:
        return run_csv_variants(args, aligner)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "alignments.jsonl"
    kw = _align_kwargs(args)
    n_ok = n_err = 0
    with open(args.csv, encoding="utf-8") as f_in, summary_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or "path" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit("CSV must have columns: path,text")
        for i, row in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break
            path, text = row["path"], row["text"]
            try:
                spans = aligner.align_file(path, text, **kw)
                f_out.write(json.dumps({"path": path, "text": text, "words": [asdict(s) for s in spans]}, ensure_ascii=False) + "\n")
                n_ok += 1
                if args.verbose:
                    print(f"OK {path} ({len(spans)} words)")
            except Exception as e:
                n_err += 1
                f_out.write(json.dumps({"path": path, "text": text, "error": str(e)}, ensure_ascii=False) + "\n")
                print(f"ERR {path}: {e}", file=sys.stderr)
    print(f"Done. ok={n_ok} err={n_err} -> {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Forced alignment with fine-tuned HF Whisper")
    p.add_argument("--model_path", required=True, help="Taraneh-WMedium or Taraneh-WLarge checkpoint")
    p.add_argument("--language", default="Persian")
    p.add_argument("--device", default=None)
    p.add_argument("--audio"); p.add_argument("--text"); p.add_argument("--csv")
    p.add_argument("--output"); p.add_argument("--output_dir", default="align_out")
    p.add_argument("--limit", type=int); p.add_argument("--verbose", action="store_true")
    p.add_argument("--aligned_unit_type", default="char", choices=["char", "subword"])
    p.add_argument("--aggr", default="topk", choices=["mean", "topk"])
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--medfilt_width", type=int, default=3)
    p.add_argument("--w_colnorm", type=float, default=1.0)
    p.add_argument("--w_rownorm", type=float, default=1.0)
    p.add_argument("--w_coverage", type=float, default=0.0)
    p.add_argument("--no_clean", action="store_true")
    p.add_argument("--no_resoftmax", action="store_true",
                   help="use HF attention probabilities directly (do not re-apply softmax)")
    p.add_argument("--heads", default=None, help="json list of [layer,head] to use (bypasses top-k selection)")
    p.add_argument("--sil_trim", action="store_true", help="clamp first/last word to energy-based speech region")
    p.add_argument("--variants", default=None, help="JSON {name: {aggr, topk, heads, sil_trim,...}} (csv mode)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.audio and not args.csv:
        raise SystemExit("Provide --audio/--text or --csv")
    if args.audio and args.text is None:
        raise SystemExit("--audio requires --text")
    print(f"Loading model from {args.model_path} ...")
    aligner = WhisperForcedAligner(args.model_path, language=args.language, device=args.device)
    print(f"Device={aligner.device} sot_sequence_len={aligner.sot_sequence_len}")
    if args.csv:
        run_csv(args, aligner)
    else:
        run_single(args, aligner)


if __name__ == "__main__":
    main()
