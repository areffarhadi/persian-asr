#!/usr/bin/env python
"""Gold-guided cross-attention head analysis for the fine-tuned Whisper aligner (FarsDat dev split).
Also compares legacy re-softmax vs raw HF probabilities."""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from alignment.qwen_head_select import batched_dtw, boundary_errors, path_to_word_times  # noqa: E402
from alignment.whisper_forced_align import (  # noqa: E402
    TOKENS_PER_SECOND, AUDIO_SAMPLES_PER_TOKEN, WhisperForcedAligner, build_decoder_ids, encode_units,
    load_audio, remove_punctuation, split_tokens_on_spaces,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--manifest", required=True, help="JSONL of gold word times, keyed by audio basename")
    ap.add_argument("--csv", required=True, help="CSV with path,text for the head-selection split")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--medfilt", type=int, default=3)
    ap.add_argument("--resoftmax", action="store_true", help="legacy behaviour (softmax over probabilities)")
    ap.add_argument("--unit", default="char", choices=["char", "subword"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip_combos", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gold = {os.path.basename(json.loads(l)["path"]): json.loads(l) for l in open(args.manifest, encoding="utf-8")}
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    al = WhisperForcedAligner(args.model_path, language="Persian", device=args.device)
    tok = al.tokenizer
    t0 = time.time()
    cache = []
    for i, r in enumerate(rows):
        audio, _ = load_audio(r["path"])
        text = remove_punctuation(r["text"])
        g = gold[os.path.basename(r["path"])]
        if len(text.split()) != len(g["words"]):
            continue
        text_tokens = encode_units(text, tok, args.unit)
        dec = build_decoder_ids(tok, text_tokens)
        max_frames = max(1, min(int(len(audio) // AUDIO_SAMPLES_PER_TOKEN), 1500))
        ws = al.get_cross_attentions(audio, dec, max_frames, medfilt_width=args.medfilt, resoftmax=args.resoftmax)
        ws = ws[:, :, al.sot_sequence_len:-1]  # rows: <|notimestamps|> + text tokens
        words, word_tokens = split_tokens_on_spaces(list(text_tokens) + [tok.eos_token_id], tok, args.unit)
        wb = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        starts_tok, ends_tok = wb[:-1], wb[1:]
        if len(starts_tok) != len(g["words"]):
            continue
        L, H, n_rows, n_fr = ws.shape
        cache.append((ws.reshape(L * H, n_rows, n_fr).half().cpu(), starts_tok, ends_tok, 1.0 / TOKENS_PER_SECOND, g["words"]))
        if (i + 1) % 50 == 0:
            print(f"cached {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    LH = cache[0][0].shape[0]
    print("clips", len(cache), "heads", LH, "L,H", L, H, flush=True)

    per_head_err = [[] for _ in range(LH)]
    for ci, (A, st_tok, en_tok, spt, gw) in enumerate(cache):
        M = A.to(args.device).float()
        M = M / M.norm(dim=-2, keepdim=True).clamp_min(1e-8)
        paths = batched_dtw(-M)
        for h, (ti, tj) in enumerate(paths):
            st, en = path_to_word_times(ti, tj, st_tok, en_tok, spt)
            per_head_err[h].append(boundary_errors(st, en, gw))
        if (ci + 1) % 50 == 0:
            print(f"per-head dtw {ci + 1}/{len(cache)} {time.time() - t0:.0f}s", flush=True)
    head_mae = np.array([np.concatenate(e).mean() for e in per_head_err])
    head_acc50 = np.array([(np.concatenate(e) <= 0.05).mean() for e in per_head_err])
    order = np.argsort(head_mae)
    ranking = [{"layer": int(h // H), "head": int(h % H), "mae": float(head_mae[h]), "acc50": float(head_acc50[h])} for h in order]
    json.dump(ranking, open(os.path.join(args.out_dir, "head_ranking.json"), "w"), indent=1)
    print("best single heads:")
    for r in ranking[:15]:
        print(r)
    results = {}
    if args.skip_combos:
        for k in [1, 2, 3, 5, 10]:
            json.dump([[int(h // H), int(h % H)] for h in order[:k]], open(os.path.join(args.out_dir, f"heads_top{k}.json"), "w"))
        print("skip combos")
        return
    for k in [1, 2, 3, 5, 10]:
        sel = order[:k]
        errs = []
        for A, st_tok, en_tok, spt, gw in cache:
            M = A[sel].to(args.device).float()
            M = M / M.norm(dim=-2, keepdim=True).clamp_min(1e-8)
            M = M.mean(0, keepdim=True)
            (ti, tj), = batched_dtw(-M)
            st, en = path_to_word_times(ti, tj, st_tok, en_tok, spt)
            errs.append(boundary_errors(st, en, gw))
        e = np.concatenate(errs)
        results[k] = {"mae": float(e.mean()), "median": float(np.median(e)), "acc50": float((e <= 0.05).mean()),
                      "acc100": float((e <= 0.1).mean())}
        print(f"top-{k}: {results[k]}", flush=True)
    best_k = min(results, key=lambda k: results[k]["mae"])
    heads = [[int(h // H), int(h % H)] for h in order[:best_k]]
    json.dump({"results": results, "best_k": best_k, "heads": heads, "resoftmax": args.resoftmax},
              open(os.path.join(args.out_dir, "combo_results.json"), "w"), indent=1)
    for k in sorted(set([best_k, 5, 10, 20])):
        json.dump([[int(h // H), int(h % H)] for h in order[:k]], open(os.path.join(args.out_dir, f"heads_top{k}.json"), "w"))
    print("best_k", best_k, "done", time.time() - t0)


if __name__ == "__main__":
    main()
