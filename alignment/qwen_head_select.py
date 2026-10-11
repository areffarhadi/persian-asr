#!/usr/bin/env python
"""Gold-guided attention-head analysis for the Qwen3-ASR internal aligner.

For every (layer, head) run DTW on that head alone (batched over heads) and measure word-boundary
error against FarsDat gold on a dev split.  Then evaluate combinations (column-normalised mean of
the top-k heads) and write the best head list for qwen_attn_align.py --heads.
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
from alignment.qwen_attn_align import QwenAttnAligner  # noqa: E402
from alignment.whisper_forced_align import median_filter, remove_punctuation  # noqa: E402


def batched_dtw(cost):
    """cost: [B, n, m] torch tensor. Returns for each b the list of (text_idx, time_idx) path arrays."""
    B, n, m = cost.shape
    inf = float("inf")
    acc = torch.full((B, n + 1, m + 1), inf, device=cost.device, dtype=torch.float32)
    acc[:, 0, 0] = 0.0
    step = torch.zeros((B, n + 1, m + 1), device=cost.device, dtype=torch.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cands = torch.stack([acc[:, i - 1, j - 1], acc[:, i - 1, j], acc[:, i, j - 1]], 0)  # [3,B]
            mn, arg = cands.min(0)
            acc[:, i, j] = cost[:, i - 1, j - 1] + mn
            step[:, i, j] = arg.to(torch.int8)
    step = step.cpu().numpy()
    paths = []
    for b in range(B):
        i, j = n, m
        ti, tj = [], []
        while i > 0 or j > 0:
            ti.append(i - 1)
            tj.append(j - 1)
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                a = step[b, i, j]
                if a == 0:
                    i, j = i - 1, j - 1
                elif a == 1:
                    i -= 1
                else:
                    j -= 1
        ti.reverse()
        tj.reverse()
        paths.append((np.array(ti), np.array(tj)))
    return paths


def path_to_word_times(text_idx, time_idx, starts_tok, ends_tok, sec_per_tok):
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    jf = time_idx[jumps]
    mx = len(jf) - 1
    st = jf[np.clip(starts_tok, 0, mx)] * sec_per_tok
    en = jf[np.clip(ends_tok, 0, mx)] * sec_per_tok
    return st, en


def boundary_errors(st, en, gold_words, skip_first_start=True):
    gs = np.array([w["start"] for w in gold_words])
    ge = np.array([w["end"] for w in gold_words])
    errs = [np.abs(en - ge)]
    errs.append(np.abs(st[1:] - gs[1:]) if skip_first_start else np.abs(st - gs))
    return np.concatenate(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--manifest", required=True, help="JSONL of gold word times, keyed by audio basename")
    ap.add_argument("--csv", required=True, help="CSV with path,text for the head-selection split")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--medfilt", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip_combos", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gold = {os.path.basename(json.loads(l)["path"]): json.loads(l) for l in open(args.manifest, encoding="utf-8")}
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    al = QwenAttnAligner(args.model_path, device=args.device)

    t0 = time.time()
    cache = []  # per clip: (A [LH, n_rows, n_audio] cpu fp16, starts_tok, ends_tok, sec_per_tok, gold words)
    for i, r in enumerate(rows):
        wav, _ = sf.read(r["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(-1)
        words = remove_punctuation(r["text"]).split()
        g = gold[os.path.basename(r["path"])]
        if len(words) != len(g["words"]):
            continue
        A, wtoks, _ = al.get_attention(wav, words)
        if args.medfilt > 1:
            A = median_filter(A, args.medfilt)
        L, H, n_rows, n_audio = A.shape
        starts_tok = np.cumsum([0] + [len(t) for t in wtoks[:-1]])
        ends_tok = np.append(starts_tok[1:], n_rows - 1)
        cache.append((A.reshape(L * H, n_rows, n_audio).half().cpu(), starts_tok, ends_tok,
                      len(wav) / 16000.0 / n_audio, g["words"]))
        if (i + 1) % 50 == 0:
            print(f"cached {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    LH = cache[0][0].shape[0]
    print("clips", len(cache), "heads", LH, "L,H", L, H, flush=True)

    # ---- per-head evaluation ----
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

    # ---- combinations of top-k heads ----
    results = {}
    if args.skip_combos:
        for k in [1,2,3,5,10]:
            json.dump([[int(h // H), int(h % H)] for h in order[:k]], open(os.path.join(args.out_dir, f"heads_top{k}.json"), "w"))
        print("skip combos"); return
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
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
    json.dump({"results": results, "best_k": best_k, "heads": heads}, open(os.path.join(args.out_dir, "combo_results.json"), "w"), indent=1)
    json.dump(heads, open(os.path.join(args.out_dir, f"heads_top{best_k}.json"), "w"))
    for k in [5, 10, 20]:
        json.dump([[int(h // H), int(h % H)] for h in order[:k]], open(os.path.join(args.out_dir, f"heads_top{k}.json"), "w"))
    print("best_k", best_k, "done", time.time() - t0)


if __name__ == "__main__":
    main()
