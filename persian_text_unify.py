#!/usr/bin/env python3
"""
Unified Persian ASR text normalization for fair WER/CER scoring.

Final pipeline (apply identically to GT and prediction):
  1) Digits -> Persian words
  2) Punctuation / ZWNJ / diacritic removal
  3) Arabic -> Persian character mapping
  4) می/نمی join, basic ها split
  5) Compound split/join canon
  6) Colloquial -> formal
  7) Clitic/suffix detachment (ترین/ها/اش/...)
  8) آ -> ا, drop standalone ezafe ی
  9) Verb plural unify, هایی -> های
 10) Optional pair-wise spacing resolve (same chars, different spaces)

Usage:
  python persian_text_unify.py \\
      --csv /path/to/inference_results.csv \\
      --output_dir /path/to/out
"""

from __future__ import annotations

import argparse
import os
import re
import string
from pathlib import Path
from typing import Iterable, List, Tuple

import evaluate
import pandas as pd

# ---------------------------------------------------------------------------
# Digit -> Persian words
# ---------------------------------------------------------------------------
_DIGIT_MAP = str.maketrans(
    "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "012345678901234567890123456789",
)
_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = [
    "ده", "یازده", "دوازده", "سیزده", "چهارده",
    "پانزده", "شانزده", "هفده", "هجده", "نوزده",
]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = [
    "", "صد", "دویست", "سیصد", "چهارصد",
    "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد",
]
_SCALES = [
    (10**12, "تریلیون"),
    (10**9, "میلیارد"),
    (10**6, "میلیون"),
    (10**3, "هزار"),
]


def _three_digits_to_words(n: int) -> str:
    if n == 0:
        return ""
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rem >= 20:
        t, o = divmod(rem, 10)
        parts.append(_TENS[t])
        if o:
            parts.append(_ONES[o])
    elif rem >= 10:
        parts.append(_TEENS[rem - 10])
    elif rem > 0:
        parts.append(_ONES[rem])
    return " و ".join(parts)


def number_to_persian_words(n: int) -> str:
    if n < 0:
        return "منفی " + number_to_persian_words(-n)
    if n == 0:
        return "صفر"
    parts = []
    for scale_val, scale_name in _SCALES:
        if n >= scale_val:
            q, n = divmod(n, scale_val)
            chunk = _three_digits_to_words(q) if q < 1000 else number_to_persian_words(q)
            parts.append(f"{chunk} {scale_name}".strip())
    if n:
        parts.append(_three_digits_to_words(n))
    return " و ".join(p for p in parts if p)


def _digit_token_to_words(token: str) -> str:
    raw = token.translate(_DIGIT_MAP)
    if re.fullmatch(r"0+", raw):
        return " ".join(["صفر"] * len(raw)) if len(raw) > 1 else "صفر"
    return number_to_persian_words(int(raw))


def convert_digits_to_persian_words(text: str) -> str:
    if not text:
        return text

    def _repl_decimal(match: re.Match) -> str:
        return f"{_digit_token_to_words(match.group(1))} ممیز {_digit_token_to_words(match.group(2))}"

    text = re.sub(
        r"([0-9۰-۹٠-٩]+)[.\u066B]([0-9۰-۹٠-٩]+)",
        _repl_decimal,
        text,
    )
    text = re.sub(r"[0-9۰-۹٠-٩]+", lambda m: _digit_token_to_words(m.group(0)), text)
    return text


# ---------------------------------------------------------------------------
# Basic normalize
# ---------------------------------------------------------------------------
def normalize_basic(text: str) -> str:
    """Digits, punct/ZWNJ, Arabic->Persian, می join, ها split."""
    if text is None:
        return ""
    text = str(text)
    text = convert_digits_to_persian_words(text)

    persian_punctuations = '''`÷×؛<>٬٫٪٭,٫،؟""⟪⟫«»٬٫:/.-!()٭،؟[]⟨⟩~*}٪{×+|️•←→‌'''
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    for char in persian_punctuations:
        text = text.replace(char, " ")
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = " ".join(text.split())

    dictionary_mapping = {
        "ك": "ک", "دِ": "د", "بِ": "ب", "زِ": "ز", "ذِ": "ذ", "شِ": "ش", "سِ": "س", "ى": "ی",
        "ي": "ی", "أ": "ا", "ؤ": "و", "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
        "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
        "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", "ﺍ": "ا", "ة": "ه",
        "ﯾ": "ی", "ﯿ": "ی", "ﺒ": "ب", "ﺖ": "ت", "ﺪ": "د", "ﺮ": "ر", "ﺴ": "س", "ﺷ": "ش",
        "ﺸ": "ش", "ﻋ": "ع", "ﻤ": "م", "ﻥ": "ن", "ﻧ": "ن", "ﻭ": "و", "ﺭ": "ر", "ﮔ": "گ",
    }
    for old_char, new_char in dictionary_mapping.items():
        text = text.replace(old_char, new_char)

    text = re.sub(r"\b(می|نمی)\s+", r"\1", text)
    text = re.sub(r"(\S+)(ها|های|هایمان)\b", r"\1 \2", text)
    text = re.sub(r"(?<=\S)هه(?=\S)", "ه ه", text)
    text = re.sub(r"(\S+ه)ی\b", r"\1 ی", text)
    return text


# ---------------------------------------------------------------------------
# Unified ortho + morph normalize (final)
# ---------------------------------------------------------------------------
def _detach_clitics(text: str) -> str:
    rules = [
        ("هایمان", 2), ("هایتان", 2), ("هایشان", 2),
        ("هایی", 2), ("هایش", 2), ("هایت", 2), ("هایم", 2), ("های", 2),
        ("ترین", 3), ("تری", 3),
        ("مان", 3), ("تان", 3), ("شان", 3),
        ("اید", 3), ("اند", 3), ("ایم", 3),
        ("ات", 3), ("اش", 3), ("ام", 3),
        ("ها", 2),
        ("تر", 3),
    ]

    def _one_pass(tok: str) -> str:
        for suf, min_stem in rules:
            if tok.endswith(suf) and len(tok) - len(suf) >= min_stem:
                stem = tok[: -len(suf)]
                if stem and stem != suf:
                    return stem + " " + suf
        return tok

    for _ in range(3):
        text = " ".join(_one_pass(t) for t in text.split())
    return text


def normalize_unified(text: str) -> str:
    """
    Final unified normalizer used for reporting fair WER/CER.
    """
    text = normalize_basic(text)
    if not text:
        return ""

    compounds = [
        (r"\bروبرو\b", "رو به رو"),
        (r"\bروبه\s+رو\b", "رو به رو"),
        (r"\bآنرا\b", "آن را"),
        (r"\bاینرا\b", "این را"),
        (r"\bپیشبینی\b", "پیش بینی"),
        (r"\bغیرقابل\b", "غیر قابل"),
        (r"\bدوبیتی\b", "دو بیتی"),
        (r"\bهیچکس\b", "هیچ کس"),
        (r"\bهمانطور\b", "همان طور"),
        (r"\bهمینطور\b", "همین طور"),
        (r"\bدرمی\b", "در می"),
        (r"\bبرمی\b", "بر می"),
        (r"\bسختگیری\b", "سخت گیری"),
        (r"\bخوشایندند\b", "خوشایند اند"),
    ]
    for pat, repl in compounds:
        text = re.sub(pat, repl, text)

    colloquial = {
        "یه": "یک",
        "اگه": "اگر",
        "داره": "دارد",
        "دارن": "دارند",
        "میشه": "میشود",
        "میکنن": "میکنند",
        "میکنه": "میکند",
        "کنه": "کند",
        "کنن": "کنند",
        "باشه": "باشد",
        "هست": "است",
        "ست": "است",
        "خب": "خوب",
        "توی": "تو",
        "برا": "برای",
        "دیگه": "دیگر",
        "اون": "آن",
        "همون": "همان",
        "اینکه": "این که",
        "آمریکا": "امریکا",
        "امریکا": "امریکا",
        "سئوال": "سوال",
        "هیچوقت": "هیچ وقت",
        "بخصوص": "به خصوص",
        "بطور": "به طور",
        "کدامیک": "کدام یک",
        "علیرغم": "علی رغم",
        "بینهایت": "بی نهایت",
        "انقدر": "اینقدر",
        "اینقدر": "اینقدر",
        "اره": "اره",
        "آره": "اره",
        "بجای": "به جای",
    }
    words = text.split()
    out: List[str] = []
    i = 0
    while i < len(words):
        if (
            i + 2 < len(words)
            and words[i] == "رو"
            and words[i + 1] == "به"
            and words[i + 2] == "رو"
        ):
            out.extend(["رو", "به", "رو"])
            i += 3
            continue
        w = words[i]
        out.append("را" if w == "رو" else colloquial.get(w, w))
        i += 1
    text = " ".join(out)

    text = _detach_clitics(text)
    text = re.sub(r"\b(می|نمی)\s+", r"\1", text)

    for a, b in {"ئ": "ی", "ؤ": "و", "أ": "ا", "إ": "ا", "ة": "ه"}.items():
        text = text.replace(a, b)
    text = text.replace("آ", "ا")
    text = " ".join(w for w in text.split() if w != "ی")

    text = re.sub(r"\b(کردند)\b", "کرده اند", text)
    text = re.sub(r"\b(بودند)\b", "بوده اند", text)
    text = re.sub(r"\b(میکنند)\b", "میکند اند", text)
    text = re.sub(r"\b(کنند)\b", "کند اند", text)
    text = re.sub(r"\b(نمیکنند)\b", "نمیکند اند", text)
    text = re.sub(r"\bهایی\b", "های", text)

    return " ".join(text.split())


def resolve_spacing_pair(gt: str, pred: str) -> Tuple[str, str]:
    """If texts differ only by spaces, force PRED to GT tokenization (keeps word count)."""
    if gt.replace(" ", "") == pred.replace(" ", "") and gt != pred:
        return gt, gt
    return gt, pred


def normalize_pair(gt: str, pred: str, resolve_spacing: bool = True) -> Tuple[str, str]:
    gt_u = normalize_unified(gt)
    pred_u = normalize_unified(pred)
    if resolve_spacing:
        return resolve_spacing_pair(gt_u, pred_u)
    return gt_u, pred_u


# ---------------------------------------------------------------------------
# Apply to inference CSV
# ---------------------------------------------------------------------------
def process_inference_csv(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
) -> dict:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or output_dir.name

    df = pd.read_csv(csv_path)
    # tolerate alternate column names
    col_map = {c.lower(): c for c in df.columns}
    gt_col = col_map.get("original_text") or col_map.get("reference") or col_map.get("gt")
    pred_col = col_map.get("generated_text") or col_map.get("prediction") or col_map.get("hyp")
    if not gt_col or not pred_col:
        raise ValueError(
            f"{csv_path}: need original_text/generated_text (or reference/prediction). "
            f"Got {list(df.columns)}"
        )

    df[gt_col] = df[gt_col].fillna("").astype(str)
    df[pred_col] = df[pred_col].fillna("").astype(str)
    # drop ERROR rows from scoring but keep in file
    valid = df[df[pred_col] != "ERROR"].copy()

    valid["gt_basic"] = valid[gt_col].map(normalize_basic)
    valid["pred_basic"] = valid[pred_col].map(normalize_basic)
    valid["gt_unified"] = valid[gt_col].map(normalize_unified)
    valid["pred_unified"] = valid[pred_col].map(normalize_unified)

    finals = [normalize_pair(g, p, True) for g, p in zip(valid[gt_col], valid[pred_col])]
    valid["gt_final"] = [a for a, _ in finals]
    valid["pred_final"] = [b for _, b in finals]

    wer_m = evaluate.load("wer")
    cer_m = evaluate.load("cer")

    def _score(refs: Iterable[str], hyps: Iterable[str]) -> Tuple[float, float]:
        refs, hyps = list(refs), list(hyps)
        return (
            100 * wer_m.compute(predictions=hyps, references=refs),
            100 * cer_m.compute(predictions=hyps, references=refs),
        )

    raw_w, raw_c = _score(valid[gt_col], valid[pred_col])
    basic_w, basic_c = _score(valid["gt_basic"], valid["pred_basic"])
    uni_w, uni_c = _score(valid["gt_unified"], valid["pred_unified"])
    fin_w, fin_c = _score(valid["gt_final"], valid["pred_final"])
    exact = float((valid["gt_final"] == valid["pred_final"]).mean() * 100)
    n_err = int((valid["gt_final"] != valid["pred_final"]).sum())

    out_csv = output_dir / "inference_results_unified_final.csv"
    valid.to_csv(out_csv, index=False)

    metrics_path = output_dir / "metrics_unified_final.txt"
    metrics_path.write_text(
        f"""Model: {model_name}
Source: {csv_path}
Samples scored: {len(valid)}/{len(df)}

Raw                 WER: {raw_w:.2f}%   CER: {raw_c:.2f}%
Basic normalize     WER: {basic_w:.2f}%   CER: {basic_c:.2f}%
Unified             WER: {uni_w:.2f}%   CER: {uni_c:.2f}%
Unified + spacing   WER: {fin_w:.2f}%   CER: {fin_c:.2f}%

Exact match (final): {exact:.2f}%
Remaining mismatches: {n_err}

Pipeline: digits->words, punct/ZWNJ, Arabic->Persian, compounds,
colloquial->formal, clitic detach, آ->ا, drop ezafe ی,
verb plural unify, هایی->های, spacing-resolve
""",
        encoding="utf-8",
    )

    # errors-only txt
    err_path = output_dir / "gt_vs_pred_unified_final_errors_only.txt"
    n = 0
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(
            f"{model_name} | unified final errors | "
            f"WER {fin_w:.2f}% CER {fin_c:.2f}%\n"
        )
        f.write("=" * 80 + "\n\n")
        for _, row in valid.iterrows():
            if row["gt_final"] == row["pred_final"]:
                continue
            n += 1
            sid = Path(str(row.get("path", n))).stem
            f.write(f"[{n:04d}] {sid}\n")
            f.write(f"GT_RAW  : {row[gt_col]}\n")
            f.write(f"PRED_RAW: {row[pred_col]}\n")
            f.write(f"GT_UNI  : {row['gt_final']}\n")
            f.write(f"PRED_UNI: {row['pred_final']}\n")
            f.write("-" * 80 + "\n")
        f.write(f"\nTotal errors: {n}\n")

    return {
        "model": model_name,
        "n": len(valid),
        "raw_wer": raw_w,
        "raw_cer": raw_c,
        "basic_wer": basic_w,
        "basic_cer": basic_c,
        "unified_wer": uni_w,
        "unified_cer": uni_c,
        "final_wer": fin_w,
        "final_cer": fin_c,
        "exact": exact,
        "errors": n_err,
        "metrics_path": str(metrics_path),
    }


def main():
    p = argparse.ArgumentParser(description="Unified Persian ASR normalization + WER/CER")
    p.add_argument("--csv", type=str, help="Single inference_results.csv")
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root with per-model subdirs each containing inference_results.csv",
    )
    p.add_argument(
        "--models",
        type=str,
        default="Taraneh-WBase,Taraneh-WMedium,Taraneh-WLarge,Taraneh-Q0.6B,Taraneh-Q1.7B",
        help="Comma-separated model subdir names under --root",
    )
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    if not args.csv and not args.root:
        p.error("pass --csv or --root")

    results = []
    if args.csv:
        results.append(process_inference_csv(args.csv, args.output_dir))
    else:
        root = Path(args.root)
        for name in [m.strip() for m in args.models.split(",") if m.strip()]:
            csv_path = root / name / "inference_results.csv"
            if not csv_path.exists():
                print(f"SKIP missing: {csv_path}")
                continue
            print(f"Processing {name} ...")
            results.append(process_inference_csv(csv_path, root / name, model_name=name))

    if not results:
        raise SystemExit("No results processed")

    # summary table
    summary_root = Path(args.root) if args.root else Path(args.output_dir or ".")
    summary_path = summary_root / "metrics_unified_final_summary.txt"
    lines = [
        "Unified final normalization summary",
        "=" * 72,
        f"{'Model':<16} {'Raw WER':>8} {'Basic':>8} {'Unified':>8} {'Final':>8} {'Final CER':>10} {'Exact%':>8}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r['model']:<16} {r['raw_wer']:8.2f} {r['basic_wer']:8.2f} "
            f"{r['unified_wer']:8.2f} {r['final_wer']:8.2f} {r['final_cer']:10.2f} {r['exact']:8.2f}"
        )
    lines.append("-" * 72)
    lines.append("Final = unified + spacing-resolve (recommended reporting number)")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
