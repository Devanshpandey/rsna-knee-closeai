#!/usr/bin/env python3
"""Phase 1: extract the 12 competition labels from multilingual radiology reports with an open LLM (vLLM).

Runs on one A100/H100. Outputs one row per study with 0/1 per label + parse status.

Usage:
  python extract_labels_llm.py --model Qwen/Qwen2.5-14B-Instruct \
      --train-csv data/raw/train.csv --output results/labels_llm_14b.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

SYSTEM = """You are an expert musculoskeletal radiologist. You will be given a knee MRI radiology report. Reports may be written in English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, or French.

Decide, for each of the 12 findings below, whether the report indicates the finding is PRESENT (1) or ABSENT (0).

Definitions:
- "ACL": anterior cruciate ligament injury — any acute or chronic tear (partial or complete), rupture, or clearly described ACL graft failure. Intact ACL or intact graft = 0.
- "MCL": medial collateral ligament injury — sprain (any grade), partial or complete tear, or edema around the MCL described as injury.
- "Medial Meniscus": medial meniscus tear — any tear (including root, radial, complex, bucket-handle, macerated). Degeneration WITHOUT tear = 0.
- "Lateral Meniscus": lateral meniscus tear — same criteria as medial.
- "Medial OA": osteoarthritis of the MEDIAL tibiofemoral compartment — cartilage loss/chondropathy, osteophytes, or degenerative change in that compartment.
- "Lateral OA": osteoarthritis of the LATERAL tibiofemoral compartment — same criteria.
- "PF OA": patellofemoral osteoarthritis — cartilage loss/chondromalacia/osteophytes in the patellofemoral compartment.
- "Effusion": joint effusion — more than trace/physiologic joint fluid.
- "Synovitis": synovitis — synovial thickening, synovial inflammation, or explicit mention of synovitis.
- "Baker's": Baker's (popliteal) cyst of any size.
- "Contusion": bone contusion / bone bruise / traumatic bone marrow edema.
- "Fracture": any fracture (including avulsion, osteochondral, insufficiency, impaction), acute or subacute.

Rules:
- Judge ONLY from the report text.
- If a finding is explicitly negated or simply not mentioned, output 0.
- "Cannot exclude X" or equivocal wording: output 1 only if the report leans toward the finding being present.
- Answer with ONLY a JSON object, exact keys as above, values 0 or 1. No explanations."""


def build_prompt(tokenizer, report: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Report:\n{report}\n\nJSON:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in LABELS:
        v = obj.get(k)
        if v not in (0, 1, "0", "1"):
            return None
        out[k] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N reports (debug)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.train_csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} reports | model {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [build_prompt(tokenizer, str(r)[:8000]) for r in df["Report"].fillna("")]

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.92, seed=42)
    sp = SamplingParams(temperature=0.0, max_tokens=160, seed=42)
    outputs = llm.generate(prompts, sp)

    rows = []
    n_fail = 0
    for (_, rec), out in zip(df.iterrows(), outputs):
        parsed = parse_json(out.outputs[0].text)
        row = {"StudyInstanceUID": rec["StudyInstanceUID"],
               "parse_ok": int(parsed is not None)}
        if parsed is None:
            n_fail += 1
            parsed = {k: None for k in LABELS}
        row.update(parsed)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"wrote {args.output} | parse failures: {n_fail}/{len(df)}", flush=True)


if __name__ == "__main__":
    main()
