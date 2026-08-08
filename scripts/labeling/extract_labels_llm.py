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

For each of the 12 findings below, output your confidence from 0 to 100 that an expert image annotator would mark the finding PRESENT on the images, given this report. The annotators used the specific thresholds below — follow them exactly.

Definitions and thresholds:
- "ACL": partial or complete tear/rupture with fiber discontinuity, or failed graft. NOT present: intact ligament, grade 1-2 signal change with intact contour/fibers, interstitial signal or "interstitial tear" with fibers intact, mucoid or cystic degeneration, bony avulsion at the insertion without ligament fiber tear.
- "MCL": grade 2 or 3 sprain, or partial/complete tear. NOT present: grade 1 sprain, periligamentous edema or fluid around an intact ligament, chronic thickening or thinning, retinaculum injury.
- "Medial Meniscus": tear of the medial meniscus reaching an articular surface, or root/radial/complex/bucket-handle/displaced/macerated tear. NOT present: intrasubstance or mucoid degeneration, grade 1-2 signal not reaching the surface, extrusion alone.
- "Lateral Meniscus": same criteria as medial meniscus.
- "Medial OA": degenerative change of the MEDIAL tibiofemoral compartment: cartilage thinning/chondropathy/chondral ulcers or defects, osteophytes, degenerative subchondral change. "Tricompartmental" or "incipient OA of all compartments" counts for all three OA labels.
- "Lateral OA": same criteria, LATERAL tibiofemoral compartment.
- "PF OA": patellofemoral compartment degenerative change INCLUDING chondromalacia / patellar or trochlear chondropathy, cartilage defects, or osteophytes.
- "Effusion": joint effusion described as moderate, large, significant, or unqualified. NOT present: trace/small/minimal/mild/"some" effusion, physiologic fluid, or effusion never mentioned. Do NOT infer effusion from injury severity.
- "Synovitis": synovitis, synovial thickening/hypertrophy/enhancement, inflamed plica, or Hoffa fat-pad edema/impingement. A moderate-large effusion usually implies synovitis unless the synovium is explicitly normal.
- "Baker\'s": Baker's (popliteal) cyst of any size.
- "Contusion": traumatic bone marrow edema / bone bruise / contusion of the distal femur, proximal tibia, patella, or fibular head. NOT present: subchondral edema under a chondral lesion or degenerative change, marrow edema immediately surrounding a fracture (that is Fracture, not Contusion), soft-tissue edema.
- "Fracture": ANY fracture: subchondral, nondisplaced, occult, trabecular, avulsion (including Segond and tibial-tuberosity avulsion), impression, insufficiency, osteochondral fracture, or a fracture line inside marrow edema. Watch for non-English terms (e.g. kirik, fraktur, fraktura, fractura, fisura, katagma).

Rules:
- Judge ONLY from the report text; do not assume findings the report does not support.
- Explicitly negated or unmentioned findings: confidence near 0.
- Equivocal wording ("cannot exclude", "suspect"): intermediate confidence reflecting the wording.
- Answer with ONLY a JSON object, exact keys as above, integer values 0-100. No explanations."""


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
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not 0 <= v <= 100:
            return None
        out[k] = v / 100.0
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
    sp = SamplingParams(temperature=0.0, max_tokens=220, seed=42)
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
