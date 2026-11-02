"""Evaluate the pipeline end to end against your Pinecone index.

    python eval/run_eval.py                       # default namespace (ingest data/sample first)
    python eval/run_eval.py --namespace handbook --model google_genai:gemini-3.8-flash
    python eval/run_eval.py --delay 4             # pause between questions (free-tier rate limits)

Metrics
- retrieval hit rate : expected source file is among the CITED sources
- answer accuracy    : every expected fact string appears in the answer
- routing accuracy   : the router chose the expected strategy (retrieve / document / direct)
- grounded rate      : answers that passed the self-verification check
- refusal accuracy   : out-of-scope questions are declined instead of hallucinated
- latency            : mean and p95 seconds per question

Writes eval/results/<timestamp>.json and .md (a table you can paste into a report).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docupilot.graph import DocuPilot  # noqa: E402
from docupilot.prompts import NO_ANSWER  # noqa: E402


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--cases", type=Path, default=Path(__file__).parent / "eval_set.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    bot = DocuPilot(namespace=args.namespace, llm_model=args.model)
    bot.cache = None  # every question must really run, or the scores would be meaningless
    rows = []

    for i, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        res = bot.ask(c["question"], thread_id=str(uuid.uuid4()))  # fresh thread per question
        latency = time.perf_counter() - t0
        answer = res["answer"]
        cited = [s["source"] for s in res["sources"] if s.get("cited")] or [s["source"] for s in res["sources"]]
        row = {"question": c["question"], "answer": answer, "route": res["route"], "expected_route": c.get("route"),
               "latency_s": round(latency, 2), "error": res["error"], "grounded": res["grounded"], "cited": cited}

        if c.get("should_decline"):
            row["kind"] = "decline"
            row["passed"] = answer.startswith(NO_ANSWER[:30]) or res["used_web"] or not res["sources"]
        elif c.get("route") == "direct":
            row["kind"] = "direct"
            row["passed"] = res["route"] == "direct" and not res["error"]
        else:
            row["kind"] = "answer"
            row["hit"] = c["source"] in cited
            row["correct"] = all(e.lower() in answer.lower() for e in c["expect"])
            row["passed"] = row["hit"] and row["correct"]
        row["route_ok"] = c.get("route") is None or res["route"] == c.get("route")
        rows.append(row)
        print(f"[{i:>2}/{len(cases)}] {'PASS' if row['passed'] else 'FAIL'}  {latency:5.1f}s  "
              f"{res['route'] or '-':8}  {c['question']}" + (f"   !! {res['error'][:80]}" if res["error"] else ""))
        if res["error"] and ("today's free-tier quota" in res["answer"] or "no credit" in res["answer"]):
            print("\nStopping: the AI provider's quota is used up. Partial results are saved below;"
                  " run again after the reset, or pass --model with another model.")
            break
        if args.delay:
            time.sleep(args.delay)

    if not rows:
        print("No questions were answered.")
        return
    ans = [r for r in rows if r["kind"] == "answer"]
    dec = [r for r in rows if r["kind"] == "decline"]
    routed = [r for r in rows if r["expected_route"]]
    lat = sorted(r["latency_s"] for r in rows)
    metrics = {
        "retrieval_hit_rate": pct(sum(r["hit"] for r in ans), len(ans)),
        "answer_accuracy": pct(sum(r["correct"] for r in ans), len(ans)),
        "routing_accuracy": pct(sum(r["route_ok"] for r in routed), len(routed)),
        "grounded_rate": pct(sum(bool(r["grounded"]) for r in ans), len(ans)),
        "questions_run": f"{len(rows)}/{len(cases)}",
        "refusal_accuracy": pct(sum(r["passed"] for r in dec), len(dec)),
        "errors": sum(bool(r["error"]) for r in rows),
        "latency_mean_s": round(statistics.mean(lat), 2),
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
    }

    print("\nMetrics")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"timestamp": stamp, "model": bot.model, "settings": {
        "embedding_model": bot.settings.embedding_model, "top_k": bot.settings.top_k,
        "fetch_k": bot.settings.fetch_k, "rerank": bot.settings.rerank, "chunk_size": bot.settings.chunk_size}}
    (out_dir / f"{stamp}.json").write_text(json.dumps({**meta, "metrics": metrics, "rows": rows}, indent=2,
                                                      ensure_ascii=False), encoding="utf-8")
    md = [f"# Evaluation run {stamp}", "", f"Model: `{bot.model}` · Embeddings: `{bot.settings.embedding_model}` · "
          f"TOP_K={bot.settings.top_k} · rerank={bot.settings.rerank}", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in metrics.items()]
    md += ["", "| # | Question | Route | Result | Latency |", "|---|---|---|---|---|"]
    md += [f"| {i} | {r['question']} | {r['route'] or '-'} | {'✅' if r['passed'] else '❌'} | {r['latency_s']}s |"
           for i, r in enumerate(rows, 1)]
    (out_dir / f"{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved eval/results/{stamp}.json and .md")


if __name__ == "__main__":
    main()
