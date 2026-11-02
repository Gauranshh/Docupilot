"""Command line interface.

    python -m docupilot.cli ingest data/sample                 # index a folder or files
    python -m docupilot.cli ask "How many casual leaves do I get?" --trace
    python -m docupilot.cli ask "Summarise this" --source leave_policy.md
    python -m docupilot.cli chat                               # multi-turn chat in the terminal
    python -m docupilot.cli docs                               # list ingested documents
    python -m docupilot.cli delete leave_policy.md             # remove a document
    python -m docupilot.cli stats
    python -m docupilot.cli graph                              # print the workflow as Mermaid
    python -m docupilot.cli train-router                       # retrain the question-router model
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .errors import friendly_error


def _print_result(res: dict, show_trace: bool) -> None:
    print("\n" + res["answer"])
    cited = [s for s in res["sources"] if s.get("cited")] or res["sources"]
    if cited:
        print("\nSources:")
        for s in cited:
            score = f"  (relevance {s['score']})" if s.get("score") is not None else ""
            print(f"  [{s['n']}] {s['label']}{score}")
    if show_trace:
        print("\nAgent trace:")
        for line in res["trace"]:
            print(f"  → {line}")
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles + emoji/✓ characters

    p = argparse.ArgumentParser(prog="docupilot", description="Agentic RAG with Pinecone + LangGraph")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Index files or folders into Pinecone")
    pi.add_argument("paths", nargs="+", type=Path)
    pi.add_argument("--namespace", default=None)

    pa = sub.add_parser("ask", help="Ask a single question")
    pa.add_argument("question")
    pa.add_argument("--namespace", default=None)
    pa.add_argument("--source", action="append", help="Restrict to this file (repeatable)")
    pa.add_argument("--model", default=None, help="Override LLM_MODEL")
    pa.add_argument("--trace", action="store_true")

    pc = sub.add_parser("chat", help="Interactive multi-turn chat")
    pc.add_argument("--namespace", default=None)
    pc.add_argument("--source", action="append")
    pc.add_argument("--model", default=None)
    pc.add_argument("--trace", action="store_true")

    pd = sub.add_parser("docs", help="List ingested documents")
    pd.add_argument("--namespace", default=None)

    px = sub.add_parser("delete", help="Delete a document from the index")
    px.add_argument("name")
    px.add_argument("--namespace", default=None)

    ps = sub.add_parser("stats", help="Show index statistics")
    ps.add_argument("--namespace", default=None)

    sub.add_parser("graph", help="Print the LangGraph workflow as Mermaid")

    pt = sub.add_parser("train-router", help="Train the question-router model on data/router_train.csv")
    pt.add_argument("--data", type=Path, default=None)

    args = p.parse_args()

    try:
        if args.cmd == "ingest":
            from .ingest import ingest_paths
            report = ingest_paths(args.paths, namespace=args.namespace)
            for w in report["warnings"]:
                print(f"  ! {w}")
            total = sum(f["chunks"] for f in report["files"])
            print(f"Done: {len(report['files'])} files, {total} chunks → namespace '{report['namespace']}'"
                  + (f" ({len(report['errors'])} failed)" if report["errors"] else ""))

        elif args.cmd == "docs":
            from .config import get_settings
            from .registry import DocumentRegistry
            ns = args.namespace or get_settings().pinecone_namespace
            docs = DocumentRegistry().list(ns)
            print(f"Namespace '{ns}': {len(docs)} document(s)")
            for name, m in docs.items():
                pages = f"{m['pages']} pages, " if m.get("pages") else ""
                print(f"  • {name} ({pages}{m['chunks']} chunks, added {m['ingested_at'][:10]})")

        elif args.cmd == "delete":
            from .ingest import delete_document
            print(f"Deleted {args.name}: {delete_document(args.name, args.namespace)} vectors removed")

        elif args.cmd == "stats":
            from .vectorstore import index_stats
            for k, v in index_stats(args.namespace).items():
                print(f"{k:>22}: {v}")

        elif args.cmd == "train-router":
            from .router_model import DEFAULT_DATA, DEFAULT_MODEL, train
            m = train(args.data or DEFAULT_DATA, DEFAULT_MODEL)
            print(f"Trained on {m['examples']} labelled questions {m['per_label']}")
            print(f"  held-out accuracy      : {m['holdout_accuracy']:.1%} ({m['holdout_size']} questions)")
            print(f"  5-fold CV accuracy     : {m['cv5_accuracy_mean']:.1%} ± {m['cv5_accuracy_std']:.1%}")
            for th, v in m["confident_cv"].items():
                print(f"  confidence ≥ {th}       : handles {v['coverage']:.0%} of questions at {v['accuracy']:.1%} accuracy")
            print(f"Saved {DEFAULT_MODEL}")

        elif args.cmd == "graph":
            from .graph import DocuPilot
            print(DocuPilot().graph.get_graph().draw_mermaid())

        elif args.cmd == "ask":
            from .graph import DocuPilot
            bot = DocuPilot(namespace=args.namespace, llm_model=args.model)
            _print_result(bot.ask(args.question, sources=args.source), args.trace)

        elif args.cmd == "chat":
            from .graph import DocuPilot
            bot, thread = DocuPilot(namespace=args.namespace, llm_model=args.model), str(uuid.uuid4())
            print("DocuPilot chat — type 'exit' to quit, 'new' to start a fresh conversation.\n")
            while True:
                try:
                    q = input("you › ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in {"exit", "quit"}:
                    break
                if q.lower() == "new":
                    thread = str(uuid.uuid4())
                    print("(new conversation)\n")
                    continue
                if q:
                    _print_result(bot.ask(q, thread_id=thread, sources=args.source), args.trace)
    except Exception as e:  # noqa: BLE001
        print("\n" + friendly_error(e, getattr(args, "model", None)), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
