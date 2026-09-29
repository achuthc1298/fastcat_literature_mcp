"""Benchmark the actual MCP RAG tools on the downloaded packed-bed papers.

uv run python scripts/benchmark_rag.py [--chunk-tokens 384 --overlap 64]
Model download is excluded if already cached. Cold time includes server-side imports/loading.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import literature as lib  # noqa: E402

PAPERS = ["d8078e47d32d773dbae7803a", "1aeda58d74687a9a65796e06", "458eea58eaaeb7cfa2389eff"]
QUESTIONS = [
    (
        "How do particle size and bed porosity affect pressure drop and heat transfer in packed-bed thermal energy storage?",
        None,
    ),
    (
        "How does decreasing particle diameter change heat transfer, pressure drop, and pumping power?",
        "A decreasing particle diameter influences",
    ),
    (
        "How does non-uniform porosity affect flow and temperature uniformity in radial-flow packed-bed storage?",
        "packing non-uniformities",
    ),
    (
        "What pressure drop and maximum overall thermal efficiency were measured in the radial-flow packed-bed experiment?",
        "71.8",
    ),
]


async def main(args):
    missing = [identifier for identifier in PAPERS if not lib.artifact_path("papers", identifier).exists()]
    if missing:
        raise SystemExit(
            "Benchmark corpus is not bundled. Download the three papers listed in "
            "reports/packed-bed-e2e-2026-09-23.md before running this benchmark. "
            f"Missing paper IDs: {', '.join(missing)}"
        )
    env = {
        **os.environ,
        "RAG_CHUNK_TOKENS": str(args.chunk_tokens),
        "RAG_CHUNK_OVERLAP": str(args.overlap),
    }
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "research_server.py")], cwd=str(ROOT), env=env
    )
    report = {
        "chunk_tokens": args.chunk_tokens,
        "overlap": args.overlap,
        "papers": PAPERS,
        "queries": [],
    }
    async with stdio_client(params) as streams:
        async with ClientSession(*streams, read_timeout_seconds=timedelta(seconds=1800)) as session:
            await session.initialize()

            async def call(name, arguments):
                start = time.perf_counter()
                result = await session.call_tool(name, arguments)
                if result.isError:
                    raise RuntimeError(str(result.content))
                data = json.loads(result.content[0].text)
                return data, round(time.perf_counter() - start, 3)

            indexed, seconds = await call("index_papers", {"paper_ids": PAPERS})
            report["index"] = indexed
            report["index_wall_seconds"] = seconds
            print("INDEX", seconds, json.dumps(indexed["papers"]), flush=True)
            cached, seconds = await call("index_papers", {"paper_ids": PAPERS})
            report["cached_index_wall_seconds"] = seconds
            assert all(p["status"] == "cached" for p in cached["papers"])
            for question, expected in QUESTIONS:
                result, seconds = await call(
                    "retrieve_evidence", {"question": question, "paper_ids": PAPERS}
                )
                hit = (
                    None
                    if expected is None
                    else any(expected.lower() in p["text"].lower() for p in result["passages"])
                )
                report["queries"].append(
                    {"wall_seconds": seconds, "expected_passage_found": hit, **result}
                )
                print(
                    "QUERY",
                    seconds,
                    "seconds; passages",
                    len(result["passages"]),
                    "anchor found",
                    hit,
                    flush=True,
                )
            first = report["queries"][0]["passages"][0]
            context, seconds = await call(
                "read_passage", {"paper_id": first["paper_id"], "chunk_index": first["chunk_index"]}
            )
            report["context_wall_seconds"] = seconds
            report["context_passages"] = len(context["passages"])
            assert any(first["text"] == p["text"] for p in context["passages"])
    path = ROOT / "reports" / f"rag-benchmark-{args.chunk_tokens}-{args.overlap}.json"
    lib.write_json(path, report)
    print("REPORT", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    asyncio.run(main(parser.parse_args()))
