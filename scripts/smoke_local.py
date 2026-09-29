"""Run an actual Qwen extraction through MCP on a clearly labeled synthetic fixture.

This downloads model weights on first use. It does not require publisher API keys.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import literature as lib  # noqa: E402


async def main():
    with tempfile.TemporaryDirectory(prefix="fastcat-smoke-") as directory:
        lib.DATA = Path(directory)
        identifier = lib.paper_id("10.1007/synthetic-test")
        lib.write_json(
            lib.artifact_path("papers", identifier),
            {
                "paper_id": identifier,
                "doi": "10.1007/synthetic-test",
                "title": "SYNTHETIC TEST: not a published paper",
                "publisher": "springer_nature",
                "sections": [
                    {
                        "section": "Results",
                        "text": "In this synthetic experiment, annealing at 600 K for 2 hours increased tensile strength from 200 MPa to 250 MPa. Only three samples were tested. No fatigue measurements were collected.",
                    }
                ],
            },
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "research_server.py")],
            env={**os.environ, "DATA_DIR": directory},
        )
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(
                    "extract_paper",
                    {
                        "paper_id": identifier,
                        "question": "How did annealing affect tensile strength, and what are the limitations?",
                    },
                )
                if result.isError:
                    raise RuntimeError(str(result.content))
                metadata = json.loads(result.content[0].text)
                print(json.dumps(metadata, indent=2))
                summary = await session.call_tool(
                    "read_summary", {"summary_id": metadata["summary_id"]}
                )
                assert not summary.isError
                text = json.loads(summary.content[0].text)["content"]
                print(text)
                saved = lib.read_json(lib.artifact_path("summaries", metadata["summary_id"]))
                assert metadata["status"] == "complete", "Extraction was incomplete"
                assert any(c["findings"] for c in saved["chunks"]), (
                    "Model returned no validated evidence"
                )
                assert "250" in text and "200" in text, (
                    "Expected synthetic measurement not extracted"
                )


if __name__ == "__main__":
    asyncio.run(main())
