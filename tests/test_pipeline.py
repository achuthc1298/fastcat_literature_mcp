import json
import sys
from types import SimpleNamespace

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import literature as lib
import research_server as server
from local_extract import parse_evidence


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    monkeypatch.setattr(lib, "DATA", tmp_path)


def test_doi_index_and_path_validation():
    assert lib.doi_normalize("https://doi.org/10.1007/ABC") == "10.1007/abc"
    assert lib.abstract_from_index({"world": [1], "Hello": [0, 2]}) == "Hello world Hello"
    with pytest.raises(ValueError):
        lib.artifact_path("papers", "../../.env")


async def test_s2_pagination_skips_missing_dois():
    def handler(request):
        offset = int(request.url.params["offset"])
        if not offset:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"title": "No DOI", "externalIds": {}},
                        {"title": "One", "externalIds": {"DOI": "10.1007/a"}},
                    ],
                    "next": 2,
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"title": "Two", "abstract": "Evidence", "externalIds": {"DOI": "10.1007/b"}},
                    {"title": "Three", "externalIds": {"DOI": "10.1007/c"}},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await lib.API(client).semantic_scholar("test", 2)
    assert len(result["papers"]) == 2
    assert result["papers"][1]["abstract"] == "Evidence"


async def test_merge_and_ranking(monkeypatch):
    async def oa(self, query, limit):
        return {
            "papers": [
                {
                    "doi": "10.1007/a",
                    "title": "A",
                    "year": 2024,
                    "abstract": None,
                    "abstract_source": None,
                    "publisher_hint": "Springer Nature",
                    "sources": ["openalex"],
                }
            ],
            "error": None,
        }

    async def s2(self, query, limit):
        item = (await oa(self, query, limit))["papers"][0]
        item.update(
            abstract="Useful abstract",
            abstract_source="semantic_scholar",
            sources=["semantic_scholar"],
        )
        return {"papers": [item], "error": "Partial source failure"}

    monkeypatch.setattr(lib.API, "openalex", oa)
    monkeypatch.setattr(lib.API, "semantic_scholar", s2)
    result = await lib.search("test", "Research question")
    assert result["counts"] == {"openalex": 1, "semantic_scholar": 1, "unique": 1}
    assert result["candidates"][0]["abstract_source"] == "semantic_scholar"
    assert not result["reranked"]
    assert "main LLM" in result["instructions"]
    assert result["errors"]["semantic_scholar"]
    ranked = server.save_ranking(
        result["run_id"], [server.RankedPaper(doi="10.1007/a", reason="Relevant abstract")]
    )
    assert ranked["ranking"][0]["rank"] == 1
    with pytest.raises(ValueError):
        server.save_ranking(
            result["run_id"], [server.RankedPaper(doi="10.1007/unknown", reason="Unknown")]
        )


JATS = b"""<response><records><article><front><article-meta>
<article-id pub-id-type="doi">10.1007/test</article-id>
<title-group><article-title>Study title</article-title></title-group>
</article-meta></front><body><sec><title>Results</title>
<p>The measured strength increased by 25 percent at 300 K.</p>
<table-wrap><caption>A table</caption><table><tr><td>25</td></tr></table></table-wrap>
</sec></body></article></records></response>"""


def test_fulltext_and_xml_safety():
    title, sections = lib.parse_fulltext(JATS, "10.1007/test", "springer_nature")
    assert title == "Study title"
    assert len(sections) == 2
    assert sections[0]["section"] == "Results"
    with pytest.raises(lib.ServiceError, match="matching DOI"):
        lib.parse_fulltext(JATS, "10.1007/wrong", "springer_nature")
    with pytest.raises(lib.ServiceError, match="body"):
        lib.parse_fulltext(
            b"<root><abstract>Abstract only</abstract></root>", "10.1016/test", "elsevier"
        )
    with pytest.raises(lib.ServiceError, match="safe"):
        lib.parse_fulltext(
            b'<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><x>&xxe;</x>',
            "10.1016/test",
            "elsevier",
        )
    xml = b'<full-text-retrieval-response xmlns:ce="urn:ce" xmlns:prism="urn:prism"><coredata><prism:doi>10.1016/test</prism:doi></coredata><originalText><article><body><ce:sections><ce:section><ce:section-title>Results</ce:section-title><ce:para>Full text with a measured value.</ce:para></ce:section></ce:sections></body></article></originalText></full-text-retrieval-response>'
    assert lib.parse_fulltext(xml, "10.1016/test", "elsevier")[1][0]["section"] == "Results"


async def test_download_and_cache(monkeypatch):
    monkeypatch.setenv("SPRINGER_NATURE_API_KEY", "test-secret")

    async def get(self, url, **kwargs):
        assert kwargs["params"]["q"] == "doi:10.1007/test"
        return httpx.Response(200, content=JATS)

    monkeypatch.setattr(lib.API, "get", get)
    result = await lib.download("10.1007/test", "springer_nature")
    assert result["status"] == "downloaded"
    assert lib.artifact_path("papers", result["paper_id"], ".xml").exists()
    assert (await lib.download("10.1007/test", "springer_nature"))["status"] == "cached"


async def test_auth_error_does_not_leak_key():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(403))
    ) as client:
        with pytest.raises(lib.ServiceError) as error:
            await lib.API(client).get("https://example.org", params={"api_key": "secret"})
    assert "secret" not in str(error.value)
    assert "403" in str(error.value)


def test_model_quotes_are_grounded():
    output = json.dumps(
        {
            "findings": [
                {"claim": "Strength rose", "quote": "strength increased by 25 percent"},
                {"claim": "Invented", "quote": "strength doubled at 500 K"},
            ],
            "limitations": "Single experiment",
        }
    )
    result = parse_evidence(output, "The measured strength increased by 25 percent at 300 K.")
    assert len(result["findings"]) == 1
    assert result["rejected_findings"] == 1


def test_summary_pagination():
    identifier = "a" * 24
    lib.write_json(
        lib.artifact_path("summaries", identifier),
        {"doi": "10.1007/test", "question": "Q", "status": "partial"},
    )
    lib.artifact_path("summaries", identifier, ".md").write_text("X" * 2500)
    first = server.read_summary(identifier, max_chars=1000)
    assert first["next_offset"] == 1000
    assert first["coverage"] == "partial"
    assert server.read_summary(identifier, offset=2000, max_chars=1000)["next_offset"] is None


async def test_mcp_stdio_handshake():
    params = StdioServerParameters(
        command=sys.executable, args=[str(lib.ROOT / "research_server.py")]
    )
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            init = await session.initialize()
            assert init.instructions == server.SERVER_INSTRUCTIONS
            assert "per_source_limit=50" in init.instructions
            assert "Place a citation immediately after" in init.instructions
            result = await session.list_tools()
            names = {t.name for t in result.tools}
            assert names == {
                "search_papers",
                "save_ranking",
                "download_elsevier_paper",
                "download_springer_nature_paper",
                "extract_paper",
                "read_summary",
                "check_configuration",
                "index_papers",
                "retrieve_evidence",
                "read_passage",
                "list_indexed_papers",
            }
            config = await session.call_tool("check_configuration", {})
            assert not config.isError


def test_extraction_covers_all_chunks_and_resumes_failures(monkeypatch):
    import local_extract

    class Tokenizer:
        def encode(self, content, **kwargs):
            return list(content)

        def decode(self, tokens):
            return "".join(tokens)

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

    calls = []
    fail_second_chunk = True

    def generate(model, tokenizer, prompt, **kwargs):
        data = json.loads(prompt)
        calls.append(data["chunk"])
        if data["chunk"] == 2 and fail_second_chunk:
            yield SimpleNamespace(text="invalid JSON", finish_reason="stop")
        else:
            yield SimpleNamespace(
                text=json.dumps(
                    {
                        "findings": [{"claim": "Test finding", "quote": data["excerpt"][:30]}],
                        "limitations": "Test fixture",
                    }
                ),
                finish_reason="stop",
            )

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(load=lambda name: (None, Tokenizer()), stream_generate=generate),
    )
    monkeypatch.setitem(
        sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_sampler=lambda **kwargs: None)
    )
    monkeypatch.setenv("CHUNK_TOKENS", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")
    identifier = lib.paper_id("10.1007/test-chunks")
    text = "Measurements and results were recorded at 300 K. " * 30
    lib.write_json(
        lib.artifact_path("papers", identifier),
        {
            "doi": "10.1007/test-chunks",
            "title": "Fixture",
            "sections": [{"section": "Results", "text": text}],
        },
    )
    first = local_extract.extract(identifier, "What was measured?")
    assert first["status"] == "partial"
    partial = lib.read_json(lib.artifact_path("summaries", first["summary_id"]))
    assert partial["chunks"][-1]["token_end"] == len("[Results] " + text)
    assert len(partial["chunks"]) > 3
    assert calls.count(2) == 2
    calls.clear()
    fail_second_chunk = False
    second = local_extract.extract(identifier, "What was measured?")
    assert second["status"] == "complete"
    assert calls == [2], "Only the previously failed chunk should run again"
    calls.clear()
    assert local_extract.extract(identifier, "What was measured?")["status"] == "complete"
    assert not calls, "Fully completed extraction should use cache"
