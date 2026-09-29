"""Offline persistence/filtering tests: no model downloads or cloud requests."""

import hashlib
import re

import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from tokenizers import Tokenizer, models, pre_tokenizers

import literature as lib
import paper_rag as rag


@pytest.fixture
def local_index(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "DATA", tmp_path)
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "128")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "16")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    monkeypatch.setattr(rag, "embedding_model", lambda: MockEmbedding(embed_dim=384))
    monkeypatch.setattr(rag, "exact_tokenizer", lambda model: tokenizer)

    def sparse(texts):
        indices, values = [], []
        for text in texts:
            tokens = sorted(
                {
                    int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
                    for t in re.findall(r"\w+", text.lower())
                }
            )
            indices.append(tokens)
            values.append([1.0] * len(tokens))
        return indices, values

    monkeypatch.setattr(
        rag,
        "vector_store",
        lambda client: QdrantVectorStore(
            client=client,
            collection_name="papers",
            enable_hybrid=True,
            index_doc_id=False,
            sparse_doc_fn=sparse,
            sparse_query_fn=sparse,
            hybrid_fusion_fn=rag.reciprocal_rank_fusion,
        ),
    )
    return tokenizer


def save_paper(suffix, text):
    doi = "10.1007/" + suffix
    identifier = lib.paper_id(doi)
    record = {
        "paper_id": identifier,
        "doi": doi,
        "title": suffix,
        "sections": [
            {"section": "Results", "text": text},
            {"section": "Limitations", "text": "Only three samples were tested."},
        ],
    }
    lib.write_json(lib.artifact_path("papers", identifier), record)
    return record


def test_index_query_scope_persistence_and_staleness(local_index):
    one = save_paper(
        "pressure", "Smaller particles increase pressure drop and improve heat transfer. " * 40
    )
    two = save_paper("unrelated", "Neural networks classify images of animals. " * 20)
    ids = [one["paper_id"], two["paper_id"]]
    indexed = rag.index_papers(ids)
    assert all(p["status"] == "indexed" for p in indexed["papers"])
    assert all(p["status"] == "cached" for p in rag.index_papers(ids)["papers"])
    # Every operation opens/closes Qdrant: this exercises disk persistence, not one in-memory client.
    results = rag.retrieve_evidence("pressure drop", paper_ids=[ids[0]])
    assert results["passages"]
    assert all(p["paper_id"] == ids[0] for p in results["passages"])
    first = results["passages"][0]
    context = rag.read_passage(ids[0], first["chunk_index"])
    assert any(p["text"] == first["text"] for p in context["passages"])
    one["sections"][0]["text"] = "Revised results: larger particles reduce pressure drop."
    lib.write_json(lib.artifact_path("papers", ids[0]), one)
    assert rag.retrieve_evidence("pressure", paper_ids=[ids[0]])["passages"] == []
    with pytest.raises(ValueError, match="stale"):
        rag.read_passage(ids[0], 0)
    assert rag.index_papers([ids[0]])["papers"][0]["status"] == "indexed"
    result = rag.retrieve_evidence("pressure", paper_ids=[ids[0]])
    assert any("Revised results" in p["text"] for p in result["passages"])
    assert not any("Smaller particles" in p["text"] for p in result["passages"])


def test_chunk_boundaries_metadata_and_complete_text(local_index):
    record = save_paper(
        "chunks", "First evidence statement. " * 100 + "FINAL_MARKER proves no tail truncation."
    )
    nodes = rag.build_nodes(record, local_index)
    assert len(nodes) > 2
    assert all(n.metadata["tokens"] <= 128 for n in nodes)
    assert any("FINAL_MARKER" in n.text for n in nodes)
    assert nodes[-1].metadata["section"] == "Limitations"
    assert all(n.metadata["doi"] == record["doi"] for n in nodes)
    assert all(set(n.excluded_embed_metadata_keys) == set(n.metadata) for n in nodes)


def test_rank_fusion_and_bad_inputs(local_index):
    a, b = TextNode(id_="a", text="a"), TextNode(id_="b", text="b")
    dense = VectorStoreQueryResult(nodes=[a, b], similarities=[0.9, 0.8])
    sparse = VectorStoreQueryResult(nodes=[b], similarities=[10])
    fused = rag.reciprocal_rank_fusion(dense, sparse)
    assert fused.ids == ["b", "a"]
    assert rag.retrieve_evidence("pressure")["passages"] == []
    with pytest.raises(ValueError):
        rag.index_papers(["../../.env"])
    with pytest.raises(ValueError):
        rag.retrieve_evidence("", top_k=99)


def test_configuration_isolates_indices(local_index, monkeypatch):
    previous = rag.index_root()
    monkeypatch.setenv("RAG_CHUNK_TOKENS", "256")
    assert rag.index_root() != previous
