# Local RAG benchmark and selected defaults

Tested through the actual global `fastcat-literature` MCP launch configuration on
2026-09-23. Corpus: the three real Elsevier packed-bed papers downloaded in the
previous test. No generated Qwen summaries were indexed; the source is parsed
publisher body text.

## Selected design

LlamaIndex SentenceSplitter + FastEmbed `BAAI/bge-small-en-v1.5` + local persistent
Qdrant, combining dense retrieval and BM25 with equal-weight reciprocal-rank fusion.

Defaults: **384 embedding-model tokens/chunk, 64-token overlap**, 32-item embedding
batches, 4 CPU threads. Sentence-aware chunks stay within section boundaries. DOI,
title, section, and chunk index are preserved, but metadata is not added to embedding
text. Token counting uses BGE's tokenizer without truncation; no chunk exceeds the
configured budget. This leaves headroom below the model's 512-token context.

Retrieve 24 dense and 24 keyword candidates; fuse with RRF constant 60; return up to
8 passages. Multi-paper searches allow up to 3 passages per paper; a paper-scoped
query can use all 8. No local generative model or cross-encoder is called. The main
LLM reads original evidence and synthesizes with citations.

## Measurements

| Chunk/overlap | Total chunks | First indexing, 3 papers | Cached index check | Mean warm query | Diagnostic anchor hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 / 32 | 227 | 13.374 s | 0.026 s | 0.405 s | 2/3 |
| **384 / 64** | **171** | **16.420 s** | **0.035 s** | **0.387 s** | **2/3** |
| 448 / 64 | 147 | 15.622 s | 0.023 s | 0.478 s | 1/3 |

Indexing timings include MCP round-trip, imports/model initialization, chunking,
embeddings, and disk writes. Dense model weights were already downloaded; the first
384-token run also fetched small BM25 assets. Runs are not controlled enough to
interpret their indexing-time differences as a speed ranking of chunk sizes.
Warm query timings average four questions. Context retrieval took about 0.003 s.

A fresh MCP process opened the existing 384-token index without reindexing and
successfully retrieved evidence. After listing the index (which imports libraries),
its first query, including embedding-model initialization, took **0.747 s**.

These timings exclude external paper search/download and the main LLM's final
answer generation. Retrieval is selective and performs a different task from
full-paper summarization. The old Qwen path took 208 and 252 seconds for two papers
and generated only partial summaries; do not read this as a controlled speedup ratio.

## Evidence review and choice of chunk size

The broad research question retrieved passages describing the heat-transfer versus
pressure-drop/pumping-power trade-off, the effect of lower void fraction on storage
density and pressure loss, and the radial-flow experiment's measured performance.
These are original source statements, avoiding the intermediate Qwen-generated
claim errors observed in the previous test.

The diagnostic checks look for three manually selected text anchors in the top
results: particle-size trade-off, local packing non-uniformity, and the measured
71.8% thermal efficiency. They are deliberately small checks, not comprehensive
scientific recall/accuracy measurements.

384-token chunks retrieved the particle-size and efficiency anchors. The porosity
query retrieved other relevant porosity passages, but not the exact selected local
packing anchor under the three-passages-per-paper cap. Repeating the query scoped
to the experimental paper retrieved that anchor too:
saved scoped evidence (local artifact; not distributed).

256-token chunks found the porosity anchor but missed the efficiency anchor;
448-token chunks missed both the specific particle-size and porosity anchors.
384/64 is therefore a practical context/precision compromise for this sample,
not a claim of universal optimality. Use focused and paper-scoped follow-ups when
individual aspects need deeper coverage; inspect neighboring chunks for conditions.

## Verification and limits

- 14 offline tests pass, including original tests and new RAG persistence, filtering,
  stale-source detection, replacement of obsolete chunks, section boundaries,
  rank fusion, configuration isolation, and MCP tool discovery.
- Live MCP tests verify indexing, cached reuse, retrieval, neighboring passages,
  configuration defaults, and persistence across server restarts.
- Qdrant is local, with an interprocess lock for safe serial access across sessions.
- The full-paper Qwen tools remain explicitly labeled legacy and optional.
- The previously found Springer DTD and Elsevier raw-text parser limitations remain
  unresolved; changing the retrieval layer does not repair publisher downloading.
- The main LLM must still interpret equations, preserve attribution to cited studies,
  and check that retrieved passages support its answer. RAG is not exhaustive reading.

Raw runs: 256/32 (local artifact; not distributed), 384/64 (local artifact; not distributed),
448/64 (local artifact; not distributed). Reproduce using `uv run python scripts/benchmark_rag.py`.


Publication note: downloaded papers, raw evidence, summaries, and raw benchmark JSON
are excluded from this repository. This report records historical observations,
not results from a fresh clone. See the README for reproduction prerequisites.
