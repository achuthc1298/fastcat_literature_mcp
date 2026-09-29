# Packed-bed research: live end-to-end test

Date: 2026-09-23. This test uses the `fastcat-literature` launch command from the
global Codex configuration and a real MCP client. All paper downloads and local
extractions were requested through MCP. No production code or API keys were changed.

Research question: **How do particle size and bed porosity affect pressure drop and
heat transfer in packed-bed thermal energy storage?**

## Search and main-LLM ranking

Primary query: `packed bed thermal energy storage`, maximum 50 results per source.

- OpenAlex: 50 DOI-bearing candidates, 17 with abstracts.
- Semantic Scholar: 0 candidates; anonymous requests returned HTTP 429 after retries.
  Its API key is not configured. This is a rate-limit failure, not zero relevant papers.
- Elsevier and Springer Nature keys are configured. No key values were printed.
- Main LLM reviewed titles/available abstracts and saved a six-paper shortlist with
  relevance reasons. Missing abstracts were labeled lower-confidence judgments.
- Two highly relevant shortlisted results belong to unsupported publishers (MDPI,
  ASME); they were not sent to the publisher download tools.

Primary search and ranking: saved JSON (local artifact; not distributed).

A supplementary query, `packed bed porosity heat transfer`, returned another 50
OpenAlex candidates, 17 with abstracts, and the same Semantic Scholar rate limit.
A relevant Springer-distributed result was selected from this search:
saved search/ranking (local artifact; not distributed).
These are separate searches; their counts must not be added as 100 unique papers.

## Publisher retrieval

| DOI | Tool | Result |
| --- | --- | --- |
| 10.1016/j.apenergy.2017.12.072 | Elsevier | Full XML downloaded; 117 extracted body blocks. |
| 10.1016/j.apenergy.2022.118672 | Elsevier | Full XML downloaded; 102 extracted body blocks. |
| 10.1016/j.solener.2017.03.032 | Elsevier | Full XML downloaded; 210 extracted body blocks. Retained as a review reference, not sent to Qwen in this bounded test. |
| 10.1016/j.egypro.2014.03.116 | Elsevier | Tool rejected response because there is no structured `body`. Diagnostic inspection found a `rawtext` element with 16,384 characters under `originalText`; this format is currently unsupported. |
| 10.1631/jzus.2006.a1422 | Springer Nature | HTTP 404. Direct diagnostic calls confirmed “No data was found for the given query” for JATS/XML/JSON. This does not establish an invalid API key. |
| 10.1007/s43937-026-00167-y | Springer Nature | HTTP 200 with XML, but the parser rejected DTD entity declarations. |
| 10.1038/s41598-025-99495-7 | Springer Nature | Tool also rejected the returned XML as unsafe/unparseable. |

The last two DOIs were found on official publisher pages as supplementary adapter
controls, not returned by the two MCP searches above:
[thermal storage/porosity study](https://link.springer.com/article/10.1007/s43937-026-00167-y),
[packed-bed morphology study](https://www.nature.com/articles/s41598-025-99495-7).

For the first Springer control, the diagnostic response was 288,576 bytes of XML
with JATS/BITS external parameter-entity declarations in the DOCTYPE. The exact
exception was `EntitiesForbidden` for the `article` DTD declaration. The existing
parser fails before it can select the article. Do not resolve this by enabling
unrestricted external entity loading or fetching remote DTDs.

## Local Qwen and main-LLM review

Model: `mlx-community/Qwen3.5-2B-4bit`, local MLX inference. Configured chunks:
1,800 input tokens, overlap 150; output limit 900 tokens. Papers were processed
sequentially through `extract_paper`, with progress notifications and checkpoints.

**Optimization paper, 10.1016/j.apenergy.2017.12.072:** all seven chunks were
attempted; only two succeeded. Five hit the output limit on both attempts. The
summary is correctly labeled `partial`. It contains 12 accepted findings and one
rejected quotation. The complete saved summary was read through `read_summary`.

Optimization paper summary (local artifact; not distributed)

Main-LLM review found a substantive error despite a text-matched quote: Qwen claims
that cross section and mass flow rate are constrained by pumping costs, while the
attached quote refers to bed height and particle diameter. It also generalizes
claims from cited earlier studies. Equation interpretations are unreliable: one
claim introduces division by duration that its quoted equation does not show.
These statements must not be reused as verified research conclusions.

**Radial-flow experimental paper, 10.1016/j.apenergy.2022.118672:** all eight chunks
were attempted; two succeeded and six exhausted the output limit on both attempts.
The saved summary is correctly `partial`, with 17 accepted findings and one
rejected quotation. All of its summary content was read through `read_summary`.

Experimental paper summary (local artifact; not distributed)

The main-LLM review found numerical errors here too: one claim says 5.7% where its
own quotation says 5.2%, and another conflates the reported 5.5% increase and 5.7%
reduction in pressure drop. Model-reported limitations also confuse measurement
uncertainty with thermal efficiency. These summaries are unsuitable as unattended
scientific evidence, even where a quotation passes exact-text validation.

Observed wall times through MCP were approximately 208 seconds for the first paper
and 252 seconds for the second, including loading, generation, and retries. Across
both papers, 4 of 15 chunks succeeded. The mean is about 3.8 minutes per paper;
extrapolating this small sample to 100 papers gives roughly 6.4 hours sequentially,
before search/download time and without obtaining complete summaries. This is not
a model tokens-per-second benchmark and is not a reliable throughput guarantee.

The 4-bit model weights are cached in `models/huggingface` (approximately 1.6 GB).
Inference runs on this Apple Silicon Mac through MLX, not a hosted model API.

## Assessment

The live test is not a clean end-to-end pass. Search, ranking persistence, real
Elsevier XML downloads, local inference, checkpoints, partial-coverage reporting,
and summary retrieval have executed. Successful synthetic tests did not reveal
the Springer XML compatibility issue or the local model's real-paper output and
factual-reliability problems.

Required follow-up work:

1. Supply a Semantic Scholar key and retest search access.
2. Handle Springer DTD declarations safely without external entity expansion.
3. Support and label Elsevier raw-text responses, while retaining full-text/DOI checks.
4. Bound the number/length of Qwen findings, adapt chunk size/output budget, and
   change failed-chunk retry behavior. Current greedy retries use the same prompt
   and settings, so they repeatedly reproduce output-limit failures.
5. Preserve attribution to cited studies and review whether each quote actually
   supports its claim. Exact text matching alone is insufficient.

No publisher restrictions were bypassed. No findings from failed chunks are counted
as covered, and unavailable papers are not treated as evidence of absence.


Publication note: downloaded papers, raw evidence, summaries, and raw benchmark JSON
are excluded from this repository. This report records historical observations,
not results from a fresh clone. See the README for reproduction prerequisites.
