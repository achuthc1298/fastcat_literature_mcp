# Fastcat literature MCP

Fastcat helps an AI assistant search scientific literature and answer questions
with citations to original paper passages. It searches **OpenAlex** and
**Semantic Scholar**, downloads supported **Elsevier** and **Springer Nature**
full texts, and builds a local searchable index using **LlamaIndex, BGE embeddings,
BM25, and Qdrant**. Your assistant reviews the evidence and writes the answer.

This is an **MCP server**: a local program that exposes tools to an assistant such
as Claude Code or Codex. It is not a website or a standalone chat application.
The entry point for this project is **`research_server.py`**.

The standard workflow uses CPU-based local embeddings. You do not need a Qdrant
account, Docker, a GPU, or an OpenAI/Anthropic API key for the server itself.
You do need to install and sign in to your chosen assistant separately. Its own
account requirements and usage charges are independent of publisher API access.
Paper passages are passed to that assistant; hosted assistants receive those passages.

## Contents

- [Requirements](#requirements)
- [Install and create the uv environment](#install-and-create-the-uv-environment)
- [Set up your private .env file](#set-up-your-private-env-file)
- [Where to get API keys](#where-to-get-api-keys)
- [Check your installation](#check-your-installation)
- [Connect to Claude Code](#connect-to-claude-code)
- [Connect to Codex](#connect-to-codex)
- [First research task](#first-research-task)
- [Settings and local files](#settings-and-local-files)
- [Troubleshooting](#troubleshooting)
- [Development and verification](#development-and-verification)
- [Prepare your own GitHub upload](#prepare-your-own-github-upload)

## Requirements

- Git, to clone the repository.
- [uv](https://docs.astral.sh/uv/), which installs Python and manages dependencies.
- Python **3.12 or 3.13**. The commands below let uv install Python 3.12 for you.
- Internet access for dependency installation, API requests, and initial model downloads.
- A writable project directory with room for downloaded papers, models, and indices.
- Claude Code and/or the Codex CLI, installed and authenticated on the same machine
  as the server.

The instructions below use a Bash-compatible terminal on Linux or macOS.
On Windows, use WSL2 and run all project and assistant commands inside the same
Linux environment. Native Windows setup is not verified here.

The optional legacy Qwen summarizer requires Apple Silicon macOS and MLX.
The standard literature search and retrieval workflow does not use Qwen.

## Install and create the uv environment

### 1. Install uv

On macOS or Linux, use the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal after installation, then check:

```bash
uv --version
git --version
```

See [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for other methods or PATH troubleshooting.

### 2. Get the project

Replace `YOUR_GITHUB_USERNAME` with the repository owner's username, and adjust
`fastcat` if the repository has a different name:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/fastcat.git
cd fastcat
```

If you downloaded a ZIP instead, extract it and open a terminal in the extracted
folder containing `pyproject.toml`. Skip `git clone` and change into that folder.

### 3. Install Python and dependencies

```bash
uv python install 3.12
uv sync --locked --python 3.12
```

`uv sync` creates a **`.venv`** directory containing this project's Python
and dependencies. `--locked` uses the committed `uv.lock` and fails if it does
not match `pyproject.toml`. Keep `uv.lock` in Git so others can reproduce the setup.
The default sync also installs the development tools used below. The project
uses uv's copy installation mode because NLTK rejects hardlinked corpus files
on Linux.

Check the environment:

```bash
uv run --locked python --version
```

You can run everything with `uv run`; activation is optional. If you prefer an
activated terminal:

```bash
source .venv/bin/activate
# Leave the environment later with:
deactivate
```

Activation affects the current terminal only. MCP configuration below uses uv
directly, so it also works when your assistant starts from another directory.

## Set up your private .env file

A `.env` file is a plain text file of `NAME=value` settings. The real file is
intentionally absent from the repository. Make your own copy:

```bash
cp .env.example .env
chmod 600 .env
```

Open `.env` in a text editor. Fill only the keys you have, for example:

```dotenv
OPENALEX_API_KEY="paste_your_own_openalex_key_here"
SEMANTIC_SCHOLAR_API_KEY="paste_your_own_semantic_scholar_key_here"
ELSEVIER_API_KEY="paste_your_own_elsevier_key_here"
SPRINGER_NATURE_API_KEY="paste_your_own_springer_open_access_key_here"
ELSEVIER_INST_TOKEN=
```

The quoted strings above are **placeholders**, not usable credentials. Keep
unavailable keys empty rather than pasting placeholders into your real file.
Keep the storage and chunk settings from `.env.example` unchanged to start.
Save the file as exactly `.env`, not `.env.txt`.

The server automatically loads `.env` beside `literature.py`, even if the assistant
starts it from a different working directory. You do not need to `source .env`
or copy keys into MCP configuration. Existing process environment variables take
precedence over `.env`; restart the MCP server after changing settings.

`.gitignore` excludes `.env` and other private `.env.*` files, while allowing
`.env.example`. Put only empty credentials and safe defaults in the example.
Do not put keys in chat messages, screenshots, README examples, or Git commits.
If you have already published a key, revoke/rotate it at its provider; deleting
it from the current file does not remove it from old Git history.

## Where to get API keys

| Variable | What uses it | How to obtain it |
| --- | --- | --- |
| `OPENALEX_API_KEY` | OpenAlex paper search; recommended for regular use | Create an OpenAlex account and copy your key from [Settings → API](https://openalex.org/settings/api). See [authentication guidance](https://help.openalex.org/api/authentication/). |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar paper search; recommended to reduce anonymous throttling | Open the [Academic Graph API portal](https://www.semanticscholar.org/product/api), choose **Request an API Key**, and submit the form. Approval is provider-controlled; the key is delivered by email. |
| `ELSEVIER_API_KEY` | `download_elsevier_paper` | Sign in/register at the [Elsevier Developer Portal](https://dev.elsevier.com/) and create an API key. Check the provider's Article Retrieval and TDM access requirements. |
| `SPRINGER_NATURE_API_KEY` | `download_springer_nature_paper` | Register at the [Springer Nature API portal](https://dev.springernature.com/) and obtain a key with **Open Access API** access. A metadata-only key is insufficient for this tool. See [Open Access API documentation](https://dev.springernature.com/docs/api-endpoints/open-access/). |
| `ELSEVIER_INST_TOKEN` | Optional institutional authentication for Elsevier downloads | Use only a token issued for your institution's access. Ask your library/Elsevier contact if one is needed; otherwise leave it empty. |

You can install the server and run offline tests with all keys empty. Search may
work without search keys, but anonymous requests can be throttled. Each publisher
key is needed only when using its download tool. Keys do not automatically grant
subscription full-text rights. Elsevier downloads remain subject to institutional
entitlements and [TDM access rules](https://dev.elsevier.com/tecdoc_text_mining.html).
Springer's downloader targets available open-access XML, not all subscription content.
Provider access policies and quotas can change; consult the linked portals.

## Check your installation

From the project directory:

```bash
uv run --locked python -c 'from research_server import check_configuration; import json; print(json.dumps(check_configuration(), indent=2))'
uv run --locked pytest -q
```

`check_configuration` reports `true`/`false` for key presence and never prints
credential values. Presence does not prove that a key works or that you have
full-text access. The tests use mocks and temporary files; they do not require
API keys or model downloads.

You can also start the server manually:

```bash
uv run --locked research_server.py
```

An apparently idle terminal is normal: this server waits for MCP messages over
standard input/output. It does not open a browser or listen on an HTTP port.
Press **Ctrl+C** to stop it. Your assistant will start its own server process.

## Connect to Claude Code

Install Claude Code using its [official setup guide](https://code.claude.com/docs/en/setup).
On macOS/Linux/WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Reopen your terminal, run `claude --version`, then run `claude` once and follow
the sign-in prompts. Exit that session before registering the server.
Then, **from the Fastcat project directory**, run:

```bash
claude mcp add --transport stdio --scope local fastcat-literature -- \
  "$(command -v uv)" run --locked --directory "$PWD" research_server.py
claude mcp list
claude
```

The shell expands the uv executable and project directory into absolute paths.
The local scope registers this connection for your current project without
creating a shared `.mcp.json`. Do not move the folder afterward without updating
the connection. Paths containing spaces remain valid because they are quoted.

Inside Claude Code, run `/mcp` to inspect the connection. Follow any client
prompts to enable/approve the server, then ask:

> Use fastcat-literature to call check_configuration and tell me which services are configured. Do not display any key values.

If you need a JSON configuration for another compatible MCP client, use
[mcp-config.example.json](mcp-config.example.json). Replace both absolute-path
placeholders before use; the example is not automatically loaded by Claude Code.
Keep your edited machine-specific configuration private.

For connection scopes and CLI options, see the [official Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

## Connect to Codex

Install the Codex CLI using the [official Codex CLI guide](https://developers.openai.com/codex/cli/).
On macOS/Linux/WSL:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Reopen your terminal, run `codex --version`, then run `codex` once and follow
the sign-in prompts. Exit that session before registering the server.
Then, **from the Fastcat project directory**, run:

```bash
codex mcp add fastcat-literature -- \
  "$(command -v uv)" run --locked --directory "$PWD" research_server.py
codex mcp list
codex
```

In the interactive Codex terminal, use `/mcp` to inspect the connection, then
ask it to call `check_configuration` as in the Claude Code example.
The CLI stores MCP configuration in `~/.codex/config.toml`; CLI and IDE extension
share this configuration. Restart existing sessions after adding the server.

For longer first-time indexing operations, edit the existing server section in
`~/.codex/config.toml` and add:

```toml
[mcp_servers.fastcat-literature]
# Keep the command and args that codex mcp add wrote here.
startup_timeout_sec = 60
tool_timeout_sec = 600
```

Do not create a second section with the same name or replace the existing command
and arguments with this partial example. Do not add publisher keys to this file;
Fastcat reads them from its private `.env`.

These instructions connect a locally running Codex client. A cloud session needs
its own checkout, installed dependencies, and separately supplied credentials.
See [official OpenAI MCP documentation](https://developers.openai.com/codex/mcp/)
for supported client configuration options.

## First research task

After connecting, try this prompt:

> Use fastcat-literature to research how particle size affects heat transfer and pressure drop in packed-bed thermal energy storage. Search with per_source_limit=50, rank candidates with reasons, download supported accessible papers, index them, and retrieve relevant passages. Inspect neighboring passages before citing numerical claims. Cite verified DOI links beside the claims they support and state the actual search and full-text coverage.

The normal sequence is:

1. **Search:** `search_papers(query, question, per_source_limit=50)` requests up to
   50 DOI-bearing results per source. Deduplication means this is not a promise
   of 100 unique papers.
2. **Rank:** the assistant screens titles/abstracts and calls `save_ranking` with
   a justified shortlist. Missing abstracts require lower-confidence judgments.
3. **Download:** use `download_elsevier_paper(doi)` or
   `download_springer_nature_paper(doi)`. A successful result includes `paper_id`.
4. **Index:** pass successful IDs to `index_papers(paper_ids)`.
   First use downloads embedding/BM25 assets; unchanged papers reuse their index.
5. **Read evidence:** call `retrieve_evidence(question, paper_ids, top_k=8)` and
   `read_passage(paper_id, chunk_index, neighbors=1)` for context.
6. **Answer:** the assistant synthesizes findings, cites DOI/section evidence,
   and distinguishes searched candidates from full texts actually inspected.

Use `list_indexed_papers()` to see the local corpus. Specify `paper_ids` in
retrieval calls to keep evidence scoped to the current task; otherwise retrieval
searches the current local index.

[RESEARCH_INSTRUCTIONS.md](RESEARCH_INSTRUCTIONS.md) contains the editable
research policy sent to the assistant when the server starts. Restart after
editing it. The host decides how to follow these instructions.

Only the two supported publishers have download tools. Search metadata can
include other publishers, but their full texts are not fetched here. Retrieval
is selective, not a complete-paper review; XML extraction can lose equations and
figure information. Known parser limitations include Springer DTD declarations
rejected by the safe parser and some Elsevier raw-text-only responses. See the
[historical live test report](reports/packed-bed-e2e-2026-09-23.md).

## Settings and local files

Leave these defaults alone until the basic workflow works:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Downloaded papers, searches, indices, and evidence. Relative to the project root. |
| `HF_HOME` | `./models/huggingface` | Hugging Face cache used by the optional legacy model. Relative to the project root. |
| `RAG_CHUNK_TOKENS` | `384` | Embedding-model tokens per chunk; allowed range 128–448. |
| `RAG_CHUNK_OVERLAP` | `64` | Chunk overlap; nonnegative and less than half the chunk size. |
| `LOCAL_MODEL` | `mlx-community/Qwen3.5-2B-4bit` | Optional legacy Qwen summarizer on Apple Silicon macOS. |
| `CHUNK_TOKENS` | `1800` | Chunk size for the legacy summarizer. |
| `CHUNK_OVERLAP` | `150` | Overlap for the legacy summarizer. |
| `EXTRACTION_MAX_TOKENS` | `900` | Maximum generated tokens per legacy summary chunk. |

Dense embeddings use `BAAI/bge-small-en-v1.5` with 384 dimensions. Hybrid retrieval
combines semantic and BM25 candidates with reciprocal-rank fusion. It normally
returns up to eight passages, with at most three per paper for multi-paper queries.
Embedding weights are stored under `models/fastembed/`. Changing RAG chunk settings
selects a separate index; call `index_papers` again to populate it.

Directories are created automatically as needed and are excluded from Git:

- `data/searches/`: search candidates and saved rankings.
- `data/papers/`: publisher XML and parsed paper records.
- `data/rag/`: Qdrant indices and chunk metadata.
- `data/evidence/`: original passages retrieved for questions.
- `data/summaries/`: optional legacy generated summaries.
- `models/`: downloaded model/cache files.

Indices are stored locally. File locks serialize database operations across local
sessions. This is intended for a small research library rather than a large
concurrent service. You may delete generated data/models to start fresh, but doing
so loses your corpus and requires downloading/reindexing again.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Reopen the terminal after installation; follow the uv PATH instructions. On Linux/macOS, `command -v uv` must print a path before registering MCP. |
| Unsupported Python version | Run `uv python install 3.12`, then `uv sync --locked --python 3.12`. |
| Server fails to connect | Run `uv run --locked research_server.py` in the project terminal to see startup errors. Verify the stored absolute paths, especially after moving the directory. |
| Keys reported absent | Confirm the file is named `.env`, saved in the project root, and contains nonempty values. Restart the MCP process. |
| An old key is still used | An exported environment variable can override `.env`; remove/update that variable in the client environment, then restart. |
| HTTP 401/403 | Check the key, enabled API product, and institutional entitlement with the provider. A present key alone does not establish access. |
| HTTP 429 | A provider rate limit was reached. Wait and check its quota policy; avoid repeated immediate retries. |
| First indexing is slow or times out | Initial model downloads need internet access. Allow them to finish and increase the client's tool timeout if needed. |
| Retrieval finds no useful evidence | Confirm papers downloaded and were indexed successfully; use focused questions and explicit paper IDs. |
| Springer/Elsevier XML parsing fails | Some publisher formats are unsupported; report the access/format limitation and continue with successfully parsed papers. |
| Legacy Qwen fails on Linux/Intel Macs | MLX is only included on Apple Silicon macOS. Use the standard indexing/retrieval workflow. |

## Development and verification

```bash
uv sync --locked --python 3.12
uv run --locked pytest -q
uv run --locked ruff check literature.py paper_rag.py local_extract.py research_server.py scripts tests
```

The offline tests cover search parsing and deduplication, safe errors, extraction
validation, MCP discovery, and RAG persistence/scoping/stale-source handling.
They use mock embeddings and do not measure live retrieval quality.

The optional live benchmark is **not** part of a fresh-clone test run. It requires
the three Elsevier papers listed in the historical live report to have been
downloaded with your own authorized access. After that:

```bash
uv run --locked python scripts/benchmark_rag.py
# Optional chunk comparison:
uv run --locked python scripts/benchmark_rag.py --chunk-tokens 256 --overlap 32
```

It launches the server directly and writes ignored JSON reports to `reports/`.
The [historical benchmark report](reports/rag-benchmark-2026-09-23.md) records a
small diagnostic sample, not a general retrieval-quality guarantee. Original
raw reports, downloaded texts, and private evidence are not distributed.

`uv run --locked python scripts/smoke_local.py` is an optional **legacy Qwen**
experiment using a synthetic paper. It downloads model weights and requires
Apple Silicon macOS; it is unnecessary for the standard workflow.

### Project layout

| File | Role |
| --- | --- |
| `research_server.py` | MCP server and tool definitions; use this entry point. |
| `literature.py` | Search APIs, publisher retrieval, XML parsing, and artifact storage. |
| `paper_rag.py` | Local embeddings, chunking, hybrid retrieval, and index persistence. |
| `local_extract.py` | Optional legacy Qwen extraction. |
| `RESEARCH_INSTRUCTIONS.md` | Research and citation guidance delivered to the assistant. |
| `.env.example` | Safe template for local configuration. |
| `mcp-config.example.json` | Generic stdio MCP configuration with path placeholders. |
| `pyproject.toml` / `uv.lock` | Dependency requirements and reproducible resolution. |
| `tests/`, `scripts/`, `reports/` | Offline tests, optional experiments, historical notes. |
| `main.py` | Separate legacy Thermo-Calc prototype; outside the supported literature setup. |

The Thermo-Calc prototype needs a licensed Thermo-Calc/TC-Python installation and
additional Google GenAI, LlamaIndex integrations, FAISS, and data dependencies
that are not part of this environment. It also expects `GOOGLE_API_KEY`,
`TC25B_HOME`, `LSHOST`, `ALLOY_COMPOSITION_CSV`, and `ALLOY_PROPERTIES_CSV`.
The CSV variables must point to your own input files. Obtain licensing/server
settings from your Thermo-Calc installation administrator and a Google key from
[Google AI Studio](https://aistudio.google.com/apikey) if working on that prototype.
These are not required for `research_server.py`; `uv sync` does not make the legacy
prototype runnable. Its code is retained for reference.

## Prepare your own GitHub upload

If working from a ZIP or a local folder without Git, initialize a repository:

```bash
git init -b main
```

Review ignored files and the prospective submission before committing:

```bash
git status --short --ignored
git check-ignore .env .venv data models
git add .
git diff --cached --stat
git diff --cached
```

Check that no credentials, personal configuration, downloaded papers, or model
weights are staged. `.env.example` and `uv.lock` should be included. Ignore rules
apply to untracked files; if you previously tracked a private file, remove it
from Git's index and deal with any exposed credentials/history before publishing.

Then commit and push to an empty repository you created on GitHub:

```bash
git commit -m "Prepare Fastcat literature MCP for publication"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/fastcat.git
git push -u origin main
```

If you cloned a repository, it already has a Git history and usually an `origin`;
inspect `git remote -v` and use your own fork instead of adding a duplicate remote.
Git may ask you to configure your author name/email before the first commit.
No license is assigned by this cleanup; the owner should choose an appropriate
license before inviting redistribution under specific terms.
