# GHA-Agent

GHA-Agent is a multi-agent framework for extracting error-related lines from GitHub Actions failure logs. It selects between timestamp-based block processing with judgment and pruning, and sliding-window parsing. Its output refers to line ranges in the original input log.

The accompanying [GHA-Bench](../GHA-Bench/README.md) supplies logs, human annotations, workflow metadata, and golden patches.

## Setup

### 1. Install dependencies

Use **Python 3.11 or newer**. Parsing requires network access to the OpenAI API and an API key with access to the configured model. The supplied implementation uses the OpenAI Python SDK; model inference is performed through the API.

From the replication package root, create and activate a virtual environment:

**Linux / macOS**

```sh
cd GHA-Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
cd GHA-Agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` for installation and subsequent commands. All commands below assume the working directory is **`GHA-Agent/`**.

### 2. Configure the API key

Copy [`.env.example`](.env.example) to `.env`:

```sh
# Linux / macOS
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Edit the new `.env` file and replace the placeholder:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Alternatively, set `OPENAI_API_KEY` in the shell environment. Existing environment variables take precedence over `.env` values. The loader looks for `GHA-Agent/.env` first, then `.env` in the working directory. `.env` is excluded by this component's `.gitignore`.

### 3. Check the configuration

The checked-in settings are in [`config/model_config.yaml`](config/model_config.yaml), and agent prompts are in [`config/prompts.yaml`](config/prompts.yaml).

| Setting | Checked-in value / behavior |
| --- | --- |
| Provider | OpenAI; the client uses `https://api.openai.com/v1` and Chat Completions |
| Agent models | `gpt-5.4` in each model-bearing agent section |
| Reasoning effort | `low` |
| Request timeout | `platforms.default.timeout_seconds: 600` |
| SDK retries | `platforms.default.max_retries: 2` |
| Completion limit | Optional `max_completion_tokens` in an individual agent's configuration |

Models are configured per agent, including nested judgment, extraction, and pruning sections. To change the model throughout the pipeline, update each relevant `model` entry. Use a model that supports the request parameters and function calling used by [`core/llm_client.py`](core/llm_client.py).

Verify the command-line entry points after installing dependencies:

```sh
python main.py --help
python scripts/parallel_full_run.py --help
```

These help commands do not make model requests or require an API key. Actual parsing makes API requests and incurs usage charges. Keep a copy of the model configuration and prompts with your experimental results so the run settings remain identifiable.

## Parse one log

Run this example against the included `psf/black` case:

```sh
python main.py --input "../GHA-Bench/cases/basic/python/psf_black/4628/13956990967/13956990967~Changelog Entry Check~whole_log.txt" --output-dir results/black-example
```

The results are written directly to `results/black-example/`. Quote log paths because job names can contain spaces and punctuation. To use another log, replace `--input` with its UTF-8 text-file path and choose a separate output directory.

## Parse GHA-Bench

Set **`--raw-data-dir ../GHA-Bench/cases`**. The runner expects the `cases/` directory, whose immediate children are `basic/` and `complex/`.

Start with one case and one worker:

```sh
python scripts/parallel_full_run.py --raw-data-dir ../GHA-Bench/cases --output-dir results/gha-bench --repository psf_black --workers 1 --limit 1
```

After checking that case's `final_summary.json`, process the complete benchmark:

```sh
python scripts/parallel_full_run.py --raw-data-dir ../GHA-Bench/cases --output-dir results/gha-bench --workers 4
```

`--workers` controls concurrent processes. The script defaults to **20 workers** when the option is omitted; set it explicitly to suit your API limits. For sequential execution, use:

```sh
python main.py --raw-data-dir ../GHA-Bench/cases --output-dir results/gha-bench-sequential
```

Both entry points support these batch options:

| Option | Meaning |
| --- | --- |
| `--raw-data-dir` | Input tree root; defaults to `raw_data`, which is not included in this directory |
| `--output-dir` | Result root; defaults to `results` for `main.py`, or `results/full_run` for the parallel script |
| `--difficulty` | Exact directory name: `basic` or `complex` |
| `--language` | Exact language directory, e.g. `python`, `java`, `"c++"`, or `"c#"` |
| `--repository` | Exact repository directory, e.g. `psf_black` |
| `--post-id` | Pull request number stored in the case path, e.g. `4628` |
| `--run-id` | Workflow run ID, e.g. `13956990967` |
| `--limit` | Maximum number of matching logs selected in sorted path order, before existing results are skipped |
| `--no-skip-existing` | Reprocess selected logs even when their `final_summary.json` already exists |

For example, to process only the Complex Python cases:

```sh
python scripts/parallel_full_run.py --raw-data-dir ../GHA-Bench/cases --output-dir results/complex-python --difficulty complex --language python --workers 4
```

Batch runs skip a log whenever its output directory already contains `final_summary.json`. This check is based on file existence, so a saved failure result is also skipped. To retry a failed case, select it with the filters and add `--no-skip-existing`, or use a new output directory. For a different model or configuration, use a separate output root to avoid reusing earlier results. Single-log execution with `--input` writes its outputs again on each run.

The batch runner discovers files matching:

```text
INPUT_ROOT/{difficulty}/{language}/{repository}/{post_id}/{run_id}/*whole_log.txt
```

It derives case identifiers from this path. The adjacent metadata JSON, human annotations, and golden patches are not required as parser inputs.

## Outputs

A batch run preserves the case hierarchy and adds a directory for each log, removing the `~whole_log.txt` suffix from its name. For the example case:

```text
results/gha-bench/
├── batch_summary.json
└── basic/python/psf_black/4628/13956990967/
    └── 13956990967~Changelog Entry Check/
        ├── final_summary.json
        ├── run_statistics.json
        └── run.log
```

| File | Contents |
| --- | --- |
| `final_summary.json` | Parse status, selected method, extracted error ranges, and statistics; failed results include an `error` message |
| `run_statistics.json` | Parser/LLM elapsed time, call count, API-reported token usage, and per-call records |
| `run.log` | Per-log console output and errors, written by the parallel runner |
| `batch_summary.json` | Batch progress, per-log result locations/statuses, timing, and aggregate token usage |

In a successful `final_summary.json`:

- `method_used: 1` denotes block judgment followed by pruning; `method_used: 2` denotes sliding-window parsing.
- `extracted_errors` is a list of objects containing `block` and `error_ranges`. Each range is `[start, end]`, with **1-based, inclusive** line numbers in the original log. Use `error_ranges` as the predicted error lines; `block` describes the enclosing block.
- `statistics` contains log/processing statistics. `run_statistics` contains the same runtime data also saved separately in `run_statistics.json`.

Read `status` and any `error` in the result to determine whether a parse succeeded. Compare the union of the predicted `error_ranges` with the matching [human annotation ranges](../GHA-Bench/README.md#human-annotations) for line-level evaluation. The runners do not compute evaluation scores.

## Source layout

```text
GHA-Agent/
├── main.py                       # Single-log and sequential batch entry point
├── scripts/parallel_full_run.py  # Parallel batch entry point
├── agents/                       # Classification, preprocessing, and parsing agents
│   └── pruning/                  # Anchor protection and pruning pipeline
├── core/                         # Model client, tools, configuration, and statistics
├── pipeline/                     # Conventional block-processing pipeline
├── utils/                        # Timestamp segmentation utilities
├── config/                       # Model settings and prompts
├── resources/keywords.json       # Error keyword resource
├── requirements.txt
└── .env.example
```
