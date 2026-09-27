# GHA-Bench

GHA-Bench is the benchmark accompanying **GHA-Agent: A Multi-Agent Framework for GitHub Actions Log Parsing**. This release contains **323 failure cases from 323 repositories**, covering eight programming languages. Each case provides one failed-job log, workflow/job metadata, manually annotated error-line ranges, and a developer patch linking the failing and subsequent fixed commits.

## Dataset composition

GHA-Bench is organized into two subsets, **Basic** (`basic/`) and **Complex** (`complex/`), based on log length as described in the paper.

| Language directory | Basic (`basic`) | Complex (`complex`) | Total |
| --- | ---: | ---: | ---: |
| `c` | 6 | 4 | 10 |
| `c#` | 13 | 4 | 17 |
| `c++` | 22 | 4 | 26 |
| `java` | 31 | 10 | 41 |
| `javascript` | 28 | 11 | 39 |
| `php` | 8 | 3 | 11 |
| `python` | 132 | 31 | 163 |
| `r` | 10 | 6 | 16 |
| **Total** | **250** | **73** | **323** |

## Directory layout

The three artifact directories use the same relative case path:

```text
{difficulty}/{language}/{repository}/{post_id}/{run_id}/
```

| Path component | Meaning | Example |
| --- | --- | --- |
| `difficulty` | Benchmark subset | `basic` |
| `language` | Repository language group | `python` |
| `repository` | Repository identifier stored as `owner_repository` | `psf_black` |
| `post_id` | GitHub pull request number | `4628` |
| `run_id` | Failed GitHub Actions workflow run ID | `13956990967` |

Use the complete relative case path to join the artifacts. Repository names can themselves contain underscores; obtain the canonical `owner/repository` from the metadata's `code_download_links.repository` field instead of splitting the directory name on every underscore.

```text
GHA-Bench/
├── cases/
│   └── {difficulty}/{language}/{repository}/{post_id}/{run_id}/
│       ├── {run_id}.json
│       └── {run_id}~{job_name}~whole_log.txt
├── human_review/
│   └── {difficulty}/{language}/{repository}/{post_id}/{run_id}/
│       └── human_review.txt
└── golden_patches/
    └── {difficulty}/{language}/{repository}/{post_id}/{run_id}/
        └── current_to_next.diff
```

For example, the `psf/black` case is represented by:

```text
cases/basic/python/psf_black/4628/13956990967/
    13956990967.json
    13956990967~Changelog Entry Check~whole_log.txt
human_review/basic/python/psf_black/4628/13956990967/
    human_review.txt
golden_patches/basic/python/psf_black/4628/13956990967/
    current_to_next.diff
```

There are 323 metadata JSON files, 323 logs, 323 annotation files, and 323 patch files in this release.

## Failure logs

`cases/.../{run_id}~{job_name}~whole_log.txt` contains the complete archived log for the selected failed job. It is a UTF-8 text file and retains GitHub Actions timestamps and any control sequences present in the log.

The job-name segment may contain spaces, parentheses, or other punctuation. Characters unsuitable for filenames may have been replaced with underscores, so this segment is not always identical to `jobs[].name`. To locate a log programmatically, select `*whole_log.txt` within the case directory instead of rebuilding its filename from the job name.

Preserve the original line order and line count when using annotations or parser outputs. Removing blank lines, filtering messages, or rewrapping text before resolving ranges changes the referenced positions.

## Metadata JSON

Each `cases/.../{run_id}.json` is a JSON object with three top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `total_count` | integer | Job count reported in the collected GitHub API response; it can differ from `len(jobs)`, so iterate the stored `jobs` array directly |
| `jobs` | array of objects | Stored job records for the workflow run, including jobs with outcomes other than failure |
| `code_download_links` | object | Repository identity, commit SHAs, source archive URLs, and collection provenance |

Every case in this release has exactly one stored job with `conclusion == "failure"`. The accompanying log corresponds to that job. Other records may have `success`, `skipped`, `cancelled`, or other conclusions.

### Job records

| Field(s) | Meaning |
| --- | --- |
| `id`, `run_id`, `run_attempt`, `node_id` | GitHub job/run identifiers and run attempt |
| `name`, `workflow_name` | Job and workflow names |
| `head_branch`, `head_sha` | Branch and commit recorded for the job |
| `status`, `conclusion` | Execution state and outcome |
| `created_at`, `started_at`, `completed_at` | Recorded timestamps |
| `run_url`, `url`, `html_url`, `check_run_url` | Links to the run, job, and check records |
| `steps` | Step records containing `name`, `number`, `status`, `conclusion`, `started_at`, and `completed_at` |
| `labels`, `runner_id`, `runner_name`, `runner_group_id`, `runner_group_name` | Runner information |

### Commit and provenance records

The `code_download_links` object contains:

| Field | Meaning |
| --- | --- |
| `repository` | Canonical repository name, e.g. `psf/black` |
| `run_id` | Workflow run ID, stored here as a string |
| `current_commit` | Failing revision: `sha`, `zipball_api_url`, and `tarball_api_url` |
| `next_commit` | Subsequent fixed revision: the same three fields |
| `matching_source` | Collection provenance and links back to the pull request and workflow run |

`matching_source` includes `post_key`, `matching_posts_path`, `pull_request_html_url`, `pull_request_diff_url`, `workflow_run_api_url`, `workflow_run_html_url`, `workflow_name`, and `workflow_conclusion`. The `matching_posts_path` value records a path used during collection; that collection file is not included in this package.

Source archives are not bundled. Use the commit-specific archive URLs to retrieve the failing or fixed repository snapshot when needed for downstream analysis. Reading the included logs, labels, and patches requires no network access.

## Human annotations

`human_review/.../human_review.txt` stores a **JSON array of line ranges**, despite its `.txt` extension:

```json
[[120, 120], [121, 121]]
```

This is the annotation for the `psf/black` example above. Each pair is `[start_line, end_line]` with **1-based, inclusive** boundaries in the corresponding original `whole_log.txt` file. A pair such as `[120, 120]` selects one line; `[120, 121]` selects two. A case can contain multiple disjoint or adjacent ranges.

The selected ranges identify human-annotated evidence for the failure. These files contain line ranges only; they do not contain free-text diagnoses or failure-category fields. For line-level evaluation, take the union of the annotated line numbers so that a line is counted once. With Python's zero-based indexing, extract an annotated range using `lines[start - 1:end]`.

## Golden patches

`golden_patches/.../current_to_next.diff` is a Git-style diff from `code_download_links.current_commit.sha` to `code_download_links.next_commit.sha`. It records the developer changes associated with the fix and can be used as a reference for downstream diagnosis or repair evaluation.

The patch is relative to the failing repository snapshot, with paths such as `a/CHANGES.md` and `b/CHANGES.md`. The `pull_request_diff_url` in the metadata points to the pull request diff; use the bundled `current_to_next.diff` for this specific commit pair.

For log-parsing evaluation, provide the failure log to the parser and compare its output with `human_review.txt`. The labels, fixed snapshot, and golden patch are reference targets rather than parser inputs.

## Load a case in Python

Run this example from the **replication package root**. It uses only the Python standard library and does not make API requests:

```python
import json
from pathlib import Path

bench = Path("GHA-Bench")
case_id = Path("basic/python/psf_black/4628/13956990967")
case_dir = bench / "cases" / case_id
run_id = case_id.name

metadata = json.loads(
    (case_dir / f"{run_id}.json").read_text(encoding="utf-8")
)
log_files = list(case_dir.glob("*whole_log.txt"))
assert len(log_files) == 1
lines = log_files[0].read_text(encoding="utf-8").splitlines()

ranges = json.loads(
    (bench / "human_review" / case_id / "human_review.txt").read_text(
        encoding="utf-8"
    )
)
golden_patch = (
    bench / "golden_patches" / case_id / "current_to_next.diff"
).read_text(encoding="utf-8")

failed_job = next(
    job for job in metadata["jobs"] if job["conclusion"] == "failure"
)
print(metadata["code_download_links"]["repository"], failed_job["name"])
for start, end in ranges:
    assert 1 <= start <= end <= len(lines)
    for line_number in range(start, end + 1):
        print(f"{line_number}: {lines[line_number - 1]}")
```

To process the data with GHA-Agent, follow its [setup and batch execution instructions](../GHA-Agent/README.md). Pass `GHA-Bench/cases` as the input tree, adjusting the path to your working directory.
