# GHA-Agent Replication Package

This package accompanies **GHA-Agent: A Multi-Agent Framework for GitHub Actions Log Parsing**. It contains the GHA-Agent implementation and GHA-Bench, a benchmark of 323 GitHub Actions failure cases from 323 repositories across eight programming languages.

## Package contents

```text
GHA-Replication-Package/
├── README.md
├── GHA-Agent/             # Parser implementation, configuration, and prompts
└── GHA-Bench/
    ├── cases/            # Failure logs and workflow/job metadata
    ├── human_review/     # Manually annotated error-line ranges
    └── golden_patches/   # Developer patches from failing to fixed commits
```

| Component | Documentation |
| --- | --- |
| **GHA-Agent** | [Installation, API configuration, single-log parsing, batch execution, and outputs](GHA-Agent/README.md) |
| **GHA-Bench** | [Directory layout, metadata fields, annotation format, patches, and data-loading example](GHA-Bench/README.md) |

## Getting started

1. Follow the [GHA-Agent setup instructions](GHA-Agent/README.md#setup) to install the Python dependencies and configure an API key.
2. Run the [single-log example](GHA-Agent/README.md#parse-one-log) to check the setup, then use the [batch runner](GHA-Agent/README.md#parse-gha-bench) for the benchmark.
3. Use the shared case path described in [GHA-Bench](GHA-Bench/README.md#directory-layout) to match predictions with their human annotations and golden patches.

GHA-Bench contains **250 Basic cases** (`basic/`) and **73 Complex cases** (`complex/`). The logs, line annotations, and patches are included locally. Repository snapshots are referenced by commit-specific download links in each case's metadata; source archives are not bundled here.

The parser reads the failure logs and produces predicted error-line ranges. Human annotations and golden patches are reference artifacts for evaluation. The included runners perform parsing and record runtime/token statistics; they do not automatically compute benchmark scores or execute repository repair workflows.
