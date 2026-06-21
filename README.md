# Llama Model Registry

A curated manifest of local LLM models for the **Aire journal app**. The app fetches `models-manifest.json` from this GitHub repository to decide which models to offer for download, verify file integrity, and recommend a default model based on the user's Mac hardware.

Models are **not** hosted here. This repo only publishes the manifest — the actual GGUF files are downloaded from upstream sources (primarily Hugging Face).

## How it works

1. The Aire journal app checks this repo for the latest `models-manifest.json`.
2. The app reads hardware tiers, model metadata, download URLs, and selection rules.
3. Based on detected unified memory, the app recommends a model and lets the user download it.
4. After download, the app verifies the file against that source's `sha256` and `size_bytes` in the manifest.

## Manifest overview

The manifest (`models-manifest.json`) is the single source of truth. Key sections:

| Section | Purpose |
|---|---|
| `manifest_version` | Schema version for the app to handle breaking changes |
| `last_updated` | Date the manifest was last changed (`YYYY-MM-DD`) |
| `app` | Target app, runtime (`llama.cpp`), platform, and use cases |
| `hardware_tiers` | RAM tiers (`tier_1`–`tier_5`) for Apple Silicon Macs |
| `models` | Curated list of models with metadata, file info, and download sources |
| `app_selection_logic` | Suggested rules for auto-recommending a model by RAM |

Each model entry includes:

- Identity: `id`, `vendor`, `family`, `name`
- Capabilities: parameters, quantization, context length, diary suitability
- Hardware guidance: recommended tiers, RAM usage, expected speed
- File identity: `file.filename`, `file.size_human` (display hint from primary source)
- Download sources: URLs with priority, resume support, and **per-source** `size_bytes` and `sha256`

### Manifest version 3

As of `manifest_version` **3**, integrity fields live on each entry in `sources[]`, not in a shared `file` block. Primary and fallback mirrors often publish different GGUF bytes for the same filename; the app must verify against the source it actually downloaded.

## Curation policy

Every model in the manifest is **manually curated and tested** by the app developer before being added. Models are chosen for:

- Quality on diary conversation and summarization use cases
- Compatibility with `llama.cpp` on macOS (Apple Silicon)
- Reasonable performance across common Mac RAM configurations
- Reliable upstream hosting (GGUF quantizations from trusted publishers)

New models are added only after they are available, evaluated, and confirmed to work well in the app. Do not add models speculatively.

## Updating the manifest

When adding, removing, or changing any model entry:

1. **Verify upstream metadata** — confirm each source's `sha256`, `size_bytes`, and download URL against Hugging Face (file page or API). Do not rely on rounded human-readable sizes alone.
2. **Test in the app** — run the model through Aire's conversation and summarization flows on representative hardware.
3. **Update `last_updated`** — set this field to the date of your change (`YYYY-MM-DD`). This must be updated on every manifest edit so the app can detect fresh content.
4. **Commit and push** — the app picks up changes from GitHub on its next manifest check.

### Checking SHA256 from Hugging Face

For files on Hugging Face, the SHA256 is shown on each file's page under pointer details, or via the API:

```
https://huggingface.co/api/models/{owner}/{repo}/tree/main
```

Look for the `lfs.oid` field on the target file — that is the SHA256 hash.

## Consumption

The app should fetch the raw manifest from GitHub:

```
https://raw.githubusercontent.com/{owner}/{repo}/main/models-manifest.json
```

Use `last_updated` or the Git commit hash to determine whether a cached copy is stale.

## Validation

A standalone test loop validates every manifest model **per download source**: download primary → verify SHA-256 → run `llama-server` → one chat completion → remove primary GGUF → repeat for fallback (when listed). Each loaded model is confirmed via `GET /v1/models` on the console.

**The Aeris app does not need to be running.** The script only borrows the pinned `llama-server` binary from `../diarySwift/Helpers/`.

```bash
# Quick smoke test (one ~5 GB model)
./tools/validate_models.sh --models gemma-4-e4b-q4_k_m

# Full suite (all 9 models, sequential download → test → delete)
./tools/validate_models.sh
```

Each run **always writes a detailed report** to `reports/models/` (JSON + Markdown):

- **Every download source tested independently** (primary then fallback): download → infer → delete before next source
- Exact download URL (after redirects), headers, HTTP status, per-source download duration
- `GET /v1/models` response after each load (console + report)
- Size and SHA-256 comparison (manifest vs downloaded file) for each source
- Summary tables at the end of the Markdown report
- llama-server command, context budget, and server log excerpt
- Full chat request body and assistant reply
- Token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`) and per-phase timing

By default each GGUF is **deleted after its test** so only one model occupies disk at a time (~5–47 GB peak). Pass `--keep-cache` to retain downloads.

Requirements: macOS Apple Silicon, `python3`, network access for Hugging Face, and the `diarySwift` repo cloned at `../diarySwift` (used to fetch `llama-server` tag `b9009`).

## Metadata watch (static check)

Fast upstream drift detection without downloading GGUFs. Queries the Hugging Face API for each manifest source (primary and fallback), compares against the last recorded snapshot in [`metadata-watch/`](metadata-watch/), and appends to per-model JSONL history.

```bash
# First run — writes baseline snapshot per model
./tools/check_upstream_metadata.sh

# Subsequent runs — appends "unchanged" (timestamp only) or "changed" (+ diff)
./tools/check_upstream_metadata.sh

# Preview without updating watch files
./tools/check_upstream_metadata.sh --dry-run
```

Each model has one file: `metadata-watch/{model_id}.jsonl`. When upstream metadata is unchanged, only a timestamp line is appended. When it changes, a diff and new snapshot are recorded.

Reports are written to `reports/metadata/metadata-check-*.json` and `.md`. The watch files in `metadata-watch/` are intended to be committed so the team can track upstream drift over time.

| | `check_upstream_metadata.py` | `validate_models.py` |
|--|-------------------------------|----------------------|
| Network | HF API only (~KB) | Full GGUF download (~GB) |
| Verifies | Upstream metadata stability | Download + inference |
| State | `metadata-watch/*.jsonl` | ephemeral cache |

Compare the live manifest against the last watch snapshot (no network). Overwrites
`reports/metadata/manifest-watch-compare.md` and `.json` each run:

```bash
./tools/compare_manifest_watch.sh
```

## License

Model weights are subject to each vendor's license (see `license` and `license_url` per model in the manifest). This repository contains only the manifest JSON and documentation.
