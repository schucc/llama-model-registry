# Llama Model Registry

A curated manifest of local LLM models for the **Aire journal app**. The app fetches `models-manifest.json` from this GitHub repository to decide which models to offer for download, verify file integrity, and recommend a default model based on the user's Mac hardware.

Models are **not** hosted here. This repo only publishes the manifest — the actual GGUF files are downloaded from upstream sources (primarily Hugging Face).

## How it works

1. The Aire journal app checks this repo for the latest `models-manifest.json`.
2. The app reads hardware tiers, model metadata, download URLs, and selection rules.
3. Based on detected unified memory, the app recommends a model and lets the user download it.
4. After download, the app verifies the file against the `sha256` and `size_bytes` fields in the manifest.

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
- File integrity: `filename`, `size_bytes`, `size_human`, `sha256`
- Download sources: URLs with priority and resume support

## Curation policy

Every model in the manifest is **manually curated and tested** by the app developer before being added. Models are chosen for:

- Quality on diary conversation and summarization use cases
- Compatibility with `llama.cpp` on macOS (Apple Silicon)
- Reasonable performance across common Mac RAM configurations
- Reliable upstream hosting (GGUF quantizations from trusted publishers)

New models are added only after they are available, evaluated, and confirmed to work well in the app. Do not add models speculatively.

## Updating the manifest

When adding, removing, or changing any model entry:

1. **Verify upstream metadata** — confirm `sha256`, `size_bytes`, and download URLs against the source (e.g. Hugging Face file page or API). Do not rely on rounded human-readable sizes alone.
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

## License

Model weights are subject to each vendor's license (see `license` and `license_url` per model in the manifest). This repository contains only the manifest JSON and documentation.
