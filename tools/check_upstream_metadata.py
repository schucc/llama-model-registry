#!/usr/bin/env python3
"""Static upstream metadata watch for manifest GGUF sources (no download).

Queries Hugging Face API for sha256/size per source, appends per-model JSONL
history under metadata-watch/, and writes a timestamped report under reports/metadata/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

USER_AGENT = "Aeris-metadata-watch/1.0 (macOS; HF API)"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "models-manifest.json"
DEFAULT_WATCH = REPO_ROOT / "metadata-watch"
DEFAULT_REPORTS = REPO_ROOT / "reports" / "metadata"

HF_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$"
)

REQUIRED_MANIFEST_VERSION = 3

COMPARE_FIELDS = (
    "upstream_sha256",
    "upstream_size_bytes",
    "upstream_last_modified",
    "file_present",
    "manifest_url",
)


def require_manifest_v3(manifest: dict[str, Any]) -> None:
    version = manifest.get("manifest_version")
    if version != REQUIRED_MANIFEST_VERSION:
        raise SystemExit(
            f"unsupported manifest_version {version!r} (expected {REQUIRED_MANIFEST_VERSION})"
        )


def validate_model_sources(model: dict[str, Any]) -> None:
    model_id = model["id"]
    file_block = model.get("file") or {}
    if not file_block.get("filename"):
        raise SystemExit(f"{model_id}: missing file.filename")
    sources = model.get("sources") or []
    if not sources:
        raise SystemExit(f"{model_id}: no sources[] entries")
    for src in sources:
        if "sha256" not in src or "size_bytes" not in src:
            raise SystemExit(
                f"{model_id}: source priority {src.get('priority')} missing sha256/size_bytes "
                "(manifest v3 requires per-source integrity pins)"
            )
        if "url" not in src:
            raise SystemExit(f"{model_id}: source priority {src.get('priority')} missing url")


def log(msg: str) -> None:
    print(msg, flush=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def report_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def yn(val: bool | None) -> str:
    if val is None:
        return "—"
    return "yes" if val else "no"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_hf_resolve_url(url: str) -> tuple[str, str] | None:
    m = HF_RESOLVE_RE.match(url.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def source_role(priority: int | None) -> str:
    return "primary" if priority == 1 else "fallback"


def manifest_sources(model: dict[str, Any]) -> list[dict[str, Any]]:
    filename = model["file"]["filename"]
    return [
        {
            "priority": s.get("priority"),
            "source_role": source_role(s.get("priority")),
            "manifest_url": s["url"],
            "filename": filename,
            "size_bytes": int(s["size_bytes"]),
            "sha256": s["sha256"].lower(),
        }
        for s in sorted(model.get("sources", []), key=lambda x: x.get("priority", 99))
    ]


def manifest_block(model: dict[str, Any]) -> dict[str, Any]:
    f = model["file"]
    return {
        "filename": f["filename"],
        "size_human": f.get("size_human"),
        "sources": manifest_sources(model),
    }


def fetch_hf_tree(repo: str) -> list[dict[str, Any]]:
    api_url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_hf_model(repo: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if repo in cache:
        return cache[repo]
    api_url = f"https://huggingface.co/api/models/{repo}"
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    info = {
        "last_modified": data.get("lastModified"),
        "sha": data.get("sha"),
    }
    cache[repo] = info
    return info


def head_resolve_url(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def fetch_source_metadata(
    source: dict[str, Any],
    filename: str,
    repo_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_url = source["url"]
    priority = source.get("priority")
    parsed = parse_hf_resolve_url(manifest_url)
    record: dict[str, Any] = {
        "source_name": source.get("name", ""),
        "source_role": source_role(priority),
        "priority": priority,
        "manifest_url": manifest_url,
        "filename": filename,
        "manifest_size_bytes": int(source["size_bytes"]),
        "manifest_sha256": source["sha256"].lower(),
        "repo": None,
        "upstream_size_bytes": None,
        "upstream_sha256": None,
        "upstream_last_modified": None,
        "file_present": False,
        "http_head_status": None,
        "fetch_error": None,
    }

    if not parsed:
        record["fetch_error"] = f"unrecognized URL pattern: {manifest_url}"
        return record

    repo, expected_path = parsed
    record["repo"] = repo
    if expected_path != filename:
        record["fetch_error"] = (
            f"URL filename {expected_path!r} != manifest filename {filename!r}"
        )

    record["http_head_status"] = head_resolve_url(manifest_url)

    try:
        model_info = fetch_hf_model(repo, repo_cache)
        record["upstream_last_modified"] = model_info.get("last_modified")
    except Exception as exc:
        record["fetch_error"] = record.get("fetch_error") or f"model info: {exc}"

    try:
        tree = fetch_hf_tree(repo)
        entry = next((x for x in tree if x.get("path") == filename), None)
        if entry is None:
            record["file_present"] = False
            record["fetch_error"] = record.get("fetch_error") or f"{filename} not in repo tree"
            return record

        record["file_present"] = True
        lfs = entry.get("lfs") or {}
        record["upstream_sha256"] = (lfs.get("oid") or "").lower() or None
        size = lfs.get("size")
        if size is None:
            size = entry.get("size")
        record["upstream_size_bytes"] = int(size) if size is not None else None
    except Exception as exc:
        record["fetch_error"] = str(exc)

    return record


def fetch_model_upstream(model: dict[str, Any]) -> list[dict[str, Any]]:
    filename = model["file"]["filename"]
    sources = sorted(model.get("sources", []), key=lambda s: s.get("priority", 99))
    repo_cache: dict[str, dict[str, Any]] = {}
    return [fetch_source_metadata(s, filename, repo_cache) for s in sources]


def compute_manifest_drift(
    manifest_sources_list: list[dict[str, Any]], upstream_sources: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest_by_priority = {s["priority"]: s for s in manifest_sources_list}
    primary = next((s for s in upstream_sources if s.get("priority") == 1), None)
    drift: dict[str, Any] = {
        "has_drift": False,
        "sha256_match": None,
        "size_match": None,
        "primary_upstream_sha256": primary.get("upstream_sha256") if primary else None,
        "primary_upstream_size_bytes": primary.get("upstream_size_bytes") if primary else None,
        "source_results": [],
        "details": [],
    }

    for upstream in upstream_sources:
        priority = upstream.get("priority")
        manifest_src = manifest_by_priority.get(priority)
        role = upstream.get("source_role") or f"priority-{priority}"
        src_result: dict[str, Any] = {
            "priority": priority,
            "source_role": role,
            "sha256_match": None,
            "size_match": None,
        }

        if manifest_src is None:
            drift["has_drift"] = True
            drift["details"].append(f"{role}: no matching manifest source")
            drift["source_results"].append(src_result)
            continue

        if not upstream.get("file_present"):
            drift["has_drift"] = True
            drift["details"].append(f"{role} file not present on Hugging Face")
            drift["source_results"].append(src_result)
            continue

        sha_ok = upstream.get("upstream_sha256") == manifest_src.get("sha256")
        size_ok = upstream.get("upstream_size_bytes") == manifest_src.get("size_bytes")
        src_result["sha256_match"] = sha_ok
        src_result["size_match"] = size_ok
        drift["source_results"].append(src_result)

        if not sha_ok:
            drift["has_drift"] = True
            drift["details"].append(
                f"{role} manifest sha256 {manifest_src.get('sha256')} != "
                f"upstream {upstream.get('upstream_sha256')}"
            )
        if not size_ok:
            drift["has_drift"] = True
            drift["details"].append(
                f"{role} manifest size {manifest_src.get('size_bytes')} != "
                f"upstream {upstream.get('upstream_size_bytes')}"
            )

    if primary:
        primary_manifest = manifest_by_priority.get(1)
        if primary_manifest:
            drift["sha256_match"] = primary.get("upstream_sha256") == primary_manifest.get("sha256")
            drift["size_match"] = (
                primary.get("upstream_size_bytes") == primary_manifest.get("size_bytes")
            )

    for upstream in upstream_sources:
        if upstream.get("priority") == 1:
            continue
        if (
            primary
            and primary.get("upstream_sha256")
            and upstream.get("upstream_sha256")
            and upstream["upstream_sha256"] != primary["upstream_sha256"]
        ):
            drift["details"].append(
                f"{upstream.get('source_role')} upstream sha256 differs from primary upstream"
            )

    return drift


def comparable_source_key(src: dict[str, Any]) -> tuple[Any, ...]:
    return (src.get("priority"), src.get("manifest_url"))


def sources_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    for field in COMPARE_FIELDS:
        if a.get(field) != b.get(field):
            return False
    return True


def diff_sources(
    old_sources: list[dict[str, Any]], new_sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    old_map = {comparable_source_key(s): s for s in old_sources}
    changes: list[dict[str, Any]] = []

    for new in new_sources:
        key = comparable_source_key(new)
        old = old_map.get(key)
        label = new.get("source_role") or f"priority-{new.get('priority')}"
        if old is None:
            changes.append(
                {
                    "source": label,
                    "field": "source_added",
                    "old": None,
                    "new": new.get("manifest_url"),
                }
            )
            continue
        for field in COMPARE_FIELDS:
            if old.get(field) != new.get(field):
                changes.append(
                    {
                        "source": label,
                        "field": field,
                        "old": old.get(field),
                        "new": new.get(field),
                    }
                )

    new_keys = {comparable_source_key(s) for s in new_sources}
    for key, old in old_map.items():
        if key not in new_keys:
            changes.append(
                {
                    "source": old.get("source_role") or f"priority-{old.get('priority')}",
                    "field": "source_removed",
                    "old": old.get("manifest_url"),
                    "new": None,
                }
            )

    return changes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def last_snapshot(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(records):
        if record.get("event") in ("snapshot", "changed") and record.get("sources"):
            return record
    return None


def watch_file_path(watch_dir: Path, model: dict[str, Any]) -> Path:
    return watch_dir / f"{model['id']}.jsonl"


def migrate_display_name_watch_file(watch_dir: Path, model: dict[str, Any]) -> None:
    """Rename display-name watch files back to `{model_id}.jsonl` if present."""
    target = watch_file_path(watch_dir, model)
    if target.is_file():
        return
    # Files created during the brief display-name naming scheme
    slug = model.get("name", "").strip().replace("(", "").replace(")", "")
    slug = re.sub(r"[\s/\\:]+", "-", slug)
    slug = re.sub(r"[^\w.\-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    legacy = watch_dir / f"{slug}.jsonl"
    if legacy.is_file():
        legacy.rename(target)


def snapshot_payload(
    model: dict[str, Any],
    sources: list[dict[str, Any]],
    manifest_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": model["id"],
        "model_name": model.get("name"),
        "sources": sources,
        "manifest": manifest_info,
    }


def check_model(
    model: dict[str, Any],
    watch_dir: Path,
    dry_run: bool,
    verbose: bool,
) -> dict[str, Any]:
    model_id = model["id"]
    migrate_display_name_watch_file(watch_dir, model)
    watch_path = watch_file_path(watch_dir, model)
    manifest_info = manifest_block(model)
    start = time.monotonic()

    result: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model.get("name"),
        "watch_file": str(watch_path.relative_to(REPO_ROOT)),
        "upstream_changed": False,
        "manifest_drift": None,
        "fetch_errors": [],
        "changes": [],
        "sources": [],
        "event_written": None,
        "last_snapshot_at": None,
        "duration_s": 0.0,
        "error": None,
    }

    log(f"\n=== {model_id} ({model.get('name', '')}) ===")
    try:
        sources = fetch_model_upstream(model)
        result["sources"] = sources

        for src in sources:
            if src.get("fetch_error"):
                result["fetch_errors"].append(
                    f"{src.get('source_role')}: {src['fetch_error']}"
                )
            if verbose:
                log(
                    f"  {src.get('source_role')} {src.get('repo')}: "
                    f"present={src.get('file_present')} "
                    f"manifest_sha256={src.get('manifest_sha256', '')[:16]}… "
                    f"upstream_sha256={src.get('upstream_sha256', '')[:16]}… "
                    f"manifest_size={src.get('manifest_size_bytes')} "
                    f"upstream_size={src.get('upstream_size_bytes')} "
                    f"lastModified={src.get('upstream_last_modified')}"
                )

        result["manifest_drift"] = compute_manifest_drift(manifest_info["sources"], sources)

        history = read_jsonl(watch_path)
        prev = last_snapshot(history)
        if prev:
            result["last_snapshot_at"] = prev.get("recorded_at")

        recorded_at = utc_now_iso()

        if prev is None:
            line = {
                "recorded_at": recorded_at,
                "event": "snapshot",
                "reason": "baseline",
                **snapshot_payload(model, sources, manifest_info),
            }
            result["event_written"] = "snapshot"
            if not dry_run:
                append_jsonl(watch_path, line)
            log(f"  baseline snapshot written")
        else:
            changes = diff_sources(prev.get("sources") or [], sources)
            if changes:
                result["upstream_changed"] = True
                result["changes"] = changes
                line = {
                    "recorded_at": recorded_at,
                    "event": "changed",
                    "changes": changes,
                    **snapshot_payload(model, sources, manifest_info),
                }
                result["event_written"] = "changed"
                if not dry_run:
                    append_jsonl(watch_path, line)
                log(f"  UPSTREAM CHANGED — {len(changes)} difference(s)")
                for c in changes:
                    log(f"    {c['source']} {c['field']}: {c.get('old')} → {c.get('new')}")
            else:
                line = {"recorded_at": recorded_at, "event": "unchanged"}
                result["event_written"] = "unchanged"
                if not dry_run:
                    append_jsonl(watch_path, line)
                log(f"  unchanged")

        if result["manifest_drift"]["has_drift"]:
            log(f"  manifest drift: {'; '.join(result['manifest_drift']['details'])}")

    except Exception as exc:
        result["error"] = str(exc)
        log(f"  ERROR: {exc}")

    result["duration_s"] = round(time.monotonic() - start, 3)
    return result


def build_summary_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        sources = r.get("sources") or []
        primary = next((s for s in sources if s.get("priority") == 1), None)
        fallback = next((s for s in sources if s.get("priority") == 2), None)
        drift = r.get("manifest_drift") or {}
        rows.append(
            {
                "model_id": r["model_id"],
                "upstream_changed": r.get("upstream_changed"),
                "manifest_drift": drift.get("has_drift"),
                "primary_sha256": (primary or {}).get("upstream_sha256"),
                "fallback_sha256": (fallback or {}).get("upstream_sha256"),
                "primary_last_modified": (primary or {}).get("upstream_last_modified"),
                "fallback_last_modified": (fallback or {}).get("upstream_last_modified"),
                "last_snapshot_at": r.get("last_snapshot_at"),
                "event_written": r.get("event_written"),
                "fetch_errors": r.get("fetch_errors"),
                "watch_file": r.get("watch_file"),
                "notes": "; ".join(drift.get("details") or []) or None,
            }
        )
    return rows


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = report["summary"]
    lines.append("# Upstream metadata check report")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Manifest:** `{report['manifest']}`")
    lines.append(f"- **Watch dir:** `{report['watch_dir']}`")
    lines.append(f"- **Dry run:** {report['dry_run']}")
    lines.append(f"- **Models checked:** {summary['total']}")
    lines.append(f"- **Upstream changed:** {summary['upstream_changed']}")
    lines.append(f"- **Manifest drift:** {summary['manifest_drift']}")
    lines.append(f"- **Fetch errors:** {summary['fetch_errors']}")
    lines.append(f"- **Duration:** {summary['total_duration_s']:.1f}s")
    lines.append("")

    for model in report["models"]:
        lines.append(f"## {model['model_id']}")
        lines.append("")
        if model.get("error"):
            lines.append(f"**Error:** {model['error']}")
        lines.append(f"- **Watch file:** `{model.get('watch_file')}`")
        lines.append(f"- **Event written:** {model.get('event_written')}")
        lines.append(f"- **Upstream changed:** {yn(model.get('upstream_changed'))}")
        drift = model.get("manifest_drift") or {}
        lines.append(f"- **Manifest drift:** {yn(drift.get('has_drift'))}")
        if drift.get("details"):
            lines.append(f"- **Drift details:** {'; '.join(drift['details'])}")
        lines.append("")

        for src in model.get("sources") or []:
            role = src.get("source_role", "?")
            lines.append(f"### {role.upper()}: {src.get('source_name')}")
            lines.append(f"- **Manifest URL:** `{src.get('manifest_url')}`")
            lines.append(f"- **Repo:** `{src.get('repo')}`")
            lines.append(f"- **File present:** {yn(src.get('file_present'))}")
            lines.append(f"- **Manifest SHA-256:** `{src.get('manifest_sha256')}`")
            lines.append(f"- **Upstream SHA-256:** `{src.get('upstream_sha256')}`")
            lines.append(f"- **Manifest size:** {src.get('manifest_size_bytes')}")
            lines.append(f"- **Upstream size:** {src.get('upstream_size_bytes')}")
            lines.append(f"- **Repo last modified:** {src.get('upstream_last_modified') or '—'}")
            lines.append(f"- **HTTP HEAD:** {src.get('http_head_status')}")
            if src.get("fetch_error"):
                lines.append(f"- **Fetch error:** {src['fetch_error']}")
            lines.append("")

        manifest_info = model.get("_manifest_snapshot") or {}
        manifest_sources_list = manifest_info.get("sources") or []
        if manifest_sources_list:
            lines.append("### Manifest vs upstream (per source)")
            lines.append("")
            lines.append("| Source | | Manifest | Upstream | SHA match | Size match |")
            lines.append("|--------|---|----------|----------|-----------|------------|")
            manifest_by_priority = {s["priority"]: s for s in manifest_sources_list}
            for upstream in model.get("sources") or []:
                priority = upstream.get("priority")
                role = upstream.get("source_role", "?")
                manifest_src = manifest_by_priority.get(priority) or {}
                src_drift = next(
                    (r for r in (drift.get("source_results") or []) if r.get("priority") == priority),
                    {},
                )
                lines.append(
                    f"| {role} | SHA-256 | `{manifest_src.get('sha256', '—')}` | "
                    f"`{upstream.get('upstream_sha256') or '—'}` | "
                    f"{yn(src_drift.get('sha256_match'))} | |"
                )
                lines.append(
                    f"| | Size | {manifest_src.get('size_bytes', '—')} | "
                    f"{upstream.get('upstream_size_bytes') or '—'} | | "
                    f"{yn(src_drift.get('size_match'))} |"
                )
            lines.append("")

        if model.get("changes"):
            lines.append("### Upstream changes vs last snapshot")
            lines.append("")
            lines.append("| Source | Field | Old | New |")
            lines.append("|--------|-------|-----|-----|")
            for c in model["changes"]:
                old = c.get("old")
                new = c.get("new")
                lines.append(f"| {c.get('source')} | {c.get('field')} | {old} | {new} |")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(
        "| Model | Upstream changed | Manifest drift | Primary SHA256 | "
        "Fallback SHA256 | Last snapshot | Event | Notes |"
    )
    lines.append(
        "|-------|------------------|----------------|----------------|"
        "-----------------|---------------|-------|-------|"
    )
    for row in report.get("summary_table") or []:
        psha = row.get("primary_sha256") or ""
        fsha = row.get("fallback_sha256") or ""
        notes = (row.get("notes") or "").replace("|", "\\|")[:60]
        lines.append(
            f"| {row.get('model_id')} | {yn(row.get('upstream_changed'))} | "
            f"{yn(row.get('manifest_drift'))} | `{psha[:16]}…` | "
            f"`{(fsha[:16] + '…') if fsha else '—'}` | "
            f"{row.get('last_snapshot_at') or '—'} | {row.get('event_written')} | "
            f"{notes or '—'} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch Hugging Face upstream metadata for manifest models (static check)."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--watch-dir", type=Path, default=DEFAULT_WATCH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models", type=str, default="", help="Comma-separated model ids")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compare and report without appending to metadata-watch JSONL",
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="metadata-check",
        help="Filename prefix for reports",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    require_manifest_v3(manifest)
    models: list[dict[str, Any]] = list(manifest.get("models", []))
    for model in models:
        validate_model_sources(model)

    if args.models.strip():
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m["id"] in wanted]
        missing = wanted - {m["id"] for m in models}
        if missing:
            raise SystemExit(f"unknown model id(s): {', '.join(sorted(missing))}")

    if not models:
        raise SystemExit("no models to check")

    log(f"Checking upstream metadata for {len(models)} model(s)")
    log(f"manifest: {args.manifest}")
    log(f"watch dir: {args.watch_dir}")
    if args.dry_run:
        log("dry-run: will not append to watch files")

    run_start = time.monotonic()
    results: list[dict[str, Any]] = []
    for model in models:
        r = check_model(model, args.watch_dir, args.dry_run, args.verbose)
        r["_manifest_snapshot"] = manifest_block(model)
        results.append(r)

    upstream_changed = sum(1 for r in results if r.get("upstream_changed"))
    manifest_drift_count = sum(
        1 for r in results if (r.get("manifest_drift") or {}).get("has_drift")
    )
    fetch_error_count = sum(1 for r in results if r.get("fetch_errors") or r.get("error"))
    total_duration = time.monotonic() - run_start

    log(f"\n{'=' * 60}")
    log(
        f"Done: {len(results)} checked, {upstream_changed} upstream changed, "
        f"{manifest_drift_count} manifest drift, {fetch_error_count} errors"
    )

    ts = report_timestamp()
    json_path = args.report_dir / f"{args.report_prefix}-{ts}.json"
    md_path = args.report_dir / f"{args.report_prefix}-{ts}.md"

    full_report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "manifest": str(args.manifest),
        "watch_dir": str(args.watch_dir),
        "dry_run": args.dry_run,
        "summary": {
            "total": len(results),
            "upstream_changed": upstream_changed,
            "manifest_drift": manifest_drift_count,
            "fetch_errors": fetch_error_count,
            "total_duration_s": round(total_duration, 3),
        },
        "summary_table": build_summary_table(results),
        "models": results,
    }

    write_json_report(json_path, full_report)
    write_markdown_report(md_path, full_report)
    log(f"\nReport JSON: {json_path}")
    log(f"Report Markdown: {md_path}")

    failed = upstream_changed > 0 or manifest_drift_count > 0 or fetch_error_count > 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
