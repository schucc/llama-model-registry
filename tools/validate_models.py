#!/usr/bin/env python3
"""Validate all manifest GGUF models: per-source download, llama-server, chat.

For each model: download primary → infer → remove primary → download fallback
→ infer → remove fallback (when present). Mirrors Aeris llama-server usage.

Always writes a detailed JSON + Markdown report under reports/models/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

USER_PROMPT = (
    "While many microorganisms are beneficial, harmful microbes (pathogens) act as "
    "agents of disease, causing illnesses like influenza, tuberculosis, and food "
    "poisoning in humans and animals. They also cause agricultural damage by "
    "blighting crops, spoil food supplies through rot, and degrade vital materials "
    "such as wood and textiles."
)

USER_AGENT = "Aeris-model-validator/1.0 (macOS; GGUF download)"

FAMILY_PROFILE = {
    "Llama 3.1": "llama",
    "Llama 3.3": "llama",
    "Qwen 2.5": "qwen",
    "Gemma 2": "gemma_2",
    "Gemma 4": "gemma_4",
}

KV_BYTES_PER_TOKEN: dict[str, int] = {
    "llama-3.1-8b-q4_k_m": 524_288,
    "qwen2.5-7b-q4_k_m": 401_408,
    "qwen2.5-14b-q4_k_m": 983_040,
    "qwen2.5-32b-q4_k_m": 1_310_720,
    "qwen2.5-72b-q4_k_m": 2_621_440,
    "llama-3.3-70b-q4_k_m": 2_621_440,
    "gemma-2-9b-q4_k_m": 602_112,
    "gemma-4-e4b-q4_k_m": 401_408,
    "gemma-4-31b-q4_k_m": 1_310_720,
}

GIB = 1_073_741_824
MIB = 1_048_576
UTILIZATION_CAP = 0.70
MIN_CONTEXT_TOKENS = 2_048
MAX_CONTEXT_TOKENS = 65_536
CONTEXT_ALIGNMENT = 256

LISTENER_TIMEOUT_S = 180
INFERENCE_TIMEOUT_S = 300
CHAT_TIMEOUT_S = 120

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "models-manifest.json"
DEFAULT_CACHE = REPO_ROOT / ".model-test-cache"
DEFAULT_REPORTS = REPO_ROOT / "reports" / "models"
DEFAULT_LLAMA_SERVER = REPO_ROOT.parent / "diarySwift" / "Helpers" / "llama-server"

REQUIRED_MANIFEST_VERSION = 3


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


@dataclass
class RunState:
    server_proc: subprocess.Popen[Any] | None = None
    keep_cache: bool = False
    verbose: bool = False
    report_paths: list[Path] = field(default_factory=list)


_state = RunState()


def log(msg: str) -> None:
    print(msg, flush=True)


def log_verbose(msg: str) -> None:
    if _state.verbose:
        log(msg)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def report_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def human_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= GIB:
        return f"{n / GIB:.2f} GB ({n} bytes)"
    if n >= MIB:
        return f"{n / MIB:.2f} MB ({n} bytes)"
    return f"{n} bytes"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def profile_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("llama_server_args_profiles", {})
    return {k: v for k, v in raw.items() if not k.startswith("comment") and isinstance(v, dict)}


def resolve_server_args(model: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> list[str]:
    inline = model.get("llama_server_args")
    if inline is not None:
        return list(inline)
    family = model.get("family", "")
    profile_key = FAMILY_PROFILE.get(family)
    if profile_key and profile_key in profiles:
        return list(profiles[profile_key].get("llama_server_args", []))
    return []


def resolve_chat_template_kwargs(
    model: dict[str, Any], profiles: dict[str, dict[str, Any]]
) -> dict[str, bool] | None:
    inline = model.get("llama_chat_template_kwargs")
    if inline:
        return {k: bool(v) for k, v in inline.items()}
    family = model.get("family", "")
    profile_key = FAMILY_PROFILE.get(family)
    if profile_key and profile_key in profiles:
        kwargs = profiles[profile_key].get("llama_chat_template_kwargs")
        if kwargs:
            return {k: bool(v) for k, v in kwargs.items()}
    return None


def physical_memory_bytes() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except (subprocess.CalledProcessError, ValueError):
        return 16 * GIB


def vm_reclaimable_bytes() -> int:
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
        out = subprocess.check_output(["vm_stat"], text=True)
    except (subprocess.CalledProcessError, ValueError):
        return physical_memory_bytes() // 3

    counts: dict[str, int] = {}
    for line in out.splitlines():
        m = re.match(r"^Pages (\w+(?:\s+\w+)?):\s+(\d+)\.", line.strip())
        if m:
            counts[m.group(1).replace(" ", "_").lower()] = int(m.group(2))

    free = counts.get("free", 0)
    inactive = counts.get("inactive", 0)
    speculative = counts.get("speculative", 0)
    return (free + inactive + speculative) * page_size


def system_overhead_bytes(physical: int | None = None) -> int:
    phys = physical if physical is not None else physical_memory_bytes()
    two_gb = 2 * GIB
    four_gb = 4 * GIB
    scaled = int(phys * 0.12)
    return min(four_gb, max(two_gb, scaled))


def estimated_kv_bytes_per_token(model_id: str, file_bytes: int) -> int:
    if model_id in KV_BYTES_PER_TOKEN:
        return KV_BYTES_PER_TOKEN[model_id]
    gb = file_bytes / GIB
    if gb < 6:
        return 401_408
    if gb < 12:
        return 983_040
    if gb < 22:
        return 1_310_720
    if gb < 35:
        return 524_288
    return 2_621_440


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return (value // alignment) * alignment


def compute_ctx_budget(model_path: Path, model_id: str) -> dict[str, Any]:
    file_bytes = model_path.stat().st_size
    kv_bytes = estimated_kv_bytes_per_token(model_id, file_bytes)
    available = vm_reclaimable_bytes()
    overhead = system_overhead_bytes()
    available_gb = available / GIB
    model_gb = file_bytes / GIB
    overhead_gb = overhead / GIB
    headroom_gb = available_gb - model_gb - overhead_gb
    kv_mb_per_token = kv_bytes / MIB

    if headroom_gb <= 0 or kv_mb_per_token <= 0:
        raw_tokens = float(MIN_CONTEXT_TOKENS)
    else:
        raw_tokens = headroom_gb * 1024.0 / kv_mb_per_token

    utilized = raw_tokens * UTILIZATION_CAP
    aligned = align_down(int(utilized), CONTEXT_ALIGNMENT)
    clamped = min(MAX_CONTEXT_TOKENS, max(MIN_CONTEXT_TOKENS, aligned))

    metrics = {
        "ctx_size": clamped,
        "available_bytes": available,
        "model_bytes": file_bytes,
        "overhead_bytes": overhead,
        "available_gb": round(available_gb, 2),
        "model_gb": round(model_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "headroom_gb": round(headroom_gb, 2),
        "kv_bytes_per_token": kv_bytes,
        "kv_mb_per_token": round(kv_mb_per_token, 4),
        "raw_context_tokens": int(raw_tokens),
        "utilization_cap": UTILIZATION_CAP,
        "physical_memory_bytes": physical_memory_bytes(),
    }
    log_verbose(
        f"[llama ctx] avail={available_gb:.1f}GB model={model_gb:.1f}GB "
        f"overhead={overhead_gb:.1f}GB kv={kv_mb_per_token:.2f}MB/tok "
        f"raw={raw_tokens:.0f} util={UTILIZATION_CAP * 100:.0f}% → ctx={clamped}"
    )
    return metrics


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def integrity_comparison(
    manifest_size: int,
    manifest_sha: str,
    actual_size: int,
    actual_sha: str,
) -> dict[str, Any]:
    return {
        "manifest_size_bytes": manifest_size,
        "manifest_size_human": human_bytes(manifest_size),
        "actual_size_bytes": actual_size,
        "actual_size_human": human_bytes(actual_size),
        "size_match": actual_size == manifest_size,
        "size_delta_bytes": actual_size - manifest_size,
        "manifest_sha256": manifest_sha.lower(),
        "actual_sha256": actual_sha.lower(),
        "sha256_match": actual_sha.lower() == manifest_sha.lower(),
    }


def model_dir_for(cache_dir: Path, model: dict[str, Any]) -> Path:
    return cache_dir / model["id"]


def source_paths(
    cache_dir: Path, model: dict[str, Any], priority: int
) -> tuple[Path, Path]:
    """Per-source GGUF and partial download paths."""
    model_dir = model_dir_for(cache_dir, model)
    filename = model["file"]["filename"]
    dest = model_dir / f"p{priority}-{filename}"
    part = model_dir / f"p{priority}-{filename}.part"
    return dest, part


def download_source(
    model: dict[str, Any],
    source: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    """Download and verify one manifest source."""
    model_dir = model_dir_for(cache_dir, model)
    model_dir.mkdir(parents=True, exist_ok=True)

    priority = source.get("priority", 99)
    manifest_url = source["url"]
    expected_sha = source["sha256"]
    expected_size = int(source["size_bytes"])
    dest_path, part_path = source_paths(cache_dir, model, priority)
    role = "primary" if priority == 1 else "fallback"

    for path in (dest_path, part_path):
        if path.exists():
            path.unlink()

    attempt: dict[str, Any] = {
        "source_name": source.get("name", ""),
        "source_role": role,
        "priority": priority,
        "manifest_url": manifest_url,
        "expected_sha256": expected_sha.lower(),
        "expected_size_bytes": expected_size,
        "exact_download_url": manifest_url,
        "supports_resume": source.get("supports_resume"),
        "request_headers": {"User-Agent": USER_AGENT},
        "http_status": None,
        "response_content_length": None,
        "success": False,
        "error": None,
        "download_duration_s": 0.0,
        "integrity": None,
        "local_path": str(dest_path),
    }

    log(f"  [{model['id']}] downloading {role} (priority {priority})")
    log(f"    URL: {manifest_url}")
    attempt_start = time.monotonic()
    try:
        http_meta = _download_url(manifest_url, part_path, expected_size, attempt["request_headers"])
        attempt["http_status"] = http_meta.get("http_status")
        attempt["response_content_length"] = http_meta.get("content_length")
        attempt["exact_download_url"] = http_meta.get("final_url") or manifest_url
        if http_meta.get("range_header"):
            attempt["request_headers"]["Range"] = http_meta["range_header"]

        actual_size = part_path.stat().st_size
        actual_sha = sha256_file(part_path)
        integrity = integrity_comparison(expected_size, expected_sha, actual_size, actual_sha)
        attempt["integrity"] = integrity

        if not integrity["sha256_match"]:
            raise RuntimeError(
                f"SHA-256 mismatch: got {actual_sha}, expected {expected_sha} "
                f"(size={actual_size}, manifest_size={expected_size})"
            )
        if not integrity["size_match"]:
            log(
                f"  [{model['id']}] WARNING [{role}]: size "
                f"{actual_size} != manifest {expected_size}, but SHA-256 OK"
            )

        part_path.rename(dest_path)
        attempt["success"] = True
        attempt["download_duration_s"] = round(time.monotonic() - attempt_start, 3)
        log(
            f"  [{model['id']}] {role} download OK in "
            f"{attempt['download_duration_s']}s — {attempt['exact_download_url']}"
        )
    except Exception as exc:
        attempt["success"] = False
        attempt["error"] = str(exc)
        attempt["download_duration_s"] = round(time.monotonic() - attempt_start, 3)
        log(f"  [{model['id']}] {role} download FAILED in {attempt['download_duration_s']}s: {exc}")
        if part_path.exists():
            part_path.unlink()

    return attempt


def remove_source_files(
    model: dict[str, Any],
    cache_dir: Path,
    priority: int,
    role: str,
) -> list[str]:
    """Delete GGUF and partial files for one source; log removals."""
    if _state.keep_cache:
        log(f"  [{model['id']}] keeping {role} GGUF on disk (--keep-cache)")
        return []

    dest_path, part_path = source_paths(cache_dir, model, priority)
    removed: list[str] = []
    for path in (dest_path, part_path):
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            removed.append(str(path))
            log(
                f"  [{model['id']}] removed {role} file: {path.name} "
                f"({human_bytes(size)})"
            )
    if not removed:
        log(f"  [{model['id']}] no {role} files on disk to remove")
    return removed


def _download_url(
    url: str,
    dest: Path,
    expected_size: int,
    base_headers: dict[str, str],
) -> dict[str, Any]:
    existing = dest.stat().st_size if dest.exists() else 0
    if existing > expected_size:
        dest.unlink()
        existing = 0

    headers = dict(base_headers)
    range_header: str | None = None
    if existing > 0:
        range_header = f"bytes={existing}-"
        headers["Range"] = range_header

    req = urllib.request.Request(url, headers=headers)
    meta: dict[str, Any] = {"range_header": range_header, "request_url": url}

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            meta["http_status"] = getattr(resp, "status", resp.getcode())
            meta["final_url"] = resp.geturl()
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                try:
                    meta["content_length"] = int(cl)
                except ValueError:
                    meta["content_length"] = cl

            status = meta["http_status"]
            mode = "ab" if existing > 0 and status == 206 else "wb"
            if mode == "wb" and dest.exists():
                dest.unlink()
                existing = 0

            downloaded = existing
            last_report = time.monotonic()
            with dest.open(mode) as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5.0:
                        pct = f"{100 * downloaded / expected_size:.1f}%" if expected_size else "?"
                        log(f"    … {downloaded / GIB:.2f} GB / {expected_size / GIB:.2f} GB ({pct})")
                        last_report = now
    except urllib.error.HTTPError as exc:
        if existing > 0 and exc.code in (416, 404):
            if dest.exists():
                dest.unlink()
            return _download_url(url, dest, expected_size, base_headers)
        meta["http_status"] = exc.code
        raise

    return meta


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def can_connect(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def probe_health(port: int) -> str:
    url = f"http://127.0.0.1:{port}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200:
                return "ready"
            if resp.status == 503 and "Loading model" in body:
                return "loading"
            return "unreachable"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 503 and "Loading model" in body:
            return "loading"
        return "unreachable"
    except (urllib.error.URLError, TimeoutError, OSError):
        return "unreachable"


def wait_for_listener(port: int, proc: subprocess.Popen[Any], deadline_s: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if proc.poll() is not None:
            log_tail = ""
            server_log = getattr(proc, "_validator_log_path", None)
            if server_log and Path(server_log).is_file():
                log_tail = Path(server_log).read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} before listening: {log_tail}"
            )
        if can_connect(port):
            return time.monotonic() - start
        time.sleep(0.15)
    raise RuntimeError(f"timed out waiting for llama-server to listen on 127.0.0.1:{port}")


def wait_for_inference(port: int, deadline_s: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if probe_health(port) == "ready":
            return time.monotonic() - start
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for inference ready on 127.0.0.1:{port}")


def start_llama_server(
    binary: Path,
    model_path: Path,
    port: int,
    extra_args: list[str],
    ctx_size: int,
    log_path: Path | None = None,
) -> tuple[subprocess.Popen[Any], list[str]]:
    argv = [
        str(binary),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--n-gpu-layers",
        "99",
        *extra_args,
    ]
    log(
        f"  starting llama-server port={port} ctx={ctx_size} "
        f"flags={' '.join(extra_args) if extra_args else '(none)'}"
    )
    log_file = None
    stderr_target: Any = subprocess.DEVNULL
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        stderr_target = log_file
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=stderr_target,
    )
    proc._validator_log_file = log_file  # type: ignore[attr-defined]
    _state.server_proc = proc
    return proc, argv


def read_server_log(proc: subprocess.Popen[Any] | None) -> str:
    if proc is None:
        return ""
    path = getattr(proc, "_validator_log_path", None)
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8", errors="replace")
    return ""


def stop_server(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log_file = getattr(proc, "_validator_log_file", None)
    if log_file is not None:
        log_file.close()
    _state.server_proc = None


def chat_completion(
    port: int,
    user_text: str,
    chat_template_kwargs: dict[str, bool] | None,
) -> tuple[str, dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": "local",
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
    )
    chat_start = time.monotonic()
    with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S) as resp:
        raw_body = resp.read().decode("utf-8")
        http_status = resp.status
    duration_s = time.monotonic() - chat_start

    body = json.loads(raw_body)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices in response: {raw_body[:500]}")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"empty assistant content: {raw_body[:500]}")

    usage = body.get("usage") or {}
    chat_report = {
        "request": {
            "method": "POST",
            "url": url,
            "headers": {"Content-Type": "application/json", "User-Agent": USER_AGENT},
            "body": payload,
        },
        "response": {
            "http_status": http_status,
            "body": body,
        },
        "assistant_reply": content,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "duration_s": round(duration_s, 3),
    }
    return content, chat_report, duration_s


def fetch_server_models(port: int) -> dict[str, Any]:
    """GET /v1/models — OpenAI-compatible list of models loaded in llama-server."""
    url = f"http://127.0.0.1:{port}/v1/models"
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return {
            "url": url,
            "http_status": resp.status,
            "body": json.loads(raw),
            "raw": raw,
        }


def log_server_models(port: int, label: str) -> dict[str, Any]:
    """Fetch and print GET /v1/models for console confirmation."""
    try:
        result = fetch_server_models(port)
        log(f"  GET /v1/models ({label}) — HTTP {result['http_status']}")
        log(json.dumps(result["body"], indent=2, ensure_ascii=False))
        return result
    except Exception as exc:
        log(f"  GET /v1/models ({label}) — failed: {exc}")
        return {"url": f"http://127.0.0.1:{port}/v1/models", "error": str(exc)}


def run_source_inference(
    model: dict[str, Any],
    model_path: Path,
    source_role: str,
    priority: int,
    cache_dir: Path,
    llama_server_bin: Path,
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Start llama-server on one GGUF, chat once, stop server."""
    model_id = model["id"]
    proc: subprocess.Popen[Any] | None = None
    result: dict[str, Any] = {
        "passed": False,
        "error": None,
        "server": None,
        "chat": None,
        "models_api": {},
        "checks": {
            "health_ready": None,
            "chat_non_empty": None,
            "echo_warning": None,
        },
        "timing": {
            "listener_wait_s": None,
            "inference_ready_s": None,
            "chat_s": None,
        },
    }

    try:
        port = pick_free_port()
        extra_args = resolve_server_args(model, profiles)
        ctx_metrics = compute_ctx_budget(model_path, model_id)
        ctx_size = ctx_metrics["ctx_size"]
        server_log_path = model_dir_for(cache_dir, model) / f"llama-server-p{priority}.log"

        log(f"  [{model_id}] starting llama-server for {source_role} …")
        proc, argv = start_llama_server(
            llama_server_bin, model_path, port, extra_args, ctx_size, log_path=server_log_path
        )
        proc._validator_log_path = server_log_path  # type: ignore[attr-defined]

        listener_s = wait_for_listener(port, proc, LISTENER_TIMEOUT_S)
        log(f"  [{model_id}] listener up on :{port}, waiting for {source_role} model load …")
        inference_s = wait_for_inference(port, INFERENCE_TIMEOUT_S)
        result["checks"]["health_ready"] = True

        result["models_api"]["after_load"] = log_server_models(port, f"{source_role} loaded")

        kwargs = resolve_chat_template_kwargs(model, profiles)
        reply, chat_report, chat_s = chat_completion(port, USER_PROMPT, kwargs)
        echo = looks_like_request_echo(USER_PROMPT, reply)
        result["checks"]["chat_non_empty"] = bool(reply.strip())
        result["checks"]["echo_warning"] = echo
        if echo:
            log(f"  [{model_id}] WARNING [{source_role}]: response may be a request echo")

        server_log = read_server_log(proc)
        result["server"] = {
            "binary": str(llama_server_bin),
            "port": port,
            "command": argv,
            "llama_server_args": extra_args,
            "chat_template_kwargs": kwargs,
            "context_budget": ctx_metrics,
            "log_path": str(server_log_path),
            "log_excerpt": server_log[-8000:] if server_log else "",
        }
        result["chat"] = chat_report
        result["timing"]["listener_wait_s"] = round(listener_s, 3)
        result["timing"]["inference_ready_s"] = round(inference_s, 3)
        result["timing"]["chat_s"] = round(chat_s, 3)

        preview = reply[:200].replace("\n", " ")
        log(f"  [{model_id}] {source_role} chat OK — preview: {preview!r}")
        usage = chat_report.get("usage") or {}
        log(
            f"  [{model_id}] {source_role} tokens: prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}"
        )
        result["passed"] = True
    except Exception as exc:
        result["error"] = str(exc)
        log(f"  [{model_id}] {source_role} inference FAILED: {exc}")
        if proc is not None:
            result["server"] = result.get("server") or {
                "log_excerpt": read_server_log(proc)[-8000:],
            }
    finally:
        stop_server(proc)
        log(f"  [{model_id}] stopped llama-server ({source_role})")

    return result


def looks_like_request_echo(request: str, response: str) -> bool:
    req = request.strip()
    resp = response.strip()
    if not req or not resp:
        return False
    if req.lower() == resp.lower():
        return True
    if resp.startswith(req) or req.startswith(resp):
        return True
    if resp.startswith("{") and resp.endswith("}") and ('"name"' in resp or '"parameters"' in resp):
        if "title:" in req.lower():
            return True
    if "title:" in req.lower() and "image_title" in resp.lower():
        return True
    if "analyze_journal_line" in resp or "describe_image" in resp:
        return True
    return False


def cleanup_model(model: dict[str, Any], cache_dir: Path) -> None:
    """Remove leftover cache files (logs, partials) for a model directory."""
    if _state.keep_cache:
        return
    model_dir = model_dir_for(cache_dir, model)
    if not model_dir.is_dir():
        return
    for path in model_dir.iterdir():
        if path.is_file():
            log(f"  [{model['id']}] removed cache file: {path.name}")
            path.unlink()
    if not any(model_dir.iterdir()):
        model_dir.rmdir()
        log(f"  [{model['id']}] removed empty cache directory")


def test_model(
    model: dict[str, Any],
    manifest: dict[str, Any],
    cache_dir: Path,
    llama_server_bin: Path,
) -> dict[str, Any]:
    model_id = model["id"]
    run_start = time.monotonic()
    profiles = profile_map(manifest)

    report: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model.get("name"),
        "family": model.get("family"),
        "passed": False,
        "failed_stage": None,
        "error": None,
        "user_prompt": USER_PROMPT,
        "checks": {
            "all_sources_download_ok": None,
            "all_sources_inference_ok": None,
            "all_sources_sha256_ok": None,
            "all_sources_size_ok": None,
            "primary_download_ok": None,
            "fallback_download_ok": None,
            "primary_inference_ok": None,
            "fallback_inference_ok": None,
            "primary_sha256_ok": None,
            "primary_size_ok": None,
            "health_ready": None,
            "chat_non_empty": None,
            "echo_warning": None,
        },
        "source_runs": [],
        "timing": {
            "download_s": None,
            "listener_wait_s": None,
            "inference_ready_s": None,
            "chat_s": None,
            "total_s": None,
        },
    }

    sources = sorted(model.get("sources", []), key=lambda s: s.get("priority", 99))
    if not sources:
        report["error"] = "no download sources in manifest"
        report["failed_stage"] = "download"
        report["timing"]["total_s"] = round(time.monotonic() - run_start, 3)
        return report

    source_runs: list[dict[str, Any]] = []
    total_download_s = 0.0
    total_listener_s = 0.0
    total_inference_ready_s = 0.0
    total_chat_s = 0.0

    try:
        log(f"\n=== {model_id} ({model.get('name', '')}) ===")

        for source in sources:
            priority = source.get("priority", 99)
            role = "primary" if priority == 1 else "fallback"
            log(f"\n--- {model_id}: {role} (priority {priority}) ---")

            attempt = download_source(model, source, cache_dir)
            total_download_s += attempt.get("download_duration_s") or 0.0

            run_record: dict[str, Any] = {
                "source_role": role,
                "priority": priority,
                "source_name": source.get("name"),
                "download": attempt,
                "inference": None,
                "removed_files": [],
                "passed": False,
            }

            if not attempt["success"]:
                source_runs.append(run_record)
                if priority == 1:
                    raise RuntimeError(f"primary download failed: {attempt.get('error')}")
                log(f"  [{model_id}] skipping {role} inference (download failed)")
                continue

            model_path = Path(attempt["local_path"])
            inference = run_source_inference(
                model,
                model_path,
                role,
                priority,
                cache_dir,
                llama_server_bin,
                profiles,
            )
            run_record["inference"] = inference
            inf_timing = inference.get("timing") or {}
            total_listener_s += inf_timing.get("listener_wait_s") or 0.0
            total_inference_ready_s += inf_timing.get("inference_ready_s") or 0.0
            total_chat_s += inf_timing.get("chat_s") or 0.0

            run_record["passed"] = bool(inference.get("passed"))
            run_record["removed_files"] = remove_source_files(model, cache_dir, priority, role)
            source_runs.append(run_record)

            if not inference.get("passed"):
                if priority == 1:
                    raise RuntimeError(inference.get("error") or f"{role} inference failed")

        report["source_runs"] = source_runs
        report["timing"]["download_s"] = round(total_download_s, 3)
        report["timing"]["listener_wait_s"] = round(total_listener_s, 3)
        report["timing"]["inference_ready_s"] = round(total_inference_ready_s, 3)
        report["timing"]["chat_s"] = round(total_chat_s, 3)

        attempts = [r["download"] for r in source_runs]
        successful_dl = [a for a in attempts if a.get("success")]
        successful_inf = [r for r in source_runs if (r.get("inference") or {}).get("passed")]

        report["checks"]["all_sources_download_ok"] = all(a["success"] for a in attempts)
        report["checks"]["all_sources_inference_ok"] = (
            len(successful_inf) == len([r for r in source_runs if r["download"].get("success")])
            if source_runs
            else None
        )
        report["checks"]["all_sources_sha256_ok"] = (
            all((a.get("integrity") or {}).get("sha256_match") for a in successful_dl)
            if successful_dl
            else None
        )
        report["checks"]["all_sources_size_ok"] = (
            all((a.get("integrity") or {}).get("size_match") for a in successful_dl)
            if successful_dl
            else None
        )

        primary_run = next((r for r in source_runs if r.get("priority") == 1), None)
        fallback_runs = [r for r in source_runs if r.get("priority") != 1]

        if primary_run:
            pdl = primary_run.get("download") or {}
            pinf = primary_run.get("inference") or {}
            report["checks"]["primary_download_ok"] = pdl.get("success")
            report["checks"]["primary_inference_ok"] = pinf.get("passed")
            report["checks"]["primary_sha256_ok"] = (pdl.get("integrity") or {}).get("sha256_match")
            report["checks"]["primary_size_ok"] = (pdl.get("integrity") or {}).get("size_match")
            report["checks"]["health_ready"] = (pinf.get("checks") or {}).get("health_ready")
            report["checks"]["chat_non_empty"] = (pinf.get("checks") or {}).get("chat_non_empty")
            report["checks"]["echo_warning"] = (pinf.get("checks") or {}).get("echo_warning")

        report["checks"]["fallback_download_ok"] = (
            all((r.get("download") or {}).get("success") for r in fallback_runs)
            if fallback_runs
            else None
        )
        report["checks"]["fallback_inference_ok"] = (
            all((r.get("inference") or {}).get("passed") for r in fallback_runs if (r.get("download") or {}).get("success"))
            if fallback_runs
            else None
        )

        report["passed"] = len(source_runs) == len(sources) and all(
            r.get("passed") for r in source_runs
        )
        if report["passed"]:
            log(f"  [{model_id}] PASS — {len(source_runs)} source run(s) completed")
    except Exception as exc:
        log(f"  [{model_id}] FAIL — {exc}")
        report["passed"] = False
        report["error"] = str(exc)
        report["source_runs"] = source_runs
        if not any(r.get("download", {}).get("success") for r in source_runs):
            report["failed_stage"] = "download"
        elif not any((r.get("inference") or {}).get("passed") for r in source_runs):
            report["failed_stage"] = "server"
        else:
            report["failed_stage"] = "chat"
    finally:
        cleanup_model(model, cache_dir)

    report["timing"]["total_s"] = round(time.monotonic() - run_start, 3)
    return report


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def yn(val: bool | None) -> str:
    if val is None:
        return "—"
    return "yes" if val else "no"


def build_summary_tables(model_reports: list[dict[str, Any]]) -> dict[str, Any]:
    download_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []

    for model in model_reports:
        model_id = model["model_id"]
        checks = model.get("checks") or {}
        timing = model.get("timing") or {}

        for run in model.get("source_runs") or []:
            attempt = run.get("download") or {}
            inference = run.get("inference") or {}
            chat = inference.get("chat") or {}
            integrity = attempt.get("integrity") or {}
            inf_checks = inference.get("checks") or {}
            inf_timing = inference.get("timing") or {}

            download_rows.append(
                {
                    "model_id": model_id,
                    "source_role": run.get("source_role"),
                    "source_name": run.get("source_name"),
                    "priority": run.get("priority"),
                    "manifest_url": attempt.get("manifest_url"),
                    "exact_download_url": attempt.get("exact_download_url"),
                    "download_duration_s": attempt.get("download_duration_s"),
                    "size_match": integrity.get("size_match"),
                    "sha256_match": integrity.get("sha256_match"),
                    "download_ok": attempt.get("success"),
                    "inference_ok": inference.get("passed"),
                    "http_status": attempt.get("http_status"),
                    "error": attempt.get("error") or inference.get("error"),
                }
            )

            usage = chat.get("usage") or {}
            inference_rows.append(
                {
                    "model_id": model_id,
                    "source_role": run.get("source_role"),
                    "priority": run.get("priority"),
                    "run_pass": run.get("passed"),
                    "download_ok": attempt.get("success"),
                    "inference_ok": inference.get("passed"),
                    "exact_download_url": attempt.get("exact_download_url"),
                    "chat_ok": inf_checks.get("chat_non_empty"),
                    "echo_warning": inf_checks.get("echo_warning"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "listener_wait_s": inf_timing.get("listener_wait_s"),
                    "inference_ready_s": inf_timing.get("inference_ready_s"),
                    "chat_s": inf_timing.get("chat_s"),
                    "error": attempt.get("error") or inference.get("error"),
                }
            )

        inference_rows.append(
            {
                "model_id": model_id,
                "source_role": "TOTAL",
                "priority": None,
                "run_pass": model.get("passed"),
                "download_ok": checks.get("all_sources_download_ok"),
                "inference_ok": checks.get("all_sources_inference_ok"),
                "exact_download_url": None,
                "chat_ok": checks.get("chat_non_empty"),
                "echo_warning": checks.get("echo_warning"),
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "listener_wait_s": timing.get("listener_wait_s"),
                "inference_ready_s": timing.get("inference_ready_s"),
                "chat_s": timing.get("chat_s"),
                "error": model.get("error"),
            }
        )

    return {"download_sources": download_rows, "inference_summary": inference_rows}


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    summary = report["summary"]
    lines.append("# Model validation report")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Manifest:** `{report['manifest']}`")
    lines.append(f"- **llama-server:** `{report['llama_server_bin']}`")
    lines.append(f"- **Models tested:** {summary['total']}")
    lines.append(f"- **Passed:** {summary['passed']}")
    lines.append(f"- **Failed:** {summary['failed']}")
    lines.append(f"- **Total duration:** {summary['total_duration_s']:.1f}s")
    lines.append("")

    for model in report["models"]:
        status = "PASS" if model["passed"] else "FAIL"
        lines.append(f"## {model['model_id']} — {status}")
        lines.append("")
        lines.append(f"**Name:** {model.get('model_name', '')}")
        if model.get("error"):
            lines.append(f"**Error:** {model['error']}")
            lines.append(f"**Failed stage:** {model.get('failed_stage')}")
        lines.append("")

        checks = model.get("checks") or {}
        lines.append("### Checks")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("|-------|--------|")
        for key, val in checks.items():
            lines.append(f"| {key} | {yn(val) if isinstance(val, bool) or val is None else val} |")
        lines.append("")

        dl = model.get("download")
        source_runs = model.get("source_runs") or []
        if source_runs:
            lines.append("### Source runs (download → infer → remove per source)")
            lines.append("")
            for run in source_runs:
                role = run.get("source_role", "?")
                lines.append(f"#### {role.upper()}: {run.get('source_name')}")
                lines.append(f"- **Run passed:** {yn(run.get('passed'))}")
                if run.get("removed_files"):
                    lines.append(f"- **Removed files:** {', '.join(f'`{p}`' for p in run['removed_files'])}")
                attempt = run.get("download") or {}
                lines.append(f"- **Manifest URL:** `{attempt.get('manifest_url')}`")
                lines.append(f"- **Exact download URL:** `{attempt.get('exact_download_url')}`")
                lines.append(f"- **Download duration:** {attempt.get('download_duration_s')}s")
                lines.append(f"- **Download success:** {yn(attempt.get('success'))}")
                if attempt.get("error"):
                    lines.append(f"- **Download error:** {attempt['error']}")
                att_int = attempt.get("integrity") or {}
                if att_int:
                    lines.append("")
                    lines.append("| | Manifest | Actual | Match |")
                    lines.append("|---|----------|--------|-------|")
                    lines.append(
                        f"| Size | {att_int.get('manifest_size_bytes')} | "
                        f"{att_int.get('actual_size_bytes')} | {yn(att_int.get('size_match'))} |"
                    )
                    lines.append(
                        f"| SHA-256 | `{att_int.get('manifest_sha256')}` | "
                        f"`{att_int.get('actual_sha256')}` | {yn(att_int.get('sha256_match'))} |"
                    )
                inference = run.get("inference") or {}
                if inference:
                    srv = inference.get("server") or {}
                    chat = inference.get("chat") or {}
                    models_api = inference.get("models_api") or {}
                    lines.append("")
                    lines.append(f"- **Inference passed:** {yn(inference.get('passed'))}")
                    if inference.get("error"):
                        lines.append(f"- **Inference error:** {inference['error']}")
                    if srv:
                        lines.append(f"- **llama-server port:** {srv.get('port')}")
                        lines.append(f"- **Context size:** {srv.get('context_budget', {}).get('ctx_size')}")
                        lines.append(f"- **Command:** `{' '.join(srv.get('command', []))}`")
                    after_load = models_api.get("after_load") or {}
                    if after_load.get("body"):
                        lines.append("- **GET /v1/models (after load):**")
                        lines.append("```json")
                        lines.append(json.dumps(after_load["body"], indent=2, ensure_ascii=False))
                        lines.append("```")
                    if chat:
                        usage = chat.get("usage") or {}
                        lines.append(
                            f"- **Chat tokens:** prompt={usage.get('prompt_tokens')} "
                            f"completion={usage.get('completion_tokens')} "
                            f"total={usage.get('total_tokens')}"
                        )
                        lines.append("")
                        lines.append("**Assistant reply:**")
                        lines.append("")
                        lines.append("```")
                        lines.append(chat.get("assistant_reply", ""))
                        lines.append("```")
                lines.append("")
        elif dl:
            lines.append("### Download (legacy report shape)")
            lines.append("")
            lines.append(json.dumps(dl, indent=2, ensure_ascii=False))
            lines.append("")

        srv = model.get("server")
        if srv and not source_runs:
            lines.append("### llama-server")
            lines.append("")
            lines.append(f"- Port: {srv.get('port')}")
            lines.append(f"- Context size: {srv.get('context_budget', {}).get('ctx_size')}")
            lines.append(f"- Command: `{' '.join(srv.get('command', []))}`")
            cb = srv.get("context_budget") or {}
            if cb:
                lines.append(
                    f"- Context budget: avail={cb.get('available_gb')}GB "
                    f"model={cb.get('model_gb')}GB overhead={cb.get('overhead_gb')}GB "
                    f"→ ctx={cb.get('ctx_size')}"
                )
            lines.append("")

        chat = model.get("chat")
        if chat and not source_runs:
            lines.append("### Chat request / response")
            lines.append("")
            req = chat.get("request") or {}
            lines.append(f"- **URL:** `{req.get('url')}`")
            lines.append(f"- **Duration:** {chat.get('duration_s')}s")
            usage = chat.get("usage") or {}
            lines.append(
                f"- **Tokens:** prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
            lines.append("")
            lines.append("**Request body:**")
            lines.append("```json")
            lines.append(json.dumps(req.get("body"), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
            lines.append("**Assistant reply:**")
            lines.append("")
            lines.append("```")
            lines.append(chat.get("assistant_reply", ""))
            lines.append("```")
            lines.append("")

        timing = model.get("timing") or {}
        lines.append("### Timing")
        lines.append("")
        lines.append("| Phase | Seconds |")
        lines.append("|-------|---------|")
        for phase in ("download_s", "listener_wait_s", "inference_ready_s", "chat_s", "total_s"):
            label = phase.replace("_s", "").replace("_", " ")
            lines.append(f"| {label} | {timing.get(phase)} |")
        lines.append("")

    tables = report.get("summary_tables") or {}
    dl_rows = tables.get("download_sources") or []
    inf_rows = tables.get("inference_summary") or []

    lines.append("---")
    lines.append("")
    lines.append("## Summary tables")
    lines.append("")
    lines.append("### Download sources")
    lines.append("")
    lines.append(
        "| Model | Role | Priority | Exact download URL | Download (s) | "
        "Size OK | SHA256 OK | DL OK | Infer OK | HTTP |"
    )
    lines.append(
        "|-------|------|----------|-------------------|--------------|"
        "---------|-----------|-------|----------|------|"
    )
    for row in dl_rows:
        url = row.get("exact_download_url") or row.get("manifest_url") or ""
        lines.append(
            f"| {row.get('model_id')} | {row.get('source_role')} | {row.get('priority')} | "
            f"`{url}` | {row.get('download_duration_s')} | {yn(row.get('size_match'))} | "
            f"{yn(row.get('sha256_match'))} | {yn(row.get('download_ok'))} | "
            f"{yn(row.get('inference_ok'))} | {row.get('http_status')} |"
        )
    lines.append("")

    lines.append("### Inference per source")
    lines.append("")
    lines.append(
        "| Model | Role | Pass | DL OK | Infer OK | "
        "Prompt tok | Completion tok | Total tok | Chat (s) | Error |"
    )
    lines.append(
        "|-------|------|------|-------|----------|"
        "-----------|----------------|-----------|----------|-------|"
    )
    for row in inf_rows:
        if row.get("source_role") == "TOTAL":
            continue
        err = (row.get("error") or "").replace("|", "\\|")[:60]
        lines.append(
            f"| {row.get('model_id')} | {row.get('source_role')} | {yn(row.get('run_pass'))} | "
            f"{yn(row.get('download_ok'))} | {yn(row.get('inference_ok'))} | "
            f"{row.get('prompt_tokens')} | {row.get('completion_tokens')} | "
            f"{row.get('total_tokens')} | {row.get('chat_s')} | {err or '—'} |"
        )
    lines.append("")

    lines.append("### Model totals")
    lines.append("")
    lines.append(
        "| Model | Pass | All DL OK | All infer OK | Listener (s) | Infer ready (s) | Chat (s) | Error |"
    )
    lines.append(
        "|-------|------|-----------|--------------|--------------|-----------------|----------|-------|"
    )
    for row in inf_rows:
        if row.get("source_role") != "TOTAL":
            continue
        err = (row.get("error") or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {row.get('model_id')} | {yn(row.get('run_pass'))} | "
            f"{yn(row.get('download_ok'))} | {yn(row.get('inference_ok'))} | "
            f"{row.get('listener_wait_s')} | {row.get('inference_ready_s')} | "
            f"{row.get('chat_s')} | {err or '—'} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signal_handler(_signum: int, _frame: Any) -> None:
    log("\nInterrupted — stopping llama-server …")
    stop_server(_state.server_proc)
    sys.exit(130)


def ensure_llama_server(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(
            f"llama-server not found at {path}\n"
            "Run tools/validate_models.sh to fetch it from diarySwift, or pass --llama-server-bin."
        )
    if not os.access(path, os.X_OK):
        raise SystemExit(f"llama-server is not executable: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and smoke-test every model in models-manifest.json via llama-server."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--llama-server-bin", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--models", type=str, default="", help="Comma-separated model ids to test")
    parser.add_argument("--keep-cache", action="store_true", help="Keep GGUFs after each test")
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="validate-models",
        help="Filename prefix for reports (default: validate-models)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _state.keep_cache = args.keep_cache
    _state.verbose = args.verbose

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if sys.platform != "darwin":
        log("WARNING: intended for macOS Apple Silicon; proceeding anyway.")

    ensure_llama_server(args.llama_server_bin)
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
        raise SystemExit("no models to test")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.monotonic()
    log(f"Testing {len(models)} model(s) from {args.manifest}")
    log(f"llama-server: {args.llama_server_bin}")
    log(f"cache: {args.cache_dir} (keep_cache={args.keep_cache})")

    model_reports: list[dict[str, Any]] = []
    for model in models:
        model_reports.append(
            test_model(model, manifest, args.cache_dir, args.llama_server_bin)
        )

    passed = sum(1 for r in model_reports if r["passed"])
    failed = len(model_reports) - passed
    total_duration = time.monotonic() - run_start

    log(f"\n{'=' * 60}")
    log(f"Results: {passed} passed, {failed} failed, {len(model_reports)} total")
    for r in model_reports:
        status = "PASS" if r["passed"] else "FAIL"
        extra = f" — {r['error']}" if r.get("error") else ""
        echo = " [echo warning]" if r.get("checks", {}).get("echo_warning") else ""
        log(f"  {status}  {r['model_id']} ({r['timing']['total_s']}s){extra}{echo}")

    ts = report_timestamp()
    json_path = args.report_dir / f"{args.report_prefix}-{ts}.json"
    md_path = args.report_dir / f"{args.report_prefix}-{ts}.md"

    full_report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "manifest": str(args.manifest),
        "llama_server_bin": str(args.llama_server_bin),
        "cache_dir": str(args.cache_dir),
        "keep_cache": args.keep_cache,
        "user_prompt": USER_PROMPT,
        "summary": {
            "total": len(model_reports),
            "passed": passed,
            "failed": failed,
            "total_duration_s": round(total_duration, 3),
        },
        "summary_tables": build_summary_tables(model_reports),
        "models": model_reports,
    }

    write_json_report(json_path, full_report)
    write_markdown_report(md_path, full_report)
    log(f"\nReport JSON: {json_path}")
    log(f"Report Markdown: {md_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
