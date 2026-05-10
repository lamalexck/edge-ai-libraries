# SPDX-License-Identifier: Apache-2.0

"""Run all ViPPET pipeline variants and export throughput metrics to CSV.

This script enumerates all pipelines and variants via ViPPET REST APIs,
launches a performance test for each variant with a fixed number of streams,
polls job status until completion/failure, and writes one CSV row per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from plot_variant_throughput import build_platform_subtitle, collect_chart_data, render_chart


TERMINAL_STATES = {"COMPLETED", "FAILED"}

CSV_FIELDNAMES = [
    "timestamp_utc",
    "pipeline_id",
    "variant_id",
    "detection_model",
    "total_streams",
    "total_fps",
    "per_stream_fps",
    "elapsed_time_ms",
]


class ApiError(RuntimeError):
    """Raised when an API request fails."""


@dataclass(frozen=True)
class VariantTarget:
    """Pipeline and variant identifiers required to launch a test job."""

    pipeline_id: str
    pipeline_name: str
    variant_id: str
    variant_name: str
    pipeline_graph: dict[str, Any]


@dataclass(frozen=True)
class DetectionModel:
    """Detection model metadata returned by /models endpoint."""

    name: str
    display_name: str
    precision: str


class VippetApiClient:
    """Minimal ViPPET API client for this benchmark sweep."""

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, payload)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url=url, method=method, data=data, headers=headers)

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            detail = raw_body
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict) and "message" in parsed:
                    detail = str(parsed["message"])
            except json.JSONDecodeError:
                pass
            raise ApiError(f"{method} {url} failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise ApiError(f"{method} {url} failed: {exc.reason}") from exc

        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError(f"{method} {url} returned non-JSON body") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep all ViPPET pipeline variants and collect throughput metrics."
    )
    parser.add_argument("--base-url", default="http://localhost:7860/api/v1", help="ViPPET API base URL (default: %(default)s)")
    parser.add_argument("--streams", type=int, default=10, help="Number of simultaneous streams per variant run (default: %(default)s)")
    parser.add_argument("--max-runtime", type=float, default=0.0, help="Execution max_runtime in seconds (0 means run to EOS; default: %(default)s)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between job status polls (default: %(default)s)")
    parser.add_argument("--ready-timeout", type=float, default=300.0, help="Seconds to wait for backend readiness (default: %(default)s)")
    parser.add_argument("--job-timeout", type=float, default=0.0, help="Per-job timeout in seconds (0 disables timeout; default: %(default)s)")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="HTTP request timeout in seconds (default: %(default)s)")
    parser.add_argument("--output-csv", default="variant_throughput_sweep.csv", help="CSV output file path (default: %(default)s)")
    parser.add_argument("--output-chart", default="", help="Chart output path (defaults to output CSV path with .png extension)")
    parser.add_argument("--pipelines", default="", help="Comma-separated pipeline IDs to run (e.g. smart-nvr,simple-nvr)")
    parser.add_argument("--variants", default="", help="Comma-separated variant IDs to run (e.g. gpu,npu)")
    parser.add_argument("--pipelines-variants", default="", help="Comma-separated list of pipeline_id/variant_id pairs to benchmark (e.g. smart-nvr/gpu,smart-parking/gpu_npu). If provided, overrides --pipelines and --variants.")
    parser.add_argument("--blacklist", "--exclude", dest="blacklist", default="", help="Comma-separated pipeline_id/variant_id pairs to exclude from benchmark (e.g. smart-nvr/gpu,smart-parking/npu). Applied after inclusion filters.")
    parser.add_argument("--blacklist-variants", "--exclude-variants", dest="blacklist_variants", default="", help="Comma-separated variant IDs to exclude across all pipelines (e.g. cpu,npu). Applied after inclusion filters.")
    parser.add_argument("--list", action="store_true", help="List available pipelines and variants, then exit")
    parser.add_argument("--list-models", "--list-detection-models", dest="list_models", action="store_true", help="List available detection models, then exit")
    parser.add_argument("--no-chart", action="store_true", help="Disable automatic chart generation after writing the CSV")
    parser.add_argument(
        "--detection-model",
        default="",
        help=(
            "Detection model override(s) used for all benchmarks (comma-separated). "
            "Accepts display name, internal model name, or internal@precision "
            "(e.g., yolo11m@INT8,yolo11m@FP16)."
        ),
    )
    parser.add_argument(
        "--detection-model-overrides",
        default="",
        help=(
            "Per-target detection model overrides: pipeline_id/variant_id=model_selector. "
            "Supports same selector formats as --detection-model. "
            "Takes precedence over --detection-model."
        ),
    )
    return parser.parse_args()


def wait_until_ready(client: VippetApiClient, timeout_seconds: float, poll_interval: float) -> None:
    start = time.monotonic()
    while True:
        status = client.get("/status")
        if isinstance(status, dict) and status.get("ready") is True:
            return
        if timeout_seconds > 0 and (time.monotonic() - start) > timeout_seconds:
            raise RuntimeError(f"Backend did not become ready within {timeout_seconds} seconds")
        time.sleep(poll_interval)


def parse_csv_id_list(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def parse_pipelines_variants_pairs(value: str) -> list[tuple[str, str]]:
    if not value:
        return []
    pairs: list[tuple[str, str]] = []
    for pair_str in value.split(","):
        normalized = pair_str.strip().lower()
        if not normalized:
            continue
        if "/" not in normalized:
            raise ValueError(
                f"Invalid pair '{pair_str}'. Expected pipeline_id/variant_id"
            )
        pipeline_id, variant_id = normalized.split("/", 1)
        pipeline_id = pipeline_id.strip()
        variant_id = variant_id.strip()
        if not pipeline_id or not variant_id:
            raise ValueError(
                f"Invalid pair '{pair_str}'. Expected pipeline_id/variant_id"
            )
        pairs.append((pipeline_id, variant_id))
    return pairs


def parse_detection_model_overrides(value: str) -> dict[tuple[str, str], str]:
    overrides: dict[tuple[str, str], str] = {}
    if not value.strip():
        return overrides
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected pipeline_id/variant_id=model_name"
            )
        pair_str, model_name = item.split("=", 1)
        model_name = model_name.strip()
        if not model_name:
            raise ValueError(f"Invalid override '{item}'. Model name cannot be empty")
        pair_values = parse_pipelines_variants_pairs(pair_str)
        if len(pair_values) != 1:
            raise ValueError(
                f"Invalid override '{item}'. Expected exactly one pipeline_id/variant_id pair"
            )
        overrides[pair_values[0]] = model_name
    return overrides


def parse_detection_model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def extract_pipeline_graph(variant: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of variant pipeline_graph in expected dict shape."""
    graph = variant.get("pipeline_graph")
    if not isinstance(graph, dict):
        raise RuntimeError("Variant is missing pipeline_graph")
    return json.loads(json.dumps(graph))


def apply_detection_model_override(
    pipeline_graph: dict[str, Any],
    detection_model_display_name: str,
) -> tuple[dict[str, Any], int]:
    """Update all gvadetect nodes model property in a pipeline graph copy."""
    graph_copy = json.loads(json.dumps(pipeline_graph))
    nodes = graph_copy.get("nodes")
    if not isinstance(nodes, list):
        return graph_copy, 0

    replaced_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type", "")).strip().lower() != "gvadetect":
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        data["model"] = detection_model_display_name
        replaced_count += 1

    return graph_copy, replaced_count


def discover_detection_models(
    client: VippetApiClient,
) -> tuple[dict[str, list[DetectionModel]], dict[str, DetectionModel]]:
    model_items = client.get("/models")
    if not isinstance(model_items, list):
        raise RuntimeError("Expected /models to return a list")

    by_internal_name: dict[str, list[DetectionModel]] = {}
    by_display_name: dict[str, DetectionModel] = {}

    for item in model_items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "")).strip().lower()
        if category != "detection":
            continue

        name = str(item.get("name", "")).strip()
        display_name = str(item.get("display_name", "")).strip()
        precision = str(item.get("precision", "")).strip()
        if not name or not display_name:
            continue

        model = DetectionModel(name=name, display_name=display_name, precision=precision)
        name_lower = name.lower()
        by_internal_name.setdefault(name_lower, []).append(model)
        by_display_name[display_name.lower()] = model

    return by_internal_name, by_display_name


def resolve_detection_model_name(
    raw_model_name: str,
    by_internal_name: dict[str, list[DetectionModel]],
    by_display_name: dict[str, DetectionModel],
) -> str:
    selector = raw_model_name.strip()
    if not selector:
        raise ValueError("Detection model name cannot be empty")

    selector_lower = selector.lower()
    if selector_lower in by_display_name:
        return by_display_name[selector_lower].display_name

    if "@" in selector:
        internal_raw, precision_raw = selector.split("@", 1)
        internal_key = internal_raw.strip().lower()
        precision_key = precision_raw.strip().upper()
        candidates = by_internal_name.get(internal_key)
        if not candidates:
            raise ValueError(f"Unknown internal model name '{internal_raw.strip()}'")
        for model in candidates:
            if model.precision.upper() == precision_key:
                return model.display_name
        available_precisions = sorted({m.precision for m in candidates})
        raise ValueError(
            f"Precision '{precision_raw.strip()}' not available for '{internal_raw.strip()}'. "
            f"Available precisions: {available_precisions}"
        )

    candidates = by_internal_name.get(selector_lower)
    if not candidates:
        available = sorted(by_internal_name.keys())
        raise ValueError(
            f"Unknown internal model name '{selector}'. Available: {available}"
        )
    if len(candidates) == 1:
        return candidates[0].display_name

    int8_candidates = [m for m in candidates if m.precision.upper() == "INT8"]
    if int8_candidates:
        print(
            f"Note: Multiple precisions found for '{selector}'. Selecting INT8 variant. "
            f"Use '{selector}@INT8' or '{selector}@FP16' to be explicit.",
            file=sys.stderr,
        )
        return int8_candidates[0].display_name

    available_forms = sorted({f"{selector_lower}@{m.precision}" for m in candidates})
    raise ValueError(
        f"Ambiguous model name '{selector}'. Use explicit selectors: {available_forms}"
    )


def discover_variants(
    client: VippetApiClient,
    pipelines: str,
    variants: str,
    pairs: list[tuple[str, str]] | None = None,
) -> list[VariantTarget]:
    pipeline_items = client.get("/pipelines")
    if not isinstance(pipeline_items, list):
        raise RuntimeError("Expected /pipelines to return a list")

    # Handle pairs mode (explicit pipeline/variant pairs)
    if pairs:
        available_pairs: dict[str, set[str]] = {}
        for pipeline in pipeline_items:
            if not isinstance(pipeline, dict):
                continue
            pipeline_id = str(pipeline.get("id", "")).lower().strip()
            if pipeline_id:
                available_pairs[pipeline_id] = {
                    str(v.get("id", "")).lower().strip() for v in pipeline.get("variants", []) if v.get("id")
                }
        
        validated_pairs: list[VariantTarget] = []
        for req_pipeline_id, req_variant_id in pairs:
            if req_pipeline_id not in available_pairs:
                print(
                    f"Warning: Pipeline '{req_pipeline_id}' not found. "
                    f"Available: {sorted(available_pairs.keys())}"
                )
                continue
            if req_variant_id not in available_pairs[req_pipeline_id]:
                print(
                    f"Warning: Variant '{req_variant_id}' not found in pipeline '{req_pipeline_id}'. "
                    f"Available: {sorted(available_pairs[req_pipeline_id])}"
                )
                continue
            
            # Find the full pipeline and variant details
            for pipeline in pipeline_items:
                if not isinstance(pipeline, dict):
                    continue
                if str(pipeline.get("id", "")).lower().strip() == req_pipeline_id:
                    pipeline_name = str(pipeline.get("name", ""))
                    for variant in pipeline.get("variants", []):
                        if str(variant.get("id", "")).lower().strip() == req_variant_id:
                            variant_name = str(variant.get("name", ""))
                            validated_pairs.append(
                                VariantTarget(
                                    pipeline_id=req_pipeline_id,
                                    pipeline_name=pipeline_name,
                                    variant_id=req_variant_id,
                                    variant_name=variant_name,
                                    pipeline_graph=extract_pipeline_graph(variant),
                                )
                            )
                            break
                    break
        
        if not validated_pairs:
            print("Error: No valid pipeline/variant pairs after validation")
            return []
        return validated_pairs

    # Cross-product mode (original behavior)
    selected_pipeline_ids = parse_csv_id_list(pipelines)
    selected_variant_ids = parse_csv_id_list(variants)

    discovered: list[VariantTarget] = []
    for pipeline in pipeline_items:
        if not isinstance(pipeline, dict):
            continue

        pipeline_id = str(pipeline.get("id", ""))
        pipeline_name = str(pipeline.get("name", ""))
        if not pipeline_id:
            continue

        if selected_pipeline_ids and pipeline_id.lower().strip() not in selected_pipeline_ids:
            continue

        pipeline_variants = pipeline.get("variants", [])
        if not isinstance(pipeline_variants, list):
            continue

        for variant in pipeline_variants:
            if not isinstance(variant, dict):
                continue

            variant_id = str(variant.get("id", ""))
            variant_name = str(variant.get("name", ""))
            if not variant_id:
                continue

            if selected_variant_ids and variant_id.lower().strip() not in selected_variant_ids:
                continue

            discovered.append(
                VariantTarget(
                    pipeline_id=pipeline_id,
                    pipeline_name=pipeline_name,
                    variant_id=variant_id,
                    variant_name=variant_name,
                    pipeline_graph=extract_pipeline_graph(variant),
                )
            )
    return discovered


def list_available_targets(client: VippetApiClient) -> None:
    pipeline_items = client.get("/pipelines")
    if not isinstance(pipeline_items, list):
        raise RuntimeError("Expected /pipelines to return a list")

    print("Available pipelines and variants")
    for pipeline in pipeline_items:
        if not isinstance(pipeline, dict):
            continue

        pipeline_id = str(pipeline.get("id", "")).strip()
        pipeline_name = str(pipeline.get("name", "")).strip()
        if not pipeline_id:
            continue

        label = pipeline_id if not pipeline_name else f"{pipeline_id} ({pipeline_name})"
        print(f"- {label}")

        pipeline_variants = pipeline.get("variants", [])
        if not isinstance(pipeline_variants, list) or not pipeline_variants:
            print("  variants: none")
            continue

        for variant in pipeline_variants:
            if not isinstance(variant, dict):
                continue

            variant_id = str(variant.get("id", "")).strip()
            variant_name = str(variant.get("name", "")).strip()
            if not variant_id:
                continue

            variant_label = variant_id if not variant_name else f"{variant_id} ({variant_name})"
            print(f"  variant: {variant_label}")


def list_available_detection_models(client: VippetApiClient) -> None:
    """List available detection models returned by backend /models endpoint."""
    model_items = client.get("/models")
    if not isinstance(model_items, list):
        raise RuntimeError("Expected /models to return a list")

    detection_models: list[dict[str, Any]] = []
    for item in model_items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "")).strip().lower()
        if category == "detection":
            detection_models.append(item)

    if not detection_models:
        print("No detection models found.")
        print("Use these values with --detection-model or --detection-model-overrides")
        return

    print("Available detection models")
    for model in detection_models:
        display_name = str(model.get("display_name", "")).strip()
        name = str(model.get("name", "")).strip()
        precision = str(model.get("precision", "")).strip()
        if precision:
            print(f"- {display_name} [name: {name}, precision: {precision}]")
        else:
            print(f"- {display_name} [name: {name}]")

    print("Use these values with --detection-model or --detection-model-overrides")


def build_performance_request(
    target: VariantTarget,
    streams: int,
    max_runtime: float,
    detection_model_override: str,
) -> dict[str, Any]:
    pipeline_source: dict[str, Any]
    if detection_model_override:
        graph_payload, replaced_count = apply_detection_model_override(
            target.pipeline_graph,
            detection_model_override,
        )
        if replaced_count == 0:
            print(
                "Warning: target "
                f"{target.pipeline_id}/{target.variant_id} has no gvadetect nodes; "
                "skipping detection model override"
            )
        pipeline_source = {
            "source": "graph",
            "graph_id": f"{target.pipeline_id}-{target.variant_id}-model-override",
            "pipeline_graph": graph_payload,
        }
    else:
        pipeline_source = {
            "source": "variant",
            "pipeline_id": target.pipeline_id,
            "variant_id": target.variant_id,
        }

    return {
        "pipeline_performance_specs": [
            {
                "pipeline": pipeline_source,
                "streams": streams,
            }
        ],
        "execution_config": {
            "output_mode": "disabled",
            "max_runtime": max_runtime,
            "metadata_mode": "disabled",
            "enable_latency_metrics": False,
        },
    }


def poll_job_status(
    client: VippetApiClient,
    job_id: str,
    poll_interval: float,
    job_timeout: float,
) -> dict[str, Any]:
    start = time.monotonic()
    while True:
        status = client.get(f"/jobs/tests/performance/{job_id}/status")
        if not isinstance(status, dict):
            raise RuntimeError(f"Unexpected status payload for job {job_id}")

        state = str(status.get("state", ""))
        if state in TERMINAL_STATES:
            return status

        if job_timeout > 0 and (time.monotonic() - start) > job_timeout:
            raise TimeoutError(f"Job {job_id} timed out after {job_timeout} seconds")

        time.sleep(poll_interval)


def flatten_stream_ids(status: dict[str, Any]) -> str:
    stream_specs = status.get("streams_per_pipeline")
    if not isinstance(stream_specs, list):
        return ""

    stream_ids: list[str] = []
    for spec in stream_specs:
        if not isinstance(spec, dict):
            continue
        spec_stream_ids = spec.get("streams_ids")
        if not isinstance(spec_stream_ids, list):
            continue
        stream_ids.extend(str(stream_id) for stream_id in spec_stream_ids)
    return ";".join(stream_ids)


def summarize_details(status: dict[str, Any]) -> str:
    details = status.get("details")
    if not isinstance(details, list):
        return ""
    return " | ".join(str(detail) for detail in details)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def initialize_csv(path: Path) -> None:
    """Create or truncate a CSV file and write the header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append one benchmark row to an existing CSV file."""
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)


def resolve_chart_output_path(output_csv: Path, output_chart: str) -> Path:
    if output_chart.strip():
        return Path(output_chart)
    return output_csv.with_suffix(".png")


def write_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    chart_rows = [
        {
            "pipeline_id": str(row.get("pipeline_id", "")),
            "variant_id": str(row.get("variant_id", "")),
            "total_fps": str(row.get("total_fps", "")),
        }
        for row in rows
    ]
    pipelines, variants, series = collect_chart_data(chart_rows)
    render_chart(
        pipelines=pipelines,
        variants=variants,
        series=series,
        output_path=path,
        title="Pipeline Throughput by Variant",
        subtitle=build_platform_subtitle(),
    )


def print_summary(rows: list[dict[str, Any]]) -> None:
    completed = sum(1 for row in rows if row.get("state") == "COMPLETED")
    failed = sum(1 for row in rows if row.get("state") != "COMPLETED")

    best_row: dict[str, Any] | None = None
    best_per_stream_fps = float("-inf")
    for row in rows:
        value = row.get("per_stream_fps")
        if isinstance(value, (int, float)) and value > best_per_stream_fps:
            best_per_stream_fps = float(value)
            best_row = row

    print("\nSweep summary")
    print(f"  total runs: {len(rows)}")
    print(f"  completed:  {completed}")
    print(f"  failed:     {failed}")
    if best_row is not None:
        print(
            "  best per-stream FPS: "
            f"{best_per_stream_fps:.3f} "
            f"({best_row['pipeline_id']} / {best_row['variant_id']})"
        )


def main() -> int:
    args = parse_args()
    if args.streams <= 0:
        print("--streams must be greater than 0", file=sys.stderr)
        return 2
    if args.poll_interval <= 0:
        print("--poll-interval must be greater than 0", file=sys.stderr)
        return 2

    detection_model_overrides: dict[tuple[str, str], str] = {}
    if args.detection_model_overrides:
        try:
            detection_model_overrides = parse_detection_model_overrides(
                args.detection_model_overrides
            )
        except ValueError as exc:
            print(f"Error parsing --detection-model-overrides: {exc}", file=sys.stderr)
            return 2

    client = VippetApiClient(base_url=args.base_url, timeout=args.request_timeout)
    rows: list[dict[str, Any]] = []
    current_job_id: str | None = None
    should_write_outputs = not (args.list or args.list_models)
    output_path = Path(args.output_csv) if should_write_outputs else None
    if output_path is not None:
        initialize_csv(output_path)

    interrupted = False

    def handle_sigint(_sig: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        print(f"Waiting for backend readiness at {args.base_url} ...")
        wait_until_ready(
            client=client,
            timeout_seconds=args.ready_timeout,
            poll_interval=args.poll_interval,
        )

        selected_global_detection_models: list[str] = []
        selected_target_detection_models: dict[tuple[str, str], str] = {}
        if args.detection_model.strip() or detection_model_overrides:
            by_internal_name, by_display_name = discover_detection_models(client)
            if not by_internal_name and not by_display_name:
                print(
                    "No detection models are available from /models; cannot apply detection model override.",
                    file=sys.stderr,
                )
                should_write_outputs = False
                return 2

            if args.detection_model.strip():
                try:
                    raw_global_models = parse_detection_model_list(args.detection_model)
                    selected_global_detection_models = [
                        resolve_detection_model_name(
                            raw_model_name,
                            by_internal_name,
                            by_display_name,
                        )
                        for raw_model_name in raw_global_models
                    ]
                except ValueError as exc:
                    print(f"Error parsing --detection-model: {exc}", file=sys.stderr)
                    should_write_outputs = False
                    return 2
                print(
                    "Using global detection model override(s): "
                    f"{', '.join(selected_global_detection_models)}"
                )

            if detection_model_overrides:
                try:
                    for target_pair, raw_model_name in detection_model_overrides.items():
                        selected_target_detection_models[target_pair] = (
                            resolve_detection_model_name(
                                raw_model_name,
                                by_internal_name,
                                by_display_name,
                            )
                        )
                except ValueError as exc:
                    print(
                        f"Error parsing --detection-model-overrides: {exc}",
                        file=sys.stderr,
                    )
                    should_write_outputs = False
                    return 2
                print(
                    "Using per-target detection model overrides for "
                    f"{len(selected_target_detection_models)} target(s)"
                )

        if args.list:
            list_available_targets(client)

        if args.list_models:
            if args.list:
                print()
            list_available_detection_models(client)

        if args.list or args.list_models:
            return 0

        pairs: list[tuple[str, str]] | None = None
        if args.pipelines_variants:
            try:
                pairs = parse_pipelines_variants_pairs(args.pipelines_variants)
                print(f"Using explicit pipeline/variant pairs: {pairs}")
            except ValueError as exc:
                print(f"Error parsing --pipelines-variants: {exc}", file=sys.stderr)
                should_write_outputs = False
                return 2

        blacklist_pairs: list[tuple[str, str]] = []
        if args.blacklist:
            try:
                blacklist_pairs = parse_pipelines_variants_pairs(args.blacklist)
                print(f"Using blacklist pairs: {blacklist_pairs}")
            except ValueError as exc:
                print(f"Error parsing --blacklist: {exc}", file=sys.stderr)
                should_write_outputs = False
                return 2

        blacklisted_variant_ids = parse_csv_id_list(args.blacklist_variants)
        if blacklisted_variant_ids:
            print(f"Using blacklisted variants: {sorted(blacklisted_variant_ids)}")

        targets = discover_variants(
            client=client,
            pipelines=args.pipelines,
            variants=args.variants,
            pairs=pairs,
        )

        if blacklist_pairs:
            blacklist_set = {(pipeline_id.lower(), variant_id.lower()) for pipeline_id, variant_id in blacklist_pairs}
            before_count = len(targets)
            targets = [
                target
                for target in targets
                if (target.pipeline_id.lower(), target.variant_id.lower()) not in blacklist_set
            ]
            print(
                "Applied blacklist filter: "
                f"removed {before_count - len(targets)} target(s), "
                f"{len(targets)} remaining"
            )

        if blacklisted_variant_ids:
            before_count = len(targets)
            targets = [
                target
                for target in targets
                if target.variant_id.lower().strip() not in blacklisted_variant_ids
            ]
            print(
                "Applied variant blacklist filter: "
                f"removed {before_count - len(targets)} target(s), "
                f"{len(targets)} remaining"
            )

        if not targets:
            print("No matching pipeline variants found.")
            return 1

        if selected_target_detection_models:
            available_target_pairs = {
                (target.pipeline_id.lower(), target.variant_id.lower())
                for target in targets
            }
            unmatched_overrides = sorted(
                pair
                for pair in selected_target_detection_models
                if pair not in available_target_pairs
            )
            for pipeline_id, variant_id in unmatched_overrides:
                print(
                    "Warning: detection override target not part of selected sweep: "
                    f"{pipeline_id}/{variant_id}"
                )

        run_plan: list[tuple[VariantTarget, str]] = []
        for target in targets:
            target_pair = (target.pipeline_id.lower(), target.variant_id.lower())
            if target_pair in selected_target_detection_models:
                run_plan.append((target, selected_target_detection_models[target_pair]))
            elif selected_global_detection_models:
                for selected_model in selected_global_detection_models:
                    run_plan.append((target, selected_model))
            else:
                run_plan.append((target, ""))

        print(
            f"Discovered {len(targets)} pipeline variants expanded to "
            f"{len(run_plan)} benchmark runs"
        )
        for index, (target, selected_model) in enumerate(run_plan, start=1):
            if interrupted:
                raise KeyboardInterrupt

            row: dict[str, Any] = {
                "timestamp_utc": utc_now_iso(),
                "pipeline_id": target.pipeline_id,
                "pipeline_name": target.pipeline_name,
                "variant_id": target.variant_id,
                "variant_name": target.variant_name,
                "detection_model": "",
                "job_id": "",
                "state": "FAILED",
                "total_streams": "",
                "total_fps": "",
                "per_stream_fps": "",
                "elapsed_time_ms": "",
                "details": "",
                "stream_ids": "",
                "error": "",
            }

            try:
                row["detection_model"] = selected_model

                model_label = selected_model if selected_model else "default"
                print(
                    f"[{index}/{len(run_plan)}] Running "
                    f"{target.pipeline_id}/{target.variant_id} with {args.streams} streams "
                    f"(detection model: {model_label})"
                )

                request_payload = build_performance_request(
                    target=target,
                    streams=args.streams,
                    max_runtime=args.max_runtime,
                    detection_model_override=selected_model,
                )
                start_response = client.post("/tests/performance", request_payload)
                if not isinstance(start_response, dict) or "job_id" not in start_response:
                    raise RuntimeError("Unexpected performance test creation response")

                current_job_id = str(start_response["job_id"])
                row["job_id"] = current_job_id

                status = poll_job_status(
                    client=client,
                    job_id=current_job_id,
                    poll_interval=args.poll_interval,
                    job_timeout=args.job_timeout,
                )

                row["state"] = str(status.get("state", ""))
                row["total_streams"] = status.get("total_streams", "")
                row["total_fps"] = status.get("total_fps", "")
                row["per_stream_fps"] = status.get("per_stream_fps", "")
                row["elapsed_time_ms"] = status.get("elapsed_time", "")
                row["details"] = summarize_details(status)
                row["stream_ids"] = flatten_stream_ids(status)
            except TimeoutError as exc:
                row["error"] = str(exc)
                if current_job_id:
                    try:
                        client.delete(f"/jobs/tests/performance/{current_job_id}")
                    except ApiError as cleanup_error:
                        row["error"] = f"{row['error']} | stop error: {cleanup_error}"
            except Exception as exc:
                row["error"] = str(exc)
            finally:
                rows.append(row)
                if output_path is not None:
                    append_csv_row(output_path, row)
                current_job_id = None

        return 0
    except KeyboardInterrupt:
        print("Interrupted by user; stopping current job and saving partial results...")
        if current_job_id:
            try:
                client.delete(f"/jobs/tests/performance/{current_job_id}")
            except ApiError as exc:
                print(f"Warning: failed to stop running job {current_job_id}: {exc}")
        return 130
    finally:
        if should_write_outputs and output_path is not None:
            print(f"Wrote {len(rows)} rows to {output_path}")
            if not args.no_chart:
                chart_path = resolve_chart_output_path(output_path, args.output_chart)
                try:
                    write_chart(chart_path, rows)
                    print(f"Wrote chart to {chart_path}")
                except RuntimeError as exc:
                    print(f"Warning: failed to generate chart: {exc}")
            print_summary(rows)


if __name__ == "__main__":
    raise SystemExit(main())