# SPDX-License-Identifier: Apache-2.0

"""Run multiple ViPPET pipeline variant combinations in batch.

This script loads a YAML config specifying multiple runs, where each run mixes
variants of the same pipeline. For each run, it launches a single performance test
containing all specified variants, polls job status until completion/failure,
and writes CSV rows with aggregated metrics.

Each configured variant can optionally set `enabled: false` to skip benchmarking
without removing it from the config file. Runs where all variants are disabled
are skipped and do not stop batch execution.
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

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


TERMINAL_STATES = {"COMPLETED", "FAILED"}

CSV_FIELDNAMES = [
    "timestamp_utc",
    "pipeline_id",
    "total_variants",
    "total_streams",
    "variants_spec",
    "total_fps",
    "per_stream_fps",
    "elapsed_time_ms",
    "job_state",
]


class ApiError(RuntimeError):
    """Raised when an API request fails."""


@dataclass(frozen=True)
class VariantSpec:
    """Single variant in a run."""

    variant_id: str
    streams: int
    detection_model: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class PipelineRun:
    """A single run: one pipeline with multiple variants."""

    pipeline_id: str
    variants: list[VariantSpec]


@dataclass(frozen=True)
class VariantTarget:
    """Pipeline and variant identifiers with metadata."""

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
    """Minimal ViPPET API client for this benchmark."""

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
        description="Run mixed variant benchmarks for pipeline(s) in batch."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file with pipeline runs",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860/api/v1",
        help="ViPPET API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=0.0,
        help="Execution max_runtime in seconds (0 means run to EOS; default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between job status polls (default: %(default)s)",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for backend readiness (default: %(default)s)",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=0.0,
        help="Job timeout in seconds (0 disables timeout; default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        default="batched_performance_report.csv",
        help="CSV output file path (default: %(default)s)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List runs from config, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and check backend availability without running benchmarks",
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


def load_config(path: Path) -> dict[str, Any]:
    """Load and parse YAML config file."""
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML dict")
        return config
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {exc}") from exc


def validate_and_parse_runs(config: dict[str, Any]) -> list[PipelineRun]:
    """Extract and validate pipeline runs from config."""
    runs_raw = config.get("runs")
    if not isinstance(runs_raw, list):
        raise ValueError("Config must have 'runs' key containing a list of pipeline runs")

    runs: list[PipelineRun] = []
    for index, run_raw in enumerate(runs_raw):
        if not isinstance(run_raw, dict):
            raise ValueError(f"runs[{index}] must be a dict, got {type(run_raw).__name__}")

        pipeline_id = str(run_raw.get("pipeline_id", "")).strip()
        variants_raw = run_raw.get("variants")

        if not pipeline_id:
            raise ValueError(f"runs[{index}] missing required key 'pipeline_id'")
        if not isinstance(variants_raw, list):
            raise ValueError(f"runs[{index}] 'variants' must be a list")
        if not variants_raw:
            raise ValueError(f"runs[{index}] 'variants' list cannot be empty")

        variants: list[VariantSpec] = []
        for var_index, variant_raw in enumerate(variants_raw):
            if not isinstance(variant_raw, dict):
                raise ValueError(
                    f"runs[{index}].variants[{var_index}] must be a dict, "
                    f"got {type(variant_raw).__name__}"
                )

            variant_id = str(variant_raw.get("variant_id", "")).strip()
            streams = variant_raw.get("streams")
            detection_model = str(variant_raw.get("detection_model", "")).strip()
            enabled = variant_raw.get("enabled", True)

            if not variant_id:
                raise ValueError(
                    f"runs[{index}].variants[{var_index}] missing required key 'variant_id'"
                )
            if not isinstance(streams, int) or streams <= 0:
                raise ValueError(
                    f"runs[{index}].variants[{var_index}] 'streams' must be a positive integer, "
                    f"got {streams}"
                )
            if type(enabled) is not bool:
                raise ValueError(
                    f"runs[{index}].variants[{var_index}] 'enabled' must be a boolean, "
                    f"got {enabled!r}"
                )

            variants.append(
                VariantSpec(
                    variant_id=variant_id,
                    streams=streams,
                    detection_model=detection_model,
                    enabled=enabled,
                )
            )

        runs.append(PipelineRun(pipeline_id=pipeline_id, variants=variants))

    if not runs:
        raise ValueError("Config 'runs' list cannot be empty")
    return runs


def extract_pipeline_graph(variant: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of variant pipeline_graph in expected dict shape."""
    graph = variant.get("pipeline_graph")
    if not isinstance(graph, dict):
        raise RuntimeError("Variant is missing pipeline_graph")
    return json.loads(json.dumps(graph))


def get_enabled_variants(run: PipelineRun) -> list[VariantSpec]:
    """Return variants that should be benchmarked for a run."""
    return [variant for variant in run.variants if variant.enabled]


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
        raise ValueError(f"Unknown internal model name '{selector}'. Available: {available}")
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
    raise ValueError(f"Ambiguous model name '{selector}'. Use explicit selectors: {available_forms}")


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


def discover_variant_targets(
    client: VippetApiClient,
    run: PipelineRun,
) -> dict[str, VariantTarget]:
    """Fetch pipeline metadata for a specific pipeline run."""
    pipeline_items = client.get("/pipelines")
    if not isinstance(pipeline_items, list):
        raise RuntimeError("Expected /pipelines to return a list")

    # Find the pipeline
    pipeline_data = None
    for pipeline in pipeline_items:
        if not isinstance(pipeline, dict):
            continue
        if str(pipeline.get("id", "")).lower().strip() == run.pipeline_id.lower():
            pipeline_data = pipeline
            break

    if not pipeline_data:
        available_ids = sorted(
            str(p.get("id", "")) for p in pipeline_items if isinstance(p, dict) and p.get("id")
        )
        raise ValueError(
            f"Pipeline '{run.pipeline_id}' not found. Available: {available_ids}"
        )

    # Build lookup map for variants in this pipeline
    by_variant_id: dict[str, VariantTarget] = {}
    pipeline_id = str(pipeline_data.get("id", "")).lower().strip()
    pipeline_name = str(pipeline_data.get("name", "")).strip()

    pipeline_variants = pipeline_data.get("variants", [])
    if not isinstance(pipeline_variants, list):
        raise RuntimeError(f"Pipeline '{run.pipeline_id}' has no variants")

    for variant in pipeline_variants:
        if not isinstance(variant, dict):
            continue

        variant_id = str(variant.get("id", "")).lower().strip()
        variant_name = str(variant.get("name", "")).strip()
        if not variant_id:
            continue

        by_variant_id[variant_id] = VariantTarget(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            variant_id=variant_id,
            variant_name=variant_name,
            pipeline_graph=extract_pipeline_graph(variant),
        )

    # Validate all variants in the run exist
    for variant_spec in get_enabled_variants(run):
        variant_key = variant_spec.variant_id.lower()
        if variant_key not in by_variant_id:
            available = sorted(by_variant_id.keys())
            raise ValueError(
                f"Variant '{variant_spec.variant_id}' not found in pipeline '{run.pipeline_id}'. "
                f"Available: {available}"
            )

    return by_variant_id


def build_mixed_performance_request(
    run: PipelineRun,
    resolved_models: dict[str, str],
    variant_targets: dict[str, VariantTarget],
    max_runtime: float,
) -> dict[str, Any]:
    """Build a single performance request with multiple variants of the same pipeline."""
    pipeline_performance_specs: list[dict[str, Any]] = []

    for variant_spec in get_enabled_variants(run):
        target = variant_targets[variant_spec.variant_id.lower()]
        detection_model = resolved_models.get(variant_spec.variant_id, "")

        pipeline_source: dict[str, Any]
        if detection_model:
            graph_payload, replaced_count = apply_detection_model_override(
                target.pipeline_graph,
                detection_model,
            )
            if replaced_count == 0:
                print(
                    f"Warning: {run.pipeline_id}/{variant_spec.variant_id} "
                    "has no gvadetect nodes; skipping detection model override"
                )
            pipeline_source = {
                "source": "graph",
                "graph_id": f"{run.pipeline_id}-{variant_spec.variant_id}-model-override",
                "pipeline_graph": graph_payload,
            }
        else:
            pipeline_source = {
                "source": "variant",
                "pipeline_id": run.pipeline_id,
                "variant_id": variant_spec.variant_id,
            }

        pipeline_performance_specs.append(
            {
                "pipeline": pipeline_source,
                "streams": variant_spec.streams,
            }
        )

    return {
        "pipeline_performance_specs": pipeline_performance_specs,
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


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


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


def list_config_runs(runs: list[PipelineRun]) -> None:
    """Print the runs from config."""
    print("Pipeline runs from config")
    for run_index, run in enumerate(runs, start=1):
        enabled_variants = get_enabled_variants(run)
        disabled_count = len(run.variants) - len(enabled_variants)
        total_streams = sum(v.streams for v in enabled_variants)
        print(
            f"  [{run_index}] {run.pipeline_id}: {len(enabled_variants)} enabled variant(s), "
            f"{total_streams} total streams"
            f"{f', {disabled_count} disabled' if disabled_count else ''}"
        )
        for var in enabled_variants:
            model_label = f" (model: {var.detection_model})" if var.detection_model else ""
            print(f"      - {var.variant_id}: {var.streams} streams{model_label}")


def format_variants_spec(run: PipelineRun) -> str:
    """Format variants spec as a summary string."""
    parts = [f"{v.variant_id}:{v.streams}" for v in get_enabled_variants(run)]
    return ";".join(parts)


def validate_runs_against_backend(
    client: VippetApiClient,
    runs: list[PipelineRun],
) -> list[dict[str, VariantTarget] | None]:
    """Verify all configured pipelines and variants before launching any job."""
    validated_targets: list[dict[str, VariantTarget] | None] = []
    for run in runs:
        if not get_enabled_variants(run):
            validated_targets.append(None)
            continue
        validated_targets.append(discover_variant_targets(client, run))
    return validated_targets


def main() -> int:
    args = parse_args()

    if args.poll_interval <= 0:
        print("--poll-interval must be greater than 0", file=sys.stderr)
        return 2

    client = None
    current_job_id: str | None = None

    try:
        # Load and validate config
        config = load_config(Path(args.config))
        runs = validate_and_parse_runs(config)
        print(f"Loaded config with {len(runs)} pipeline run(s)")

        if args.list_runs:
            list_config_runs(runs)
            return 0

        # Connect to API and wait for readiness
        client = VippetApiClient(base_url=args.base_url, timeout=args.request_timeout)

        print(f"Waiting for backend readiness at {args.base_url} ...")
        wait_until_ready(
            client=client,
            timeout_seconds=args.ready_timeout,
            poll_interval=args.poll_interval,
        )
        print("Backend is ready")

        print("Validating configured pipelines and variants ...")
        validated_variant_targets = validate_runs_against_backend(client, runs)
        executable_runs = sum(1 for targets in validated_variant_targets if targets is not None)
        skipped_runs = len(validated_variant_targets) - executable_runs
        print(
            f"Validated {executable_runs} executable pipeline run(s) against backend"
            f"{f'; {skipped_runs} skipped (no enabled variants)' if skipped_runs else ''}"
        )

        if args.dry_run:
            print("Dry-run: config, pipelines, and variants verified. Backend is ready.")
            return 0

        # Initialize output CSV
        output_path = Path(args.output_csv)
        initialize_csv(output_path)

        results: list[dict[str, Any]] = []
        interrupted = False

        def handle_sigint(_sig: int, _frame: Any) -> None:
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, handle_sigint)

        # Process each run
        for run_index, (run, variant_targets) in enumerate(
            zip(runs, validated_variant_targets, strict=True),
            start=1,
        ):
            if interrupted:
                raise KeyboardInterrupt

            try:
                enabled_variants = get_enabled_variants(run)
                if not enabled_variants:
                    print(
                        f"\n[{run_index}/{len(runs)}] Skipping {run.pipeline_id}: "
                        "no enabled variants"
                    )
                    continue
                if variant_targets is None:
                    raise RuntimeError(
                        f"Internal error: validation targets missing for run '{run.pipeline_id}'"
                    )
                print(
                    f"\n[{run_index}/{len(runs)}] Running {run.pipeline_id} "
                    f"with {len(enabled_variants)} enabled variant(s)"
                )

                # Resolve detection models if specified
                resolved_models: dict[str, str] = {}
                for variant_spec in enabled_variants:
                    if variant_spec.detection_model:
                        by_internal_name, by_display_name = discover_detection_models(client)
                        try:
                            resolved_models[variant_spec.variant_id] = resolve_detection_model_name(
                                variant_spec.detection_model,
                                by_internal_name,
                                by_display_name,
                            )
                        except ValueError as exc:
                            print(
                                f"Error resolving detection model for {run.pipeline_id}/{variant_spec.variant_id}: {exc}",
                                file=sys.stderr,
                            )
                            return 2

                # Print run summary
                total_streams = sum(v.streams for v in enabled_variants)
                for variant_spec in enabled_variants:
                    model_label = (
                        f" (model: {resolved_models.get(variant_spec.variant_id, 'default')})"
                        if variant_spec.detection_model
                        else ""
                    )
                    print(f"  - {variant_spec.variant_id}: {variant_spec.streams} streams{model_label}")
                print(f"  Total streams: {total_streams}")

                # Build and submit mixed benchmark job
                print("  Launching performance benchmark job...")
                request_payload = build_mixed_performance_request(
                    run=run,
                    resolved_models=resolved_models,
                    variant_targets=variant_targets,
                    max_runtime=args.max_runtime,
                )

                start_response = client.post("/tests/performance", request_payload)
                if not isinstance(start_response, dict) or "job_id" not in start_response:
                    raise RuntimeError("Unexpected performance test creation response")

                job_id = str(start_response["job_id"])
                current_job_id = job_id
                print(f"  Job started with ID: {job_id}")

                # Poll until completion
                status = poll_job_status(
                    client=client,
                    job_id=job_id,
                    poll_interval=args.poll_interval,
                    job_timeout=args.job_timeout,
                )

                job_state = str(status.get("state", ""))
                elapsed_time_ms = status.get("elapsed_time", "")
                aggregate_total_fps = status.get("total_fps", "")
                aggregate_per_stream_fps = status.get("per_stream_fps", "")

                print(f"  Job completed with state: {job_state}")
                if job_state == "COMPLETED":
                    print(f"  Metrics: {aggregate_total_fps} total_fps, {aggregate_per_stream_fps} per_stream_fps")

                # Record result
                row: dict[str, Any] = {
                    "timestamp_utc": utc_now_iso(),
                    "pipeline_id": run.pipeline_id,
                    "total_variants": len(enabled_variants),
                    "total_streams": total_streams,
                    "variants_spec": format_variants_spec(run),
                    "total_fps": aggregate_total_fps,
                    "per_stream_fps": aggregate_per_stream_fps,
                    "elapsed_time_ms": elapsed_time_ms,
                    "job_state": job_state,
                }

                results.append(row)
                append_csv_row(output_path, row)
                current_job_id = None

            except Exception as exc:
                print(f"Error running {run.pipeline_id}: {exc}", file=sys.stderr)
                if current_job_id:
                    try:
                        client.delete(f"/jobs/tests/performance/{current_job_id}")
                    except ApiError as cleanup_error:
                        print(f"Warning: failed to stop job {current_job_id}: {cleanup_error}", file=sys.stderr)
                current_job_id = None
                return 1

        print(f"\nWrote {len(results)} result row(s) to {output_path}")

        completed = sum(1 for row in results if row.get("job_state") == "COMPLETED")
        failed = len(results) - completed
        print(f"Summary: {completed} completed, {failed} failed")

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user; stopping current job and exiting...", file=sys.stderr)
        if current_job_id and client:
            try:
                client.delete(f"/jobs/tests/performance/{current_job_id}")
            except ApiError as exc:
                print(f"Warning: failed to stop job {current_job_id}: {exc}", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, ApiError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
