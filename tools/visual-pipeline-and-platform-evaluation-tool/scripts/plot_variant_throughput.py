# SPDX-License-Identifier: Apache-2.0

"""Render grouped bar charts from ViPPET throughput sweep CSV data with matplotlib."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a grouped bar chart from ViPPET throughput sweep CSV data."
    )
    parser.add_argument(
        "--input-csv",
        default="variant_throughput_sweep.csv",
        help="Input CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-image",
        default="variant_throughput_sweep.png",
        help="Output chart image path (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default="Pipeline Throughput by Variant",
        help="Chart title (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI (default: %(default)s)",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def collect_chart_data(rows: list[dict[str, str]]) -> tuple[list[str], list[str], dict[str, list[float]]]:
    pipelines: list[str] = []
    variants: list[str] = []
    values_by_variant: dict[str, dict[str, float]] = {}

    for row in rows:
        pipeline_id = row.get("pipeline_id", "").strip()
        variant_id = row.get("variant_id", "").strip()
        total_fps_raw = row.get("total_fps", "").strip()
        if not pipeline_id or not variant_id or not total_fps_raw:
            continue

        try:
            total_fps = float(total_fps_raw)
        except ValueError:
            continue

        if pipeline_id not in pipelines:
            pipelines.append(pipeline_id)
        if variant_id not in variants:
            variants.append(variant_id)

        variant_values = values_by_variant.setdefault(variant_id, {})
        variant_values[pipeline_id] = total_fps

    if not pipelines or not variants:
        raise RuntimeError("No valid pipeline_id, variant_id, and total_fps data found in CSV")

    series = {
        variant_id: [values_by_variant.get(variant_id, {}).get(pipeline_id, 0.0) for pipeline_id in pipelines]
        for variant_id in variants
    }
    return pipelines, variants, series


def get_cpu_model_name() -> str:
    """Return CPU model name from host system information."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()

    processor_name = platform.processor().strip()
    if processor_name:
        return processor_name
    return "Unknown CPU"


def get_total_memory_gb() -> int | None:
    """Return total system memory in rounded GiB, or None if unavailable."""
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None

    for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("MemTotal:"):
            continue
        match = re.search(r"(\d+)", line)
        if match is None:
            return None

        total_kb = int(match.group(1))
        if total_kb <= 0:
            return None

        gib = total_kb / (1024 * 1024)
        return round(gib)
    return None


def build_platform_subtitle() -> str:
    """Build subtitle text with CPU model and system memory."""
    cpu_name = get_cpu_model_name()
    total_memory_gb = get_total_memory_gb()
    memory_label = f"{total_memory_gb} GB" if total_memory_gb is not None else "Unknown"
    return f"CPU: {cpu_name} | RAM: {memory_label}"


def render_chart(
    pipelines: list[str],
    variants: list[str],
    series: dict[str, list[float]],
    output_path: Path,
    title: str,
    subtitle: str = "",
    dpi: int = 180,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dpi <= 0:
        raise ValueError("dpi must be greater than 0")

    legend_columns = min(4, max(len(variants), 1))
    legend_rows = max(math.ceil(len(variants) / legend_columns), 1)
    figure_width = max(9.0, len(pipelines) * 2.6, len(variants) * 1.45)
    figure_height = 6.0 + (legend_rows - 1) * 0.45 + (0.3 if subtitle.strip() else 0.0)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)

    group_positions = [float(index) for index in range(len(pipelines))]
    group_width = 0.78
    bar_width = group_width / max(len(variants), 1)
    group_start_offset = -group_width / 2 + bar_width / 2

    max_value = max(max(values) for values in series.values())
    y_max = max_value * 1.15 if max_value > 0 else 1.0

    bar_containers = []
    for series_index, variant_id in enumerate(variants):
        color = PALETTE[series_index % len(PALETTE)]
        offsets = [position + group_start_offset + (series_index * bar_width) for position in group_positions]
        container = ax.bar(
            offsets,
            series[variant_id],
            width=bar_width * 0.92,
            label=variant_id,
            color=color,
            edgecolor="none",
        )
        bar_containers.append(container)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)
    if subtitle.strip():
        ax.text(
            0.5,
            1.00,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#444444",
        )

    ax.set_xlabel("Pipeline", fontsize=11)
    ax.set_ylabel("Total FPS", fontsize=11)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(pipelines, rotation=25, ha="right")
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)

    for container in bar_containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=7)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=legend_columns,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_image)

    if not input_path.is_file():
        print(
            f"Input CSV not found: {input_path}. "
            "Run sweep_pipeline_variants.py first or pass --input-csv with an existing file.",
            file=sys.stderr,
        )
        return 2

    rows = load_rows(input_path)
    pipelines, variants, series = collect_chart_data(rows)
    render_chart(
        pipelines=pipelines,
        variants=variants,
        series=series,
        output_path=output_path,
        title=args.title,
        subtitle=build_platform_subtitle(),
        dpi=args.dpi,
    )
    print(f"Wrote chart to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())