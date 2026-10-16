#!/usr/bin/env python3
"""Laboratory work 1, variant 7: binary erosion with a 3x3 element."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


KERNEL_SIZE = 3
BORDER_VALUE = 255


@dataclass(frozen=True)
class TimingStats:
    minimum_ms: float
    median_ms: float
    mean_ms: float
    stdev_ms: float
    samples_ms: list[float]


def validate_binary_image(image: np.ndarray) -> None:
    """Validate the data contract shared by both erosion implementations."""
    if image.ndim != 2:
        raise ValueError("Expected a single-channel image")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 pixels")
    if image.size == 0:
        raise ValueError("Image must not be empty")
    if not np.logical_or(image == 0, image == 255).all():
        raise ValueError("Image must be binary (only 0 and 255 are allowed)")


def erode_opencv(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a binary image through OpenCV's optimized implementation."""
    validate_binary_image(binary)
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)
    return cv2.erode(
        binary,
        kernel,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=BORDER_VALUE,
    )


def _erode_native_once(binary: np.ndarray) -> np.ndarray:
    """One erosion pass using explicit Python loops and no morphology calls."""
    height, width = binary.shape
    source = binary.tolist()
    result = [[255] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            output_value = 255

            for offset_y in (-1, 0, 1):
                neighbor_y = y + offset_y
                if neighbor_y < 0 or neighbor_y >= height:
                    continue  # Pixels outside the image have the neutral value 255.

                source_row = source[neighbor_y]
                for offset_x in (-1, 0, 1):
                    neighbor_x = x + offset_x
                    if neighbor_x < 0 or neighbor_x >= width:
                        continue
                    if source_row[neighbor_x] == 0:
                        output_value = 0
                        break

                if output_value == 0:
                    break

            result[y][x] = output_value

    return np.asarray(result, dtype=np.uint8)


def erode_native(binary: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode a binary image with a native, readable Python implementation."""
    validate_binary_image(binary)
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    result = binary.copy()
    for _ in range(iterations):
        result = _erode_native_once(result)
    return result


def create_demo_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a deterministic grayscale image when no input file is supplied."""
    if width < 160 or height < 120:
        raise ValueError("Demo image must be at least 160x120 pixels")

    image = np.full((height, width), 28, dtype=np.uint8)
    cv2.rectangle(image, (45, 55), (width // 2 - 20, height - 65), 225, -1)
    cv2.circle(image, (3 * width // 4, height // 3), min(width, height) // 6, 235, -1)
    cv2.line(
        image,
        (width // 2, 4 * height // 5),
        (width - 45, 3 * height // 5),
        210,
        max(5, min(width, height) // 32),
    )
    cv2.putText(
        image,
        "CV",
        (width // 8, 3 * height // 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(1.2, min(width, height) / 250),
        35,
        max(3, min(width, height) // 80),
        cv2.LINE_AA,
    )

    generator = np.random.default_rng(7)
    noise_count = max(1, width * height // 80)
    noise_y = generator.integers(0, height, size=noise_count)
    noise_x = generator.integers(0, width, size=noise_count)
    image[noise_y, noise_x] = generator.choice([0, 255], size=noise_count)
    return image


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def binarize(grayscale: np.ndarray, threshold: int) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in the range 0..255")
    _, binary = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _time_function(
    function: Callable[[np.ndarray, int], np.ndarray],
    image: np.ndarray,
    iterations: int,
    repeats: int,
) -> TimingStats:
    samples_ms: list[float] = []
    for _ in range(repeats):
        started_ns = time.perf_counter_ns()
        function(image, iterations)
        elapsed_ns = time.perf_counter_ns() - started_ns
        samples_ms.append(elapsed_ns / 1_000_000)

    return TimingStats(
        minimum_ms=min(samples_ms),
        median_ms=statistics.median(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
        stdev_ms=statistics.pstdev(samples_ms),
        samples_ms=samples_ms,
    )


def benchmark(
    binary: np.ndarray, iterations: int, repeats: int
) -> tuple[TimingStats, TimingStats]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    # Warm-up excludes one-time import, allocation, and dispatch effects.
    erode_opencv(binary, iterations)
    erode_native(binary, iterations)
    opencv_stats = _time_function(erode_opencv, binary, iterations, repeats)
    native_stats = _time_function(erode_native, binary, iterations, repeats)
    return opencv_stats, native_stats


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def _caption(image: np.ndarray, label: str) -> np.ndarray:
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.copyMakeBorder(
        color, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    cv2.putText(
        panel,
        label,
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return panel


def make_comparison_image(
    grayscale: np.ndarray,
    binary: np.ndarray,
    opencv_result: np.ndarray,
    native_result: np.ndarray,
) -> np.ndarray:
    top = np.hstack((_caption(grayscale, "Input (grayscale)"), _caption(binary, "Binary")))
    bottom = np.hstack(
        (
            _caption(opencv_result, "Erosion: OpenCV"),
            _caption(native_result, "Erosion: native Python"),
        )
    )
    return np.vstack((top, bottom))


def make_benchmark_chart(
    opencv_stats: TimingStats, native_stats: TimingStats
) -> np.ndarray:
    width, height = 900, 520
    chart = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(
        chart,
        "Erosion benchmark (mean time, lower is better)",
        (35, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    values = [opencv_stats.mean_ms, native_stats.mean_ms]
    labels = ["OpenCV", "Native Python"]
    colors = [(211, 134, 52), (73, 102, 190)]
    maximum = max(values) or 1.0
    baseline_y, chart_top = 430, 90
    bar_width = 230
    x_positions = [150, 520]

    cv2.line(chart, (65, baseline_y), (850, baseline_y), (80, 80, 80), 2)
    for x, value, label, color in zip(x_positions, values, labels, colors):
        bar_height = max(2, round((baseline_y - chart_top) * value / maximum))
        cv2.rectangle(
            chart,
            (x, baseline_y - bar_height),
            (x + bar_width, baseline_y),
            color,
            -1,
        )
        value_label = f"{value:.3f} ms"
        cv2.putText(
            chart,
            value_label,
            (x + 18, max(chart_top + 24, baseline_y - bar_height - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            chart,
            label,
            (x + 22, baseline_y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

    speedup = native_stats.mean_ms / opencv_stats.mean_ms
    cv2.putText(
        chart,
        f"OpenCV speedup: {speedup:.1f}x",
        (300, 505),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return chart


def write_benchmark_csv(
    path: Path, opencv_stats: TimingStats, native_stats: TimingStats, repeats: int
) -> None:
    fieldnames = [
        "implementation",
        "minimum_ms",
        "median_ms",
        "mean_ms",
        "stdev_ms",
        "repeats",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in (("OpenCV", opencv_stats), ("Native Python", native_stats)):
            writer.writerow(
                {
                    "implementation": name,
                    "minimum_ms": f"{stats.minimum_ms:.6f}",
                    "median_ms": f"{stats.median_ms:.6f}",
                    "mean_ms": f"{stats.mean_ms:.6f}",
                    "stdev_ms": f"{stats.stdev_ms:.6f}",
                    "repeats": repeats,
                }
            )


def run_self_tests() -> None:
    block = np.zeros((7, 7), dtype=np.uint8)
    block[1:6, 1:6] = 255
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 255
    if not np.array_equal(erode_native(block), expected):
        raise AssertionError("Known 5x5 block test failed")

    all_white = np.full((5, 6), 255, dtype=np.uint8)
    if not np.array_equal(erode_native(all_white), all_white):
        raise AssertionError("Constant border handling test failed")

    generator = np.random.default_rng(7)
    for shape in ((1, 1), (2, 3), (17, 23), (64, 48)):
        binary = (generator.integers(0, 2, size=shape, dtype=np.uint8) * 255).astype(
            np.uint8
        )
        for iterations in (1, 2, 3):
            expected_cv = erode_opencv(binary, iterations)
            actual_native = erode_native(binary, iterations)
            if not np.array_equal(expected_cv, actual_native):
                raise AssertionError(
                    f"Mismatch for shape={shape}, iterations={iterations}"
                )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Binary erosion with a 3x3 structuring element: OpenCV and native Python"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="input image; if omitted, a deterministic demo image is generated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="directory for images, CSV, and JSON (default: results)",
    )
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--self-test", action="store_true", help="run deterministic correctness tests first"
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")

    if args.self_test:
        run_self_tests()
        print("Self-tests: OK")

    grayscale = load_grayscale(args.input) if args.input else create_demo_image()
    binary = binarize(grayscale, args.threshold)
    opencv_result = erode_opencv(binary, args.iterations)
    native_result = erode_native(binary, args.iterations)
    exact_match = np.array_equal(opencv_result, native_result)
    if not exact_match:
        raise AssertionError("OpenCV and native results differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_image(args.output_dir / "input_grayscale.png", grayscale)
    _write_image(args.output_dir / "binary.png", binary)
    _write_image(args.output_dir / "erosion_opencv.png", opencv_result)
    _write_image(args.output_dir / "erosion_native.png", native_result)
    difference = cv2.absdiff(opencv_result, native_result)
    _write_image(args.output_dir / "difference.png", difference)
    _write_image(
        args.output_dir / "comparison.png",
        make_comparison_image(grayscale, binary, opencv_result, native_result),
    )

    opencv_stats, native_stats = benchmark(
        binary, iterations=args.iterations, repeats=args.repeats
    )
    write_benchmark_csv(
        args.output_dir / "benchmark.csv",
        opencv_stats,
        native_stats,
        args.repeats,
    )
    _write_image(
        args.output_dir / "benchmark.png",
        make_benchmark_chart(opencv_stats, native_stats),
    )

    summary = {
        "variant": 7,
        "operation": "binary erosion",
        "kernel": "3x3 rectangle",
        "input": str(args.input) if args.input else "generated deterministic demo",
        "image_width": int(binary.shape[1]),
        "image_height": int(binary.shape[0]),
        "threshold": args.threshold,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "border_value": BORDER_VALUE,
        "exact_match": bool(exact_match),
        "different_pixels": int(np.count_nonzero(difference)),
        "foreground_pixels_before": int(np.count_nonzero(binary)),
        "foreground_pixels_after": int(np.count_nonzero(opencv_result)),
        "opencv": asdict(opencv_stats),
        "native_python": asdict(native_stats),
        "opencv_speedup": native_stats.mean_ms / opencv_stats.mean_ms,
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Results: {args.output_dir.resolve()}")
    print(f"Exact match: {exact_match} ({summary['different_pixels']} different pixels)")
    print(
        "Mean time: "
        f"OpenCV={opencv_stats.mean_ms:.3f} ms, "
        f"native Python={native_stats.mean_ms:.3f} ms"
    )
    print(f"OpenCV speedup: {summary['opencv_speedup']:.1f}x")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
