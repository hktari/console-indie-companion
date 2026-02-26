import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.vlm.analyze import SceneAnalyzer
from src.utils.cost_tracker import CostTracker
from src.utils.logging_config import setup_logging


@dataclass
class FrameMetric:
    index: int
    filename: str
    capture_ts: float
    analyze_start_ts: float
    analyze_end_ts: float
    frame_bytes: int
    ok: bool
    error: Optional[str]

    @property
    def analyze_latency_s(self) -> float:
        return self.analyze_end_ts - self.analyze_start_ts


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def run_benchmark(
    screenshot_dir: str,
    model: str,
    limit: int,
    sleep_between_frames_s: float,
    output_path: Optional[str],
) -> dict[str, Any]:
    load_dotenv()

    p = Path(screenshot_dir)
    if not p.is_dir():
        raise SystemExit(f"Screenshot directory not found: {screenshot_dir}")

    image_files = sorted(
        [x for x in p.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_files:
        raise SystemExit(f"No images found in: {screenshot_dir}")

    if limit > 0:
        image_files = image_files[:limit]

    cost_tracker = CostTracker()
    analyzer = SceneAnalyzer(model=model, cost_tracker=cost_tracker)

    metrics: list[FrameMetric] = []

    t_run_start = time.perf_counter()
    for idx, img_path in enumerate(image_files, 1):
        capture_ts = time.perf_counter()
        data = img_path.read_bytes()
        suffix = img_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"

        analyze_start_ts = time.perf_counter()
        ok = True
        err: Optional[str] = None
        try:
            result = analyzer.analyze_screenshot(data, mime_type=mime_type)
            if not isinstance(result, dict) or "error" in result:
                ok = False
                if isinstance(result, dict):
                    err = str(result.get("error"))
                else:
                    err = "invalid result"
        except Exception as e:
            ok = False
            err = str(e)
        analyze_end_ts = time.perf_counter()

        metrics.append(
            FrameMetric(
                index=idx,
                filename=img_path.name,
                capture_ts=capture_ts,
                analyze_start_ts=analyze_start_ts,
                analyze_end_ts=analyze_end_ts,
                frame_bytes=len(data),
                ok=ok,
                error=err,
            )
        )

        if sleep_between_frames_s > 0:
            time.sleep(sleep_between_frames_s)

    t_run_end = time.perf_counter()

    latencies = [m.analyze_latency_s for m in metrics]
    inter_frame = [
        metrics[i].analyze_start_ts - metrics[i - 1].analyze_start_ts
        for i in range(1, len(metrics))
    ]

    cost = cost_tracker.get_session_cost()

    report: dict[str, Any] = {
        "config": {
            "screenshot_dir": screenshot_dir,
            "model": model,
            "frames": len(metrics),
            "limit": limit,
            "sleep_between_frames_s": sleep_between_frames_s,
        },
        "timing": {
            "run_wall_time_s": t_run_end - t_run_start,
            "throughput_fps": (len(metrics) / (t_run_end - t_run_start)) if metrics else 0.0,
            "latency_s": {
                "mean": statistics.mean(latencies) if latencies else 0.0,
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p95": _percentile(latencies, 95),
                "max": max(latencies) if latencies else 0.0,
                "min": min(latencies) if latencies else 0.0,
            },
            "inter_frame_s": {
                "mean": statistics.mean(inter_frame) if inter_frame else 0.0,
                "p50": _percentile(inter_frame, 50),
                "p90": _percentile(inter_frame, 90),
                "p95": _percentile(inter_frame, 95),
                "max": max(inter_frame) if inter_frame else 0.0,
                "min": min(inter_frame) if inter_frame else 0.0,
            },
        },
        "cost": cost,
        "calls": cost_tracker.get_calls_as_dicts(),
        "frames": [
            {
                "index": m.index,
                "filename": m.filename,
                "frame_bytes": m.frame_bytes,
                "analyze_latency_s": round(m.analyze_latency_s, 4),
                "ok": m.ok,
                "error": m.error,
            }
            for m in metrics
        ],
    }

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemini VLM analysis throughput/cost on a screenshot set")
    parser.add_argument("--screenshot-dir", default="data/screenshots/", help="Directory with PNG/JPG screenshots")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", choices=["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of frames (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between frames (seconds) to simulate capture interval")
    parser.add_argument("--output", type=str, default="data/benchmarks/vlm_benchmark.json", help="Write JSON report to this path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    report = run_benchmark(
        screenshot_dir=args.screenshot_dir,
        model=args.model,
        limit=args.limit,
        sleep_between_frames_s=args.sleep,
        output_path=args.output,
    )

    # Compact stdout summary
    timing = report["timing"]
    cost = report["cost"]
    print("\n=== VLM BENCHMARK SUMMARY ===")
    print(f"Frames: {report['config']['frames']}")
    print(f"Model: {report['config']['model']}")
    print(f"Wall time: {timing['run_wall_time_s']:.2f}s")
    print(f"Throughput: {timing['throughput_fps']:.3f} fps")
    print("Latency (s):", timing["latency_s"])
    print("Inter-frame (s):", timing["inter_frame_s"])
    print(f"Estimated total cost: ${cost['total_cost']:.4f} (calls={cost['call_count']})")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
