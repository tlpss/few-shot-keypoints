#!/usr/bin/env python3
import argparse
import time
from typing import Callable, Dict, List

import torch

# Featurizers
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg in {"cuda", "cpu"}:
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return device_arg
    raise ValueError(f"Unknown device: {device_arg}")


def build_featurizer(name: str, device: str):
    if name in FeaturizerRegistry.list():
        return FeaturizerRegistry.create(name, device=device)
    raise ValueError(f"Unsupported featurizer: {name}")


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_single_featurizer(
    featurizer,
    img_shape: tuple[int, int],
    warmup: int,
    repeats: int,
    device: str,
) -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        image = torch.rand(1, 3, img_shape[0], img_shape[1], device=device)
        _ = featurizer.extract_features(image)
    synchronize(device)

    # Timing
    start = time.perf_counter()
    for _ in range(repeats):
        image = torch.rand(1, 3, img_shape[0], img_shape[1], device=device)
        _ = featurizer.extract_features(image)
    synchronize(device)
    end = time.perf_counter()

    avg_seconds = (end - start) / max(1, repeats)
    return avg_seconds


def format_seconds(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f} ms"


def main():
    parser = argparse.ArgumentParser(description="Measure inference time of featurizers.")
    parser.add_argument(
        "--featurizers",
        type=str,
        default=",".join(FeaturizerRegistry.list()),
        help="Comma-separated list of featurizers: dinov2-s,dinov2-b,dinov2-l,dinov3-s,dinov3-b,dinov3-l,radio-b,radio-l",
    )
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--repeats", type=int, default=50, help="Timed repetitions")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on",
    )

    args = parser.parse_args()

    device = get_device(args.device)

    requested = [s.strip() for s in args.featurizers.split(",") if s.strip()]

    results: Dict[str, float] = {}

    for name in requested:
        try:
            featurizer = build_featurizer(name, device=device)
            featurizer_name = type(featurizer).__name__
            avg_sec = time_single_featurizer(
                featurizer=featurizer,
                img_shape=(args.height, args.width),
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            results[featurizer_name] = avg_sec
        except Exception as e:
            results[f"{name} (error)"] = float("nan")
            print(f"[ERROR] {name}: {e}")

    # Report
    print("Featurizer inference time (per batch):")
    for k, v in results.items():
        if v == v:  # not NaN
            per_image = v / max(1, args.batch_size)
            throughput = (args.batch_size / v) if v > 0 else float("inf")
            print(
                f"- {k}: {format_seconds(v)} per batch | {format_seconds(per_image)} per image | {throughput:.2f} images/s"
            )
        else:
            print(f"- {k}: failed")


if __name__ == "__main__":
    main()
