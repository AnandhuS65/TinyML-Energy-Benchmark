import argparse
import csv
import glob
import os
import random
import time
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


def preprocess_window(raw_window: np.ndarray, mean: np.ndarray, std: np.ndarray, in_det: dict) -> np.ndarray:
    norm = ((raw_window - mean) / std).astype(np.float32)
    x = np.expand_dims(norm, axis=0)
    if in_det["dtype"] == np.int8:
        scale, zero_pt = in_det["quantization"]
        x = np.clip(np.round(x / scale + zero_pt), -128, 127).astype(np.int8)
    return x


def benchmark_model(model_path, mean, std, repeats=20, warmup=5, num_threads=1, prep_pin=None, infer_pin=None):
    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]

    rng = np.random.default_rng(42)
    sample_pool = rng.standard_normal((50, 128, 9), dtype=np.float32)

    for i in range(warmup):
        x_in = preprocess_window(sample_pool[i % len(sample_pool)], mean, std, in_det)
        interp.set_tensor(in_det["index"], x_in)
        interp.invoke()

    latencies_ms = []
    for i in range(repeats):
        raw_win = sample_pool[i % len(sample_pool)]

        if prep_pin is not None:
            prep_pin.on()
        x_in = preprocess_window(raw_win, mean, std, in_det)
        if prep_pin is not None:
            prep_pin.off()

        if infer_pin is not None:
            infer_pin.on()
        t0 = time.perf_counter_ns()
        interp.set_tensor(in_det["index"], x_in)
        interp.invoke()
        t1 = time.perf_counter_ns()
        if infer_pin is not None:
            infer_pin.off()

        latencies_ms.append((t1 - t0) / 1e6)

    return {
        "model_file": os.path.basename(model_path),
        "threads": num_threads,
        "repeats": repeats,
        "median_ms": round(float(np.median(latencies_ms)), 4),
        "mean_ms": round(float(np.mean(latencies_ms)), 4),
        "std_ms": round(float(np.std(latencies_ms, ddof=1)), 4) if repeats > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Pi 5 LiteRT HAR Latency Benchmark")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--scaler", default="models/scaler_params.npz")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--gpio", action="store_true")
    parser.add_argument("--out-csv", default="data/processed/pi5_latency_dryrun.csv")
    args = parser.parse_args()

    scaler = np.load(args.scaler)
    mean = scaler["mean"].reshape(1, 9).astype(np.float32)
    std = (scaler["std"].reshape(1, 9) + 1e-8).astype(np.float32)

    prep_pin = LED(17) if (args.gpio and GPIO_AVAILABLE) else None
    infer_pin = LED(27) if (args.gpio and GPIO_AVAILABLE) else None

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "har_*.tflite")))
    assert model_paths, f"No har_*.tflite models found in {args.models_dir}"

    rows = []
    try:
        for t in args.threads:
            ordered = list(model_paths)
            random.Random(42 + t).shuffle(ordered)
            for mp in ordered:
                res = benchmark_model(mp, mean, std, repeats=args.repeats, num_threads=t, prep_pin=prep_pin, infer_pin=infer_pin)
                rows.append(res)
                print(f"{res['model_file']:24s} | threads={res['threads']} | median={res['median_ms']:.4f} ms | std={res['std_ms']:.4f} ms")
    finally:
        if prep_pin is not None:
            prep_pin.close()
        if infer_pin is not None:
            infer_pin.close()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {args.out_csv}")


if __name__ == "__main__":
    main()
