import argparse
import csv
import glob
import os
import random
import subprocess
import time
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    from tflite_runtime.interpreter import Interpreter, OpResolverType

try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


def get_cpu_temp_c() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        return float(out.split("=")[1].replace("'C", ""))
    except Exception:
        return -1.0


def load_interpreter(model_path: str, num_threads: int, delegate_mode: str) -> Interpreter:
    if delegate_mode == "reference":
        interp = Interpreter(
            model_path=model_path,
            num_threads=num_threads,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
        )
    else:
        interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def preprocess_window(raw_window: np.ndarray, mean: np.ndarray, std: np.ndarray, in_det: dict) -> np.ndarray:
    norm = ((raw_window - mean) / std).astype(np.float32)
    x = np.expand_dims(norm, axis=0)
    if in_det["dtype"] == np.int8:
        scale, zero_pt = in_det["quantization"]
        x = np.clip(np.round(x / scale + zero_pt), -128, 127).astype(np.int8)
    return x


def evaluate_accuracy(interp, X_test, y_test, mean, std) -> float:
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    correct = 0
    for i in range(len(X_test)):
        x_in = preprocess_window(X_test[i], mean, std, in_det)
        interp.set_tensor(in_det["index"], x_in)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out_det["index"])[0]))
        if pred == int(y_test[i]):
            correct += 1
    return round(100.0 * correct / len(X_test), 2)


def benchmark_model(
    model_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    delegate_mode: str,
    num_threads: int,
    sweeps: int,
    repeats: int,
    warmup: int = 10,
    prep_pin=None,
    infer_pin=None,
    check_acc: bool = False,
):
    interp = load_interpreter(model_path, num_threads=num_threads, delegate_mode=delegate_mode)
    in_det = interp.get_input_details()[0]

    # Warm-up
    for i in range(warmup):
        x_in = preprocess_window(X_test[i % len(X_test)], mean, std, in_det)
        interp.set_tensor(in_det["index"], x_in)
        interp.invoke()

    temp_before = get_cpu_temp_c()
    all_latencies_ms = []

    for s in range(sweeps):
        for i in range(repeats):
            raw_win = X_test[(s * repeats + i) % len(X_test)]

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

            all_latencies_ms.append((t1 - t0) / 1e6)

    temp_after = get_cpu_temp_c()
    acc_pct = evaluate_accuracy(interp, X_test, y_test, mean, std) if check_acc else ""

    arr = np.array(all_latencies_ms, dtype=np.float64)
    return {
        "model_file": os.path.basename(model_path),
        "delegate": delegate_mode,
        "threads": num_threads,
        "total_runs": sweeps * repeats,
        "median_ms": round(float(np.median(arr)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms": round(float(np.std(arr, ddof=1)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "temp_before_c": temp_before,
        "temp_after_c": temp_after,
        "test_acc_pct": acc_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Pi 5 Full LiteRT HAR Benchmark")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--scaler", default="models/scaler_params.npz")
    parser.add_argument("--test-data", default="data/raw/har_test.npz")
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--delegates", nargs="+", default=["xnnpack", "reference"])
    parser.add_argument("--gpio", action="store_true")
    parser.add_argument("--skip-acc", action="store_true")
    parser.add_argument("--out-csv", default="data/processed/pi5_latency_full.csv")
    args = parser.parse_args()

    scaler = np.load(args.scaler)
    mean = scaler["mean"].reshape(1, 9).astype(np.float32)
    std = (scaler["std"].reshape(1, 9) + 1e-8).astype(np.float32)

    data = np.load(args.test_data)
    X_test, y_test = data["X_test"], data["y_test"]

    prep_pin = LED(17) if (args.gpio and GPIO_AVAILABLE) else None
    infer_pin = LED(27) if (args.gpio and GPIO_AVAILABLE) else None
    if args.gpio and not GPIO_AVAILABLE:
        print("WARNING: --gpio requested but gpiozero is unavailable.")

    model_paths = sorted(glob.glob(os.path.join(args.models_dir, "har_*.tflite")))
    assert model_paths, f"No har_*.tflite models found in {args.models_dir}"

    rows = []
    acc_checked = set()
    try:
        for delegate in args.delegates:
            for t in args.threads:
                ordered = list(model_paths)
                random.Random(42 + t).shuffle(ordered)
                for mp in ordered:
                    do_acc = (not args.skip_acc) and (mp not in acc_checked)
                    res = benchmark_model(
                        mp, mean, std, X_test, y_test,
                        delegate_mode=delegate,
                        num_threads=t,
                        sweeps=args.sweeps,
                        repeats=args.repeats,
                        prep_pin=prep_pin,
                        infer_pin=infer_pin,
                        check_acc=do_acc,
                    )
                    if do_acc:
                        acc_checked.add(mp)
                    rows.append(res)
                    print(
                        f"{res['model_file']:22s} | {res['delegate']:9s} | t={res['threads']} | "
                        f"med={res['median_ms']:7.4f} ms | p95={res['p95_ms']:7.4f} ms | "
                        f"temp={res['temp_after_c']:4.1f}C | acc={res['test_acc_pct']}"
                    )
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
    print(f"\nSaved full results to {args.out_csv}")


if __name__ == "__main__":
    main()
