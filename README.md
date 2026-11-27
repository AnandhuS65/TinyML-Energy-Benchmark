# Benchmarking Energy and Latency of TinyML Models on Low-Cost Embedded Systems

**BEng (Hons) Robotics Engineering and AI — Final-Year Project (ER6010)**  
Kingston University London | **Author:** Anandhu Santhosh

---

## Overview
Running neural networks directly on edge hardware avoids network latency and the high energy cost of wireless transmission. However, many TinyML studies report classification accuracy alone or rely on specialist laboratory power analysers without quantifying measurement uncertainty.

This project designs, calibrates, and validates a **repeatable, low-cost, inference-only measurement setup** (targeting $\le 5\%$ expanded uncertainty, $k = 2$) to benchmark **accuracy, latency, and energy per inference** across bare-metal microcontrollers and an edge single-board computer (SBC).

## Research Question
> *Can a repeatable, low-cost measurement setup produce trustworthy latency and energy data, and what do those measurements reveal about the trade-offs between accuracy, speed, and energy across low-cost embedded platforms, model sizes, and quantisation levels?*

---

## Hardware Platforms & Benchmarking Scope

| Platform | Architecture | Runtime & Kernel Comparison | Role |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi Pico H** | Arm Cortex-M0+ (no SIMD) | TensorFlow Lite Micro (Reference) | Non-SIMD bare-metal MCU baseline |
| **ESP32-S3** | Xtensa LX7 (Vector/SIMD) | TensorFlow Lite Micro (Reference vs. **ESP-NN**) | Non-Arm MCU & literature cross-check |
| **Raspberry Pi 5 (8 GB)** | Quad-core Arm Cortex-A76 | LiteRT / TFLite (Reference vs. **XNNPACK / NEON**) | Headless Linux edge SBC comparison |
| **STM32** *(optional)* | Arm Cortex-M | TensorFlow Lite Micro (Reference vs. **CMSIS-NN**) | Optional Arm SIMD MCU target |

### Workload & Model Variants
* **Dataset:** UCI Human Activity Recognition (UCI HAR) — 10,299 labelled samples across 6 activities (fixed 30-volunteer train/test split).
* **Models:** 6 variants of a 1D-CNN/MLP architecture — **Small, Medium, and Large** configurations, each exported in **FP32** and **INT8** via Post-Training Quantisation (PTQ).

---

## Measurement Architecture
To prevent measurement overhead and host-PC back-powering from skewing results, power and timing are captured externally:
* **Power Sensing:** High-side **Adafruit INA219** (TI INA219B, $0.1\ \Omega$ 1% shunt) wired in series with a regulated DC bench supply (DUT USB data lines disconnected; direct 18–20 AWG power leads for Raspberry Pi 5).
* **Dedicated Logger:** A separate **Raspberry Pi Pico H** polls the INA219 over I2C and timestamps **two 3.3 V GPIO marker pins** (separating pre-processing from inference), streaming raw CSV logs to the Host PC over USB Serial.
* **Calibration & Protocol:** Calibrated against a laboratory digital multimeter across known resistor loads. Every board, model, and kernel combination is executed at least **5 times in randomised order** (aligned with MLPerf Tiny conventions), reporting medians, standard deviations, energy–delay products, and kernel speedup ratios.

---

## Repository Structure
```text
tinyml-energy-benchmark/
├── models/             # UCI HAR training scripts, FP32/INT8 .tflite models & C headers
├── firmware/           # Deployment code for each target & logger
│   ├── pico/           # Raspberry Pi Pico H (TFLite Micro)
│   ├── esp32s3/        # ESP32-S3 (ESP-IDF + TFLite Micro / ESP-NN)
│   ├── pi5/            # Raspberry Pi 5 (LiteRT + gpiozero marker pins)
│   └── logger/         # Dedicated Pico H INA219 + GPIO timestamp logger
├── data/
│   ├── raw/            # Archived read-only CSV measurement logs
│   └── processed/      # Aggregated metrics & uncertainty budgets
├── analysis/           # Python scripts (pandas, SciPy, Matplotlib) for plots & stats
└── docs/               # Weekly progress log, wiring guides & calibration notes
