# gait_gravity


# Gait Cycle Segmentation Pipeline

This repository provides a robust pipeline for reading gait force data, filtering noise, detecting heel-strike events, and segmenting the signal into fixed‑length gait cycles. It is designed to handle edge cases gracefully—ensuring usable cycles even when data is noisy or trials are very short.

## Features

- **Automatic File I/O**: Load CSV files with flexible column name handling (strips whitespace).
- **Adaptive Filtering**: Zero‑phase 4th‑order Butterworth low‑pass filter at 20 Hz.
- **Heel‑Strike Detection**: Two‑stage peak detection on tangential (`TgF`) and vertical (`gFz`) force signals with adaptive thresholding.
- **Fixed‑Length Cycles**: Segment gait cycles into exactly 150 samples via padding or truncation.
- **Fallback Strategies**: Windowing and padding ensures at least one cycle per trial, even without detectable peaks.

## Dependencies

- Python 3.7+
- numpy
- pandas
- scipy

Install with:

```bash
pip install numpy pandas scipy
```

## Usage

1. Place your CSV file(s) in a data directory.
2. Update the script parameters:
   - `fs`: Sampling frequency (Hz).
   - `cycle_len`: Desired cycle length (default: 150 samples).
3. Run the segmentation script:

```python
from gait_pipeline import process_file

fn = "path/to/your/data.csv"
out = process_file(fn, fs=100, cycle_len=150)
```

4. The output is a NumPy array of shape `(n_cycles, cycle_len, n_features)`.

## Pipeline Steps

1. **File I/O & Feature Alignment**

   - Load CSV and strip whitespace from column names.
   - Automatically select all non-`time` channels into a 2D array `(n_samples, n_features)`.

2. **Low‑Pass Filtering & Edge‑Effect Handling**

   - Design a 4th‑order Butterworth filter with cutoff at 20 Hz:
     ```python
     b, a = butter(4, 20/(fs/2), btype="low")
     padlen = 3 * (max(len(a), len(b)) - 1)
     ```
   - If `n_samples <= padlen`, skip filtering and proceed directly to padding.
   - Otherwise, apply zero‑phase filtering via `filtfilt`.

3. **Heel‑Strike Peak Detection**

   - Try detecting peaks on the filtered `TgF` signal using:
     ```python
     p, _ = find_peaks(sig, height=np.mean(sig), distance=fs*0.5)
     ```
   - If fewer than two peaks, retry without a height threshold.
   - If still insufficient, repeat on filtered `gFz`.
   - Enforces a 0.5 s refractory period (`distance=fs*0.5`).

4. **Cycle Segmentation & Normalization**

   - For each adjacent peak pair `(p_i, p_{i+1})`, extract the segment `data[p_i:p_{i+1}, :]`.
   - If segment length < `cycle_len`, pad with zeros; if longer, truncate.
   - Produces cycles of shape `(cycle_len, n_features)`.

5. **Robust Fallbacks**

   - **Windowing**: If no peaks are found, split the full signal into as many non-overlapping 150‑sample windows as possible.
   - **Partial Pad**: Pad any remaining samples (< 150) into one final cycle.
   - **Empty Trials**: Return an empty array when there are zero samples.

## Example

```python
import numpy as np
from gait_pipeline import process_file

# Parameters
fs = 100  # Hz
cycle_len = 150

# Process a single file
dry_path = "~/data/gait_trial1.csv"
cycles = process_file(dry_path, fs=fs, cycle_len=cycle_len)

print(f"Extracted {cycles.shape[0]} gait cycles,")
print(f"Each of shape {cycles.shape[1:]} -> (samples, features)")
```


