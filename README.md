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


# Custom Subject-Wise Train/Validation/Test Split

This document describes a subject-wise splitting strategy for time-series or cycle data, ensuring that every subject appears in the training, validation, and test sets—even if they only have a small number of cycles.

## Overview

Rather than shuffling all cycles globally, we group by subject ID (`y`) and split each subject’s cycles according to the number of cycles they have:

- **> 5 cycles**:  80% train, 10% val, 10% test  
- **3–5 cycles**: fallback to 80/10/10 (may produce very small val/test sets)  
- **2 cycles**: 1 → train, both 2nd occurrences → val & test  
- **1 cycle**: duplicate the single cycle into train/val/test  

This guarantees every subject contributes to each split and preserves approximate proportions for larger subjects.

---

## Requirements

- Python 3.6+  
- `numpy`  
- `scikit-learn`  

```bash
pip install numpy scikit-learn
```

---

## `custom_split` Function

```python
from sklearn.model_selection import train_test_split
import numpy as np

def custom_split(X: np.ndarray, 
                 y: np.ndarray, 
                 seed: int = 42):
    """
    Perform a subject-wise train/val/test split on cycle data.

    Parameters
    ----------
    X : np.ndarray, shape (N_cycles, T, C)
        Array of cycle data (e.g., time‐series per subject).
    y : np.ndarray, shape (N_cycles,)
        Integer subject IDs for each cycle.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_tr, y_tr : train set
    X_val, y_val : validation set
    X_te, y_te : test set
    """
    train_idx, val_idx, test_idx = [], [], []

    for sid in np.unique(y):
        idx = np.where(y == sid)[0]
        n   = len(idx)

        if n > 5:
            # >5 cycles: 80% train, 10% val, 10% test
            i_tr, i_tmp = train_test_split(idx, train_size=0.8,
                                           random_state=seed, shuffle=True)
            i_val, i_te = train_test_split(i_tmp, train_size=0.5,
                                           random_state=seed, shuffle=True)

        elif n == 2:
            # exactly 2 cycles: 1→train, 2→val & test
            i_tr  = idx[:1]
            i_val = idx[1:]
            i_te  = idx[1:]

        elif n == 1:
            # only 1 cycle: duplicate into all sets
            i_tr = i_val = i_te = idx

        else:
            # 3–5 cycles: fallback to 80/10/10
            i_tr, i_tmp = train_test_split(idx, train_size=0.8,
                                           random_state=seed, shuffle=True)
            i_val, i_te = train_test_split(i_tmp, train_size=0.5,
                                           random_state=seed, shuffle=True)

        train_idx += i_tr.tolist()
        val_idx   += i_val.tolist()
        test_idx  += i_te.tolist()

    X_tr, y_tr   = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx],   y[val_idx]
    X_te, y_te   = X[test_idx],  y[test_idx]

    return X_tr, y_tr, X_val, y_val, X_te, y_te
```

---

## How It Works

1. **Grouping by subject**  
   Iterate over each unique subject ID in `y`.

2. **Branch by cycle count**  
   - **> 5 cycles**: Perform a true random 80/10/10 split.  
   - **3–5 cycles**: Use the same 80/10/10 logic (though val/test may be very small).  
   - **2 cycles**: Use the first cycle for training; the second cycle is shared between validation and test.  
   - **1 cycle**: Duplicate the single cycle across all splits so that the model sees at least one example of that subject during training, validation, and testing.

3. **Index collection**  
   Collect the train/val/test indices for each subject, then concatenate all indices and index into `X` and `y` at the end.

---

## Usage Example

```python
import numpy as np

# Suppose you have 100 cycles, each of length 200 timesteps with 3 channels:
X = np.random.randn(100, 200, 3)
# y contains subject IDs (e.g., 0–9, each subject has varying # of cycles)
y = np.random.randint(0, 10, size=100)

X_tr, y_tr, X_val, y_val, X_te, y_te = custom_split(X, y, seed=123)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")
```

This approach ensures robust, subject-aware splitting and helps prevent data leakage across splits while still including every subject in all phases.


