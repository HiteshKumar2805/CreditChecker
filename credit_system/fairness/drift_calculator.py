import numpy as np
import pandas as pd

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.
    PSI < 0.1: No significant drift
    PSI < 0.2: Moderate drift, monitor
    PSI >= 0.2: Significant drift, action required
    """
    def scale_range(input, min_val, max_val):
        input += -(np.min(input))
        input /= np.max(input) / (max_val - min_val)
        input += min_val
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid zero division
    def sub_zero(x): return [value if value != 0 else 0.0001 for value in x]
    
    expected_percents = sub_zero(expected_percents)
    actual_percents = sub_zero(actual_percents)

    psi_values = (np.array(expected_percents) - np.array(actual_percents)) * np.log(np.array(expected_percents) / np.array(actual_percents))
    return float(np.sum(psi_values))

def compute_model_drift(training_probs: np.ndarray, recent_probs: np.ndarray) -> dict:
    """
    Computes drift indicators for the regulator report.
    """
    if len(recent_probs) < 50:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "Need at least 50 recent decisions to compute reliable drift.",
            "psi": 0.0,
            "drift_detected": False
        }

    psi = compute_psi(training_probs, recent_probs)
    
    drift_detected = psi >= 0.2
    status = "CRITICAL" if psi >= 0.2 else ("WARNING" if psi >= 0.1 else "HEALTHY")
    
    return {
        "status": status,
        "psi_score": round(psi, 4),
        "drift_detected": drift_detected,
        "baseline_mean_prob": round(float(np.mean(training_probs)), 4),
        "recent_mean_prob": round(float(np.mean(recent_probs)), 4),
        "message": "Significant drift detected." if drift_detected else "Model output distribution is stable."
    }
