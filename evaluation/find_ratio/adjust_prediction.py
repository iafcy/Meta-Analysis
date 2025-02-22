import json as json
import re as re
import numpy as np
from scipy.stats import t

def empty_values(data):
    empty_indices = []
    for idx, val in enumerate(data):
        if str(val).strip().lower() == "empty response":
            empty_indices.append(idx)
    return empty_indices

def grubbs_test_outliers(data, alpha=0.05):
    original_data = [(val, idx) for idx, val in enumerate(data)]
    outlier_indices = [] 

    def critical_value_grubbs(n, alpha):
        t_crit = t.ppf(1 - alpha/(2*n), df=n - 2)
        numerator = (n - 1)
        denominator = np.sqrt(n)
        inside_sqrt = t_crit**2 / (n - 2 + t_crit**2)
        return numerator / denominator * np.sqrt(inside_sqrt)

    while True:
        n = len(original_data)
        if n < 3:
            break

        values = np.array([item[0] for item in original_data])
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) 

        abs_diffs = np.abs(values - mean_val)
        max_idx = np.argmax(abs_diffs)  
        max_val, max_val_original_idx = original_data[max_idx]
        G = abs(max_val - mean_val) / std_val  
        G_crit = critical_value_grubbs(n, alpha)

        if G > G_crit:
            outlier_indices.append(max_val_original_idx)  
            original_data.pop(max_idx)                    
        else:
            break

    return sorted(outlier_indices)

def bootlier_plot_outliers(data, alpha=0.05, n_boot=1000, random_state=None):
    data = np.array(data)
    n = len(data)
    if n < 2:
        return []

    if random_state is not None:
        np.random.seed(random_state)

    boot_medians = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_medians.append(np.median(sample))
    boot_medians = np.array(boot_medians)

    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    median_lower = np.quantile(boot_medians, lower_q)
    median_upper = np.quantile(boot_medians, upper_q)
    outlier_mask = (data < median_lower) | (data > median_upper)
    outlier_indices = np.where(outlier_mask)[0]

    return list(outlier_indices)

