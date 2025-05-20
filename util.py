from datetime import datetime
import time
import numpy as np
from scipy.stats import ttest_rel, ks_2samp, t

def time_str_to_seconds(time_str: str) -> int:
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def seconds_to_time_str(seconds: float) -> str:
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def empty_values(data):
    empty_indices = []
    for idx, val in enumerate(data):
        val_str = str(val).strip().lower()
        if val_str == "empty response" or val_str == "" or val_str == "none" or val is None:
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

# Paired t-test 
def paired_t_test(ground_truth, prediction):
    try:
        if len(ground_truth) < 2 or len(prediction) < 2:
            return np.nan, np.nan
        
        if len(ground_truth) != len(prediction):
            return np.nan, np.nan
        
        t_stat, p_value = ttest_rel(ground_truth, prediction)
        
        if np.isnan(t_stat) or np.isnan(p_value):
            return np.nan, np.nan
            
        return t_stat, p_value
    except Exception:
        return np.nan, np.nan

# Kolmogorov-Smirnov test
def ks_test(ground_truth, prediction):
    try:
        if len(ground_truth) < 1 or len(prediction) < 1:
            return np.nan, np.nan
        
        stat, pvalue = ks_2samp(ground_truth, prediction)
        
        if np.isnan(stat) or np.isnan(pvalue):
            return np.nan, np.nan
            
        return stat, pvalue
    except Exception:
        return np.nan, np.nan

def stat_test(ground_truth, prediction):
    if not ground_truth or not prediction:
        return [np.nan, np.nan, np.nan, np.nan]
    
    try:
        ground_truth = np.array(ground_truth, dtype=float)
        prediction = np.array(prediction, dtype=float)
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]
    
    if np.isnan(ground_truth).any() or np.isnan(prediction).any() or np.isinf(ground_truth).any() or np.isinf(prediction).any():
        valid_indices = ~(np.isnan(ground_truth) | np.isnan(prediction) | np.isinf(ground_truth) | np.isinf(prediction))
        ground_truth = ground_truth[valid_indices]
        prediction = prediction[valid_indices]
    
    t_stat, t_pvalue = paired_t_test(ground_truth, prediction)
    ks_stat, ks_pvalue = ks_test(ground_truth, prediction)

    return [t_stat, t_pvalue, ks_stat, ks_pvalue]

def fixed_effects_pooled_estimate(effect_sizes, weights):
    effect_sizes = np.array(effect_sizes)
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0
        
    pooled_estimate = np.sum(weights * effect_sizes) / sum_weights
    return pooled_estimate