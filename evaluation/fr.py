import pandas as pd
import numpy as np
import json
import os

from utils import grubbs_test_outliers, empty_values, stat_test, fixed_effects_pooled_estimate

# Helper function to calculate metrics for a list of meta-analysis objects from one parser.
def _calculate_metrics_for_meta_list(list_of_meta_prediction_objects):
    processed_meta_results = []

    for meta_pred_obj in list_of_meta_prediction_objects:
        weighted_abs_diff_sum = 0.0
        weighted_percentage_diff_sum = 0.0
        total_weight_for_meta_avg = 0.0
        
        all_truths_for_meta = []
        all_preds_for_meta = []
        
        meta_level_empty_num = 0
        meta_level_valid_task_count = 0
        
        pmid = str(meta_pred_obj.get('pmid', 'N/A'))
        forest_plots_data_from_input = meta_pred_obj.get('data', []) 
        
        meta_forest_plot_stats = [] # Stores stats for each FP within this meta-analysis
        
        plot_num_counter = 0
        # For calculating average t-test and ks-test for this meta-analysis
        meta_t_stat_sum, meta_t_pvalue_sum, meta_ks_stat_sum, meta_ks_pvalue_sum = 0.0, 0.0, 0.0, 0.0
        meta_valid_t_test_count, meta_valid_ks_test_count = 0, 0

        for single_fp_sample_list in forest_plots_data_from_input: # single_fp_sample_list is List[Dict] for one FP
            plot_num_counter += 1
            
            # Extract raw data for THIS forest plot
            raw_predictions_str = [sample.get('prediction') for sample in single_fp_sample_list]
            raw_ground_truths_str = [sample.get('truth') for sample in single_fp_sample_list]
            raw_sample_weights_str = [sample.get('weight', '1.0') for sample in single_fp_sample_list]

            # Convert ground truths and weights to float early, assuming they are always valid numbers
            current_fp_ground_truths_float = []
            for gt_s in raw_ground_truths_str:
                try: current_fp_ground_truths_float.append(float(gt_s))
                except (ValueError, TypeError): pass # Skip if truth is not a valid number
            
            fp_task_num = len(current_fp_ground_truths_float)
            if fp_task_num == 0: # No valid ground truths for this forest plot
                meta_forest_plot_stats.append({
                    "plot_num": plot_num_counter, "find_ratio_task_num": 0, 
                    "valid_task_count": 0, "t_paired_test": None, "ks_test": None
                })
                continue

            # Ensure predictions list matches ground truth length if initially empty
            if not raw_predictions_str and current_fp_ground_truths_float:
                raw_predictions_str = ["Empty Response"] * len(current_fp_ground_truths_float)
            elif len(raw_predictions_str) != len(current_fp_ground_truths_float):
                raw_predictions_str = (raw_predictions_str + ["Empty Response"] * fp_task_num)[:fp_task_num]

            current_fp_initial_weights_float = []
            for w_s in raw_sample_weights_str:
                try: current_fp_initial_weights_float.append(float(w_s))
                except (ValueError, TypeError): current_fp_initial_weights_float.append(1.0) # Default weight
            current_fp_initial_weights_float = (current_fp_initial_weights_float + [1.0] * fp_task_num)[:fp_task_num]

            fp_empty_responses_count = sum(1 for val_str in raw_predictions_str if val_str == "Empty Response")
            meta_level_empty_num += fp_empty_responses_count
            
            fp_valid_tasks_count = fp_task_num - fp_empty_responses_count
            meta_level_valid_task_count += fp_valid_tasks_count

            # Outlier detection
            empty_outlier_indices = empty_values(raw_predictions_str) 
            
            numeric_prediction_for_grubbs, numeric_indices_original_pos = [], []
            temp_non_float_indices = [] 

            for idx, pred_str in enumerate(raw_predictions_str):
                if idx in empty_outlier_indices: continue
                try:
                    numeric_prediction_for_grubbs.append(float(pred_str))
                    numeric_indices_original_pos.append(idx)
                except (ValueError, TypeError): temp_non_float_indices.append(idx)
            
            grubbs_outliers_original_indices = []
            if len(numeric_prediction_for_grubbs) > 2: # Grubbs needs at least 3 data points
                try:
                    grubbs_outliers_relative = grubbs_test_outliers(numeric_prediction_for_grubbs)
                    for rel_idx in grubbs_outliers_relative:
                        if rel_idx < len(numeric_indices_original_pos):
                            grubbs_outliers_original_indices.append(numeric_indices_original_pos[rel_idx])
                except Exception: pass # Grubbs failed, continue without these outliers
            
            final_outlier_indices = list(set(empty_outlier_indices + temp_non_float_indices + grubbs_outliers_original_indices))
            
            fp_cleaned_predictions_float, fp_cleaned_ground_truths_float, fp_cleaned_weights_float = [], [], []

            for idx in range(len(raw_predictions_str)):
                if idx >= fp_task_num: break
                if idx in final_outlier_indices: continue
                try:
                    fp_cleaned_predictions_float.append(float(raw_predictions_str[idx]))
                    fp_cleaned_ground_truths_float.append(current_fp_ground_truths_float[idx])
                    fp_cleaned_weights_float.append(current_fp_initial_weights_float[idx])
                except (ValueError, TypeError): continue
            
            current_fp_t_paired_test, current_fp_ks_test = None, None

            if fp_cleaned_weights_float:
                df = pd.DataFrame({
                    'Ground_Truth': fp_cleaned_ground_truths_float,
                    'Predicted': fp_cleaned_predictions_float,
                    'Weights': fp_cleaned_weights_float 
                })

                truth_pooled = fixed_effects_pooled_estimate(df['Ground_Truth'], df['Weights'])
                pred_pooled = fixed_effects_pooled_estimate(df['Predicted'], df['Weights'])

                abs_diff = abs(truth_pooled - pred_pooled)
                percentage_diff = abs_diff / abs(truth_pooled) if abs(truth_pooled) > 1e-9 else (abs_diff if abs_diff > 0 else 0.0)

                if fp_valid_tasks_count > 0:
                    weighted_abs_diff_sum += abs_diff * fp_valid_tasks_count
                    weighted_percentage_diff_sum += percentage_diff * fp_valid_tasks_count
                    total_weight_for_meta_avg += fp_valid_tasks_count

                all_truths_for_meta.extend(fp_cleaned_ground_truths_float)
                all_preds_for_meta.extend(fp_cleaned_predictions_float)
                
                if len(fp_cleaned_ground_truths_float) > 1 and len(fp_cleaned_predictions_float) > 1:
                    t_stat, t_p, ks_stat, ks_p = stat_test(fp_cleaned_ground_truths_float, fp_cleaned_predictions_float)
                    if not (np.isnan(t_stat) or np.isnan(t_p) or np.isnan(ks_stat) or np.isnan(ks_p)):
                        current_fp_t_paired_test = {"t_stat": t_stat, "p_value": t_p}
                        current_fp_ks_test = {"ks_stat": ks_stat, "p_value": ks_p}
                        meta_t_stat_sum += t_stat; meta_t_pvalue_sum += t_p; meta_valid_t_test_count += 1
                        meta_ks_stat_sum += ks_stat; meta_ks_pvalue_sum += ks_p; meta_valid_ks_test_count +=1
            
            meta_forest_plot_stats.append({
                "plot_num": plot_num_counter, 
                "find_ratio_task_num": fp_task_num, 
                "valid_task_count": fp_valid_tasks_count, 
                "t_paired_test": current_fp_t_paired_test, 
                "ks_test": current_fp_ks_test
            })

        # After all forest plots for a meta-analysis
        final_meta_avg_abs_diff = weighted_abs_diff_sum / total_weight_for_meta_avg if total_weight_for_meta_avg > 0 else 0.0
        final_meta_avg_percentage_diff = weighted_percentage_diff_sum / total_weight_for_meta_avg if total_weight_for_meta_avg > 0 else 0.0

        gt_mean, gt_variance = (np.mean(all_truths_for_meta), np.var(all_truths_for_meta)) if all_truths_for_meta else (0.0, 0.0)
        pred_mean, pred_variance = (np.mean(all_preds_for_meta), np.var(all_preds_for_meta)) if all_preds_for_meta else (0.0, 0.0)
        
        avg_t_stat, avg_t_p = (meta_t_stat_sum / meta_valid_t_test_count, meta_t_pvalue_sum / meta_valid_t_test_count) if meta_valid_t_test_count > 0 else (0.0, 0.0)
        avg_ks_stat, avg_ks_p = (meta_ks_stat_sum / meta_valid_ks_test_count, meta_ks_pvalue_sum / meta_valid_ks_test_count) if meta_valid_ks_test_count > 0 else (0.0, 0.0)

        result_for_this_meta = {
            "pmid": pmid,
            "total_number": meta_level_valid_task_count, # Total valid (non-empty) tasks for this meta
            "avg_absolute_diff": final_meta_avg_abs_diff,
            "avg_percentage_diff": final_meta_avg_percentage_diff,
            "pred_mean": pred_mean,
            "pred_variance": pred_variance,
            "gt_mean": gt_mean,
            "gt_variance": gt_variance,
            "forest_plot_stat": meta_forest_plot_stats,
            "avg_pair_t_test": {"stat": avg_t_stat, "p_value": avg_t_p}, 
            "avg_ks_test": {"stat": avg_ks_stat, "p_value": avg_ks_p},
            "empty_number": meta_level_empty_num # Total empty responses for this meta
        }
        processed_meta_results.append(result_for_this_meta)
    
    return processed_meta_results

# Helper function to calculate overall performance metrics for a parser.
def _calculate_overall_parser_performance(list_of_evaluated_meta_results):
    if not list_of_evaluated_meta_results:
        return [0.0, 0.0, 0.0, 0.0]

    valid_results_for_sorting = [res for res in list_of_evaluated_meta_results if isinstance(res.get("avg_percentage_diff"), (int, float))]
    
    sorted_results = sorted(valid_results_for_sorting, key=lambda x: abs(x["avg_percentage_diff"]), reverse=True)
    
    neglect_count = int(len(sorted_results) * 0.1)
    results_for_overall_calc = sorted_results[neglect_count:]

    def _calc_weighted_stats_for_key(results_list, key_name):
        total_w = 0.0
        weighted_s = 0.0
        
        items_for_calc = []
        for r in results_list:
            val = r.get(key_name)
            weight = r.get("total_number") 
            if isinstance(val, (int, float)) and isinstance(weight, (int, float)) and weight > 0:
                 items_for_calc.append({"value": val, "weight": weight})

        if not items_for_calc: return 0.0, 0.0

        for item in items_for_calc:
            total_w += item["weight"]
            weighted_s += item["value"] * item["weight"]
        
        weighted_avg = weighted_s / total_w if total_w > 0 else 0.0
        
        variance_num = 0.0
        for item in items_for_calc:
            variance_num += item["weight"] * ((item["value"] - weighted_avg) ** 2)
        
        variance = variance_num / total_w if total_w > 0 else 0.0
        return weighted_avg, variance

    avg_abs_diff, var_abs_diff = _calc_weighted_stats_for_key(results_for_overall_calc, "avg_absolute_diff")
    avg_perc_diff, var_perc_diff = _calc_weighted_stats_for_key(results_for_overall_calc, "avg_percentage_diff")
    
    return [avg_abs_diff, var_abs_diff, avg_perc_diff, var_perc_diff]

def evaluate_fr(data_dir: str):
    input_json_path = os.path.join(data_dir, "fr_predictions.json") 

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            full_input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_json_path}")
        return 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    # Extract the 'results_by_parser' dictionary from the loaded data
    input_results_by_parser = full_input_data.get("results_by_parser", {})
    
    output_json_structure = {} # To store evaluated results for cc.json
    parser_overall_performance_data = {} # To store overall stats for terminal print

    for parser_name, list_of_meta_preds_for_parser in input_results_by_parser.items():
        # list_of_meta_preds_for_parser is List[MetaPredictionObject]
        # Each MetaPredictionObject has "pmid", "data": List[List[SampleDict]], "time_used"
        
        evaluated_meta_list_for_this_parser = _calculate_metrics_for_meta_list(list_of_meta_preds_for_parser)
        output_json_structure[parser_name] = evaluated_meta_list_for_this_parser
        
        # Calculate overall performance for this parser using the evaluated list
        overall_perf_metrics = _calculate_overall_parser_performance(evaluated_meta_list_for_this_parser)
        parser_overall_performance_data[parser_name] = {
            "Avg_Abs_Diff": overall_perf_metrics[0],
            "Var_Abs_Diff": overall_perf_metrics[1],
            "Avg_Percentage_Diff": overall_perf_metrics[2],
            "Var_Percentage_Diff": overall_perf_metrics[3]
        }

    # Save the structured results to cc.json
    output_cc_json_path = os.path.join(data_dir, 'cc.json')
    try:
        with open(output_cc_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_structure, f, indent=4, ensure_ascii=False)
        print(f"Detailed evaluation results saved to: {output_cc_json_path}")
    except IOError:
        print(f"Error: Could not write to {output_cc_json_path}")
    
    # Print overall performance summary for each parser to the terminal
    print("\n--- Overall Performance by Parser ---")
    if not parser_overall_performance_data:
        print("No parser data processed or found to display overall performance.")
    else:
        for parser_name, metrics in parser_overall_performance_data.items():
            print(f"Parser: {parser_name}")
            print(f"  Average Absolute Difference: {metrics['Avg_Abs_Diff']:.4f}")
            print(f"  Variance of Absolute Difference: {metrics['Var_Abs_Diff']:.4f}")
            print(f"  Average Percentage Difference: {metrics['Avg_Percentage_Diff']:.4%}") # Display as percentage
            print(f"  Variance of Percentage Difference: {metrics['Var_Percentage_Diff']:.4f}")
            print("-" * 30)
