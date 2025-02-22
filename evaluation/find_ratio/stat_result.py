from scipy.stats import ttest_rel, ks_2samp


# Paired t-test 
def paired_t_test(ground_truth, prediction):
    t_stat, p_value = ttest_rel(ground_truth, prediction)
    return t_stat, p_value

# Kolmogorov-Smirnov test
def ks_test(ground_truth, prediction):
    stat, pvalue = ks_2samp(ground_truth, prediction)
    return stat, pvalue

def stat_test(ground_truth, prediction):
    # calculate the statistics
    t_stat, t_pvalue = paired_t_test(ground_truth, prediction)
    ks_stat, ks_pvalue = ks_test(ground_truth, prediction)

    return [t_stat, t_pvalue, ks_stat, ks_pvalue]







