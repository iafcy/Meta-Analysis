import json
from util import time_str_to_seconds, seconds_to_time_str

def evaluate_cc(dir: str):
    results = json.load(open(f'{dir}/cc.json'))

    precision = []
    recall = []
    f1 = []
    time_used = []

    for meta_analysis in results['results']:
        subgroup_performance = {}
        ma_precision = []
        ma_recall = []
        ma_f1 = []

        for x in meta_analysis['data']:
            if x['subgroup'] not in subgroup_performance.keys():
                subgroup_performance[x['subgroup']] = {
                    'tp': 0,
                    'tn': 0,
                    'fn': 0,
                    'fp': 0
                }

            is_key = True if x['truth'] == "True" else False

            if x['pred'] == True and is_key == True:
                subgroup_performance[x['subgroup']]['tp'] += 1
            elif x['pred'] == False and is_key == False:
                subgroup_performance[x['subgroup']]['tn'] += 1
            elif x['pred'] == True and is_key == False:
                subgroup_performance[x['subgroup']]['fp'] += 1
            elif x['pred'] == False and is_key == True:
                subgroup_performance[x['subgroup']]['fn'] += 1

        for plot in subgroup_performance.keys():
            tp = subgroup_performance[x['subgroup']]['tp']
            fn = subgroup_performance[x['subgroup']]['fn']
            fp = subgroup_performance[x['subgroup']]['fp']

            subgroup_precision = tp / (tp + fp) if tp + fp > 0 else 0
            subgroup_recall = tp / (tp + fn) if tp + fn > 0 else 0
            subgroup_f1 = (2 * subgroup_precision * subgroup_recall) / (subgroup_precision + subgroup_recall) if subgroup_precision + subgroup_recall > 0 else 0

            ma_precision.append(subgroup_precision)
            ma_recall.append(subgroup_recall)
            ma_f1.append(subgroup_f1)

        ma_precision = sum(ma_precision) / len(ma_precision)
        ma_recall = sum(ma_recall) / len(ma_recall)
        ma_f1 = sum(ma_f1) / len(ma_f1)

        meta_analysis['precision'] = ma_precision
        meta_analysis['recall'] = ma_recall
        meta_analysis['f1'] = ma_f1

        precision.append(ma_precision)
        recall.append(ma_recall)
        f1.append(ma_f1)
        time_used.append(time_str_to_seconds(meta_analysis['time_used']))

    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f1 = sum(f1) / len(f1)
    time_used = sum(time_used) / len(time_used)

    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['time_used'] = seconds_to_time_str(time_used)

    print(f'Characteristics classification - {results["model_name"]}')
    print(f'Average precision                  : {precision:.4f}')
    print(f'Average recall                     : {recall:.4f}')
    print(f'Average f1                         : {f1:.4f}')
    print(f'Average time used per Meta-analysis: {results["time_used"]}')

    with open(f'{dir}/cc.json', 'w') as f:
        json.dump(results, f, indent=4)
