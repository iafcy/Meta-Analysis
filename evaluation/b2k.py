import json
from util import time_str_to_seconds, seconds_to_time_str

def evaluate_b2k(dir: str):
    results = json.load(open(f'{dir}/b2k.json'))

    precision = []
    recall = []
    f1 = []
    time_used = []

    for meta_analysis in results['results']:
        tp, fp, tn, fn = 0, 0, 0, 0

        for x in meta_analysis['data']:
            is_key = True if x['truth'] == "True" else False

            if x['pred'] == True and is_key == True:
                tp += 1
            elif x['pred'] == False and is_key == False:
                tn += 1
            elif x['pred'] == True and is_key == False:
                fp += 1
            elif x['pred'] == False and is_key == True:
                fn += 1

        ma_precision = tp / (tp + fp) if tp + fp > 0 else 0
        ma_recall = tp / (tp + fn) if tp + fn > 0 else 0
        ma_f1 = (2 * ma_precision * ma_recall) / (ma_precision + ma_recall) if ma_precision + ma_recall > 0 else 0

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

    print(f'Base to key - {results["model_name"]}')
    print(f'Average precision                  : {precision:.4f}')
    print(f'Average recall                     : {recall:.4f}')
    print(f'Average f1                         : {f1:.4f}')
    print(f'Average time used per Meta-analysis: {results["time_used"]}')

    with open(f'{dir}/b2k.json', 'w') as f:
        json.dump(results, f, indent=4)