import json
from util import time_str_to_seconds, seconds_to_time_str

def evaluate_grade(dir: str):
    results = json.load(open(f'{dir}/grade.json'))

    accuracy = []
    time_used = []

    for meta_analysis in results['results']:
        correct = 0

        for x in meta_analysis['data']:
            correct += 1 if x['pred'] == x['truth'] else 0

        ma_accuracy = correct / len(meta_analysis['data'])

        meta_analysis['accuracy'] = ma_accuracy

        accuracy.append(ma_accuracy)
        time_used.append(time_str_to_seconds(meta_analysis['time_used']))

    accuracy = sum(accuracy) / len(accuracy)
    time_used = sum(time_used) / len(time_used)

    results['accuracy'] = accuracy
    results['time_used'] = seconds_to_time_str(time_used)

    print(f'GRADE - {results["model_name"]}')
    print(f'Average accuracy                   : {accuracy:.4f}')
    print(f'Average time used per Meta-analysis: {results["time_used"]}')

    with open(f'{dir}/grade.json', 'w') as f:
        json.dump(results, f, indent=4)
