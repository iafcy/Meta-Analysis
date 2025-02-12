import json

def evaluate_grade(dir: str):
    results = json.load(open(f'{dir}/grade.json'))

    accuracy = []

    for meta_analysis in results['results']:
        correct = 0

        for x in meta_analysis['data']:
            correct += 1 if x['pred'] == x['truth'] else 0

        ma_accuracy = correct / len(meta_analysis['data'])

        meta_analysis['accuracy'] = ma_accuracy

        accuracy.append(ma_accuracy)

    accuracy = sum(accuracy) / len(accuracy)

    results['accuracy'] = accuracy

    print(f'GRADE - {results["model_name"]}')
    print(f'Average accuracy: {accuracy:.4f}')

    with open(f'{dir}/grade.json', 'w') as f:
        json.dump(results, f, indent=4)
