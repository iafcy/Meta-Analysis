import os
import json
from tqdm import tqdm
import time
from timeit import default_timer as timer
from prompt import generate_b2k_prompt
from model import Bot

def load_data_to_prompt():
    with open('./data/b2k_test.json', 'r') as file:
        data = json.load(file)
    
    for meta_analysis in data:
        for x in meta_analysis['data']:
            x['prompt'] = generate_b2k_prompt(x)
    
    return data

def predict_b2k(model: Bot, output_dir: str):
    data = load_data_to_prompt()

    for meta_analysis in tqdm(data, desc='Prediction for Base to key:'):
        start_time = timer()

        for x in tqdm(meta_analysis['data'], desc=f'{meta_analysis["pmid"]}'):
            messages = [{'role': 'user', 'content': x['prompt']}]
            response = model.query(messages=messages)

            if 'True' in response and 'False' not in response:
                pred = True
            elif 'True' not in response and 'False' in response:
                pred = False
            else:
                print('Invalid response: ', response)
                pred = False
            
            x['pred'] = pred

        end_time = timer()

        meta_analysis['time_used'] = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/b2k.json', 'w') as f:
        json.dump({
            'model_name': model.model_name,
            'results': data
        }, f, indent=4)
