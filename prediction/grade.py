import os
import json
from tqdm import tqdm
import time
from timeit import default_timer as timer
from model import Bot
from prompt import generate_grade_prompt
from util import seconds_to_time_str

def load_data_to_prompt():
    with open('./data/grade_test.json', 'r') as file:
        data = json.load(file)
    
    for meta_analysis in data:
        for x in meta_analysis['data']:
            x['prompt'] = generate_grade_prompt(x)
    
    return data

def predict_grade(model: Bot, output_dir: str):
    data = load_data_to_prompt()

    for meta_analysis in tqdm(data, desc='Prediction for GRADE'):
        start_time = timer()

        for x in tqdm(meta_analysis['data'], desc=f'{meta_analysis["pmid"]}'):
            messages = [{'role': 'user', 'content': x['prompt']}]
            response = model.query(messages=messages)
            
            x['pred'] = response

        end_time = timer()

        meta_analysis['time_used'] = seconds_to_time_str(end_time - start_time)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/grade.json', 'w') as f:
        json.dump({
            'model_name': model.model_name,
            'results': data
        }, f, indent=4)
