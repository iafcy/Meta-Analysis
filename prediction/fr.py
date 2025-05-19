import os
import json
import tqdm
import time
from timeit import default_timer as timer
from model import Bot
from prompt import generate_fr_prompt
from utils import seconds_to_time_str

def load_data_to_prompt():
    with open('./data/fr_test.json', "r", encoding="utf-8") as file:
        parsers_data = json.load(file)

    for parser_name, meta_analyses_list in parsers_data.items():
        for meta_analysis in meta_analyses_list:
            for forest_plot_samples in meta_analysis['data']:
                for sample in forest_plot_samples:
                    sample['prompt'] = generate_fr_prompt(sample)
    
    return parsers_data

# If you would like to use LlamaIndex : Define the model = model.get_llamaindex_llm()
def predict_fr(model: Bot, output_dir: str):
    parsers_data = load_data_to_prompt()
    all_results = {}

    for parser_name, meta_analyses_list in parsers_data.items():
        parser_results = []

        for meta_analysis in tqdm.tqdm(meta_analyses_list, desc=f'Prediction for {parser_name}'):
            start_time = timer()

            for forest_plot_samples in tqdm.tqdm(meta_analysis['data'], desc=f'Forest Plots in {meta_analysis.get("pmid", "N/A")}', leave=False):
                for sample_item in forest_plot_samples:
                    messages = [{'role': 'user', 'content': sample_item['prompt']}]
                    response = model.query(messages=messages)

                    MAX_REASONABLE_LENGTH = 20

                    if len(response) > MAX_REASONABLE_LENGTH:
                        sample_item["prediction"] = "Empty Response"
                    elif response.strip() == "Empty Response":
                        sample_item["prediction"] = "Empty Response"
                    else:
                        sample_item["prediction"] = response.strip()
            
            meta_analysis['time_used'] = seconds_to_time_str(timer() - start_time)
            parser_results.append(meta_analysis)
        all_results[parser_name] = parser_results

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_data_structure = {
        'model_name': model.model_name,
        'results_by_parser': all_results
    }

    with open(os.path.join(output_dir, 'fr_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(output_data_structure, f, indent=4, ensure_ascii=False)