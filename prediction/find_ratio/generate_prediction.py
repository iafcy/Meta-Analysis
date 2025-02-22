import os
import json
import tqdm
from timeit import default_timer as timer
from ai_tool import find_ratio_text
from helper import get_txt_path_of_key, get_txt


def generate_prediction_python_parser(pdf_parser_name, model_name):
    with open('./data/or_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file) 
        
    model_name_file = ""
    if model_name == "claude-3-5-sonnet-20241022":
        model_name_file = "Claude-3.5"
    elif model_name == "gpt-4o-mini":
        model_name_file = "4o-mini"
    elif model_name == "gpt-4o":
        model_name_file = "4o"
        
    failed_cases = []
    results = []

    # test only one meta
    # data = [meta for meta in data if meta.get('pmid', "") == "21673296"]

    for meta in tqdm.tqdm(data, desc="Processing the meta_analysis", leave=False):
        # if meta.get('pmid', "") not in ["33811827", "37851443", "37983028"]:
        #     continue

        res = {
        "pmid" : "",
        "time_used" : 0,
        "token_used" : 0,
        "find_ratio_tasks" : 0
        }

        author_name_list = meta.get('author_name_list', {})
        meta_abstract = meta.get('summerize_abstract', "")

        time_start = timer()
        find_ratio_tasks = 0
        token_used = 0

        for forest_plot in tqdm.tqdm(meta.get('forest_plot_title', []), desc="Processing the forest_plot", leave=False):
            table_num = 0

            try:
                forest_plot_title = forest_plot.get('title', "")
                authors = forest_plot.get('author', [])
                find_ratio_tasks += len(authors)

                if "Predicted_ratio" not in forest_plot:
                    forest_plot["Predicted_ratio"] = []

                for author_key in tqdm.tqdm(authors, desc="Generating the predictions", leave=False):
                    if isinstance(author_key, list) and len(author_key) == 2:
                        actual_author_key = author_key[0]
                        additional_info = author_key[1]
                    else:
                        actual_author_key = author_key
                        additional_info = "NONE"

                    author_name = author_name_list.get(str(actual_author_key), "")
                    table_num += 1
                    txt_path = get_txt_path_of_key(meta['pmid'], author_name, pdf_parser_name)
                    key_full_text = get_txt(txt_path)
                    prediction, token = find_ratio_text(model_name, key_full_text, forest_plot_title, meta_abstract, additional_info)
                    forest_plot["Predicted_ratio"].append(prediction)
                    token_used += token

            except Exception as e:
                failed_cases.append((meta['pmid'], table_num))
                print(f"Error: {e}")
                break
        
        time_end = timer()

        res["pmid"] = meta['pmid']
        res["time_used"] = time_end - time_start
        res["token_used"] = token_used
        res["find_ratio_tasks"] = find_ratio_tasks

        results.append(res)

        if not os.path.exists("./prediction_results"):
            os.makedirs("./prediction_results")
        
        if not os.path.exists("./usage_results"):
            os.makedirs("./usage_results")

        with open(f'./prediction_results/{pdf_parser_name}_{model_name_file}.json', 'w', encoding='utf-8') as json_file:
                    json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        with open(f'./usage_results/{pdf_parser_name}_{model_name_file}.json', 'w', encoding='utf-8') as json_file: 
            json.dump(results, json_file, ensure_ascii=False, indent=4)

    with open("failed_cases.json", "w", encoding="utf-8") as json_file:
        json.dump(failed_cases, json_file, ensure_ascii=False, indent=4)




