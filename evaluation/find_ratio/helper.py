import os
import json
import tiktoken


def combine_messages_to_prompt(messages):
    prompt = "\n".join(message['content'].strip() for message in messages if 'content' in message)
    return prompt

def encoder(model_name, txt):
    enc = tiktoken.encoding_for_model(f"{model_name}")
    return len(enc.encode(f"{txt}"))
                
def get_txt_path_of_key(pmid, name, pdf_parser):
    pmid = str(pmid)
    key_path = os.path.join("./data/OR_data", pmid, "key_references")
    key_full_text_path = os.path.join(key_path, pdf_parser, name + ".txt")
    
    return key_full_text_path

def get_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_prediction():
    with open('./data/or_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file) 
    
    for meta in data:
        for forest_plot in meta.get('forest_plot_title', []):
            forest_plot["Predicted_ratio"] = []
    
    with open('./data/or_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def extract_test_results(results):
    test_data_pmid_list = [
    '38879695',
    '37847505',
    '36346627',
    '36318133',
    '36264575',
    '35780792',
    '35438755',
    '33574573',
    '29279934',
    '21427375',
    '20951422',
    '38594408',
    '36306133',
    '28697252',
    '19531786']
    test_results = []
    for result in results:
        if result.get("pmid", "") in test_data_pmid_list:
            test_results.append(result)
    return test_results

