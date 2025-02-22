import os
import json
import tiktoken
from tqdm import tqdm
from timeit import default_timer as timer
from helper import clean_prediction


os.environ["OPENAI_API_KEY"] = "API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.org/v1/"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


def get_key_ref_path(pmid, name):
    pmid = str(pmid)
    key_path = os.path.join("./data/OR_data", pmid, "key_references")
    key_full_text_path = os.path.join(key_path, name + ".pdf")
    return key_full_text_path

def generate_prediction_llamaindex_default_parser(model_name):
    output_file_name = ""
    if model_name == "gpt-4o-mini":
        output_file_name = "gpt-4o-mini.json"
    elif model_name == "gpt-4o":
        output_file_name = "gpt-4o.json"
    else:
        output_file_name = "claude-3.5-sonnet.json"

    tokenizer_name = model_name
    if model_name not in ["gpt-4o-mini", "gpt-4o"]:
        tokenizer_name = "gpt-4o"
        
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(f"{tokenizer_name}").encode
    )
    llm = OpenAI(model=f"{model_name}")

    Settings.callback_manager = CallbackManager([token_counter])

    results = []

    with open('./data/or_data.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for meta in tqdm(data, desc="Processing the meta_analysis", leave=False):

        res = {
        "pmid" : "",
        "time_used" : 0,
        "token_used" : 0,
        "find_ratio_tasks" : 0
        }

        author_name_list = meta.get('author_name_list', {})
        meta_abstract = meta.get('summerize_abstract', "")

        fails = []

        find_ratio_tasks = 0
        start_time = timer()

        for forest_plot in tqdm(meta.get('forest_plot_title', []), desc="Processing the forest_plot", leave=False):
            forest_plot_title = forest_plot.get('title', "")
            authors = forest_plot.get('author', [])
            prediction = forest_plot.get('Predicted_ratio', [])

            find_ratio_tasks += len(authors)

            for author_key in tqdm(authors, desc="Generating the predictions", leave=False):
                try:
                    if isinstance(author_key, list) and len(author_key) == 2:
                        actual_author_key = author_key[0]
                        additional_info = author_key[1]
                    else:
                        actual_author_key = author_key
                        additional_info = "NONE"

                    author_name = author_name_list.get(str(actual_author_key), "")

                    key_ref_path = get_key_ref_path(meta['pmid'], author_name)

                    # Load pdf
                    # pdf will be chunked using SentenceSplitter by default
                    # Chunk size is 1024 and chunk overlap is 200 tokens by default
                    documents = SimpleDirectoryReader(input_files=[key_ref_path]).load_data()

                    index = VectorStoreIndex.from_documents(documents, llm=llm)

                    query_engine = index.as_query_engine(verbose=True)

                    # Change to add additional info if there is any
                    query_prompt = ( f""" 
                        You are a medical expert that excels in extracting context specific statistic odds ratio (OR), risk ratio (RR), hazard ratio (HR) from clinical literatures. The derived ratios should be strictly selected under the analytic goal of each specified medical meta-analysis. Find the best matching ratio that related to the clinical aim of the forest_plot_title and meta_abstract. Detailly check the Abstract, Methods, Results and Conclusion and don't make up any value as the output.
                        
                        Information Provided: 
                        (1) Meta-Analysis Abstract
                        {meta_abstract}

                        (2) Forest Plot Table Title: 
                        {forest_plot_title}

                        (3) Additional Information:
                        {additional_info}

                        To enable the best alignment between extracted ratio and the specific clinical aim. 
                        You are strictly regulated by the following principles: 

                        1: Extract the ratio as numeric form without additional commentary or explanation.
                        2: You should only return value that is from the key reference.
                        3: If no relevant ratio is found, return the most closely related numerical value available from the provided content and the output should be a single, standalone number e.g 1.50, 2.00, 3.52, etc. 
                        """   
                    )

                    response = query_engine.query(query_prompt)
                    prediction.append(response.response)

                except Exception as e:
                    fail_m = next((m for m in fails if m['pmid'] == meta['pmid']), None)
                    if fail_m == None:
                        fails.append({
                            'pmid': meta['pmid'],
                            'refs': [author_name]
                        })
                    else:
                        fail_m['refs'].append(author_name)

                    print("Error: ", e)

        end_time = timer()
        time_cost = end_time - start_time
        token = token_counter.total_llm_token_count

        res["pmid"] = meta['pmid']
        res["time_used"] = time_cost
        res["token_used"] = token
        res["find_ratio_tasks"] = find_ratio_tasks

        results.append(res)
        token_counter.reset_counts()

        if not os.path.exists("./prediction_results"):
            os.makedirs("./prediction_results")
        
        if not os.path.exists("./usage_results"):
            os.makedirs("./usage_results")

        with open(f"./prediction_results/llamaIndex_{output_file_name}", 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
            
        with open(f"./usage_results/llamaIndex_{output_file_name}", 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)




