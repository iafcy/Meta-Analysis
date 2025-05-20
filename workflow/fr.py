from prompt import generate_fr_prompt
from workflow.meta import Subgroup
from model import Bot
import os

MAX_LENGTH = len("Empty Response")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_ratio(model: Bot, ma_abstract: str, subgroup: Subgroup):
    ratios = []

    for ref in subgroup.key_references:
        if os.path.exists(ref.md_path):
            with open(ref.md_path, "r") as file:
                ref_content = file.read()
            
            messages = [{
                'role': 'user',
                'content': generate_fr_prompt({
                    "meta_abstract": ma_abstract,
                    "forest_plot_title": subgroup.name,
                    "additional_information": "None",
                    "key_full_text": ref_content
                })
            }]

            response = model.query(messages)
            if len(response) > MAX_LENGTH:
                response = "Empty Response"
                ratios.append(response)
            else:
                if is_number(response):
                    ratios.append(float(response))
                else:
                    ratios.append(response)
        else:
            ratios.append("Missing MD")

    return ratios

