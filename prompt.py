def generate_b2k_prompt(data):
    prompt = (
        "You are a medical expert.\n"
        f"""You are going to write a meta-analysis with the title "{data['meta_analysis_title']}".\n"""
        "Based on the inclusion and exclusion criteria, determine whether the given study should be included in the meta-analysis by reading the title and abstract of the study.\n\n"

        "Inclusion and exclusion criteria:\n"
        f"{data['criteria']}\n\n"

        "Title:\n"
        f"{data['title']}\n\n"

        "Abstract:\n"
        f"{data['abstract']}\n\n"

        f"""Only return "True" or "False".\n"""
    )

    return prompt

def generate_cc_prompt(data):
    prompt = (
        "You are a medical expert.\n"
        f"""You are going to write a meta-analysis with the title "{data['meta_analysis_title']}".\n"""
        f"""The major question of intend of the meta-analysis is "{data['meta_analysis_question']}".\n\n"""

        "The inclusion and exclusion criteria of the meta-analysis is:\n"
        f"{data['criteria']}\n\n"

        f"""You are going to conduct a subgroup selection for the subgroup "{data['subgroup']}".\n"""
        "Based on the provided title and abstract of a study, determine whether the given study should be included in the subgroup.\n"
        "Only return True or False.\n\n"

        "Title:\n"
        f"{data['key_reference_title']}\n\n"

        "Abstract:\n"
        f"{data['key_reference_abstract']}"
    )

    return prompt

def generate_qa_prompt(data):
    prompt = (
        "You are a medical expert.\n"
        f"""You are going to write a meta-analysis with the title "{data['meta_analysis_title']}".\n"""
        "You are given the Title, Abstract, Methods, and Results sections of one of the included study, and you need to conduct quality assessment on it.\n\n"
                             
        "Title:\n"
        f"{data['title']}\n\n"
        
        "Abstract:\n"
        f"{data['abstract']}\n\n"
                                
        "Methods:\n"
        f"{data['methods']}\n\n"
        
        "Results:\n"
        f"{data['results']}\n\n"

        "You need to fill in the below column in the quality assessment table.\n"
        "Judge only based on the given content.\n\n"

        f"Column: {data['column_title']}\n\n"

        f"{data['instruction']}\n\n"

        f"Only return the {data['type']} you filled in the cell."
    )

    return prompt

def generate_grade_prompt(data):
    tiab_list = []
    for ref in data['tiab_list']:
        tiab_list.append(f"Source: {ref['source']}\nSummary: {ref['summary']}\n")
    tiab_list = "\n".join(tiab_list)

    options = "\n".join([f"- {label}" for label in data['labels']])

    prompt = (
        "You are a medical expert.\n"
        f"""You are going to write a meta-analysis with the title "{data['meta_analysis_title']}".\n"""
        f"""The question of intend of the meta-analysis is "{data['question']}".\n"""
        f"""Below are the included studies associated with the {data['title_name']} "{data['title']}".\n\n"""

        f"{tiab_list}\n"
        f"{data['characteristics_table']}\n"
        f"{data['qa_table']}\n"
        f"{data['summary_table']}\n\n"

        "You need to judge the quality of evidence using the GRADE approach.\n" 
        "You need to fill in the below column in the GRADE quality of evidence table.\n"
        f"Column: {data['column_title']}\n"
        "Choose the most suitable label from the following:\n"
        f"""{options}\n\n"""

        "Only return the label you chosen."
    )

    return prompt