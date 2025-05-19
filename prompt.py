def generate_search_prompt(data):
    prompt = (
        """You are a medical expert.\n"""
        f"""You are going to write a meta-analysis with the title "{data['title']}".\n"""
        f"""Generate a search query to find relevant papers in {data['platform']}.\n"""
        """Your search query should be broad and designed to retrieve a significant number of papers, while being relevant and adhere to the given inclusion and exclusion criteria.\n"""

        """Inclusion and exclusion criteria:\n"""
        f"""{data['criteria']}\n\n"""

        """Only return the search query."""
    )

    return prompt

def generate_refine_search_prompt(num):
    prompt = (
        f"""The search result only has {num} papers.\n"""
        """Please refine your search query so that it is broad enough to search for a significant number of papers.\n"""
        """Only return the search query."""
    )

    return prompt

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

def genereate_char_prompt(data):
    prompt = (
        "You are a medical expert."
        f"""You are going to write a meta-analysis with the title "{data['title']}"\n\n"""

        "You are going to make the Characteristics Table for the meta-analysis.\n"
        "You will be given the content of an included study and the columns you need to fill in.\n"
        "Extract the relevant information based on the given content.\n"
        f"Columns in the Characteristics Table: {data['columns']}\n\n"
        
        "There are some things you need to note:\n"
        """1. You are only allowed to fill in the content of each cell separated by "|" without any explanation.\n"""
        "2. Your result should only include the summarized result, and should not include the column title.\n"
        "3. If you cannot find the result in the given string, just return NR.\n"
        "4. Some results may need to be calculated from the information.\n"
        """5. You should focus on the "Methods" and "Results" section.\n"""
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

def generate_extract_section_prompt(data):
    return [
        {
            'role': "system",
            'content': (
                "You are a helpful assistant.\n"
                f"You will be given an article, please help me extract the \"{data['section']}\" section.\n"
                "You must not modify the content."
            )
        },
        {
            "role": "user",
            "content": data['full_text'],
        }
    ]

def generate_summary_prompt(data):
    return (
        'You are a medical expert.\n'
        f"You are writing a meta-analysis with the title {data['ma_title']}\n."
        'You are going to conduct GRADE assessment in the meta-analysis and you are reviewing one of the included studies.\n'
        'Summarize the key points from the given study based on its title, abstract and discussion, which will be used for GRADE assessment.\n\n'
        f"Title: {data['title']}\n"
        f"Abstract: {data['abstract']}\n"
        f"Discussion: {data['discussion']}"
    )

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

def generate_fr_prompt(data):
    prompt = (
        "You are a medical expert that excels in extracting context specific statistic odds ratio (OR), risk ratio (RR), hazard ratio (HR) from clinical literatures. The derived ratios should be strictly selected under the analytic goal of each specified medical meta-analysis. Find the best matching ratio that related to the clinical aim of the forest_plot_title and meta_abstract. Detailly check the Abstract, Methods, Results and Conclusion and don't make up any value as the output. \n\n"

        "Information Provided: \n"
        f"(1) Meta-Analysis Abstract: \n"
        f"{data['meta_abstract']} \n\n"

        f"(2) Forest Plot Table Title: \n"
        f"{data['forest_plot_title']} \n\n"

        f"(3) Additional Information: \n"
        f"{data['additional_information']} \n\n"

        f"(4) Key Reference Full Text: \n"
        f"{data['key_full_text']} \n\n"

        "To enable the best alignment between extracted ratio and the specific clinical aim. You are strictly regulated by the following principles: \n"
        "1: Extract the ratio as numeric form without additional commentary or explanation. \n"
        "2: You should only return value that is from the key reference. \n"
        "3: If no relevant ratio is found, return the most closely related numerical value available from the provided content and the output should be a single, standalone number e.g 1.50, 2.00, 3.52, etc. \n"
        "4: No additional text should be included in the output. \n"
    )

    return prompt