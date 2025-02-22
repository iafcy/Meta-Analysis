import os
from openai import OpenAI
from helper import encoder, combine_messages_to_prompt


client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="https://api.chatanywhere.org/v1/"
)

def find_ratio_text(model_name, key_full_text, forest_plot_title, meta_abstract, additional_information):
    try:

        messages = [
            {
                'role': "system",
                'content': """
                    You are a medical expert that excels in extracting context specific statistic odds ratio (OR), risk ratio (RR), hazard ratio (HR) from clinical literatures. The derived ratios should be strictly selected under the analytic goal of each specified medical meta-analysis. Find the best matching ratio that related to the clinical aim of the forest_plot_title and meta_abstract. Detailly check the Abstract, Methods, Results and Conclusion and don't make up any value as the output.
                """
            },
            {
                'role': "user",
                'content': f"""
                    Information Provided: 
                    (1) Meta-Analysis Abstract
                    {meta_abstract}

                    (2) Forest Plot Table Title: 
                    {forest_plot_title}

                    (3) Additional Information:
                    {additional_information}

                    (4) Key Reference Full Text:
                    {key_full_text}

                    To enable the best alignment between extracted ratio and the specific clinical aim. 
                    You are strictly regulated by the following principles: 

                    1: Extract the ratio as numeric form without additional commentary or explanation.
                    2: You should only return value that is from the key reference.
                    3: If no relevant ratio is found, return the most closely related numerical value available from the provided content and the output should be a single, standalone number e.g 1.50, 2.00, 3.52, etc. 
                    4: No additional text should be included in the output. 
                """
            }
        ]

        prompt = combine_messages_to_prompt(messages)

        chat_completion = client.chat.completions.create(
            model=f"{model_name}",
            messages=messages,
            temperature=0
        )

        if model_name == 'claude-3-5-sonnet-20241022':
            model_name = 'gpt-4o'

        return chat_completion.choices[0].message.content, encoder(f"{model_name}", prompt)

    except Exception as e:
        print(f"Error when finding Ratio: {e}")
        return []
    
    
