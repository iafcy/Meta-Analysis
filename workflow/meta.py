import os
import json
from tqdm import tqdm
from model import Bot
from prompt import (
    generate_b2k_prompt,
    generate_qa_prompt,
    generate_cc_prompt,
    generate_grade_prompt
)
from workflow.sr import (
    search_pubmed_pmids,
    search_pubmed_single_article,
    generate_pubmed_query
)
from workflow.char import extract_characteristics
from workflow.qa import RubricsColumn
from workflow.grade import (
    format_grade_characteristics,
    format_quality_table,
)
from workflow.classes import (
    Reference,
    KeyReference,
    Subgroup
)

class MetaAnalysis():
    """
    Usage:
        ma = MetaAnalysis(
            title="Title",
            criteria="Criteria",
            question="Question",
            output_dir="./workflow/test"
        )
        ma.generate_pubmed_query(model: Bot, max_year: int | None = None)
        ma.search_references(max_year: int | None = None)
        ma.base_to_key(model: Bot)
        ma.download_references()
        ma.extract_sections(model: Bot)
        ma.characteristics_table(model: Bot, columns: list[str])
        ma.quality_assessment(model: Bot, rubrics: list[RubricsColumn])
        ma.set_subgroups(subgroups: list[str])
        ma.characteristics_classification(model: Bot)
        ma.find_ratio(model: Bot)
        ma.grade_assessment(model: Bot)
    """

    def __init__(
        self,
        title: str,
        criteria: str,
        question: str,
        output_dir: str,
    ):
        self.title = title
        self.criteria = criteria
        self.question = question
        self.output_dir = output_dir
        self.base_references: list[Reference] = []
        self.key_references: list[KeyReference] = []
        self.subgroups: list[Subgroup] = []
        self.query: str = ""
        self.characteristics_columns: list[str] | None = None
        self.qa_rubrics: list[RubricsColumn] | None = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f"{output_dir}/key_references"):
            os.makedirs(f"{output_dir}/key_references")

    @classmethod
    def resume(cls, path) -> "MetaAnalysis":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' not found")
    
        try:
            with open(f"{path}/data.json", "r") as f:
                data = json.load(f)

            ma = cls(
                title=data["title"],
                criteria=data["criteria"],
                question=data["question"],
                output_dir=path
            )

            if "query" in data.keys() and not (data["query"] == ""):
                ma.query = data["query"]
            else:
                print("Resuming. Start from generating query for 'Search reference'")
                return ma

            if "base_references" in data.keys() and len(data["base_references"]) > 0:
                ma.base_references = [Reference(
                    doi=ref["doi"],
                    pmid=ref["pmid"],
                    title=ref["title"],
                    abstract=ref["abstract"],
                    publication_year=int(ref["publication_year"]),
                    citation=ref["citation"]
                ) for ref in data["base_references"]]
            else:
                print("Resuming. Start from searching base references for 'Search reference'")
                return ma
            
            if "key_references" in data.keys() and len(data["key_references"]) > 0:
                ma.key_references = [KeyReference(
                    reference=Reference(
                        doi=ref["doi"],
                        pmid=ref["pmid"],
                        title=ref["title"],
                        abstract=ref["abstract"],
                        publication_year=int(ref["publication_year"]),
                        citation=ref["citation"]
                    ),
                    index=int(ref["index"])
                ) for ref in data["key_references"]]
            else:
                print("Resuming. Start from 'Base to key'")
                return ma
            
            for key_ref, saved_ref in zip(ma.key_references, data["key_references"]):
                key_ref.dir = saved_ref["dir"]
                key_ref.pdf_path = saved_ref["pdf_path"]
                key_ref.md_path = saved_ref["md_path"]
                key_ref.set_characteristics(saved_ref["characteristics"])
                key_ref.set_quality(saved_ref["quality"])
            
            if any(not ref.dir for ref in ma.key_references):
                print("Resuming. Start from 'Download full text pdf'")
                return ma
            
            if all(os.path.exists(f"{ref.dir}/sections.json") for ref in ma.key_references):
                for ref in ma.key_references:
                    with open(f"{ref.dir}/sections.json", "r") as f:
                        sections = json.load(f)
                        ref.set_methods(sections["methods"])
                        ref.set_results(sections["results"])
                        ref.set_discussion(sections["discussion"])
                        ref.set_summary(sections["summary"])
            else:
                print("Resuming. Start from 'Extract sections'")
                return ma
            
            if all(ref.characteristics for ref in ma.key_references) and "characteristics_columns" in data.keys():
                ma.characteristics_columns = data["characteristics_columns"]
            else:
                print("Resuming. Start from 'Characteristics Table'")
                return ma
            
            if all(ref.quality for ref in ma.key_references) and "qa_rubrics" in data.keys():
                ma.qa_rubrics = [RubricsColumn(
                    title=col["title"],
                    options=col["options"],
                    type=col["type"]
                ) for col in data["qa_rubrics"]]
            else:
                print("Resuming. Start from 'Quality Assessment'")
                return ma
            
            if "subgroup" in data.keys() and len(data["subgroup"]) > 0:
                ma.set_subgroups([subgroup["name"] for subgroup in data["subgroup"]])
            else:
                print("Resuming. Start from 'Setting subgroups'")
                return ma
            
            if all(len(subgroup["key_references"]) > 0 for subgroup in data["subgroup"]):
                for subgroup, saved_subgroup in zip(ma.subgroups, data["subgroup"]):
                    for saved_ref in saved_subgroup["key_references"]:
                        key_ref = next((ref for ref in ma.key_references if ref.index == saved_ref["index"]), None)
                        if key_ref:
                            subgroup.add_reference(key_ref)
            else:
                print("Resuming. Start from 'Characteristics classification'")
                return ma
            
            if all(all("ratio" in ref.keys() for ref in subgroup["key_references"]) for subgroup in data["subgroup"]):
                for subgroup, saved_subgroup in zip(ma.subgroups, data["subgroup"]):
                    subgroup.set_ratio([ref["ratio"] for ref in saved_subgroup["key_references"]])
            else:
                print("Resuming. Start from 'Find ratio'")
                return ma
            
            if all(subgroup["grade"] for subgroup in data["subgroup"]):
                for subgroup, saved_subgroup in zip(ma.subgroups, data["subgroup"]):
                    subgroup.set_grade(saved_subgroup["grade"])
            else:
                print("Resuming. Start from 'GRADE assessment'")
                return ma
            
            return ma
                
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file '{path}/data.json' not found")

    def to_json(self):
        with open(f"{self.output_dir}/data.json", 'w') as f:
            data = {
                'title': self.title,
                'criteria': self.criteria,
                'question': self.question,
                'query': self.query,
                'base_references': [ref.to_dict() for ref in self.base_references],
                'key_references': [ref.to_dict() for ref in self.key_references],
                'subgroup': [{
                    "name": subgroup.name,
                    "key_references": [{
                        "index": ref.index,
                        "ratio": ratio
                    } for ref, ratio in zip(subgroup.key_references, subgroup.ratios)]
                    if len(subgroup.key_references) == len(subgroup.ratios) else [{
                        "index": ref.index,
                    } for ref in subgroup.key_references],
                    "grade": subgroup.grade
                } for subgroup in self.subgroups],
            }
            if self.characteristics_columns:
                data['characteristics_columns'] = self.characteristics_columns,
            if self.qa_rubrics:
                data['qa_rubrics'] = [col.to_dict() for col in self.qa_rubrics],
            json.dump(data, f, indent=4)

    def generate_pubmed_query(self, model: Bot, max_year: int | None = None):
        query = generate_pubmed_query(model, self.title, self.criteria, max_year)
        self.query = query

    def search_references(self, max_year: int | None = None):
        pmids = search_pubmed_pmids(self.query, max_year)

        for pmid in tqdm(pmids, desc="Retrieving references data"):
            doi, title, abstract, publication_year, citation = search_pubmed_single_article(pmid)
            self.base_references.append(Reference(
                doi=doi,
                pmid=pmid,
                title=title,
                abstract=abstract,
                publication_year=publication_year,
                citation=citation
            ))

    def base_to_key(self, model: Bot):
        i = 1

        for ref in tqdm(self.base_references, desc=f'Base to key'):
            messages = [{
                'role': 'user',
                'content': generate_b2k_prompt({
                    'meta_analysis_title': self.title,
                    'criteria': self.criteria,
                    'title': ref.title,
                    'abstract': ref.abstract
                })
            }]
            response = model.query(messages=messages)

            if 'True' in response and 'False' not in response:
                self.key_references.append(KeyReference(ref, i))
                i += 1
            
    def download_references(self):
        for ref in self.key_references:
            ref.download_full_text(f"{self.output_dir}/key_references")

    def extract_sections(self, model: Bot):
        for ref in self.key_references:
            ref.extract_sections(model)
            ref.summarize(model, self.title)
            ref.save_sections_to_json()

    def characteristics_table(self, model: Bot, columns: list[str]):
        self.characteristics_columns = columns
        for ref in self.key_references:
            if os.path.exists(ref.pdf_path):
                ref.set_characteristics(extract_characteristics(model, columns, self.title, ref.pdf_path))
            else:
                print(f"PDF does not exist, fail to extract characteristics for the reference ({ref.index}) with doi {ref.doi}, pmid {ref.pmid}, title: \"{ref.title}\"")

    def quality_assessment(self, model: Bot, rubrics: list[RubricsColumn]):
        self.qa_rubrics = rubrics

        for ref in self.key_references:
            scores = []

            for col in rubrics:
                messages = [{
                    'role': 'user',
                    'content': generate_qa_prompt({
                        'meta_analysis_title': self.title,
                        'title': ref.title,
                        'abstract': ref.abstract,
                        'methods': ref.methods,
                        'results': ref.results,
                        'column_title': col.title,
                        'instruction': col.instruction,
                        'type': col.type
                    })
                }]

                response = model.query(messages=messages)
                scores.append(response)

            ref.quality = scores

    def set_subgroups(self, subgroups: list[str]):
        for subgroup in subgroups:
            self.subgroups.append(Subgroup(subgroup))

    def characteristics_classification(self, model: Bot):
        for subgroup in self.subgroups:
            for ref in self.key_references:
                messages = [{
                    'role': 'user',
                    'content': generate_cc_prompt({
                        'meta_analysis_title': self.title,
                        'meta_analysis_question': self.question,
                        'criteria': self.criteria,
                        'subgroup': subgroup.name,
                        'key_reference_title': ref.title,
                        'key_reference_abstract': ref.abstract
                    })
                }]
                response = model.query(messages=messages)

                if 'True' in response and 'False' not in response:
                    subgroup.add_reference(ref)
                
    def find_ratio(self, model: Bot):
        for subgroup in self.subgroups:
            subgroup.set_ratio([1.1])

    def grade_assessment(self, model: Bot):
        for subgroup in self.subgroups:
            tiab_list = [{ 'source': ref.citation, 'summary': ref.summary } for ref in subgroup.key_references]
            sources = [ref.citation for ref in subgroup.key_references]
            qa_scores = [ref.quality for ref in subgroup.key_references]
            characteristics_table = [ref.characteristics for ref in subgroup.key_references]

            messages = [{
                'role': 'user',
                'content': generate_grade_prompt({
                    'meta_analysis_title': self.title,
                    'question': self.question,
                    'title_name': "subgroup",
                    'title': subgroup.name,
                    'tiab_list': tiab_list,
                    'characteristics_table': format_grade_characteristics(
                        self.characteristics_columns,
                        sources,
                        characteristics_table
                    ),
                    'qa_table': format_quality_table(
                        self.qa_rubrics,
                        sources,
                        qa_scores
                    ),
                    'summary_table': None,
                    'column_title': "Certainty of Evidence",
                    'labels': ["High", "Moderate", "Low", "Very low"]
                })
            }]
            response = model.query(messages=messages)

            subgroup.set_grade(response)
