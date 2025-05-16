import os
import json
from scidownl import scihub_download
import pymupdf4llm
from model import Bot
from prompt import generate_extract_section_prompt, generate_summary_prompt

class Reference():
    def __init__(
        self,
        doi: str | None,
        pmid: int | None,
        title: str,
        abstract: str,
        publication_year: int,
        citation: str
    ):
        self.doi = doi
        self.pmid = pmid
        self.title = title
        self.abstract = abstract
        self.publication_year = publication_year
        self.citation = citation

    def to_dict(self):
        return {
            "doi": self.doi,
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "publication_year": self.publication_year,
            "citation": self.citation
        }

class KeyReference(Reference):
    def __init__(self, reference: Reference, index: int):
        super().__init__(
            reference.doi,
            reference.pmid,
            reference.title,
            reference.abstract,
            reference.publication_year,
            reference.citation
        )
        self.index: int = index

        self.dir: str | None = None
        self.pdf_path: str | None = None
        self.md_path: str | None = None
        
        self.methods: str | None = None
        self.results: str | None = None
        self.discussion: str | None = None
        self.summary: str | None = None

        self.characteristics: str | None = None
        self.quality: list[int | str] | None = None

    def to_dict(self):
        return {
            "doi": self.doi,
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "publication_year": self.publication_year,
            "citation": self.citation,
            "index": self.index,
            "dir": self.dir,
            "pdf_path": self.pdf_path,
            "md_path": self.md_path,
            "characteristics": self.characteristics,
            "quality": self.quality
        }
        
    def download_full_text(self, output_dir: str):
        self.dir = f"{output_dir}/{self.index}"

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        pdf_path = f"{self.dir}/full_text.pdf"
        print(pdf_path)
        if self.doi:
            scihub_download(keyword=self.doi, paper_type="doi", out=pdf_path)
            if os.path.exists(pdf_path):
                self.pdf_path = pdf_path
        elif self.pmid:
            scihub_download(keyword=self.pmid, paper_type="pmid", out=pdf_path)
            if os.path.exists(pdf_path):
                self.pdf_path = pdf_path
        
        md_path = f"{self.dir}/full_text.md"
        if self.pdf_path:
            md_text = pymupdf4llm.to_markdown(self.pdf_path)
            with open(md_path, "w") as file:
                file.write(md_text)
                self.md_path = md_path
        else:
            print(f"Fail to download the reference ({self.index}) with doi {self.doi}, pmid {self.pmid}, title: \"{self.title}\"")
            self.md_path = None

    def set_methods(self, methods: str):
        self.methods = methods
    
    def set_results(self, results: str):
        self.results = results
    
    def set_discussion(self, discussion: str):
        self.discussion = discussion
    
    def set_summary(self, summary: str):
        self.summary = summary

    def set_characteristics(self, characteristics: str):
        self.characteristics = characteristics

    def set_quality(self, scores: list[int | str]):
        self.quality = scores

    def extract_sections(self, model: Bot):
        if self.md_path:
            with open(self.md_path, "r") as file:
                full_text = file.read()
            self.set_methods(model.query(generate_extract_section_prompt({ "section": "Methods", "full_text": full_text })))
            self.set_results(model.query(generate_extract_section_prompt({ "section": "Results", "full_text": full_text })))
            self.set_discussion(model.query(generate_extract_section_prompt({ "section": "Discussion", "full_text": full_text })))
        else:
            print(f"Fail to read and exract sections for the reference ({self.index}) with doi {self.doi}, pmid {self.pmid}, title: \"{self.title}\"")
    
    def summarize(self, model: Bot, ma_title: str):
        if not self.md_path or not self.discussion:
            print(f"Fail to summarize the reference ({self.index}) with doi {self.doi}, pmid {self.pmid}, title: \"{self.title}\"")
        else:
            self.set_summary(model.query(generate_summary_prompt({
                "ma_title": ma_title,
                "title": self.title,
                "abstract": self.abstract,
                "discussion": self.discussion
            })))

    def save_sections_to_json(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        with open(f"{self.dir}/sections.json", 'w') as f:
            json.dump({
                "methods": self.methods,
                "results": self.results,
                "discussion": self.discussion,
                "summary": self.summary
            }, f, indent=4)

class Subgroup():
    def __init__(self, name: str):
        self.name = name
        self.key_references: list[KeyReference] = []
        self.ratios: list[float | str] = []
        self.grade: str | None = None

    def add_reference(self, reference: KeyReference):
        self.key_references.append(reference)

    def set_ratio(self, ratios: list[float | str]):
        self.ratios = ratios

    def set_grade(self, grade: str):
        self.grade = grade
