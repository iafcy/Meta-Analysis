import requests
from xml.etree import ElementTree
import os
from tqdm import tqdm
from dotenv import load_dotenv
from model import Bot
from prompt import generate_search_prompt, generate_refine_search_prompt

load_dotenv()

MIN_YEAR = 1900
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY')

def search_pubmed_pmids(search_term, max_year=None):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': "pubmed",
        'term': search_term,
        'retmax': 1000,
        'retmode': 'xml',
        'sort': 'relevance'
    }

    if max_year:
        params['maxdate'] = max_year
        params['mindate'] = MIN_YEAR
    
    if PUBMED_API_KEY:
        params['api_key'] = PUBMED_API_KEY

    response = requests.get(url, params=params)
    xml_response = ElementTree.fromstring(response.content)
    return [e.text for e in xml_response.find("IdList").findall("Id")]

def search_pubmed_single_article(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'retmode': 'xml',
    }
    if PUBMED_API_KEY:
        params['api_key'] = PUBMED_API_KEY
    
    response = requests.get(url, params=params)
    xml_response = ElementTree.fromstring(response.content)

    try:
        title = xml_response.find(".//ArticleTitle").text
    except Exception as e:
        title = None

    doi = None
    try:
        for article_id in xml_response.find('.//PubmedData').find('ArticleIdList').findall('ArticleId'):
            if article_id.get('IdType') == 'doi':
                doi = article_id.text
                break
    except Exception as e:
        doi = None

    abstract_elements = xml_response.findall(".//AbstractText")
    abstract = ''.join([f"""## {a.attrib['Label']}\n{a.text}\n""" if "Label" in a.attrib else f"{a.text}\n" for a in abstract_elements])

    date_element = xml_response.find(".//PubMedPubDate[@PubStatus='pubmed']")
    publication_year = date_element.find('.//Year').text

    authors = [author.find("LastName").text for author in xml_response.findall(".//Author")]
    if len(authors) == 1:
        citation = f"{authors[0]}, {publication_year}"
    elif len(authors) == 2:
        citation = f"{authors[0]} & {authors[1]}, {publication_year}"
    else:
        citation = f"{authors[0]} et al., {publication_year}"

    return doi, title, abstract, publication_year, citation

def get_pubmed_references_data(search_query, max_year: int | None = None):
    pmids = search_pubmed_pmids(search_query, max_year)

    results = []
    
    for pmid in tqdm(pmids, desc="Retrieving papers data"):
        doi, title, abstract, publication_year, citation = search_pubmed_single_article(pmid)
        results.append({
            'doi': doi,
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'publication_year': publication_year,
            'citation': citation
        })

    return results

def generate_pubmed_query(model: Bot, title: str, criteria: str, max_year: int | None = None):
    messages = [{
        'role': 'user',
        'content': generate_search_prompt({
            "title": title,
            "citeria": criteria,
            "platform": "Pubmed"
        })
    }]
    query = model.query(messages)

    n = len(search_pubmed_pmids(query, max_year))
    if n < 100:
        return query
    
    messages.append({
        'role': 'assistant',
        'content': query,
    })
    messages.append({
        'role': 'user',
        'content': generate_refine_search_prompt(n)
    })
    
    refined_query = model.query(messages)
    return refined_query
