from pyserini.index.lucene import IndexReader
import xml.etree.ElementTree as ET
import math
import pandas as pd
import argparse
import numpy as np
from pyserini.search import SimpleSearcher
import re


"""
df : int
    Document frequency, the number of documents in the collection that contains the term.
cf : int
    Collection frequency, the number of times the term occurs in the entire collection.  This value is equal to the
    sum of all the term frequencies of the term across all documents in the collection.
"""


# Inverse Document Frequency
def idf(term):
    # Analyze the term
    analyzed = index_reader.analyze(term)

    if len(analyzed) == 0:
        return 0

    # print(f'The analyzed form of "{term}" is "{analyzed[0]}"')
    # Skip term analysis (already performed)
    df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)

    if df == 0:
        return 0

    idf = math.log(N / df)
    # print(f"Term: {term}, Inverse Document Frequency: {idf}")
    return idf


def avg_idf(q):
    query_split = q.split()
    if len(query_split) == 0:
        print("Empty query")
        exit()

    idf_query = [idf(term) for term in query_split]
    return sum(idf_query) / len(query_split)


def max_idf(q):
    query_split = q.split()
    idf_query = [idf(term) for term in query_split]
    return max(idf_query)


# Collection Query Similarity
def scq(term):
    # Analyze the term
    analyzed = index_reader.analyze(term)

    if len(analyzed) == 0:
        return 0

    # print(f'The analyzed form of "{term}" is "{analyzed[0]}"')
    # Skip term analysis (already performed)
    df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)

    if df == 0:   # Then cf is also 0 
        return 0

    idf = math.log(N / df)
    scq = (1 + math.log(cf)) * idf
    # print(f"Term: {term}, Scaled Collection Query: {scq}")
    return scq


def avg_scq(q):
    query_split = q.split()
    if len(query_split) == 0:
        print("Empty query")
        exit()

    scq_query = [scq(term) for term in query_split]
    return sum(scq_query) / len(query_split)


def max_scq(q):
    query_split = q.split()
    scq_query = [scq(term) for term in query_split]
    return max(scq_query)

def ictf(term):
    analyzed = index_reader.analyze(term)

    if len(analyzed) == 0:
        return 0

    df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)

    if cf == 0:   
        return 0

    ictf = math.log(N / cf)
    return ictf


def avg_ictf(q):
    query_split = q.split()
    if len(query_split) == 0:
        print("Empty query")
        exit()

    ictf_query = [ictf(term) for term in query_split]
    return sum(ictf_query) / len(query_split)


# Simplified Clarity Score
def scs(q):
    # We check that all the terms in the query are different
    query_split = q.split()
    if len(query_split) == 0:
        print("Empty query")
        exit()

    if len(query_split) != len(set(query_split)):
        print(f"WARNING - Repeated terms in the query {q}")

    return math.log(1/len(query_split)) + avg_ictf(q)



def sigma_1_term(analyzed_term, df):    
    weights = []

    # Find the documents that contain the given term
    postings_list = index_reader.get_postings_list(analyzed_term, analyzer=None)
    postings_list_len = len(postings_list)
    for posting in postings_list:
        fdt = posting.tf
        if fdt == 0: 
            print("error: fdt=0")
            exit()

        w = 1 + math.log(fdt) * math.log(1 + N/df)
        weights.append(w)

    weights = np.array(weights)
    #bar_w = sum(weights) / weights.size    
    bar_w = np.mean(weights)

    tot_sum = 0
    for w in weights:
        tot_sum += (w - bar_w)*(w - bar_w)
    sigma_1 = np.sqrt(tot_sum / df)
    return sigma_1, postings_list_len


def sigma_1_term_nopostings(not_analyzed_term):
    weights = []

    # Find the documents that contain the given term
    hits = searcher.search(not_analyzed_term)
    regex_term = fr"\b{not_analyzed_term}\b"
    pattern = re.compile(regex_term, flags=re.IGNORECASE)

    df = len(hits)
    if df == 0:
        return 0

    for i in range(len(hits)):
        raw_doc = hits[i].lucene_document.get('raw')
        matches = pattern.findall(raw_doc)
        fdt = len(matches)

        if fdt == 0:
            print("error: fdt=0")
            df = df - 1
            if df == 0:
                return 0
            continue

        w = 1 + math.log(fdt) * math.log(1 + N/df)
        weights.append(w)

    weights = np.array(weights)
    #bar_w = sum(weights) / weights.size    
    bar_w = np.mean(weights)

    tot_sum = 0
    for w in weights:
        tot_sum += (w - bar_w)*(w - bar_w)
    sigma_1 = np.sqrt(tot_sum / df)
    return sigma_1




def var(q):
    query_split = q.split()
    if len(query_split) == 0:
        print("Empty query")
        exit()

    valid_terms = len(query_split)
    failures = []

    sigma_1_list = []
    postings_list_lens = []
    for term in query_split:
        # Analyze the term
        analyzed = index_reader.analyze(term)

        if len(analyzed) == 0:
            valid_terms -= 1
            continue

        # print(f'The analyzed form of "{term}" is "{analyzed[0]}"')
        # Skip term analysis (already performed)
        df, cf = index_reader.get_term_counts(analyzed[0], analyzer=None)

        if df == 0:
            valid_terms -= 1
            continue

        #if corpus not in ["C4-2021", "C4-2022"]:
        #sigma_1_list.append(sigma_1_term(analyzed[0], df))
        #else:
        #sigma_1_list.append(sigma_1_term_nopostings(term))


        try:
            sigma_1, postings_list_len = sigma_1_term(analyzed[0], df)
            sigma_1_list.append(sigma_1)
            postings_list_lens.append(postings_list_len)
        except:
            print(f"Error computing sigma_1 for term: {term}")
            valid_terms -= 1
            failures.append(term)
            posting_list_lens.append(np.NaN)
            continue


    sigma_1_list = np.array(sigma_1_list)
    sigma_1 = np.sum(sigma_1_list)
    sigma_2 = sigma_1 / valid_terms
    sigma_3 = np.max(sigma_1_list)

    print(f"TOTAL: {len(failures)} failures")
    if len(failures) > 0:
        print(f"Failures: {failures}")


    return sigma_1, sigma_2, sigma_3, postings_list_lens




fields = {
    'misinfo-2020': 'title',
    'C4-2021': 'query',
    'C4-2022': 'query',
    'CLEF': 'title',
    'clef': 'title'
}

indexes = {
    'misinfo-2020': '/mnt/beegfs/home/xiana.carrera/irgroup/indexes/misinfo-2020',
    'C4-2021': '/mnt/beegfs/home/xiana.carrera/irgroup/indexes/C4',
    'C4-2022': '/mnt/beegfs/home/xiana.carrera/irgroup/indexes/C4',
    'CLEF': '/mnt/beegfs/home/xiana.carrera/irgroup/indexes/clueweb-b13',
    'clef': '/mnt/beegfs/home/xiana.carrera/irgroup/indexes/clueweb-b13'
}

topics_paths = {
    'misinfo-2020': '/mnt/beegfs/home/xiana.carrera/misinfo-2020/TREC_2020_BEIR/original-misinfo-resources-2020/topics/misinfo-2020-topics.xml',
    'C4-2021': '/mnt/beegfs/home/xiana.carrera/C4-2021/TREC_2021_BEIR/original-misinfo-resources-2021/topics/misinfo-2021-topics.xml',
    'C4-2022': '/mnt/beegfs/home/xiana.carrera/C4-2022/TREC_2022_BEIR/original-misinfo-resources-2022/topics/misinfo-2022-topics.xml',
    'CLEF': '/mnt/beegfs/home/xiana.carrera/CLEF/CLEF/queries2016_corregidas.xml',
    'clef': '/mnt/beegfs/home/xiana.carrera/CLEF/CLEF/queries2016_corregidas.xml'
}



metrics_funcs = {
    'avg_idf': avg_idf,
    'max_idf': max_idf,
    'avg_scq': avg_scq,
    'max_scq': max_scq,
    'avg_ictf': avg_ictf,
    'scs': scs,
    "var": var
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str, help="Corpus to work with")
    parser.add_argument("metric", type=str, help="Metric to compute")
    args = parser.parse_args()

    corpus = args.corpus
    metric = args.metric


    if corpus not in indexes:
        print(f"{corpus} is not a valid corpus name")
        exit()

    if metric not in metrics_funcs:
        print(f"{metric} is not a valid metric")
        exit()

    # Initialize the index reader from an index path
    index_reader = IndexReader(indexes[corpus])
    searcher = SimpleSearcher(indexes[corpus])


    # Get general statistics (total terms, number of documents, number of non-empty documents, unique terms)
    stats = index_reader.stats()
    print(stats) 
    N = stats['documents']

    field = fields[corpus]
    topics_path = topics_paths[corpus]
    topics_metric = {}
    with open(topics_path) as f:
        root = ET.parse(topics_path).getroot()

        topic_tag = 'query' if corpus == 'CLEF' else 'topic'
        for topic in root.findall(topic_tag):
            qid_tag = "id" if corpus == "CLEF" else "number"
            qid = topic.find(qid_tag).text
            query = topic.find(field).text

            if metric == "var":
                sigma_1, sigma_2, sigma_3, posting_list_lens = metrics_funcs[metric](query)
                print(f"QID: {qid}, query: {query}, sigma_1: {sigma_1}, sigma_2: {sigma_2}, sigma_3: {sigma_3}")
                print(f"Posting list lengths: {posting_list_lens}")
                topics_metric[qid] = [sigma_1, sigma_2, sigma_3, posting_list_lens]
                cols = ["sigma_1", "sigma_2", "sigma_3", "posting_list_lens"]

            else:
                metric_res = metrics_funcs[metric](query)
                print(f"QID: {qid}, query: {query}, {metric}: {metric_res}")
                topics_metric[qid] = metric_res
                cols = [metric]


    df = pd.DataFrame.from_dict(topics_metric, orient='index', columns=cols)
    df.to_csv(f'/mnt/beegfs/home/xiana.carrera/qpp_progs/{corpus}_{metric}_{field}.csv')
