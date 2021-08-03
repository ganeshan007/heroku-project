from re import template
import numpy as np
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import pandas as pd
import random
import logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel
import os
from enum import Enum
os.system('systemctl start elasticsearch')
os.system('systemctl enable elasticsearch')
os.system('curl -sX GET "localhost:9200/"')

corpus_path = "corpus.jsonl"
query_path = "queries.jsonl"
qrels_path = "qrels.tsv"

corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path, 
    query_file=query_path, 
    qrels_file=qrels_path).load_custom()


tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-base")
labels_mapping = ['contradiction','entailment','neutral']

app = FastAPI()

@app.get('/predict',response_class=HTMLResponse)
def take_input():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Enter claim to be tested" />
        <input type="submit" />'''





def get_all_results(q):
    #### Provide parameters for elastic-search
    hostname = "localhost" 
    index_name = "ourdata" 
    initialize = False # True, will delete existing index with same name and reindex all documents

    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, q)
    return results


def get_top10_results(q):
    query_text = str(q['q'])
    results = get_all_results(q)
    top_k=10
    # print(results)
    ranking_scores = results['q']
    # print(query_id)
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # print("Query : %s\n" % queries[query_id])
    doc_ids = []
    labels = []
    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc_ids.append(doc_id)
        # print(corpus[doc_id].get('text'))
        passage_text = corpus[doc_id].get('text')
        enc = tokenizer(query_text,passage_text,return_tensors='pt',truncation=True)
        with torch.no_grad():
            label = torch.argmax(model(**enc).logits).item()
            labels.append(label)
        
        # print(doc_id+'-------------',label)
    
    # print('\n')
    # , corpus[doc_ids[0]].get('text')
    return doc_ids, labels


@app.post('/predict')
def predict(text:str = Form(...)):
    q = {'q':text}
    doc_ids, labels = get_top10_results(q)
    passages = []
    for id in doc_ids:
        print(id)
        passages.append(corpus[id].get('text'))

    return {
        'query' : text,
        'passages retrieved' : passages,
        'Labels' : labels
    }




