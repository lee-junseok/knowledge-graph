import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import re

nlp = spacy.load('en_core_web_sm')

def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]

def get_entities(sent):
  ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

  #############################################################

    for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
          # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
                      prefix = prv_tok_text + " "+ tok.text

              # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                      modifier = prv_tok_text + " "+ tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier +" "+ prefix + " "+ tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier +" "+ prefix +" "+ tok.text

          ## chunk 5
          # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
      #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    #define the pattern
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

    matcher.add("matching_1", None, pattern)

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return(span.text)

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate a Knowledge Graph from a Webpage.")
    parser.add_argument('--no-graph', dest="no_graph",
                        default=False, action="store_true",
                        help="Do not draw a graph.")
    parser.add_argument('--no-save', dest="no_save",
                        default=False, action="store_true",
                        help="Do not save a triple data as a file.")
    parser.add_argument('-u', '--url', type=str, default=None,
                        help="Provide an URL containing text to parse.")
    parser.add_argument('-s', '--source', type=str, default=None, nargs='*',
                        help="Draw a Knowledge Graph only for the given source word.")
    parser.add_argument('-t', '--target', type=str, default=None, nargs='*',
                        help="Draw a Knowledge Graph only for the given target word.")
    parser.add_argument('-e', '--edge', type=str, default=None, nargs='*',
                        help="Draw a Knowledge Graph only for the given edge word.")
    parser.add_argument('-f', '--from-file', dest='file', default='',
                        help="Give a file path and draw a Knowledge Graph plot from the file.")

    args = parser.parse_args()
    if args.file:
        kg_df = pd.read_csv(args.file)
    else:
        headers = {'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"}
        url = args.url
        page = requests.get(url, headers = headers)
        soup = BeautifulSoup(page.text, 'lxml') #using lxml parser. You can use Python’s html.parser.
        paragraphs = []
        for i, item in enumerate(soup.find_all("p")):
            paragraphs.append(item.text)
        text = ''.join(paragraphs)
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = text.rstrip("\n")
        sentences = getSentences(text)
        triples = []
        entity_pairs = []
        relations = []
        for sent in sentences:
            entity_pairs.append(get_entities(sent))
            relations.append(get_relation(sent))

        # extract subject
        source = [i[0] for i in entity_pairs]

        # extract object
        target = [i[1] for i in entity_pairs]

        kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    if not args.no_save:
        fname = '_'.join(url.rsplit('/',2)[-2:])
        kg_df.to_csv('data/'+fname+'.csv',index = False)
    if not args.no_graph:
        condition = np.array([True]*len(kg_df))
        if args.source:
            condition &= kg_df['source'].isin(args.source)
        if args.target:
            condition &= kg_df['target'].isin(args.target)
        if args.edge:
            condition &= kg_df['edge'].isin(args.edge)
        G=nx.from_pandas_edgelist(kg_df[condition], "source", "target",
                            edge_attr=True, create_using=nx.MultiDiGraph())


    plt.figure(figsize=(12,8))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    plt.show()
