from src.GraphModels import Cooccurence
from src.GraphModels import ISG
from src import GraphExtraction as grext
from sklearn.model_selection import train_test_split
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from collections import Counter
import networkx as nx
import json
import pandas as pd
import numpy as np
import random
import re
import emoji
import nltk



RM_HTML = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
RM_BLANK_SPACES = re.compile(r'^\s+|\s+$')
PUNCT = r"[!@#$%^&*()[]{};:,./<>?\|`~-=_+]"
PATTERN_TOKENIZER = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
CUSTOM_TOKENIZER = RegexpTokenizer(PATTERN_TOKENIZER)

def text_to_graph(corpus_data, graph_rep_type, graph_config_params):
    graphs_representation_options = {
        'co-occurrence-word': Cooccurence.main,
        'co-occurrence-pos': Cooccurence.main, #lexico
        'integrated-syntactic-graph': ISG.main #lexico, semantico, sintactico, morfologico
    }
    return graphs_representation_options[graph_rep_type](corpus_data, graph_config_params)


def read_dataset():
    main_path = 'datasets/PAN_22/'
    pairs_path = main_path + 'pairs.jsonl'
    truth_path = main_path + 'truth.jsonl'
    pairs_data, truth_data = {}, {}
    cnt = 0
    with open(pairs_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            pairs_data[json_line['id']] = {
                'id': json_line['id'],
                'discourse_types':  json_line['discourse_types'], 
                'pair': json_line['pair'] 
            }
    with open(truth_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            pairs_data[json_line['id']]['same'] = json_line['same']
            pairs_data[json_line['id']]['authors'] = json_line['authors']

    pairs_data = [pair_value for pair_value in pairs_data.values()]
    # cutoff dataset for dev purposes
    pairs_data = pairs_data[:int(len(pairs_data)*0.01)] 
    return pairs_data

def handle_contractions(text):
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def normalization(text):
    # emoticons
    clean_text = emoji.demojize(text, delimiters=(" emoji_", " "))
    # punctuation
    # html tags
    clean_text = re.sub(RM_HTML, ' ', clean_text)
    #clean_text = re.sub(PUNCT, 'punct', clean_text)
    # contractions
    #clean_text = handle_contractions(clean_text)
    # blank spaces
    clean_text = re.sub(RM_BLANK_SPACES, '', clean_text)
    # non ascii char
    clean_text = unidecode(clean_text)
    # lowercase
    clean_text = clean_text.lower()
    # sentence tokenizer
    clean_sent_tokens = nltk.sent_tokenize(clean_text)
    # work tokenizer
    clean_word_tokens = []
    for sent in clean_sent_tokens:
        words_tokens = [re.sub(RM_BLANK_SPACES, '', t) for t in CUSTOM_TOKENIZER.tokenize(sent)]
        clean_word_tokens.extend(words_tokens)
    clean_word_tokens = list(set(clean_word_tokens))
    # pos tags
    clean_word_tokens_tags = nltk.pos_tag(clean_word_tokens) 
    # set node_structure
    nodes = [(t[0], {'pos_tag': t[1]}) for t in clean_word_tokens_tags]
    return nodes

def co_occurrence(corpus, window_size=1):
    for d in corpus[:]:
        d_cocc = defaultdict(int)
        vocab = set()
        doc = [*d['nodes'][0]]
        #doc = [*d['nodes'][0], *d['nodes'][1]]
        for i in range(len(doc)):
            word = doc[i][0]
            next_word = doc[i+1 : i+1+window_size]
            vocab.add(word)
            for t in next_word:
                #key = tuple(sorted([t[0], word]))
                key = tuple([word, t[0]])
                d_cocc[key] += 1
                #print(key)

        #print(len(vocab), len(d_cocc))
        d['edges'] = []
        for key, value in d_cocc.items():
            d['edges'].append((key[0], key[1], {'freq': value}))
        #print(d['edges'])
    
    return corpus
        
    #edges = [(node_pre, node_act, (c / total))
    #            for (node_pre, node_act), c in Counter(edges_list).items()]
    #vocab = sorted(vocab) # sort vocab
    '''df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d_cocc.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    print(df)
    print(df.info())
    return df'''

if __name__ == "__main__":
    '''#co-ocurr types: words, words_pos
    corpus_data = 'Momo, also known as The Grey Gentlemen or The Men in Grey, First, each text'
    graph_config_params = {'graph_type': 'undirected', 'window_size': 1}
    G = text_to_graph(corpus_data, graph_rep_type='co-occurrence-word', graph_config_params=graph_config_params)
    print(type(G), G)
    #print(grext.to_adjacency_matrix(G))
    grext.show_graph(G)'''

    # Testing PAN 22
    # 1) *** Read dataset
    data = read_dataset()
    print('dataset_len: ', len(data))

    # 2) *** Split dataset
    x_train, x_test = train_test_split(data,test_size=0.3) 
    print('x_train: ', len(x_train))
    print('x_test: ', len(x_test))
    #print("x_train[0]: \n", x_train[0])
    
    # 3) *** Normalization
    x_train_norm = []
    for x in x_train[:]:
        #if 'email' in x['discourse_types']:
        x_pairs_1 = normalization(x['pair'][0])
        x_pairs_2 = normalization(x['pair'][1])
        x['nodes'] = [x_pairs_1, x_pairs_2]
        x_train_norm.append(x)
    #print("x_train_norm[0]: \n", x_train_norm[0])
    
    # 4) *** Text2Graph
    x_train_cooc = co_occurrence(x_train_norm[:1])
    #print("x_train_cooc: \n", x_train_cooc[0]['nodes'])

    # 5) *** Build graph
    G = nx.Graph()
    G.add_nodes_from(x_train_cooc[0]['nodes'][0])
    G.add_edges_from(x_train_cooc[0]['edges'])
    print('nodes:\n', list(G.nodes.data())[:10])
    print('edges: \n', list(G.edges.data())[:10])
    A = nx.adjacency_matrix(G)
    print(A.todense())
    # 6) Graph Neural Network, learn embeddings

    # Clasification