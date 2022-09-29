

from cgitb import text
from src import Normalization as norm
from src import GraphExtraction as grext
import scipy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as img
from itertools import combinations
from collections import Counter


def coocur_words_pos(text_data_tokens):
    text_data_pos = norm.part_of_speech_tag(text_data_tokens)

def coocur_words(text_data_tokens, window_size=1):
    vocabulary = {}
    data, row, col = [], [], []
    for pos, token in enumerate(text_data_tokens):
        #print(pos, token)
        i = vocabulary.setdefault(token, len(vocabulary))
        start = max(0, pos-window_size)
        end = min(len(text_data_tokens), pos+window_size+1)
        for pos2 in range(start, end):
            if pos2 == pos:
                continue
            j = vocabulary.setdefault(text_data_tokens[pos2], len(vocabulary))
            data.append(1.)
            row.append(i)
            col.append(j)

    return vocabulary, (data, (row, col))

def coocur_words_v2():
    text_data = [norm.clean_text(sen) for sen in text_data]
    vocabulary = set(norm.word_tokenize(' '.join(text_data)))
    text_data_tokens_list = [norm.word_tokenize(sen) for sen in text_data]
    print('vocab: ',vocabulary)
    print('text_data_tokens_list: ', text_data_tokens_list)
    co_occ = {ii:Counter({jj:0 for jj in vocabulary if jj!=ii}) for ii in vocabulary}
    k=2
    #print(co_occ)
    for sen in text_data_tokens_list:
        for ii in range(len(sen)):
            if ii < k:
                c = Counter(sen[0:ii+k+1])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c
            elif ii > len(sen)-(k+1):
                c = Counter(sen[ii-k::])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c
            else:
                c = Counter(sen[ii-k:ii+k+1])
                del c[sen[ii]]
                co_occ[sen[ii]] = co_occ[sen[ii]] + c

    co_occ = {ii:dict(co_occ[ii]) for ii in vocabulary}
    

def main(text_data, params):
    #text_data = [norm.clean_text(sen) for sen in text_data]
    text_data = norm.clean_text(text_data) 
    #print(text_data)
    vocabulary = set(norm.word_tokenize(text_data))
    text_data_tokens = norm.word_tokenize(text_data)
    print('vocab: ',vocabulary)
    #print('vocab: ',text_data_tokens)

    #if (params['type'] == 'words'):
    vocabs, co_occ = coocur_words(text_data_tokens, params['window_size'])
    co_occ_matrix_sparse = scipy.sparse.coo_matrix(co_occ)
    df_co_occ  = pd.DataFrame(co_occ_matrix_sparse.todense(),
                        index=vocabs.keys(),
                        columns = vocabs.keys())

    df_co_occ = df_co_occ.sort_index()[sorted(vocabs.keys())]
    df_co_occ.style.applymap(lambda x: 'color: red' if x>0 else '')
    #print(df_co_occ)
    graph = grext.build_from_numpy_matrix(df_co_occ)
    graph = grext.relabel_nodes(graph, dict(enumerate(df_co_occ.columns)))
    return graph
    #grext.show_graph(graph)

    #if (params['type'] == 'words_pos'):
    #    coocur_words_pos(text_data_tokens)

