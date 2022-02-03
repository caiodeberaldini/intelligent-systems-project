import os
import re
import sys
import time
import spacy
import pickle
import string
import unidecode

import pandas as pd

pt_br_stop_words = []
with open('/usr/src/data/stop_words.txt', 'r') as f:
    pt_br_stop_words = (
        [word if len(word.split(' ')) == 1 else word.split(' ')[1] for word in f.read().split(',')]
    )

    f.close()

nlp = spacy.load('pt_core_news_sm')

def processText(df_column, stop_words=pt_br_stop_words, lemma_dict=nlp):

    # Disable case sensitivity
    df_column = df_column.apply(
        lambda seq: ' '.join([word.lower() for word in seq.split(' ')])
    )

    # Removing stop words
    df_column = df_column.apply(
        lambda seq: ' '.join([word for word in seq.split(' ') if word not in pt_br_stop_words])
    )

    # Remove numbers
    df_column = df_column.apply(
        lambda seq: ' '.join([re.sub(r'\d+', '', word) for word in seq.split(' ')])
    )

    # Remove punctuation marks
    df_column = df_column.apply(
        lambda seq: ' '.join([
            word.translate(
                str.maketrans('','', string.punctuation)) for word in seq.split(' ')
        ])
    )

    # Remove accent marks
    df_column = df_column.apply(
        lambda seq: ' '.join([unidecode.unidecode(word) for word in seq.split(' ')])
    )

    # Remove duplicates
    df_column = df_column.apply(
        lambda seq: ' '.join(list(set(seq.split(' '))))
    )

    # Lemmatization
    df_column = df_column.apply(
        lambda seq: ' '.join([
            word.lemma_ if word.pos_ == 'VERB' else str(word) for word in lemma_dict(seq) 
        ])
    )

    # Remove single char words
    df_column = df_column.apply(
        lambda seq: ' '.join([
            word for word in seq.split(' ') if len(word) > 1
        ])
    )

    return df_column