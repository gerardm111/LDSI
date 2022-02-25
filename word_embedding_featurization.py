from preprocessing import one_sentence_tokenizer
from data_loader import train_data_loader, make_span_data

import json
import numpy as np
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree

def vectorization(spans_txt):
    spacy_tfidf_vectorizer = TfidfVectorizer(tokenizer=one_sentence_tokenizer,
                                         min_df=3,
                                         ngram_range=(1,1))
    spacy_tfidf_vectorizer = spacy_tfidf_vectorizer.fit(spans_txt)
    #tfidf_features_spacy = spacy_tfidf_vectorizer.get_feature_names()
    print("Vectorizer done")
    return spacy_tfidf_vectorizer

corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
data = json.load(open(corpus_fpath))
data_train = train_data_loader(data, "labeled/curated_annotations_split.yml")
train_spans, train_spans_labels, train_spans_txt = make_span_data(data_train)
spacy_tfidf_vectorizer = vectorization(train_spans_txt)

def make_feature_vectors_and_labels(spans, vectorizer):
    # function takes long to execute
    # note: we un-sparse the matrix here to be able to manipulate it
    list_nb_of_tokens = []
    for elem in spans:
        list_nb_of_tokens.append(len(elem['tokens_spacy']))
    mean_nb_tokens = mean(list_nb_of_tokens)
    std_nb_tokens = np.std(np.array(list_nb_of_tokens))

    tfidf = spacy_tfidf_vectorizer.transform([s['txt'] for s in spans]).toarray()
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    nb_tokens_normalized = np.array([(len(s['tokens_spacy'])-mean_nb_tokens)/std_nb_tokens for s in spans])
    y = np.array([s['type'] for s in spans])
    temp = np.concatenate((tfidf, np.expand_dims(starts_normalized, axis=1)), axis=1)
    X = np.concatenate((temp, np.expand_dims(nb_tokens_normalized, axis=1)), axis=1)
    print("X and y created")
    return X, y

train_X, train_y = make_feature_vectors_and_labels(train_spans, spacy_tfidf_vectorizer)
print(f'{train_X.shape} {train_y.shape}')

clf = tree.DecisionTreeClassifier(max_depth=12)
clf = clf.fit(train_X, train_y)