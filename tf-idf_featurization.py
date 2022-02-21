import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import random

from data_loader import train_data_loader, make_span_data

def featurize(spans_txt):
    # spans_txt = list of sentences
    # spans = dict with for each sentence: document, type, start, end
    vectorizer = TfidfVectorizer(min_df=5)
    vectorizer = vectorizer.fit(spans_txt)
    tfidf_features_skl = vectorizer.get_feature_names()
    tfidf_skl = vectorizer.transform(spans_txt).toarray()
    return tfidf_features_skl, tfidf_skl

def top_tfidf_features(row, features, top_n=15):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_features_in_doc(Xtr, features, row_id, top_n=15):
    ''' Top tfidf features in specific document (matrix row) '''
    xtr_row = Xtr[row_id]
    if type(xtr_row) is not np.ndarray:
        xtr_row = xtr_row.toarray()
    row = np.squeeze(xtr_row)
    return top_tfidf_features(row, features, top_n)


def top_mean_features(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids]
    else:
        D = Xtr
    if type(D) is not np.ndarray:
        D = D.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_features(tfidf_means, features, top_n)


def top_features_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = {}
    labels = np.unique(y)
    y = np.array(y)
    for label in labels:
        ids = np.nonzero(y==label)
        feats_df = top_mean_features(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs[label] = feats_df
    return dfs


def span_top_tfidf(spans_txt, spans_tfidf, features, index):
    print('span text:\n'+spans_txt[index]+'\n')
    print(top_features_in_doc(spans_tfidf, features, index))

## TO DO: find what is spans_txt (train_spans_txt from workshop) & spans (train_spans from workshop)
corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
data = json.load(open(corpus_fpath))
data_train = train_data_loader(data, "labeled/curated_annotations_split.yml")
data_dev = train_data_loader(data, "labeled/curated_annotations_split.yml", set_of_data="dev")
data_test = train_data_loader(data, "labeled/curated_annotations_split.yml", set_of_data="test")

train_spans, train_spans_labels, train_spans_txt = make_span_data(data_train)
dev_spans, dev_spans_labels, dev_spans_txt = make_span_data(data_dev)
test_spans, test_spans_labels, test_spans_txt = make_span_data(data_test)

train_tfidf_features_skl, train_tfidf_skl = featurize(train_spans_txt)
dev_tfidf_features_skl, dev_tfidf_skl = featurize(dev_spans_txt)
test_tfidf_features_skl, test_tfidf_skl = featurize(test_spans_txt)

#print(train_tfidf_skl.shape)
span_top_tfidf(train_spans_txt, train_tfidf_skl, train_tfidf_features_skl, random.randint(0, len(train_spans)))
dfs = top_features_by_class(train_tfidf_skl, train_spans_labels, train_tfidf_features_skl)
#print(dfs.keys())
#print(dfs['Citation'])