from preprocessing import spans_add_tokens
from data_loader import train_data_loader, make_span_data
from classification_metrics import plot_confusion_matrix

import json
import numpy as np
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fasttext
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def make_feature_vectors_and_labels(spans):
    # function takes long to execute
    # note: we un-sparse the matrix here to be able to manipulate it
    list_nb_of_tokens = []
    fasttext_vectorizer = []
    for sent in spans:
        temp_feature = []
        list_nb_of_tokens.append(len(sent['tokens_spacy']))
        for token in sent['tokens_spacy']:
            temp_feature.append(model.get_word_vector(token))
        np.array(temp_feature)
        fasttext_vectorizer.append(np.average(temp_feature, axis=0)) #possible to weight it with tfidf
    mean_nb_tokens = mean(list_nb_of_tokens)
    std_nb_tokens = np.std(np.array(list_nb_of_tokens))
    fasttext_vectorizer  =np.array(fasttext_vectorizer)

    print("fasttext vect: ", fasttext_vectorizer.shape)
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    print("starts: ", starts_normalized.shape)
    nb_tokens_normalized = np.array([(len(s['tokens_spacy'])-mean_nb_tokens)/std_nb_tokens for s in spans])
    print("number: ", nb_tokens_normalized.shape)
    y = np.array([s['type'] for s in spans])
    temp = np.concatenate((fasttext_vectorizer, np.expand_dims(starts_normalized, axis=1)), axis=1)
    X = np.concatenate((temp, np.expand_dims(nb_tokens_normalized, axis=1)), axis=1)
    print("X and y created")
    return X, y

def make_feature_vectors(spans, model):
    # function takes long to execute
    # note: we un-sparse the matrix here to be able to manipulate it
    list_nb_of_tokens = []
    fasttext_vectorizer = []
    for sent in spans:
        temp_feature = []
        list_nb_of_tokens.append(len(sent['tokens_spacy']))
        for token in sent['tokens_spacy']:
            temp_feature.append(model.get_word_vector(token))
        np.array(temp_feature)
        fasttext_vectorizer.append(np.average(temp_feature, axis=0)) #possible to weight it with tfidf
    mean_nb_tokens = mean(list_nb_of_tokens)
    std_nb_tokens = np.std(np.array(list_nb_of_tokens))
    fasttext_vectorizer  =np.array(fasttext_vectorizer)

    print("fasttext vect: ", fasttext_vectorizer.shape)
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    nb_tokens_normalized = np.array([(len(s['tokens_spacy'])-mean_nb_tokens)/std_nb_tokens for s in spans])
    temp = np.concatenate((fasttext_vectorizer, np.expand_dims(starts_normalized, axis=1)), axis=1)
    X = np.concatenate((temp, np.expand_dims(nb_tokens_normalized, axis=1)), axis=1)
    print("X created")
    return X


"""
model = fasttext.load_model("model_fasttext.bin")

corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
data = json.load(open(corpus_fpath))
print("****TRAIN****")
data_train = train_data_loader(data, "labeled/curated_annotations_split.yml")
train_spans, train_spans_labels, train_spans_txt = make_span_data(data_train)
spans_add_tokens(train_spans)
train_X, train_y = make_feature_vectors_and_labels(train_spans)
print(f'{train_X.shape} {train_y.shape}')
print("****DEV****")
data_dev = train_data_loader(data, "labeled/curated_annotations_split.yml", set_of_data="dev")
dev_spans, dev_spans_labels, dev_spans_txt = make_span_data(data_dev)
spans_add_tokens(dev_spans)
dev_X, dev_y = make_feature_vectors_and_labels(dev_spans)
print(f'{dev_X.shape} {dev_y.shape}')

#clf = tree.DecisionTreeClassifier(max_depth=12)
#clf = LogisticRegression()
#clf = RandomForestClassifier(max_depth=12)
clf = make_pipeline(StandardScaler(), SVC(kernel='poly', gamma='auto'))
clf = clf.fit(train_X, train_y)
#clf2 = RandomForestClassifier()
#clf2 = clf2.fit(train_X, train_y)
print('TRAIN LR:\n'+classification_report(train_spans_labels, clf.predict(train_X)))
print('DEV TEST LR:\n'+classification_report(dev_spans_labels, clf.predict(dev_X)))
plot_confusion_matrix(dev_spans_labels, clf.predict(dev_X), classes=list(clf.classes_),
                      title='Confusion matrix for dev data (Logistic regression)')
print(clf.get_params())

print('TRAIN RF:\n'+classification_report(train_spans_labels, clf2.predict(train_X)))
print('DEV TEST RF:\n'+classification_report(dev_spans_labels, clf2.predict(dev_X)))
plot_confusion_matrix(dev_spans_labels, clf2.predict(dev_X), classes=list(clf2.classes_),
                      title='Confusion matrix for dev data (Random Forest)')
forest_trees = [estimator.tree_.max_depth for estimator in clf2.estimators_]
print(len(forest_trees), forest_trees)

plt.show()
"""