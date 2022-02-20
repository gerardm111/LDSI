import json
from random import sample
import os
import luima_sbd.sbd_utils as luima
import matplotlib.pyplot as plt

#corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
#data = json.load(open(corpus_fpath))

# dataset randomly splitting
def dataset_splitting(data, outcome_type):
    seq = []
    for document in data['documents']:
        if document['outcome']==outcome_type:
            seq.append(document['_id'])
    selec = sample(seq, 14)
    selec_test = sample(selec, 7)
    selec_dev = list(set(selec)-set(selec_test))

    print("test "+outcome_type)
    for elem in selec_test:
        print('- ' +elem)
    print("dev "+outcome_type)
    for elem in selec_dev:
        print('- ' + elem)

#dataset_splitting(data, 'granted')
#dataset_splitting(data, 'denied')
##test set & dev set IDs are stored in curated_annotations_split.yml

def unlabeled_preprocessing(folder):
    """
    :param folder: path to the folder containing the data (ie. .txt files)
    :type folder: str

    :output:    1: number of sentence for each document (dict: filename -> int)
                2: total number of documents (int)
    """
    nb_sentence_per_doc = {}
    total_nb = 0
    dict = {}
    cpt = 1
    for filename in os.listdir(folder):
        if filename[-4:] == ".txt":
            print(filename, " in progress: ", cpt, "/30 000", end="\r")
            with open(folder+'/'+filename, encoding='ISO-8859-1') as f:
                text = f.read()       
                list_sentences = luima.text2sentences(text, offsets=False)
                nb= len(list_sentences)
                nb_sentence_per_doc[filename]= nb
                total_nb += nb
                dict[filename]=list_sentences
        cpt += 1
    with open('unlabeled/ldsi_unlabeled_annotations.json', "a+") as file:
        json.dump(dict, file, indent=2)
    print('---All sentences segmented in all files')
    return total_nb, nb_sentence_per_doc


def display_histogram(total_nb_sent, dict_nb_sent_per_doc):
    print("Total number of sentences in the unlabeled corpus", total_nb_sent)
    list_of_sent_nb = []
    for key in dict_nb_sent_per_doc:
        list_of_sent_nb.append(dict_nb_sent_per_doc[key])
    plt.hist(list_of_sent_nb, bins=100)
    plt.title('Frequency histogram of sentence number')
    plt.show()

total_nb_sent, dict_nb_sent_per_doc = unlabeled_preprocessing('unlabeled')
display_histogram(total_nb_sent, dict_nb_sent_per_doc)

def tokenizer(sentence):
    """
    :param sentence: sentence to tokenize
    :type sentence: str

    :output: list of tokens from the sentence input
    """
    list = []
    return list