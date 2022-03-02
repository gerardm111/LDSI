import json
from random import sample, choice
import os
import sys
sys.path.append('/home/mahaut/LDSI/luima_sbd')
import luima_sbd.sbd_utils as luima
import matplotlib.pyplot as plt
import spacy
from spacy.attrs import ORTH

#corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
#data = json.load(open(corpus_fpath))

# dataset randomly splitting
def dataset_splitting(data, outcome_type):
    """
    :param data:
    :type data: dictionary
    :param outcome_type: 'granted' or 'denied' cases
    :type outcom_type: str (one of the two above values)

    :output: none
    """
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


def display_histogram_from_dict(total_nb_sent, dict_nb_sent_per_doc):
    print("Total number of sentences in the unlabeled corpus", total_nb_sent)
    list_of_sent_nb = []
    for key in dict_nb_sent_per_doc:
        list_of_sent_nb.append(dict_nb_sent_per_doc[key])
    plt.hist(list_of_sent_nb, bins=100)
    plt.title('Frequency histogram of sentence number')
    plt.show()

#total_nb_sent, dict_nb_sent_per_doc = unlabeled_preprocessing('unlabeled')
#display_histogram_from_dict(total_nb_sent, dict_nb_sent_per_doc)

def tokenizer(doc_sentences, nlp):
    """
    :param sentence: sentence to tokenize
    :type sentence: str
    :param nlp: nlp from spacy

    :output: dict of tokens from the doc input with one key per sentence
    """
    clean_tokens = {}
    docs = nlp.pipe(doc_sentences, n_process=4)
    cpt = 0
    for doc in docs:
        clean_tokens[cpt]=[]
        tokens = list(doc)
        par_removed = ''
        for t in tokens:
            if t.pos_ == 'PUNCT':
                pass
            elif t.pos_ == 'NUM':
                clean_tokens[cpt].append(f'<NUM{len(t)}>')
            elif t.lemma_ == "'s":
                pass
            elif '(' in t.lemma_:
                par_split = t.lemma_.split('(')
                for elem in par_split:
                    par_removed = par_removed + elem
                par_split = tokenizer([par_removed], nlp)
                for elem in par_split:
                    clean_tokens[cpt].append(elem)
            elif "\n" in t.lemma_:
                par_split = t.lemma_.split('\n')
                for elem in par_split:
                    if elem != ' ' and elem != '':
                        par_removed = par_removed + ' ' + elem
                par_split = tokenizer([par_removed], nlp)
                for elem in par_split:
                    clean_tokens[cpt].append(elem)
            else:
                clean_tokens[cpt].append(t.lemma_.lower())
        cpt += 1
    return clean_tokens

def tokenize_documents(path):
    """
    :param path: path to the json file containing segmented sentences for each document
    :type path: str

    :output: list of int: number of tokens in each sentence
    """
    f = open(path)
    doc = json.load(f)
    f.close()
    nb_tokens_list = []
    cpt1 = 0

    # Create our nlp instance
    ## TO DO: put it in another function
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
    nlp.tokenizer.add_special_case('Veterans Law Judge', [{ORTH: 'Veterans Law Judge'}])
    nlp.tokenizer.add_special_case('Veterans Affairs', [{ORTH: 'Veterans Affairs'}])
    nlp.tokenizer.add_special_case("Veterans' Appeals", [{ORTH: "Veterans' Appeals"}])
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[{"TEXT": "["}], [{"TEXT": "\n"}], [{"TEXT": "'"}], [{"TEXT": "\r"}], [{"TEXT": "\t"}]]
    attrs = {"POS": "PUNCT"}
    ruler.add(patterns=patterns, attrs=attrs, index=0)
    nlp.disable_pipes('parser')

    for key in doc:
        print("In progress, doc ", cpt1, "/30 000", end="\r")
        doc_tokens_sent = tokenizer(doc[key], nlp)
        for sent_key in doc_tokens_sent:
            nb_tokens_list.append(len(doc_tokens_sent[sent_key]))
        with open('unlabeled/ldsi_unlabeled_annotations_tokens.json', "a+") as file:
            file.write(',\n')
            file.write(""" " """ +str(key)+""" " """+':')
            json.dump(doc_tokens_sent, file, indent=2)
        cpt1 += 1
    print("---All documents sentences tokenized! ")
    return nb_tokens_list

def one_sentence_tokenizer(txt, nlp):
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    par_removed = ''
    for t in tokens:
        if t.pos_ == 'PUNCT':
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        elif t.lemma_ == "'s":
            pass
        elif t.lemma_ == "'" or t.lemma_ == "\n" or t.lemma_ == "\r" or t.lemma_ == "\t":
            pass
        elif '(' in t.lemma_:
            par_split = t.lemma_.split('(')
            for elem in par_split:
                par_removed = par_removed + elem
            par_split = one_sentence_tokenizer(par_removed, nlp)
            for elem in par_split:
                clean_tokens.append(elem)
        elif "\n" in t.lemma_:
            par_split = t.lemma_.split('\n')
            for elem in par_split:
                if elem != ' ' and elem != '':
                    par_removed = par_removed + ' ' + elem
            par_split = one_sentence_tokenizer(par_removed, nlp)
            for elem in par_split:
                clean_tokens.append(elem)
        else:
            clean_tokens.append(t.lemma_.lower())
    return clean_tokens

def spans_add_tokens(spans):
    cpt = 0
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
    nlp.tokenizer.add_special_case('Veterans Law Judge', [{ORTH: 'Veterans Law Judge'}])
    nlp.tokenizer.add_special_case('Veterans Affairs', [{ORTH: 'Veterans Affairs'}])
    nlp.tokenizer.add_special_case("Veterans' Appeals", [{ORTH: "Veterans' Appeals"}])
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[{"TEXT": "["}], [{"TEXT": "\n"}], [{"TEXT": "'"}], [{"TEXT": "\r"}], [{"TEXT": "\t"}]]
    attrs = {"POS": "PUNCT"}
    ruler.add(patterns=patterns, attrs=attrs, index=0)
    nlp.disable_pipes('parser')
    for s in spans:
        s['tokens_spacy'] = one_sentence_tokenizer(s['txt'], nlp)
        cpt += 1
        print("span: ", cpt, " / 14 291", end="\r")
    print("---Number of tokens key added to spans")

def display_histogram_from_list(list_nb_sent_per_doc):
    plt.hist(list_nb_sent_per_doc, bins=100)
    plt.title('Frequency histogram of tokens number')
    plt.show()

#display_histogram_from_list(tokenize_documents('unlabeled/ldsi_unlabeled_annotations.json'))

def write_tokenized_sentences_randomly(path):
    """
    :param path: path to the json file containing tokenized sentences for each document
    :type path: str

    :output: None
    """
    # readind the json file
    f = open(path)
    corpus = json.load(f)
    cpt = 0
    while (len(corpus)) != 0:
        #choose randomly a doc
        print("sentence ", cpt, " / 3360495", end="\r")
        random_key = choice(list(corpus))
        # choose randomly a sentence
        random_sent = choice(list(corpus[random_key]))
        # remove the sentence
        temp = corpus[random_key].pop(random_sent)
        ## write this sentence if nb_tokens >= 5 in txt file with white spaces between each tokens
        if len(temp) >= 5:
            with open('unlabeled/tokenized_sentences_2.txt', 'a+') as file:
                for elem in temp:
                    file.write(str(elem) + ' ')
                file.write('\n')
        ## if last sentence of doc remove doc
        if len(corpus[random_key]) == 0:
            del corpus[random_key]
        cpt += 1
    print("---All tokenized sentences written in txt file")
    f.close()

#write_tokenized_sentences_randomly('unlabeled/ldsi_unlabeled_annotations_tokens.json')