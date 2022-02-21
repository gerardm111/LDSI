import spacy
import json
from statistics import mean, stdev
from spacy.language import Language

from data_loader import train_data_loader
import luima_sbd.sbd_utils as luima

def test1(path):
    #nlseg = NewLineSegmenter()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('sentencizer')
    text = open(path).read()
    doc = nlp(text)
    for sent in doc.sents:
        print('***')
        print(sent.text)
        print(len(sent.text))
#test1('test.txt')

corpus_fpath = 'labeled/ldsi_w21_curated_annotations_v2.json'
data = json.load(open(corpus_fpath))
data_train = train_data_loader(data, "labeled/curated_annotations_split.yml")
#print(len(data_train['documents']))

@Language.component('spatial_separators')
def set_custom_Sentence_end_points_spatial(doc):
    for token in doc[:-1]:
        if " \r\n\r" in token.text:
            doc[token.i].is_sent_start = True
        elif "\r\n\r" in token.text:
            doc[token.i+1].is_sent_start = True
        elif "\t" in token.text:
            doc[token.i+1].is_sent_start = False
    return doc

@Language.component('footnote')
def set_custom_Sentence_end_points_footnote(doc):
    for token in doc[:-1]:
        if "_" in token.text:
            doc[token.i+1].is_sent_start = False
    return doc

def sentence_segment_starting_ind(data, type_of_segmenter):
    """
    :param data: dict corresponding to the document corpus
    :type data: dictionary
    :param type_of_segmenter: "standard" for standard, "enhanced" for standard with custom fundtions, "luima" for the luima_sbd
    :type type_of_segmenter: str

    :output: dictionary with starting sentence index for each document
    """
    if type_of_segmenter != "standard" and type_of_segmenter != "enhanced" and type_of_segmenter != "luima":
        print("Type of segmenter in function 'sentence_segment_starting_ind' wrong -> luima used by default !!!")
    starting_indexes = {}
    if type_of_segmenter != 'luima':
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe('sentencizer')
        if type_of_segmenter == 'enhanced':
            nlp.add_pipe('spatial_separators', before='parser')
            nlp.add_pipe('footnote', before='parser')
            print("Enhanced segmenter used")
        for document in data['documents']:
            cpt_string = 0
            text = document['plainText']
            doc = nlp(text)
            print(document['_id'], " in progress", end="\r")
            for sentence in doc.sents:
                if document['_id'] in starting_indexes:
                    starting_indexes[document['_id']].append(cpt_string)
                else:
                    starting_indexes[document['_id']]=[cpt_string]
                cpt_string +=len(sentence.text)
    else:
        print("Luima segmenter used")
        for document in data['documents']:
            temp = luima.text2sentences(document['plainText'], offsets=True)
            print(document['_id'], " in progress", end="\r")
            for elem in temp:
                if document['_id'] in starting_indexes:
                    starting_indexes[document['_id']].append(elem[0])
                else:
                    starting_indexes[document['_id']]=[elem[0]]
    print('---All sentences segmented in documents')
    return starting_indexes


def create_list_of_sentence_boundaries(data):
    """
    :param data: dict corresponding to the document corpus, including sentence annotation
    :type data: dictionary

    :output: dictionary with starting sentence index for each document
    """
    sentence_boundaries = {}
    for annotation in data['annotations']:
        document = annotation['document']
        if document in sentence_boundaries:
            sentence_boundaries[document].append(annotation['start'])
        else:
            sentence_boundaries[document]=[annotation['start']]
    for key in sentence_boundaries:
        sentence_boundaries[key].sort()
    return sentence_boundaries

generated_split= sentence_segment_starting_ind(data_train, "luima")
true_split = create_list_of_sentence_boundaries(data_train)

def scores_computation(true, prediction):
    """
    :param true: for one document, true list of sentence start
    :type true: list
    :param prediction: for one document, predicted list of sentence start
    :type prediction: list

    :output: precision, recall and f1 score
    """
    true_positiv = 0
    false_positiv = 0
    false_negativ = 0
    for elem in prediction:
        if elem in set(true):
            true_positiv += 1
        elif elem + 1 in set(true):
            true_positiv += 1
        elif elem + 2 in set(true):
            true_positiv += 1
        elif elem + 3 in set(true):
            true_positiv += 1
        elif elem - 1 in set(true):
            true_positiv += 1
        elif elem - 2 in set(true):
            true_positiv += 1
        elif elem - 3 in set(true):
            true_positiv += 1
        else:
            false_positiv += 1
    false_negativ = len(true) - true_positiv
    precision = true_positiv / (false_positiv + true_positiv)
    recall = true_positiv / (false_negativ + true_positiv)
    f1 = (2*precision*recall) / (precision + recall)
    return precision, recall, f1

def dict_of_scores(true_split, generated_split):
    """
    :param true_split: dict containing for each document (key=document['_id']) the list of true sentence start
    :type true_split: dictionary
    :param generated_split: dict containing for each document (key=document['_id']) the list of generated sentence start
    :type generated_split: dictionary

    :output: dict containing precision, recall, f1 scores for each document
    """
    scores = {}
    for key in true_split:
        precision, recall, f1 = scores_computation(true_split[key], generated_split[key])
        scores[key]=[precision, recall, f1]
        print(key, " done", end="\r")
    print("---Precision, recall & f1 calculated for all documents")
    return scores

scores = dict_of_scores(true_split, generated_split)

#print(scores)
#print(true_split['61aea55c97ad59b4cfc412a1'])
#print(generated_split['61aea55c97ad59b4cfc412a1'])

def two_min(two_min_list, new_elem, two_min_list_id, new_elem_id):
    if new_elem < two_min_list[0]:
        two_min_list[1] = two_min_list[0]
        two_min_list_id[1] = two_min_list_id[0]
        two_min_list[0] = new_elem
        two_min_list_id[0] = new_elem_id
    elif new_elem < two_min_list[1]:
        two_min_list[1] = new_elem
        two_min_list_id[1] = new_elem_id
    return two_min_list, two_min_list_id

def listing_of_each_type_of_score(scores):
    """
    :param scores: dict with for each document precision, recall and f1 score
    :type scores: dictionary

    :output: lists of precision, of recall and of f1 scores & 2 min values of precision with ids & 2 min values of recall with ids
    """
    precisions = []
    min_precision = [1, 1]
    min_precision_id = ['', '']
    recalls = []
    min_recall = [1, 1]
    min_recall_id = ['', '']
    f1s = []
    for key in scores:
        precisions.append(scores[key][0])
        recalls.append(scores[key][1])
        f1s.append(scores[key][2])
        min_precision, min_precision_id = two_min(min_precision, scores[key][0], min_precision_id, key)
        min_recall, min_recall_id = two_min(min_recall, scores[key][1], min_recall_id, key)
    return precisions, recalls, f1s, min_precision, min_precision_id, min_recall, min_recall_id

precisions, recalls, f1s, min_precision, min_precision_id, min_recall, min_recall_id = listing_of_each_type_of_score(scores)
print("PRECISION--- mean: ", mean(precisions), " standard deviation: ", stdev(precisions))
print("RECALL--- mean: ", mean(recalls), " standard deviation: ", stdev(recalls))
print("F1--- mean: ", mean(f1s), " standard deviation: ", stdev(f1s))
print("2 min precision: ", min_precision, " IDs: ", min_precision_id)
print("2 min recall: ", min_recall, " IDs: ", min_recall_id)
#precisions.sort()
#recalls.sort()
#print("VERIF-- precision min: ", precisions[0], " recall min: ", recalls[0])