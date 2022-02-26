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
import fasttext
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
from transformers import  BertForSequenceClassification, BertTokenizerFast
from transformers import Trainer, TrainingArguments

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

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 512
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_spans_labels))
class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def conversion_labels_to_num(labels_set):
    converted = []
    dict_conversion = {'CaseFooter': 1, 'CaseHeader': 2, 'CaseIssue': 3,
                        'Citation': 4, 'ConclusionOfLaw': 5, 'Evidence': 6,
                        'EvidenceBasedOrIntermediateFinding': 7,
                        'EvidenceBasedReasoning': 8, 'Header': 9, 'LegalRule': 10,
                        'LegislationAndPolicy': 11, 'PolicyBasedReasoning': 12, 
                        'Procedure': 13, 'RemandInstructions': 14}
    for label in labels_set:
        converted.append(dict_conversion[label])
    
    return converted

# convert our tokenized data into a torch Dataset
dict_train_X = {'input_ids': train_X.tolist()}
print(dict_train_X)
dict_dev_X = {'input_ids': dev_X.tolist()}
train_dataset = NewsGroupsDataset(dict_train_X, conversion_labels_to_num(train_y))
valid_dataset = NewsGroupsDataset(dict_dev_X, conversion_labels_to_num(dev_y))
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
trainer = Trainer(
    model=bert_model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)
# train the model
trainer.train()
# evaluate the current model after training
trainer.evaluate()